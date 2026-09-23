//! Integration tests for `greentic-dw-knowledge-chronicle`.
//!
//! Uses `chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm}` so no real
//! database or network is needed. Tests build a [`KnowledgeChronicle`] via
//! [`KnowledgeChronicle::from_parts`] and exercise the full ingest → search
//! round-trip through the [`Knowledge`] trait.

use std::sync::Arc;

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};
use greentic_dw_knowledge::{Knowledge, KnowledgeChunk, KnowledgeQuery};
use greentic_dw_knowledge_chronicle::KnowledgeChronicle;
use greentic_types::{EnvId, TenantCtx, TenantId};

const EMB_DIM: usize = 4;

fn tenant(id: &str) -> TenantCtx {
    TenantCtx::new(
        EnvId::try_from("dev").unwrap(),
        TenantId::try_from(id).unwrap(),
    )
}

async fn make_knowledge(search_limit: usize) -> KnowledgeChronicle {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![]));
    KnowledgeChronicle::from_parts(
        Arc::clone(&driver) as Arc<dyn chronicle_core::driver::GraphDriver>,
        llm,
        embedder,
        1,
        search_limit,
    )
    .await
    .unwrap()
}

fn chunk(doc_id: &str, index: usize, text: &str) -> KnowledgeChunk {
    KnowledgeChunk {
        doc_id: doc_id.to_string(),
        chunk_index: index,
        text: text.to_string(),
        metadata: serde_json::Map::new(),
        embedding: None,
    }
}

/// Ingest two chunks and assert we receive back exactly two non-empty chunk ids.
#[tokio::test]
async fn ingest_two_chunks_returns_two_ids() {
    let kb = make_knowledge(5).await;
    let t = tenant("tenant-integ");

    let chunks = vec![
        chunk("doc-a", 0, "First chunk of document A."),
        chunk("doc-a", 1, "Second chunk of document A."),
    ];

    let outcome = kb.ingest(&t, chunks).await.unwrap();
    assert_eq!(outcome.chunk_ids.len(), 2, "expected exactly 2 chunk ids");
    assert!(outcome.chunk_ids.iter().all(|id| !id.is_empty()));
}

/// Re-ingesting the same chunks must return the same deterministic UUIDs.
#[tokio::test]
async fn ingest_same_chunks_idempotent() {
    let kb = make_knowledge(5).await;
    let t = tenant("tenant-integ");

    let chunks = vec![chunk("doc-b", 0, "Idempotent text here.")];
    let first = kb.ingest(&t, chunks.clone()).await.unwrap();
    let second = kb.ingest(&t, chunks).await.unwrap();
    assert_eq!(
        first.chunk_ids, second.chunk_ids,
        "chunk ids must be stable across re-ingest"
    );
}

/// After ingestion, a search query must complete without error and return
/// properly mapped [`RetrievedChunk`] values (score ≥ 0, text non-empty).
#[tokio::test]
async fn search_after_ingest_no_error_and_mapped_fields() {
    let kb = make_knowledge(5).await;
    let t = tenant("tenant-integ");

    let chunks = vec![
        chunk("doc-c", 0, "Chronicle enables hybrid document retrieval."),
        chunk("doc-c", 1, "BM25 and cosine similarity are fused with RRF."),
    ];
    kb.ingest(&t, chunks).await.unwrap();

    let query = KnowledgeQuery {
        query: "hybrid document retrieval".to_string(),
        limit: Some(5),
    };
    let hits = kb.search(&t, query).await.unwrap();

    for hit in &hits {
        assert!(!hit.text.is_empty(), "hit text must be non-empty");
        assert!(
            hit.score >= 0.0,
            "hit score must be non-negative, got {}",
            hit.score
        );
        // Provenance fields are Optional[String] / Optional[usize] — just verify
        // they deserialise correctly (they come from node attributes in the FakeDriver).
    }
}

/// An empty text query must return an empty result list (Chronicle short-circuits
/// empty queries before touching the driver).
#[tokio::test]
async fn search_empty_query_returns_no_hits() {
    let kb = make_knowledge(5).await;
    let t = tenant("tenant-integ");
    let query = KnowledgeQuery {
        query: String::new(),
        limit: None,
    };
    let hits = kb.search(&t, query).await.unwrap();
    assert!(hits.is_empty(), "empty query must return empty hits");
}

/// Chunks ingested by tenant-A must not be visible to tenant-B.
#[tokio::test]
async fn cross_tenant_isolation() {
    let kb = make_knowledge(10).await;
    let a = tenant("tenant-aa");
    let b = tenant("tenant-bb");

    kb.ingest(&a, vec![chunk("doc-x", 0, "Private data for tenant AA.")])
        .await
        .unwrap();

    // Positive control: tenant-aa MUST see its own ingested chunk. Without this,
    // the isolation assertion below could pass vacuously (e.g. if retrieval were
    // silently broken and returned nothing for everyone).
    let hits_a = kb
        .search(
            &a,
            KnowledgeQuery {
                query: "Private data".to_string(),
                limit: Some(10),
            },
        )
        .await
        .unwrap();
    assert!(
        hits_a
            .iter()
            .any(|h| h.text == "Private data for tenant AA."),
        "tenant-aa must see its own data (positive control) — got {} hits",
        hits_a.len()
    );

    let hits_b = kb
        .search(
            &b,
            KnowledgeQuery {
                query: "Private data".to_string(),
                limit: Some(10),
            },
        )
        .await
        .unwrap();

    for hit in &hits_b {
        assert_ne!(
            hit.text, "Private data for tenant AA.",
            "tenant-bb must not see tenant-aa's data"
        );
    }
}

/// The `search_limit` field passed at construction must cap the result count.
#[tokio::test]
async fn search_limit_is_respected() {
    let limit = 2usize;
    let kb = make_knowledge(limit).await;
    let t = tenant("tenant-integ");

    // Ingest more chunks than the limit
    let chunks: Vec<KnowledgeChunk> = (0..5)
        .map(|i| {
            chunk(
                "doc-d",
                i,
                &format!("Chunk number {i} with some text for retrieval."),
            )
        })
        .collect();
    kb.ingest(&t, chunks).await.unwrap();

    let query = KnowledgeQuery {
        query: "chunk number".to_string(),
        limit: None, // use the provider's default (= limit)
    };
    let hits = kb.search(&t, query).await.unwrap();
    assert!(
        hits.len() <= limit,
        "result count {count} must not exceed search_limit={limit}",
        count = hits.len()
    );
}
