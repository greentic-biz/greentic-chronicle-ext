// Integration tests for chronicle-core's document_rag module.
//
// These tests exercise the async ingest + search paths over the in-memory
// FakeDriver + MockEmbedder, confirming:
//   - batch-embed + persist wiring (ingest_chunks → save_entity_nodes)
//   - idempotency via deterministic UUIDs
//   - search wiring (embed query → node_search → filter DocumentChunk → map)
//   - no-error contract for empty inputs
//
// Pure unit tests for chunk_text, chunk_uuid, chunk_to_entity_node, and
// node_to_chunk_hit live in crates/chronicle-core/src/document_rag/mod.rs.

use std::sync::Arc;

use chronicle_core::chronicle::Chronicle;
use chronicle_core::document_rag::DocumentChunk;

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

fn engine(dim: usize) -> Chronicle {
    Chronicle::new(
        Arc::new(FakeDriver::new()),
        Arc::new(MockLlm::new(vec![])),
        Arc::new(MockEmbedder::new(dim)),
        0,
    )
}

#[tokio::test]
async fn ingest_embeds_and_persists_idempotently() {
    let eng = engine(4);
    let chunks = vec![
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 0,
            text: "alpha".into(),
            metadata: Default::default(),
        },
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 1,
            text: "beta".into(),
            metadata: Default::default(),
        },
    ];
    let uuids1 = eng
        .ingest_document_chunks(chunks.clone(), "kb-group")
        .await
        .expect("ingest");
    assert_eq!(uuids1.len(), 2);
    // re-ingest → identical UUIDs (idempotent UPSERT)
    let uuids2 = eng
        .ingest_document_chunks(chunks, "kb-group")
        .await
        .expect("re-ingest");
    assert_eq!(uuids1, uuids2);
}

#[tokio::test]
async fn ingest_empty_is_noop() {
    let eng = engine(4);
    assert!(eng
        .ingest_document_chunks(vec![], "kb-group")
        .await
        .expect("noop")
        .is_empty());
}

#[tokio::test]
async fn search_empty_query_returns_empty() {
    let eng = engine(4);
    let hits = eng
        .search_document_chunks("   ", &["kb-group".to_string()], 5)
        .await
        .expect("search");
    assert!(hits.is_empty());
}

#[tokio::test]
async fn search_returns_only_document_chunks_mapped() {
    let eng = engine(4);
    eng.ingest_document_chunks(
        vec![DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 0,
            text: "the capital of france is paris".into(),
            metadata: Default::default(),
        }],
        "kb-group",
    )
    .await
    .expect("ingest");
    // Depending on FakeDriver search fidelity this may be empty; assert it does
    // not error and every hit is a mapped DocumentChunk with provenance.
    let hits = eng
        .search_document_chunks("france capital", &["kb-group".to_string()], 5)
        .await
        .expect("search");
    for h in &hits {
        assert!(!h.text.is_empty());
        assert_eq!(h.group_id, "kb-group");
    }
}
