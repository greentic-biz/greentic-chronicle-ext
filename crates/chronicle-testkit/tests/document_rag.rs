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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chronicle_core::chronicle::Chronicle;
use chronicle_core::document_rag::{DocumentChunk, chunk_uuid};
use chronicle_core::driver::EntityNodeOps;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};
use chronicle_core::errors::ChronicleError;

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

/// Embedder test double that counts `create_batch` invocations and records the
/// exact texts it was asked to embed, so tests can assert that precomputed
/// chunks are NOT re-embedded. Returns a fixed, dimension-`dim` vector distinct
/// from any caller-supplied vector used in the tests.
struct CountingEmbedder {
    dim: usize,
    batch_calls: Arc<AtomicUsize>,
    batch_inputs: Arc<Mutex<Vec<String>>>,
}

impl CountingEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            batch_calls: Arc::new(AtomicUsize::new(0)),
            batch_inputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The fixed vector this embedder produces for any text.
    fn fresh_vector(&self) -> Vec<f32> {
        vec![0.1_f32; self.dim]
    }
}

#[async_trait]
impl EmbedderClient for CountingEmbedder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    async fn create(&self, _input: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(vec![0.1_f32; self.dim])
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut recorded) = self.batch_inputs.lock() {
            recorded.extend(inputs.iter().cloned());
        }
        Ok(inputs.iter().map(|_| vec![0.1_f32; self.dim]).collect())
    }
}

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
            embedding: None,
        },
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 1,
            text: "beta".into(),
            metadata: Default::default(),
            embedding: None,
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
    assert!(
        eng.ingest_document_chunks(vec![], "kb-group")
            .await
            .expect("noop")
            .is_empty()
    );
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
            embedding: None,
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

// ---------------------------------------------------------------------------
// Precomputed-vector ingest (Slice 3, Task 1)
// ---------------------------------------------------------------------------

const GROUP: &str = "kb-group";

/// Test A: mixed batch — one chunk carries a precomputed vector, one does not.
/// The embedder must be called exactly once, with ONLY the without-vector
/// chunk's text; the precomputed chunk must be stored with its SUPPLIED vector.
#[tokio::test]
async fn ingest_skips_reembedding_for_precomputed_chunk() {
    let dim = 4;
    let supplied = vec![0.9_f32; dim];

    let embedder = CountingEmbedder::new(dim);
    let batch_calls = embedder.batch_calls.clone();
    let batch_inputs = embedder.batch_inputs.clone();
    let fresh = embedder.fresh_vector();

    let driver = Arc::new(FakeDriver::new());
    let eng = Chronicle::new(
        driver.clone(),
        Arc::new(MockLlm::new(vec![])),
        Arc::new(embedder),
        0,
    );

    let chunks = vec![
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 0,
            text: "precomputed text".into(),
            metadata: Default::default(),
            embedding: Some(supplied.clone()),
        },
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 1,
            text: "needs embedding".into(),
            metadata: Default::default(),
            embedding: None,
        },
    ];

    let uuids = eng
        .ingest_document_chunks(chunks, GROUP)
        .await
        .expect("ingest");

    // Order preserved: returned UUIDs follow input chunk order.
    assert_eq!(uuids.len(), 2);
    assert_eq!(uuids[0], chunk_uuid(GROUP, "d", 0));
    assert_eq!(uuids[1], chunk_uuid(GROUP, "d", 1));

    // Embedder called exactly once, with ONLY the without-vector text.
    assert_eq!(batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *batch_inputs.lock().expect("inputs lock"),
        vec!["needs embedding".to_string()]
    );

    // Precomputed chunk stored with the SUPPLIED vector (not a re-embedded one).
    let node0 = driver
        .get_entity_node(&uuids[0])
        .await
        .expect("get node0")
        .expect("node0 stored");
    assert_eq!(node0.name_embedding, Some(supplied));

    // Without-vector chunk stored with the freshly embedded vector.
    let node1 = driver
        .get_entity_node(&uuids[1])
        .await
        .expect("get node1")
        .expect("node1 stored");
    assert_eq!(node1.name_embedding, Some(fresh));
}

/// Test B: a precomputed vector whose dimension does not match the embedder is
/// rejected (InvalidInput) and nothing is persisted.
#[tokio::test]
async fn ingest_rejects_wrong_dimension_precomputed_vector() {
    let dim = 4;
    let embedder = CountingEmbedder::new(dim);
    let batch_calls = embedder.batch_calls.clone();

    let driver = Arc::new(FakeDriver::new());
    let eng = Chronicle::new(
        driver.clone(),
        Arc::new(MockLlm::new(vec![])),
        Arc::new(embedder),
        0,
    );

    let chunks = vec![DocumentChunk {
        doc_id: "d".into(),
        chunk_index: 0,
        text: "wrong dim".into(),
        metadata: Default::default(),
        embedding: Some(vec![0.9_f32; dim + 1]), // 5 != 4
    }];

    let err = eng
        .ingest_document_chunks(chunks, GROUP)
        .await
        .expect_err("expected dimension-mismatch error");
    match err {
        ChronicleError::InvalidInput(msg) => {
            assert!(msg.contains("dimension"), "unexpected message: {msg}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // Nothing embedded, nothing stored.
    assert_eq!(batch_calls.load(Ordering::SeqCst), 0);
    assert!(
        driver
            .get_entity_node(&chunk_uuid(GROUP, "d", 0))
            .await
            .expect("get node")
            .is_none()
    );
}

/// Test C (regression): all chunks lack precomputed vectors → behaves exactly
/// as before, i.e. `create_batch` is called once with every chunk's text and
/// every chunk is persisted in order.
#[tokio::test]
async fn ingest_all_without_vectors_embeds_all() {
    let dim = 4;
    let embedder = CountingEmbedder::new(dim);
    let batch_calls = embedder.batch_calls.clone();
    let batch_inputs = embedder.batch_inputs.clone();

    let driver = Arc::new(FakeDriver::new());
    let eng = Chronicle::new(
        driver.clone(),
        Arc::new(MockLlm::new(vec![])),
        Arc::new(embedder),
        0,
    );

    let chunks = vec![
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 0,
            text: "alpha".into(),
            metadata: Default::default(),
            embedding: None,
        },
        DocumentChunk {
            doc_id: "d".into(),
            chunk_index: 1,
            text: "beta".into(),
            metadata: Default::default(),
            embedding: None,
        },
    ];

    let uuids = eng
        .ingest_document_chunks(chunks, GROUP)
        .await
        .expect("ingest");

    assert_eq!(uuids.len(), 2);
    assert_eq!(batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *batch_inputs.lock().expect("inputs lock"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    // Both persisted, in order.
    assert_eq!(uuids[0], chunk_uuid(GROUP, "d", 0));
    assert_eq!(uuids[1], chunk_uuid(GROUP, "d", 1));
    assert!(
        driver
            .get_entity_node(&uuids[0])
            .await
            .expect("get node0")
            .is_some()
    );
    assert!(
        driver
            .get_entity_node(&uuids[1])
            .await
            .expect("get node1")
            .is_some()
    );
}
