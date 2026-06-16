# Chronicle Lite Document-RAG (Knowledge/RAG epic — W2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a "lite" document-RAG path to `chronicle-core` — ingest plain-text chunks (embed → store as nodes, NO LLM entity extraction) and retrieve top-k by hybrid BM25+cosine, reusing existing driver primitives.

**Architecture:** A new `document_rag` module in `chronicle-core` with: a pure `chunk_text` util, a deterministic `chunk_to_entity_node` builder (chunks stored as `EntityNode` labelled `DocumentChunk` under a dedicated knowledge `group_id`; UUID = blake2(group_id|doc_id|chunk_idx) → idempotent UPSERT), `ingest_chunks` (batch-embed → `save_entity_nodes`), and `search_chunks` (embed query → `node_search` BM25+cosine RRF → map). Two thin `Chronicle` methods wrap them. Bypasses `add_episode` entirely. Zero schema change (reuses the `entity` table's existing HNSW + BM25 indexes).

**Tech stack:** Rust edition 2024, workspace `1.2.0-research`, rust 1.95. Deps already in `chronicle-core`: `uuid` (v4+serde), `blake2` 0.10, `chrono`, `serde_json`, `async-trait`, `tokio`. No new deps.

**Epic context / decisions (locked):** Knowledge/RAG rides the Chronicle substrate (epic spec in greentic-designer `docs/superpowers/specs/2026-06-16-dw-knowledge-rag-design.md`). Chunks isolated per knowledge `group_id` (idiomatic Chronicle isolation) → retrieval scoped to that group cleanly separates them from real entities. The dw.embedding→`EmbedderClient` adapter is **W3's** job (chronicle-core must not depend on dw-embedding). Dedicated `DocumentChunk` node *type* (vs reusing `EntityNode`) deferred — not worth a 3-driver schema fork.

**Verified API (research line):**
- `Clients { pub driver: Arc<dyn GraphDriver>, pub llm, pub embedder: Arc<dyn EmbedderClient>, semaphore }`; `Chronicle { clients: Clients, .. }` (private field; wrapper methods live in `chronicle.rs`).
- `EntityNode::new(name: String, group_id: String, created_at: DateTime<Utc>) -> Self` (uuid=v4, labels=`["Entity"]`, name_embedding=None). Fields all `pub`.
- `EmbedderClient { fn embedding_dim()->usize; async fn create(&str)->Result<Vec<f32>,EmbedderError>; async fn create_batch(&[String])->Result<Vec<Vec<f32>>,EmbedderError> }`.
- `driver.save_entity_nodes(&[EntityNode]) -> Result<(),DriverError>` (UPSERT by uuid).
- `chronicle_core::search::node_search::node_search(driver: &dyn GraphDriver, cross_encoder: Option<&dyn CrossEncoderClient>, query: &str, query_vector: &[f32], group_ids: &[String], config: Option<&NodeSearchConfig>, filters: &SearchFilters, center_node_uuid: Option<&str>, bfs_origin_node_uuids: Option<&[String]>, limit: usize, reranker_min_score: f64) -> Result<(Vec<EntityNode>, Vec<f64>), ChronicleError>`.
- `NodeSearchConfig::default()` = BM25 + CosineSimilarity + RRF. `SearchFilters::default()`.
- `ChronicleError`: `Driver(#[from])`, `Embedder(#[from])`, `InvalidInput(String)`, `Serde(#[from])`, … (use `InvalidInput` for our validation; driver/embedder errors auto-convert via `?`).
- testkit: `FakeDriver::new()`, `MockEmbedder::new(dim: usize)`, `MockLlm::new(vec![])`. Build engine: `Chronicle::new(Arc::new(FakeDriver::new()), Arc::new(MockLlm::new(vec![])), Arc::new(MockEmbedder::new(4)), 0)`.

**Conventions:** English only; `#![forbid(unsafe_code)]` already at crate root; no `unwrap()/panic!()` in non-test code; Conventional Commits `feat(doc-rag):`; NO Claude co-author trailer. Build/test SCOPED to core (`cargo test -p chronicle-core`) — do NOT build the surreal driver (its RocksDB needs `BINDGEN_EXTRA_CLANG_ARGS`, a known sandbox caveat; not part of W2).

---

## File Structure

### New
```
crates/chronicle-core/src/document_rag/mod.rs    types + chunk_text + builder + ingest + search free fns
```
### Modified
```
crates/chronicle-core/src/lib.rs        + pub mod document_rag; + re-exports
crates/chronicle-core/src/chronicle.rs  + Chronicle::ingest_document_chunks / search_document_chunks
```

---

## Task 0: Worktree (DONE)
Worktree exists at `~/Works/worktrees/chronicle-docrag` (branch `feat/chronicle-doc-rag` off `origin/research`). Set:
```
cd ~/Works/worktrees/chronicle-docrag
export CARGO_TARGET_DIR=$HOME/.cache/cargo-target/chronicle-docrag
```
Sanity: `cargo build -p chronicle-core` green before starting.

---

## Task 1: `document_rag` module — types + `chunk_text` util (pure, TDD)

**Files:** create `crates/chronicle-core/src/document_rag/mod.rs`; modify `lib.rs` (`pub mod document_rag;`).

- [ ] **Step 1 — failing tests.** Create `document_rag/mod.rs` with the test module first:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_empty_or_zero_returns_empty() {
        assert!(chunk_text("   ", 100, 10).is_empty());
        assert!(chunk_text("hello", 0, 0).is_empty());
    }
    #[test]
    fn chunk_text_short_returns_single_trimmed() {
        assert_eq!(chunk_text("  hello world  ", 100, 10), vec!["hello world".to_string()]);
    }
    #[test]
    fn chunk_text_splits_with_overlap_and_terminates() {
        let text = "aaaa bbbb cccc dddd eeee ffff"; // 29 chars
        let chunks = chunk_text(text, 10, 4);
        assert!(chunks.len() >= 3);                       // multiple windows
        assert!(chunks.iter().all(|c| c.chars().count() <= 10));
        assert!(chunks.iter().all(|c| !c.is_empty()));    // no empties, no infinite loop
    }
    #[test]
    fn chunk_text_prefers_whitespace_boundary() {
        let chunks = chunk_text("alpha beta gamma", 8, 0);
        assert_eq!(chunks[0], "alpha");                   // broke at space, not mid-word
    }
}
```
- [ ] **Step 2 — run, expect FAIL** (`chunk_text` undefined): `cargo test -p chronicle-core document_rag`
- [ ] **Step 3 — implement** the module header + types + `chunk_text` (above the test module):
```rust
//! Lite document-RAG: chunk → embed → store as nodes (no LLM extraction) +
//! hybrid retrieval. See docs/superpowers/plans/2026-06-16-chronicle-doc-rag-w2.md.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::ChronicleError;
use crate::pipeline::clients::Clients;
use crate::search::config::NodeSearchConfig;
use crate::search::filters::SearchFilters;
use crate::search::node_search::node_search;
use crate::types::node::EntityNode;

/// Label marking an EntityNode as a stored document chunk (in addition to "Entity").
pub const DOCUMENT_CHUNK_LABEL: &str = "DocumentChunk";

/// One pre-chunked unit of source text to ingest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub doc_id: String,
    pub chunk_index: usize,
    pub text: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// A retrieval hit: chunk text + relevance score + provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunkHit {
    pub text: String,
    pub score: f64,
    pub group_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Split text into overlapping windows, preferring whitespace boundaries.
/// Heuristic char-window (not a tokenizer). Empty text / `max_chars==0` → empty.
#[must_use]
pub fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() || max_chars == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return vec![text.trim().to_string()];
    }
    let overlap = overlap.min(max_chars - 1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let hard_end = (start + max_chars).min(chars.len());
        let mut end = hard_end;
        if end < chars.len() {
            if let Some(ws) = (start + 1..end).rev().find(|&i| chars[i].is_whitespace()) {
                end = ws;
            }
        }
        let piece: String = chars[start..end].iter().collect::<String>().trim().to_string();
        if !piece.is_empty() {
            chunks.push(piece);
        }
        if end >= chars.len() {
            break;
        }
        // guarantee forward progress even with large overlap
        let next = end.saturating_sub(overlap);
        start = if next > start { next } else { end };
    }
    chunks
}
```
- [ ] **Step 4 — run, expect PASS.** Add `pub mod document_rag;` to `lib.rs` (alongside siblings). `cargo test -p chronicle-core document_rag`
- [ ] **Step 5 — commit:** `feat(doc-rag): document_rag module types + chunk_text util`

---

## Task 2: deterministic node builder + hit mapper (pure, TDD)

**Files:** modify `document_rag/mod.rs`.

- [ ] **Step 1 — failing tests** (append to test module):
```rust
    use crate::types::node::EntityNode;
    use chrono::Utc;

    fn sample_chunk() -> DocumentChunk {
        let mut md = serde_json::Map::new();
        md.insert("source".into(), serde_json::json!("kb.pdf"));
        DocumentChunk { doc_id: "doc1".into(), chunk_index: 2, text: "hello world".into(), metadata: md }
    }

    #[test]
    fn chunk_uuid_is_deterministic_and_stable() {
        let a = chunk_uuid("g1", "doc1", 2);
        let b = chunk_uuid("g1", "doc1", 2);
        let c = chunk_uuid("g1", "doc1", 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 36); // hyphenated uuid
    }

    #[test]
    fn build_node_sets_text_embedding_label_attrs() {
        let node = chunk_to_entity_node(&sample_chunk(), "g1", vec![0.1, 0.2, 0.3], Utc::now());
        assert_eq!(node.name, "hello world");
        assert_eq!(node.group_id, "g1");
        assert_eq!(node.name_embedding, Some(vec![0.1, 0.2, 0.3]));
        assert!(node.labels.contains(&"Entity".to_string()));
        assert!(node.labels.contains(&DOCUMENT_CHUNK_LABEL.to_string()));
        assert_eq!(node.attributes.get("doc_id").and_then(|v| v.as_str()), Some("doc1"));
        assert_eq!(node.attributes.get("chunk_index").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(node.attributes.get("source").and_then(|v| v.as_str()), Some("kb.pdf"));
        assert_eq!(node.uuid, chunk_uuid("g1", "doc1", 2)); // deterministic
    }

    #[test]
    fn map_hit_extracts_provenance() {
        let node = chunk_to_entity_node(&sample_chunk(), "g1", vec![0.0; 3], Utc::now());
        let hit = node_to_chunk_hit(node, 0.87);
        assert_eq!(hit.text, "hello world");
        assert_eq!(hit.score, 0.87);
        assert_eq!(hit.doc_id.as_deref(), Some("doc1"));
        assert_eq!(hit.chunk_index, Some(2));
        assert_eq!(hit.group_id, "g1");
    }
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** (add to module body):
```rust
use blake2::{Blake2b512, Digest};

/// Deterministic chunk UUID (stable across re-ingest → idempotent UPSERT).
#[must_use]
pub fn chunk_uuid(group_id: &str, doc_id: &str, chunk_index: usize) -> String {
    let mut hasher = Blake2b512::new();
    hasher.update(group_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(doc_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(chunk_index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Uuid::from_bytes(bytes).to_string()
}

/// Build an EntityNode for a chunk: text in `name` (→ name_embedding), labelled
/// `["Entity","DocumentChunk"]`, provenance in `attributes`. Keeps the "Entity"
/// label so the existing entity-table BM25/HNSW search finds it.
#[must_use]
pub fn chunk_to_entity_node(
    chunk: &DocumentChunk,
    group_id: &str,
    embedding: Vec<f32>,
    created_at: DateTime<Utc>,
) -> EntityNode {
    let mut node = EntityNode::new(chunk.text.clone(), group_id.to_string(), created_at);
    node.uuid = chunk_uuid(group_id, &chunk.doc_id, chunk.chunk_index);
    node.labels = vec!["Entity".to_string(), DOCUMENT_CHUNK_LABEL.to_string()];
    node.name_embedding = Some(embedding);
    node.attributes.insert("doc_id".to_string(), Value::String(chunk.doc_id.clone()));
    node.attributes.insert("chunk_index".to_string(), Value::from(chunk.chunk_index));
    for (k, v) in &chunk.metadata {
        node.attributes.entry(k.clone()).or_insert_with(|| v.clone());
    }
    node
}

/// Map a retrieved node + score into a hit, pulling provenance from attributes.
#[must_use]
pub fn node_to_chunk_hit(node: EntityNode, score: f64) -> DocumentChunkHit {
    let doc_id = node.attributes.get("doc_id").and_then(|v| v.as_str()).map(str::to_string);
    let chunk_index = node
        .attributes
        .get("chunk_index")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    DocumentChunkHit { text: node.name, score, group_id: node.group_id, doc_id, chunk_index, metadata: node.attributes }
}
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(doc-rag): deterministic chunk node builder + hit mapper`

---

## Task 3: `ingest_chunks` (free fn) + `Chronicle::ingest_document_chunks`

**Files:** modify `document_rag/mod.rs`, `chronicle.rs`.

- [ ] **Step 1 — failing test** (append; uses testkit):
```rust
    use crate::Chronicle;
    use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};
    use std::sync::Arc;

    fn engine(dim: usize) -> Chronicle {
        Chronicle::new(Arc::new(FakeDriver::new()), Arc::new(MockLlm::new(vec![])), Arc::new(MockEmbedder::new(dim)), 0)
    }

    #[tokio::test]
    async fn ingest_embeds_and_persists_idempotently() {
        let eng = engine(4);
        let chunks = vec![
            DocumentChunk { doc_id: "d".into(), chunk_index: 0, text: "alpha".into(), metadata: Default::default() },
            DocumentChunk { doc_id: "d".into(), chunk_index: 1, text: "beta".into(),  metadata: Default::default() },
        ];
        let uuids1 = eng.ingest_document_chunks(chunks.clone(), "kb-group").await.expect("ingest");
        assert_eq!(uuids1.len(), 2);
        // re-ingest → identical UUIDs (idempotent UPSERT)
        let uuids2 = eng.ingest_document_chunks(chunks, "kb-group").await.expect("re-ingest");
        assert_eq!(uuids1, uuids2);
    }

    #[tokio::test]
    async fn ingest_empty_is_noop() {
        let eng = engine(4);
        assert!(eng.ingest_document_chunks(vec![], "kb-group").await.expect("noop").is_empty());
    }
```
> If `MockEmbedder::create_batch` returns vectors of the configured `dim` for each input (verify in testkit), the assertions hold. If `MockEmbedder` only implements `create` (single), add a batch loop fallback in `ingest_chunks` — but prefer `create_batch`; confirm testkit supports it (the EmbedderClient trait requires it).

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the free fn in `document_rag/mod.rs`:
```rust
/// Ingest pre-chunked text: batch-embed, build nodes, persist. Returns chunk UUIDs.
/// Idempotent: deterministic UUIDs + driver UPSERT. No LLM extraction.
pub async fn ingest_chunks(
    clients: &Clients,
    chunks: Vec<DocumentChunk>,
    group_id: &str,
) -> Result<Vec<String>, ChronicleError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = clients.embedder.create_batch(&texts).await?;
    if embeddings.len() != chunks.len() {
        return Err(ChronicleError::InvalidInput(format!(
            "embedder returned {} vectors for {} chunks",
            embeddings.len(),
            chunks.len()
        )));
    }
    let created_at = Utc::now();
    let mut nodes = Vec::with_capacity(chunks.len());
    let mut uuids = Vec::with_capacity(chunks.len());
    for (chunk, embedding) in chunks.into_iter().zip(embeddings) {
        let node = chunk_to_entity_node(&chunk, group_id, embedding, created_at);
        uuids.push(node.uuid.clone());
        nodes.push(node);
    }
    clients.driver.save_entity_nodes(&nodes).await?;
    Ok(uuids)
}
```
And the wrapper in `chronicle.rs` (inside `impl Chronicle`, near other methods — uses `self.clients`):
```rust
    /// Ingest pre-chunked documents into the knowledge store (no LLM extraction).
    /// See [`crate::document_rag`].
    pub async fn ingest_document_chunks(
        &self,
        chunks: Vec<crate::document_rag::DocumentChunk>,
        group_id: &str,
    ) -> Result<Vec<String>, ChronicleError> {
        crate::document_rag::ingest_chunks(&self.clients, chunks, group_id).await
    }
```
- [ ] **Step 4 — run, expect PASS.** (If `FakeDriver` does not retain saved nodes, the idempotency assert still holds — it only compares returned UUIDs, which are deterministic regardless of driver retention.)
- [ ] **Step 5 — commit:** `feat(doc-rag): ingest_document_chunks (batch-embed + persist, idempotent)`

---

## Task 4: `search_chunks` (free fn) + `Chronicle::search_document_chunks`

**Files:** modify `document_rag/mod.rs`, `chronicle.rs`.

- [ ] **Step 1 — failing test** (append):
```rust
    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let eng = engine(4);
        let hits = eng.search_document_chunks("   ", &["kb-group".to_string()], 5).await.expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_returns_only_document_chunks_mapped() {
        let eng = engine(4);
        eng.ingest_document_chunks(
            vec![DocumentChunk { doc_id: "d".into(), chunk_index: 0, text: "the capital of france is paris".into(), metadata: Default::default() }],
            "kb-group",
        ).await.expect("ingest");
        // Depending on FakeDriver search fidelity this may be empty; assert it does
        // not error and every hit is a mapped DocumentChunk with provenance.
        let hits = eng.search_document_chunks("france capital", &["kb-group".to_string()], 5).await.expect("search");
        for h in &hits {
            assert!(!h.text.is_empty());
            assert_eq!(h.group_id, "kb-group");
        }
    }
```
> The `node_to_chunk_hit` mapping is already proven pure in Task 2; this test guards the wiring (embed → node_search → filter `DocumentChunk` → map) against panics and confirms shape. If `FakeDriver` implements node search and returns the stored node, `hits` will be non-empty and provenance asserted; if it returns empty, the test still validates the no-error contract.

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the free fn:
```rust
/// Retrieve top-k document chunks for a query via hybrid BM25+cosine (RRF),
/// scoped to the given knowledge group_id(s). Returns mapped hits, newest-API
/// node ordering preserved.
pub async fn search_chunks(
    clients: &Clients,
    query: &str,
    group_ids: &[String],
    limit: usize,
) -> Result<Vec<DocumentChunkHit>, ChronicleError> {
    if query.trim().is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let query_vector = clients.embedder.create(&query.replace('\n', " ")).await?;
    let config = NodeSearchConfig::default(); // BM25 + Cosine, RRF
    let (nodes, scores) = node_search(
        clients.driver.as_ref(),
        None, // no cross-encoder in the lite path
        query,
        &query_vector,
        group_ids,
        Some(&config),
        &SearchFilters::default(),
        None,
        None,
        limit,
        0.0,
    )
    .await?;
    let hits = nodes
        .into_iter()
        .zip(scores)
        .filter(|(node, _)| node.labels.iter().any(|l| l == DOCUMENT_CHUNK_LABEL))
        .map(|(node, score)| node_to_chunk_hit(node, score))
        .collect();
    Ok(hits)
}
```
And the wrapper in `chronicle.rs`:
```rust
    /// Hybrid (BM25 + cosine) retrieval of stored document chunks, scoped to
    /// knowledge group_id(s). See [`crate::document_rag`].
    pub async fn search_document_chunks(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<crate::document_rag::DocumentChunkHit>, ChronicleError> {
        crate::document_rag::search_chunks(&self.clients, query, group_ids, limit).await
    }
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `feat(doc-rag): search_document_chunks (hybrid retrieval + chunk filter)`

---

## Task 5: exports, docs, gate

**Files:** modify `lib.rs`; this plan doc already present.

- [ ] **Step 1 — re-export public surface** in `lib.rs` (match how other modules re-export):
```rust
pub use document_rag::{
    chunk_text, DocumentChunk, DocumentChunkHit, DOCUMENT_CHUNK_LABEL,
};
```
- [ ] **Step 2 — module doc.** Ensure `document_rag/mod.rs` top doc-comment explains the bypass-extraction rationale + group_id isolation (1–3 lines; already drafted in Task 1).
- [ ] **Step 3 — gate (scoped to core).** Run:
```
cargo test -p chronicle-core
cargo clippy -p chronicle-core --all-targets -- -D warnings
cargo fmt -p chronicle-core -- --check   # or `cargo fmt --all` then re-check
```
All must pass. Do NOT run workspace `--all-features` / the surreal driver build (RocksDB needs `BINDGEN_EXTRA_CLANG_ARGS`; out of scope, known sandbox caveat). If `bash ci/local_check.sh` is attempted and fails ONLY on the surreal/RocksDB bindgen build, document it as the pre-existing environmental caveat — not a W2 failure.
- [ ] **Step 4 — commit:** `feat(doc-rag): export public surface + module docs`
- [ ] **Step 5 — push (await controller).** Do NOT push or open a PR yourself — the controller handles finishing (push + PR to `research`).

---

## Self-Review
- **Coverage:** chunk_text (Task1), deterministic uuid + builder + mapper (Task2), ingest+idempotency (Task3), search wiring + chunk filter (Task4), exports+gate (Task5). The retrieval *mapping* is proven pure (Task2) so search coverage doesn't depend on FakeDriver search fidelity.
- **No placeholders:** all code blocks complete; the only conditional is FakeDriver search behavior, handled by a no-error-contract test + the pure mapper test.
- **Layering:** chronicle-core gains zero new deps and no dependency on dw-embedding (adapter is W3). No schema change.
- **Type consistency:** `DocumentChunk`/`DocumentChunkHit`/`chunk_uuid`/`chunk_to_entity_node`/`node_to_chunk_hit`/`ingest_chunks`/`search_chunks` names consistent across module, tests, and `Chronicle` wrappers.
- **Confirm at build time:** `MockEmbedder::create_batch` returns one `dim`-vector per input (EmbedderClient requires it); the exact `lib.rs` module-declaration ordering; that `node_search` is re-exported at `crate::search::node_search::node_search` (adjust path if it's `crate::search::node_search` module fn).
