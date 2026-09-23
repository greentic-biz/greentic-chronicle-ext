# greentic-dw-knowledge-chronicle

Chronicle-backed knowledge (document-RAG) provider for Greentic DW.

## Capability

- URI: `cap://dw.knowledge`
- Pack capability ID: `greentic.cap.knowledge`
- Operations: `knowledge.ingest`, `knowledge.search`
- Provider type: `dw.knowledge.chronicle`

## Overview

Implements the `Knowledge` trait from `greentic-dw-knowledge` using Chronicle's
lite doc-RAG path (W2):

- **Ingest**: batch-embeds pre-chunked text via `Chronicle::ingest_document_chunks`
  (BM25-indexed + HNSW vector store, no LLM extraction).
- **Search**: hybrid BM25 + cosine retrieval via `Chronicle::search_document_chunks`,
  returning ranked `RetrievedChunk` hits with provenance.

### Tenant isolation

Every `TenantCtx` is mapped to a Chronicle `group_id` of the form
`knowledge:<tenant_id>`. This scopes all ingest and retrieval per tenant and
avoids collision with the long-term memory graph family (which uses the bare
tenant id as its group).

### Embedder bridge

`DwEmbedderBridge` adapts the synchronous `EmbeddingProvider` (W1) to
Chronicle's async `EmbedderClient` via `tokio::task::spawn_blocking`, mirroring
the `DwLlmBridge` pattern used by the long-term memory family.

The embedding dimension (`KnowledgeConfig::embedding_dim`) is required at
construction time because `EmbeddingProviderFeatures` carries no `dim` field.

## Driver features

The crate is **driver-agnostic** by default: no real database dependency is
compiled in unless a feature is enabled.

| Feature | Driver | Dependency |
|---|---|---|
| `default` (none) | `chronicle_testkit::FakeDriver` (test only) | none |
| `surreal` | SurrealDB | `chronicle-driver-surreal` (RocksDB/bindgen) |
| `neo4j` | Neo4j (Bolt) | `chronicle-driver-neo4j` |

> Do NOT enable `surreal` or `neo4j` in CI sandboxes — those features require
> bindgen / RocksDB native libraries.

## Usage (tests / injection)

```rust
use std::sync::Arc;
use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};
use greentic_dw_knowledge::Knowledge;
use greentic_dw_knowledge_chronicle::KnowledgeChronicle;

let kb = KnowledgeChronicle::from_parts(
    Arc::new(FakeDriver::new()),
    Arc::new(MockLlm::new(vec![])),
    Arc::new(MockEmbedder::new(4)),
    0,  // max_concurrency (0 = Chronicle default)
    10, // search_limit
)
.await?;
```

## Epic spec

`greentic-designer/docs/superpowers/specs/2026-06-16-dw-knowledge-rag-design.md`
