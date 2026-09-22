#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::unwrap_used, clippy::expect_used)]

//! Chronicle-backed knowledge (document-RAG) provider for Greentic DW.
//!
//! [`KnowledgeChronicle`] adapts the [`Knowledge`] trait from
//! `greentic-dw-knowledge` onto the Chronicle lite doc-RAG path:
//! chunk → embed → store (no LLM extraction) + hybrid BM25+cosine retrieval.
//!
//! Tenant isolation is enforced by mapping every [`TenantCtx`] to a Chronicle
//! `group_id` (`knowledge:<tenant_id>`); all reads and writes are scoped to that
//! group.
//!
//! The crate is **driver-agnostic**: the caller injects an `Arc<dyn GraphDriver>`
//! so tests can use [`chronicle_testkit::FakeDriver`] while production wires in
//! SurrealDB or Neo4j (behind the optional `surreal` / `neo4j` crate features).

mod bridge;
mod config;

pub use bridge::DwEmbedderBridge;
pub use config::{DEFAULT_MAX_CONCURRENCY, DEFAULT_SEARCH_LIMIT, KnowledgeConfig};

use std::sync::Arc;

use async_trait::async_trait;
use chronicle_core::chronicle::Chronicle;
use chronicle_core::document_rag::DocumentChunk as CDocumentChunk;
use chronicle_core::driver::GraphDriver;
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::llm::{LlmClient, LlmConfig};
use chronicle_llm_openai::{OpenAiEmbedder, OpenAiEmbedderConfig, OpenAiLlm};
use greentic_dw_knowledge::{
    IngestOutcome, Knowledge, KnowledgeChunk, KnowledgeError, KnowledgeQuery, KnowledgeResult,
    RetrievedChunk,
};
use greentic_types::TenantCtx;

/// Chronicle-backed knowledge (document-RAG) provider.
///
/// Driver-agnostic: callers inject a [`GraphDriver`] via [`KnowledgeChronicle::from_parts`].
/// Production deployments enable the `surreal` or `neo4j` crate feature and wire
/// the appropriate driver; tests use [`chronicle_testkit::FakeDriver`].
pub struct KnowledgeChronicle {
    chronicle: Chronicle,
    search_limit: usize,
}

impl KnowledgeChronicle {
    /// Builds a [`KnowledgeChronicle`] from fully-injected Chronicle clients.
    ///
    /// Provisions the backend's indices and constraints via
    /// [`Chronicle::build_indices_and_constraints`]. Pass `max_concurrency = 0`
    /// to use the Chronicle default ([`chronicle_core::helpers::SEMAPHORE_LIMIT`]).
    /// Pass `search_limit = 0` to use [`DEFAULT_SEARCH_LIMIT`].
    ///
    /// This is the primary testable seam: tests inject a [`chronicle_testkit::FakeDriver`]
    /// and a [`chronicle_testkit::MockEmbedder`] without needing a real database.
    pub async fn from_parts(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
        search_limit: usize,
    ) -> KnowledgeResult<Self> {
        let chronicle = Chronicle::new(driver, llm, embedder, max_concurrency);
        chronicle
            .build_indices_and_constraints(false)
            .await
            .map_err(|e| KnowledgeError::Backend(e.to_string()))?;
        Ok(Self {
            chronicle,
            search_limit: if search_limit == 0 {
                DEFAULT_SEARCH_LIMIT
            } else {
                search_limit
            },
        })
    }

    /// Builds a [`KnowledgeChronicle`] from a [`KnowledgeConfig`] plus an injected
    /// graph driver.
    ///
    /// The embedder and LLM client are constructed from config (OpenAI /
    /// OpenAI-compatible, mirroring the long-term-memory family). The driver stays
    /// **injected** so this crate remains driver-agnostic and free of the optional
    /// `surreal` / `neo4j` (bindgen / RocksDB) build: the runner-host edge — where
    /// those features and `BINDGEN_EXTRA_CLANG_ARGS` live — constructs the driver
    /// and passes it here.
    ///
    /// The LLM is wired but unused on the lite doc-RAG path (entity extraction is
    /// off), so it constructs even without an LLM key (the key, if any, is only
    /// consulted at call time).
    pub async fn from_config(
        config: &KnowledgeConfig,
        driver: Arc<dyn GraphDriver>,
    ) -> KnowledgeResult<Self> {
        let embedder = Self::openai_embedder(config)?;
        let llm = Self::openai_llm(config)?;
        Self::from_parts(
            driver,
            llm,
            embedder,
            config.max_concurrency,
            config.search_limit,
        )
        .await
    }

    /// Builds the OpenAI (or OpenAI-compatible) embedder from config. The
    /// `embedding_dim` is load-bearing: it sizes the bridge and the index and must
    /// match the provider's actual output.
    fn openai_embedder(config: &KnowledgeConfig) -> KnowledgeResult<Arc<dyn EmbedderClient>> {
        let mut embedder_config = OpenAiEmbedderConfig {
            api_key: config.openai_api_key.clone(),
            base_url: config.openai_base_url.clone(),
            ..OpenAiEmbedderConfig::default()
        };
        if let Some(model) = &config.embedding_model {
            embedder_config.embedding_model = model.clone();
        }
        embedder_config.embedding_dim = config.embedding_dim;
        let embedder = OpenAiEmbedder::new(embedder_config)
            .map_err(|e| KnowledgeError::Backend(e.to_string()))?;
        Ok(Arc::new(embedder))
    }

    /// Builds the OpenAI (or OpenAI-compatible) LLM client from config. Falls back
    /// to the embedder's `openai_api_key` when no dedicated `llm_api_key` is set.
    fn openai_llm(config: &KnowledgeConfig) -> KnowledgeResult<Arc<dyn LlmClient>> {
        let llm_config = LlmConfig {
            api_key: config
                .llm_api_key
                .clone()
                .or_else(|| config.openai_api_key.clone()),
            model: config.llm_model.clone(),
            base_url: config.openai_base_url.clone(),
            ..LlmConfig::default()
        };
        let llm = OpenAiLlm::new(llm_config).map_err(|e| KnowledgeError::Backend(e.to_string()))?;
        Ok(Arc::new(llm))
    }
}

/// Validates the tenant identifier and returns the Chronicle `group_id`.
///
/// The group_id format is `knowledge:<tenant_id>`, matching the pattern used by
/// the long-term memory family which uses the bare tenant id. Using a
/// `knowledge:` prefix ensures knowledge chunks do not collide with graph
/// entities from the memory family even when they share a backend.
///
/// The identifier must be non-empty and contain only ASCII alphanumerics, `_`,
/// or `-`. Anything else is rejected as [`KnowledgeError::InvalidTenant`].
fn tenant_group_id(tenant: &TenantCtx) -> KnowledgeResult<String> {
    let id = tenant.tenant_id.as_str();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if valid {
        Ok(format!("knowledge:{id}"))
    } else {
        Err(KnowledgeError::InvalidTenant(id.to_string()))
    }
}

#[async_trait]
impl Knowledge for KnowledgeChronicle {
    async fn ingest(
        &self,
        tenant: &TenantCtx,
        chunks: Vec<KnowledgeChunk>,
    ) -> KnowledgeResult<IngestOutcome> {
        let group_id = tenant_group_id(tenant)?;

        let cchunks: Vec<CDocumentChunk> = chunks
            .into_iter()
            .map(|c| CDocumentChunk {
                doc_id: c.doc_id,
                chunk_index: c.chunk_index,
                text: c.text,
                metadata: c.metadata,
                embedding: c.embedding,
            })
            .collect();

        let chunk_ids = self
            .chronicle
            .ingest_document_chunks(cchunks, &group_id)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "chronicle ingest_document_chunks failed");
                KnowledgeError::Backend(e.to_string())
            })?;

        Ok(IngestOutcome { chunk_ids })
    }

    async fn search(
        &self,
        tenant: &TenantCtx,
        query: KnowledgeQuery,
    ) -> KnowledgeResult<Vec<RetrievedChunk>> {
        let group_id = tenant_group_id(tenant)?;
        let limit = query.limit.unwrap_or(self.search_limit);

        let hits = self
            .chronicle
            .search_document_chunks(&query.query, &[group_id], limit)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "chronicle search_document_chunks failed");
                KnowledgeError::Backend(e.to_string())
            })?;

        Ok(hits
            .into_iter()
            .map(|h| RetrievedChunk {
                text: h.text,
                score: h.score,
                doc_id: h.doc_id,
                chunk_index: h.chunk_index,
                metadata: h.metadata,
            })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};
    use greentic_types::{EnvId, TenantId};

    const EMB_DIM: usize = 4;

    fn tenant(value: &str) -> TenantCtx {
        TenantCtx::new(
            EnvId::try_from("dev").expect("env id"),
            TenantId::try_from(value).expect("tenant id"),
        )
    }

    async fn knowledge(search_limit: usize) -> KnowledgeChronicle {
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
        let llm = Arc::new(MockLlm::new(vec![]));
        KnowledgeChronicle::from_parts(
            Arc::clone(&driver) as Arc<dyn GraphDriver>,
            llm,
            embedder,
            1,
            search_limit,
        )
        .await
        .expect("from_parts succeeds")
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

    #[test]
    fn tenant_group_id_prefixes_correctly() {
        let t = tenant("tenant-a");
        let gid = tenant_group_id(&t).expect("valid tenant");
        assert_eq!(gid, "knowledge:tenant-a");
    }

    #[test]
    fn invalid_tenant_is_rejected() {
        let mut t = tenant("tenant-a");
        t.tenant_id = TenantId("bad tenant!".to_string());
        let err = tenant_group_id(&t).expect_err("invalid tenant");
        assert!(matches!(err, KnowledgeError::InvalidTenant(_)));
    }

    #[tokio::test]
    async fn ingest_returns_chunk_ids() {
        let kb = knowledge(5).await;
        let t = tenant("tenant-a");
        let chunks = vec![
            chunk("doc-1", 0, "Rust is a systems programming language."),
            chunk("doc-1", 1, "It guarantees memory safety without a GC."),
        ];
        let outcome = kb.ingest(&t, chunks).await.expect("ingest succeeds");
        assert_eq!(outcome.chunk_ids.len(), 2, "expected 2 chunk ids");
        // UUIDs are non-empty strings
        for id in &outcome.chunk_ids {
            assert!(!id.is_empty(), "chunk id must not be empty");
        }
    }

    #[tokio::test]
    async fn ingest_is_idempotent() {
        let kb = knowledge(5).await;
        let t = tenant("tenant-a");
        let chunks = vec![chunk("doc-1", 0, "Idempotent chunk text.")];
        let first = kb.ingest(&t, chunks.clone()).await.expect("first ingest");
        let second = kb.ingest(&t, chunks).await.expect("second ingest");
        // Deterministic UUIDs → same ids on re-ingest
        assert_eq!(
            first.chunk_ids, second.chunk_ids,
            "re-ingest must be idempotent"
        );
    }

    #[tokio::test]
    async fn ingest_empty_chunks_returns_empty_ids() {
        let kb = knowledge(5).await;
        let t = tenant("tenant-a");
        let outcome = kb.ingest(&t, vec![]).await.expect("empty ingest ok");
        assert!(outcome.chunk_ids.is_empty());
    }

    #[tokio::test]
    async fn search_after_ingest_returns_retrieved_chunks() {
        let kb = knowledge(5).await;
        let t = tenant("tenant-a");
        let chunks = vec![
            chunk("doc-1", 0, "Rust is fast and memory safe."),
            chunk("doc-1", 1, "Chronicle is a knowledge graph engine."),
        ];
        kb.ingest(&t, chunks).await.expect("ingest succeeds");

        let query = KnowledgeQuery {
            query: "memory safe language".to_string(),
            limit: Some(5),
        };
        let hits = kb.search(&t, query).await.expect("search succeeds");
        // FakeDriver BM25+cosine may return 0 or more hits; we just assert no error
        // and that every returned hit is a properly mapped RetrievedChunk.
        for hit in &hits {
            assert!(!hit.text.is_empty(), "hit text must not be empty");
            assert!(hit.score >= 0.0, "score must be non-negative");
        }
    }

    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let kb = knowledge(5).await;
        let t = tenant("tenant-a");
        // The chronicle search_chunks implementation returns empty for empty queries
        let query = KnowledgeQuery {
            query: String::new(),
            limit: None,
        };
        let hits = kb.search(&t, query).await.expect("empty query ok");
        assert!(hits.is_empty(), "empty query should return empty results");
    }

    #[tokio::test]
    async fn search_uses_default_limit_when_none() {
        let kb = knowledge(3).await;
        let t = tenant("tenant-a");
        // Ingest more chunks than the default limit
        let chunks: Vec<KnowledgeChunk> = (0..6)
            .map(|i| chunk("doc-1", i, &format!("chunk text {i}")))
            .collect();
        kb.ingest(&t, chunks).await.expect("ingest succeeds");
        let query = KnowledgeQuery {
            query: "chunk text".to_string(),
            limit: None,
        };
        // Should not error; FakeDriver may return fewer hits than the limit
        let hits = kb.search(&t, query).await.expect("search succeeds");
        assert!(
            hits.len() <= 3,
            "results must not exceed search_limit=3, got {}",
            hits.len()
        );
    }

    #[tokio::test]
    async fn cross_tenant_isolation() {
        let kb = knowledge(5).await;
        let tenant_a = tenant("tenant-a");
        let tenant_b = tenant("tenant-b");

        // Ingest as tenant-a
        let chunks = vec![chunk("doc-1", 0, "Secret document for tenant A only.")];
        kb.ingest(&tenant_a, chunks)
            .await
            .expect("ingest as tenant-a");

        // Tenant-b search must not surface tenant-a's chunks
        let query = KnowledgeQuery {
            query: "Secret document".to_string(),
            limit: Some(10),
        };
        let hits_b = kb
            .search(&tenant_b, query)
            .await
            .expect("search as tenant-b");
        for hit in &hits_b {
            assert_ne!(
                hit.text, "Secret document for tenant A only.",
                "tenant-b must not see tenant-a's chunks"
            );
        }
    }

    #[tokio::test]
    async fn invalid_tenant_rejected_on_ingest() {
        let kb = knowledge(5).await;
        let mut t = tenant("tenant-a");
        t.tenant_id = TenantId("bad tenant!".to_string());
        let err = kb
            .ingest(&t, vec![chunk("d", 0, "x")])
            .await
            .expect_err("invalid tenant must be rejected");
        assert!(matches!(err, KnowledgeError::InvalidTenant(_)));
    }

    #[tokio::test]
    async fn invalid_tenant_rejected_on_search() {
        let kb = knowledge(5).await;
        let mut t = tenant("tenant-a");
        t.tenant_id = TenantId("bad tenant!".to_string());
        let err = kb
            .search(
                &t,
                KnowledgeQuery {
                    query: "q".to_string(),
                    limit: None,
                },
            )
            .await
            .expect_err("invalid tenant must be rejected");
        assert!(matches!(err, KnowledgeError::InvalidTenant(_)));
    }

    #[tokio::test]
    async fn from_config_constructs_with_injected_driver() {
        // from_config builds the OpenAI embedder + LLM from config and provisions
        // indices on the injected driver. The real OpenAI clients construct without
        // a live key (the key is only consulted at call time), so this asserts
        // construction only — it does not call ingest/search (which would hit the
        // network).
        let driver: Arc<dyn GraphDriver> = Arc::new(FakeDriver::new());
        let config = KnowledgeConfig::new(EMB_DIM).with_openai_api_key("test-key");
        let kb = KnowledgeChronicle::from_config(&config, driver).await;
        assert!(kb.is_ok(), "from_config should construct: {:?}", kb.err());
    }
}
