#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Chronicle-backed implementation of the Greentic DW long-term memory contract.
//!
//! [`ChronicleLongTermMemory`] adapts the [`LongTermMemory`] trait from
//! `greentic-dw-memory-long-term` onto the Chronicle bi-temporal knowledge-graph
//! engine. Episodes are ingested into the operator-selected graph store — Neo4j,
//! FalkorDB, or embedded SurrealDB (see [`ChronicleBackend`]) — and recalled
//! through Chronicle's hybrid (BM25 + cosine, RRF-fused) edge search.
//!
//! Tenant isolation is enforced by mapping every [`TenantCtx`] to a Chronicle
//! `group_id`; all reads and writes are scoped to that group.

mod bridge;
mod bridge_embedder;
mod config;

pub use bridge::DwLlmBridge;
pub use bridge_embedder::DwEmbedderBridge;
pub use config::{
    ChronicleBackend, ChronicleMemoryConfig, DEFAULT_FALKOR_GRAPH, DEFAULT_MAX_CONCURRENCY,
    DEFAULT_NEO4J_DATABASE, DEFAULT_RECALL_LIMIT,
};

use std::sync::Arc;

use async_trait::async_trait;
use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::driver::GraphDriver;
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::llm::{LlmClient, LlmConfig};
use chronicle_core::search::edge_hybrid_search_rrf;
use chronicle_core::types::EpisodeType;
use chronicle_driver_falkor::FalkorDriver;
use chronicle_driver_neo4j::Neo4jDriver;
use chronicle_driver_surreal::SurrealDriver;
use chronicle_llm_openai::{OpenAiEmbedder, OpenAiEmbedderConfig, OpenAiLlm};
use greentic_dw_embedding::{DEFAULT_EMBEDDING_DIM, EmbeddingProvider};
use greentic_dw_llm::LlmProvider;
use greentic_dw_memory_long_term::{
    EpisodeIngest, EpisodeSource, IngestOutcome, LongTermMemory, LongTermMemoryError, RecallQuery,
    RecalledFact,
};
use greentic_types::TenantCtx;

/// Chronicle-backed long-term memory provider.
pub struct ChronicleLongTermMemory {
    chronicle: Chronicle,
    recall_limit: usize,
}

impl ChronicleLongTermMemory {
    /// Connects to the configured backend and builds a Chronicle engine using
    /// OpenAI for both extraction (LLM) and embeddings, then provisions the
    /// backend's indices and constraints.
    pub async fn connect(config: ChronicleMemoryConfig) -> Result<Self, LongTermMemoryError> {
        let driver = Self::connect_driver(&config).await?;
        let llm = Self::openai_llm(&config)?;
        let embedder = Self::openai_embedder(&config)?;

        Self::assemble(
            driver,
            llm,
            embedder,
            config.max_concurrency,
            config.recall_limit,
        )
        .await
    }

    /// Connects to the configured backend and builds a Chronicle engine that
    /// drives entity / edge extraction through the supplied DW [`LlmProvider`]
    /// (via [`DwLlmBridge`]).
    ///
    /// Embeddings still use OpenAI: there is no DW embeddings family yet, so the
    /// vector side of recall continues to rely on an OpenAI-compatible embedder.
    pub async fn connect_with_dw_llm(
        config: ChronicleMemoryConfig,
        provider: Arc<dyn LlmProvider>,
        tenant: TenantCtx,
    ) -> Result<Self, LongTermMemoryError> {
        let driver = Self::connect_driver(&config).await?;
        let llm: Arc<dyn LlmClient> = Arc::new(DwLlmBridge::new(provider, tenant));
        let embedder = Self::openai_embedder(&config)?;

        Self::assemble(
            driver,
            llm,
            embedder,
            config.max_concurrency,
            config.recall_limit,
        )
        .await
    }

    /// Connects to Neo4j and builds a Chronicle engine driven entirely by
    /// provider-neutral Greentic DW backends: the supplied [`LlmProvider`] for
    /// entity / edge extraction (via [`DwLlmBridge`]) and the supplied
    /// [`EmbeddingProvider`] for vector recall (via [`DwEmbedderBridge`]).
    ///
    /// No hard dependency on any single provider — the operator chooses which
    /// LLM and embedding backends (and endpoints) to wire in. The embedding
    /// dimension comes from `config.embedding_dim`, falling back to
    /// [`DEFAULT_EMBEDDING_DIM`]; it MUST match the configured provider's output.
    pub async fn connect_with_dw_providers(
        config: ChronicleMemoryConfig,
        llm_provider: Arc<dyn LlmProvider>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        tenant: TenantCtx,
    ) -> Result<Self, LongTermMemoryError> {
        let driver = Self::connect_driver(&config).await?;
        let llm: Arc<dyn LlmClient> = Arc::new(DwLlmBridge::new(llm_provider, tenant.clone()));
        let embedding_dim = config.embedding_dim.unwrap_or(DEFAULT_EMBEDDING_DIM);
        let embedder: Arc<dyn EmbedderClient> = Arc::new(DwEmbedderBridge::new(
            embedding_provider,
            tenant,
            embedding_dim,
        ));

        Self::assemble(
            driver,
            llm,
            embedder,
            config.max_concurrency,
            config.recall_limit,
        )
        .await
    }

    /// Builds a provider directly from pre-constructed Chronicle clients.
    ///
    /// Intended for tests and dependency injection; does not provision indices or
    /// constraints (the caller's driver is assumed ready).
    #[must_use]
    pub fn from_parts(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
        recall_limit: usize,
    ) -> Self {
        let chronicle = Chronicle::new(driver, llm, embedder, max_concurrency);
        Self {
            chronicle,
            recall_limit,
        }
    }

    /// Connects to the configured graph store and returns it as a trait object,
    /// so the rest of the engine is backend-agnostic. The embedding dimension
    /// (used to size the vector index on the Falkor / Surreal backends) comes
    /// from `config.embedding_dim`, falling back to [`DEFAULT_EMBEDDING_DIM`].
    async fn connect_driver(
        config: &ChronicleMemoryConfig,
    ) -> Result<Arc<dyn GraphDriver>, LongTermMemoryError> {
        let embedding_dim = config.embedding_dim.unwrap_or(DEFAULT_EMBEDDING_DIM);
        let driver: Arc<dyn GraphDriver> = match &config.backend {
            ChronicleBackend::Neo4j {
                uri,
                user,
                password,
                database,
            } => Arc::new(
                Neo4jDriver::connect(uri, user, password, database.clone())
                    .await
                    .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?,
            ),
            ChronicleBackend::Falkor { connection, graph } => Arc::new(
                FalkorDriver::connect(connection, graph, embedding_dim)
                    .await
                    .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?,
            ),
            ChronicleBackend::SurrealEmbedded { path } => Arc::new(
                SurrealDriver::connect_embedded(path, embedding_dim)
                    .await
                    .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?,
            ),
            ChronicleBackend::SurrealMemory => Arc::new(
                SurrealDriver::connect_memory(embedding_dim)
                    .await
                    .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?,
            ),
        };
        Ok(driver)
    }

    fn openai_llm(
        config: &ChronicleMemoryConfig,
    ) -> Result<Arc<dyn LlmClient>, LongTermMemoryError> {
        let llm_config = LlmConfig {
            api_key: config.openai_api_key.clone(),
            model: config.model.clone(),
            small_model: config.small_model.clone(),
            base_url: config.openai_base_url.clone(),
            ..LlmConfig::default()
        };
        let llm =
            OpenAiLlm::new(llm_config).map_err(|e| LongTermMemoryError::Backend(e.to_string()))?;
        Ok(Arc::new(llm))
    }

    fn openai_embedder(
        config: &ChronicleMemoryConfig,
    ) -> Result<Arc<dyn EmbedderClient>, LongTermMemoryError> {
        let mut embedder_config = OpenAiEmbedderConfig {
            api_key: config.openai_api_key.clone(),
            base_url: config.openai_base_url.clone(),
            ..OpenAiEmbedderConfig::default()
        };
        if let Some(model) = &config.embedding_model {
            embedder_config.embedding_model = model.clone();
        }
        if let Some(dim) = config.embedding_dim {
            embedder_config.embedding_dim = dim;
        }
        let embedder = OpenAiEmbedder::new(embedder_config)
            .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?;
        Ok(Arc::new(embedder))
    }

    async fn assemble(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
        recall_limit: usize,
    ) -> Result<Self, LongTermMemoryError> {
        let chronicle = Chronicle::new(driver, llm, embedder, max_concurrency);
        chronicle
            .build_indices_and_constraints(false)
            .await
            .map_err(|e| LongTermMemoryError::Backend(e.to_string()))?;
        Ok(Self {
            chronicle,
            recall_limit,
        })
    }
}

/// Maps a contract [`EpisodeSource`] onto Chronicle's [`EpisodeType`].
fn map_source(source: EpisodeSource) -> EpisodeType {
    match source {
        EpisodeSource::Message => EpisodeType::Message,
        EpisodeSource::Text => EpisodeType::Text,
        EpisodeSource::Json => EpisodeType::Json,
    }
}

/// Extracts and validates the tenant identifier used as the Chronicle `group_id`.
///
/// The identifier must be non-empty and contain only ASCII alphanumerics, `_`, or
/// `-`. Anything else is rejected as [`LongTermMemoryError::InvalidTenant`] before
/// any backend call is made.
fn tenant_group_id(tenant: &TenantCtx) -> Result<String, LongTermMemoryError> {
    let id = tenant.tenant_id.as_str();
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if valid {
        Ok(id.to_string())
    } else {
        Err(LongTermMemoryError::InvalidTenant(id.to_string()))
    }
}

#[async_trait]
impl LongTermMemory for ChronicleLongTermMemory {
    async fn ingest_episode(
        &self,
        tenant: &TenantCtx,
        episode: EpisodeIngest,
    ) -> Result<IngestOutcome, LongTermMemoryError> {
        let group_id = tenant_group_id(tenant)?;

        let request = AddEpisodeRequest {
            name: episode.name,
            episode_body: episode.body,
            source: map_source(episode.source),
            source_description: episode.source_description.unwrap_or_default(),
            reference_time: episode.reference_time,
            group_id,
            uuid: None,
            previous_episode_uuids: None,
            entity_types: None,
            custom_extraction_instructions: None,
            // Phase-4 additive fields (`update_communities`, `saga`,
            // `saga_previous_episode_uuid`). This provider does not yet expose
            // communities or saga; defaulting preserves prior behaviour exactly
            // (`update_communities = false`, no saga threading).
            ..Default::default()
        };

        let results = self.chronicle.add_episode(request).await.map_err(|e| {
            tracing::error!(error = %e, "chronicle add_episode failed");
            LongTermMemoryError::Backend(e.to_string())
        })?;

        Ok(IngestOutcome {
            episode_id: results.episode.uuid,
            fact_count: results.edges.len(),
            entity_count: results.nodes.len(),
        })
    }

    async fn recall(
        &self,
        tenant: &TenantCtx,
        query: RecallQuery,
    ) -> Result<Vec<RecalledFact>, LongTermMemoryError> {
        let group_id = tenant_group_id(tenant)?;

        let mut config = edge_hybrid_search_rrf();
        config.limit = query.limit.unwrap_or(self.recall_limit);

        let edges = self
            .chronicle
            .search(&query.query, &[group_id], &config)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "chronicle search failed");
                LongTermMemoryError::Backend(e.to_string())
            })?;

        Ok(edges
            .into_iter()
            .map(|edge| RecalledFact {
                fact: edge.fact,
                relation: edge.name,
                valid_at: edge.valid_at,
                invalid_at: edge.invalid_at,
                source_episode_ids: edge.episodes,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};
    use chrono::{TimeZone, Utc};
    use greentic_types::{EnvId, TenantId};

    const EMB_DIM: usize = 8;

    fn tenant(value: &str) -> TenantCtx {
        TenantCtx::new(
            EnvId::new("dev").expect("env id"),
            TenantId::new(value).expect("tenant id"),
        )
    }

    fn fixed_ts() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .expect("valid timestamp")
    }

    // Scripted LLM responses for a single fresh-store episode producing two
    // entities (Alice, Acme) and one edge (Alice WORKS_AT Acme). Mirrors the call
    // order in chronicle-testkit's add_episode_e2e: extract_nodes → extract_edges →
    // summarize × 2 (one per node), with concurrency pinned to 1.
    fn scripted_llm() -> MockLlm {
        let extracted_entities = serde_json::json!({
            "extracted_entities": [
                {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
                {"name": "Acme", "entity_type_id": 0, "episode_indices": [0]}
            ]
        });
        let extracted_edge = serde_json::json!({
            "edges": [{
                "source_entity_name": "Alice",
                "target_entity_name": "Acme",
                "relation_type": "WORKS_AT",
                "fact": "Alice works at Acme.",
                "valid_at": "2025-01-01T00:00:00Z",
                "invalid_at": null,
                "episode_indices": [0]
            }]
        });
        let summary = serde_json::json!({"summary": "Employment relationship."});

        MockLlm::new(vec![
            extracted_entities,
            extracted_edge,
            summary.clone(),
            summary,
        ])
    }

    fn memory_with_scripted_llm() -> (ChronicleLongTermMemory, Arc<FakeDriver>) {
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
        let llm = Arc::new(scripted_llm());
        let memory = ChronicleLongTermMemory::from_parts(
            Arc::clone(&driver) as Arc<dyn GraphDriver>,
            llm,
            embedder,
            1, // serialize the summary fan-out
            5,
        );
        (memory, driver)
    }

    #[tokio::test]
    async fn ingest_then_recall_maps_fields() {
        let (memory, _driver) = memory_with_scripted_llm();
        let t = tenant("tenant-a");

        let episode = EpisodeIngest {
            name: "ep1".into(),
            body: "Alice works at Acme.".into(),
            source: EpisodeSource::Message,
            source_description: Some("test".into()),
            reference_time: fixed_ts(),
        };

        let outcome = memory
            .ingest_episode(&t, episode)
            .await
            .expect("ingest succeeds");
        assert_eq!(outcome.fact_count, 1, "one WORKS_AT edge expected");
        assert_eq!(outcome.entity_count, 2, "Alice + Acme expected");
        assert!(!outcome.episode_id.is_empty());

        let facts = memory
            .recall(
                &t,
                RecallQuery {
                    query: "where does Alice work".into(),
                    limit: None,
                },
            )
            .await
            .expect("recall succeeds");

        let acme = facts
            .iter()
            .find(|f| f.fact == "Alice works at Acme.")
            .expect("recall surfaces the Acme fact");
        assert_eq!(acme.relation, "WORKS_AT");
        assert_eq!(acme.valid_at, Some(fixed_ts()));
        assert!(acme.invalid_at.is_none());
        assert!(
            !acme.source_episode_ids.is_empty(),
            "fact must carry its source episode id"
        );
    }

    #[tokio::test]
    async fn invalid_tenant_is_rejected_without_backend_calls() {
        // FakeDriver + empty LLM queue: any backend/LLM call would surface as an
        // error other than InvalidTenant, so reaching InvalidTenant proves no call
        // was made.
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
        let llm = Arc::new(MockLlm::new(vec![]));
        let memory = ChronicleLongTermMemory::from_parts(
            Arc::clone(&driver) as Arc<dyn GraphDriver>,
            llm,
            embedder,
            1,
            5,
        );

        // greentic-types validates TenantId on construction, so build a ctx with a
        // valid id and then overwrite the inner newtype with a deliberately invalid
        // value to exercise our own validator.
        let mut t = tenant("tenant-a");
        t.tenant_id = TenantId("bad tenant!".to_string());

        let episode = EpisodeIngest {
            name: "ep".into(),
            body: "x".into(),
            source: EpisodeSource::Message,
            source_description: None,
            reference_time: fixed_ts(),
        };

        let err = memory
            .ingest_episode(&t, episode)
            .await
            .expect_err("invalid tenant must be rejected");
        assert!(matches!(err, LongTermMemoryError::InvalidTenant(_)));

        let err = memory
            .recall(
                &t,
                RecallQuery {
                    query: "anything".into(),
                    limit: None,
                },
            )
            .await
            .expect_err("invalid tenant must be rejected on recall");
        assert!(matches!(err, LongTermMemoryError::InvalidTenant(_)));
    }

    /// Tenant B must not see facts ingested by tenant A, even when both share the
    /// same in-memory [`FakeDriver`] instance. Tenant A recall is included as a
    /// positive control to confirm the fact is actually stored.
    #[tokio::test]
    async fn cross_tenant_isolation() {
        let (memory, _driver) = memory_with_scripted_llm();

        let tenant_a = tenant("tenant-a");
        let tenant_b = tenant("tenant-b");

        // Ingest an episode as tenant-a.
        let episode = EpisodeIngest {
            name: "ep1".into(),
            body: "Alice works at Acme.".into(),
            source: EpisodeSource::Message,
            source_description: Some("test".into()),
            reference_time: fixed_ts(),
        };
        memory
            .ingest_episode(&tenant_a, episode)
            .await
            .expect("ingest as tenant-a succeeds");

        // Tenant-b must not see tenant-a's facts.
        let facts_b = memory
            .recall(
                &tenant_b,
                RecallQuery {
                    query: "where does Alice work".into(),
                    limit: None,
                },
            )
            .await
            .expect("recall as tenant-b succeeds");
        assert!(
            facts_b.is_empty(),
            "tenant-b must not see tenant-a's facts; got: {facts_b:?}"
        );

        // Positive control: tenant-a should still see the fact.
        let facts_a = memory
            .recall(
                &tenant_a,
                RecallQuery {
                    query: "where does Alice work".into(),
                    limit: None,
                },
            )
            .await
            .expect("recall as tenant-a succeeds");
        assert!(
            !facts_a.is_empty(),
            "tenant-a must see its own ingested facts"
        );
    }

    /// Exercises a **real** embedded (in-memory) SurrealDB backend through the
    /// full Chronicle pipeline — no `FakeDriver` stub, no external infra. Proves
    /// the Surreal driver wiring end-to-end: connect → provision indices → ingest
    /// → recall. The LLM/embedder remain mocked (extraction is deterministic and
    /// backend-independent); only the graph store is real.
    #[tokio::test]
    async fn surreal_memory_backend_ingest_then_recall() {
        // `connect_memory` already provisions the schema + HNSW / search indices
        // (via `schema_ddl`), so the store is ready to ingest and recall.
        let driver = SurrealDriver::connect_memory(EMB_DIM)
            .await
            .expect("connect in-memory surreal");
        let memory = ChronicleLongTermMemory::from_parts(
            Arc::new(driver) as Arc<dyn GraphDriver>,
            Arc::new(scripted_llm()),
            Arc::new(MockEmbedder::new(EMB_DIM)),
            1, // serialize the summary fan-out (matches scripted_llm ordering)
            5,
        );

        let t = tenant("tenant-surreal");
        let outcome = memory
            .ingest_episode(
                &t,
                EpisodeIngest {
                    name: "ep1".into(),
                    body: "Alice works at Acme.".into(),
                    source: EpisodeSource::Message,
                    source_description: Some("test".into()),
                    reference_time: fixed_ts(),
                },
            )
            .await
            .expect("ingest into surreal succeeds");
        assert_eq!(outcome.entity_count, 2, "Alice + Acme expected");
        assert_eq!(outcome.fact_count, 1, "one WORKS_AT edge expected");

        let facts = memory
            .recall(
                &t,
                RecallQuery {
                    query: "where does Alice work".into(),
                    limit: None,
                },
            )
            .await
            .expect("recall from surreal succeeds");
        assert!(
            facts.iter().any(|f| f.fact == "Alice works at Acme."),
            "surreal recall must surface the ingested fact; got {facts:?}"
        );
    }
}
