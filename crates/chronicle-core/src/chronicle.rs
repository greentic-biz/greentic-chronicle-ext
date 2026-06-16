// Public facade over the Phase-1 core loop. Mirrors the user-facing surface of
// upstream `graphiti_core/graphiti.py::Graphiti` for the operations the Rust port
// supports today: add_episode, retrieve_episodes, search, and schema setup.

use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;
use crate::helpers::SEMAPHORE_LIMIT;
use crate::llm::LlmClient;
use crate::pipeline::add_episode::add_episode;
use crate::pipeline::bulk::{
    AddBulkEpisodeResults, RawEpisode, add_episode_bulk as pipeline_add_episode_bulk,
};
use crate::pipeline::clients::Clients;
use crate::pipeline::community_ops::rebuild_communities;
use crate::pipeline::maintenance::{
    AddTripletResults, add_triplet as pipeline_add_triplet,
    get_nodes_and_edges_by_episode as pipeline_get_nodes_and_edges_by_episode,
    remove_episode as pipeline_remove_episode,
};
use crate::pipeline::saga::summarize_saga as pipeline_summarize_saga;
use crate::search::filters::SearchFilters;
use crate::search::recipes::{
    combined_hybrid_search_cross_encoder, edge_hybrid_search_node_distance, edge_hybrid_search_rrf,
};
use crate::search::results::SearchResults;
use crate::search::{SearchConfig, edge_search_simple, search};
use crate::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    SagaNode,
};

/// Request to ingest a single episode.
///
/// Mirrors upstream `add_episode`'s keyword arguments. A [`Default`] impl is
/// provided so callers can construct with `..Default::default()` and remain
/// source-compatible when additive optional fields are introduced later. The
/// six required fields (`name` / `episode_body` / `source` / `source_description`
/// / `reference_time` / `group_id`) default to empty/`Text`/epoch and MUST be set
/// by the caller.
///
/// Phase-4 additive fields (`update_communities`, `saga`,
/// `saga_previous_episode_uuid`) are new this release. The Phase-5 dw-providers
/// crate pins v0.2.0 and constructs this struct by named fields; it is unaffected
/// until it bumps to v0.3.0, at which point it should adopt
/// `..Default::default()` to stay forward-compatible.
pub struct AddEpisodeRequest {
    pub name: String,
    pub episode_body: String,
    pub source: EpisodeType,
    pub source_description: String,
    pub reference_time: DateTime<Utc>,
    pub group_id: String,
    /// When set, update an existing episode instead of creating a new one.
    pub uuid: Option<String>,
    /// When set, use these episodes as prior context instead of retrieving the
    /// last-N by reference time.
    pub previous_episode_uuids: Option<Vec<String>>,
    /// Custom entity-type registry (Phase-2 attribute extraction). Passed through
    /// to node extraction; `None` uses the default `Entity` type only.
    pub entity_types: Option<serde_json::Value>,
    /// Extra free-text instructions appended to extraction prompts.
    pub custom_extraction_instructions: Option<String>,
    /// Phase-4 additive: when `true`, run `update_community` for each final node
    /// after ingestion (neighbour-vote community membership). Defaults to `false`
    /// (upstream `update_communities=False`).
    pub update_communities: bool,
    /// Phase-4 additive: when set, thread this episode into the named saga
    /// (get-or-create by `(name, group_id)`, chain NEXT_EPISODE/HAS_EPISODE).
    pub saga: Option<String>,
    /// Phase-4 additive: explicit previous-episode UUID for saga chaining; skips
    /// the driver query when the caller already knows the prior episode.
    pub saga_previous_episode_uuid: Option<String>,
}

impl Default for AddEpisodeRequest {
    fn default() -> Self {
        Self {
            name: String::new(),
            episode_body: String::new(),
            source: EpisodeType::Text,
            source_description: String::new(),
            reference_time: DateTime::<Utc>::from(std::time::UNIX_EPOCH),
            group_id: String::new(),
            uuid: None,
            previous_episode_uuids: None,
            entity_types: None,
            custom_extraction_instructions: None,
            update_communities: false,
            saga: None,
            saga_previous_episode_uuid: None,
        }
    }
}

/// Result of [`Chronicle::add_episode`]. Mirrors upstream `AddEpisodeResults`.
///
/// `communities` / `community_edges` are populated only when the request set
/// `update_communities = true`; otherwise they are empty (upstream parity).
pub struct AddEpisodeResults {
    pub episode: EpisodicNode,
    pub episodic_edges: Vec<EpisodicEdge>,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
    /// Communities updated during this ingest (empty unless
    /// `update_communities` was requested).
    pub communities: Vec<CommunityNode>,
    /// HAS_MEMBER edges created/updated during this ingest (empty unless
    /// `update_communities` was requested).
    pub community_edges: Vec<CommunityEdge>,
}

/// Chronicle engine. Bundles the driver / LLM / embedder clients and the shared
/// concurrency semaphore, plus an optional cross-encoder reranker.
pub struct Chronicle {
    clients: Clients,
    /// Optional cross-encoder reranker. Wired via [`Chronicle::with_cross_encoder`].
    /// Required for cross-encoder recipes (e.g. the `search_()` default
    /// `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`); absent → those paths surface
    /// [`ChronicleError::InvalidInput`].
    cross_encoder: Option<Arc<dyn CrossEncoderClient>>,
}

impl Chronicle {
    /// Build a Chronicle over the given clients with a concurrency cap of
    /// `max_concurrency` (pass `0` to fall back to [`SEMAPHORE_LIMIT`]).
    pub fn new(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
    ) -> Self {
        let permits = if max_concurrency == 0 {
            SEMAPHORE_LIMIT
        } else {
            max_concurrency
        };
        Self {
            clients: Clients::new(driver, llm, embedder, permits),
            cross_encoder: None,
        }
    }

    /// Attach a cross-encoder reranker, enabling cross-encoder recipes (notably
    /// the [`Chronicle::search_`] default `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`).
    /// Builder-style: returns `self` for chaining off [`Chronicle::new`].
    #[must_use]
    pub fn with_cross_encoder(mut self, cross_encoder: Arc<dyn CrossEncoderClient>) -> Self {
        self.cross_encoder = Some(cross_encoder);
        self
    }

    /// Ingest a single episode. See [`crate::pipeline::add_episode::add_episode`].
    pub async fn add_episode(
        &self,
        req: AddEpisodeRequest,
    ) -> Result<AddEpisodeResults, ChronicleError> {
        add_episode(&self.clients, req).await
    }

    /// Ingest multiple episodes in one batch with cross-episode entity/edge
    /// dedup. See [`crate::pipeline::bulk::add_episode_bulk`].
    ///
    /// `group_id` scopes the batch; `saga` optionally threads all episodes into a
    /// named saga (valid_at-ordered NEXT_EPISODE chain). Communities are never
    /// updated in the bulk path (upstream — the result's `communities` /
    /// `community_edges` are always empty).
    pub async fn add_episode_bulk(
        &self,
        episodes: Vec<RawEpisode>,
        group_id: &str,
        saga: Option<&str>,
    ) -> Result<AddBulkEpisodeResults, ChronicleError> {
        pipeline_add_episode_bulk(&self.clients, episodes, group_id, saga, None, None).await
    }

    /// Insert a single `(source)-[edge]->(target)` triplet with full node/edge
    /// resolution + invalidation, but no episode/episodic-edge/community side
    /// effects. See [`crate::pipeline::maintenance::add_triplet`].
    pub async fn add_triplet(
        &self,
        source_node: EntityNode,
        edge: EntityEdge,
        target_node: EntityNode,
    ) -> Result<AddTripletResults, ChronicleError> {
        pipeline_add_triplet(&self.clients, source_node, edge, target_node).await
    }

    /// Remove an episode and the graph state it solely contributed (primary-
    /// source edges + single-mention nodes + the episode). See
    /// [`crate::pipeline::maintenance::remove_episode`].
    pub async fn remove_episode(&self, episode_uuid: &str) -> Result<(), ChronicleError> {
        pipeline_remove_episode(&self.clients, episode_uuid).await
    }

    /// Rebuild communities for the given groups (DETACH DELETE all `:Community` →
    /// detect → summarize → persist). Empty `group_ids` rebuilds over the whole
    /// graph. See [`crate::pipeline::community_ops::rebuild_communities`].
    pub async fn build_communities(
        &self,
        group_ids: &[String],
    ) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), ChronicleError> {
        rebuild_communities(&self.clients, group_ids).await
    }

    /// Incrementally summarize a saga using only episodes added since the last
    /// run. See [`crate::pipeline::saga::summarize_saga`].
    pub async fn summarize_saga(&self, saga_uuid: &str) -> Result<SagaNode, ChronicleError> {
        pipeline_summarize_saga(&self.clients, saga_uuid).await
    }

    /// Collect the nodes and entity edges attributed to a single episode. See
    /// [`crate::pipeline::maintenance::get_nodes_and_edges_by_episode`].
    pub async fn get_nodes_and_edges_by_episode(
        &self,
        episode_uuid: &str,
    ) -> Result<SearchResults, ChronicleError> {
        pipeline_get_nodes_and_edges_by_episode(&self.clients, episode_uuid).await
    }

    /// Retrieve the last-`last_n` episodes with `valid_at <= reference_time`,
    /// in chronological order.
    pub async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
    ) -> Result<Vec<EpisodicNode>, ChronicleError> {
        Ok(self
            .clients
            .driver
            .retrieve_episodes(reference_time, last_n, group_ids, None)
            .await?)
    }

    /// Edge-only hybrid search with explicit [`SearchConfig`] control.
    ///
    /// Returns edges only (drops reranker scores). The wired cross-encoder (if
    /// any) is forwarded, so cross-encoder edge recipes work once
    /// [`Chronicle::with_cross_encoder`] has been called. No filters are applied
    /// — for advanced filtering / multi-scope results use [`Chronicle::search_`].
    pub async fn search(
        &self,
        query: &str,
        group_ids: &[String],
        config: &SearchConfig,
    ) -> Result<Vec<EntityEdge>, ChronicleError> {
        edge_search_simple(
            self.clients.driver.as_ref(),
            self.clients.embedder.as_ref(),
            self.cross_encoder.as_deref(),
            query,
            group_ids,
            config,
            &SearchFilters::default(),
        )
        .await
    }

    /// Edge-only hybrid search with upstream `search()` recipe-routing (R13).
    ///
    /// Mirrors upstream `graphiti_core/graphiti.py::Graphiti.search`:
    /// - `center_node_uuid == None` → `EDGE_HYBRID_SEARCH_RRF`;
    /// - `center_node_uuid == Some(..)` → `EDGE_HYBRID_SEARCH_NODE_DISTANCE`
    ///   (the center is forwarded to the node-distance reranker);
    /// - `limit` = `num_results`.
    ///
    /// Returns edges only. The base [`Chronicle::search`] is kept for callers that
    /// want direct [`SearchConfig`] control.
    pub async fn search_with_center(
        &self,
        query: &str,
        group_ids: &[String],
        num_results: usize,
        center_node_uuid: Option<&str>,
    ) -> Result<Vec<EntityEdge>, ChronicleError> {
        let mut config = match center_node_uuid {
            None => edge_hybrid_search_rrf(),
            Some(_) => edge_hybrid_search_node_distance(),
        };
        config.limit = num_results;

        // Forward the center via the full top-level search so the node-distance
        // reranker receives it (edge_search_simple does not take a center).
        let results = search(
            self.clients.driver.as_ref(),
            self.clients.embedder.as_ref(),
            self.cross_encoder.as_deref(),
            query,
            group_ids,
            &config,
            &SearchFilters::default(),
            center_node_uuid,
            None,
        )
        .await?;
        Ok(results.edges)
    }

    /// Advanced multi-scope search (R13 `search_()`).
    ///
    /// Mirrors upstream `graphiti_core/graphiti.py::Graphiti.search_`:
    /// - `config == None` → default `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`
    ///   (requires a wired cross-encoder — otherwise the cross-encoder scope
    ///   surfaces [`ChronicleError::InvalidInput`]);
    /// - `filters == None` → default (empty) [`SearchFilters`];
    /// - returns the full [`SearchResults`] (edges / nodes / episodes + scores).
    #[allow(clippy::too_many_arguments)]
    pub async fn search_(
        &self,
        query: &str,
        config: Option<&SearchConfig>,
        group_ids: &[String],
        center_node_uuid: Option<&str>,
        bfs_origin_node_uuids: Option<&[String]>,
        filters: Option<&SearchFilters>,
    ) -> Result<SearchResults, ChronicleError> {
        let default_config = combined_hybrid_search_cross_encoder();
        let config = config.unwrap_or(&default_config);
        let default_filters = SearchFilters::default();
        let filters = filters.unwrap_or(&default_filters);

        search(
            self.clients.driver.as_ref(),
            self.clients.embedder.as_ref(),
            self.cross_encoder.as_deref(),
            query,
            group_ids,
            config,
            filters,
            center_node_uuid,
            bfs_origin_node_uuids,
        )
        .await
    }

    /// Ingest pre-chunked documents into the knowledge store (no LLM extraction).
    /// See [`crate::document_rag`].
    pub async fn ingest_document_chunks(
        &self,
        chunks: Vec<crate::document_rag::DocumentChunk>,
        group_id: &str,
    ) -> Result<Vec<String>, ChronicleError> {
        crate::document_rag::ingest_chunks(&self.clients, chunks, group_id).await
    }

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

    /// Build the backend's indices / constraints, optionally dropping existing
    /// data first.
    pub async fn build_indices_and_constraints(
        &self,
        delete_existing: bool,
    ) -> Result<(), ChronicleError> {
        Ok(self
            .clients
            .driver
            .build_indices_and_constraints(delete_existing)
            .await?)
    }
}
