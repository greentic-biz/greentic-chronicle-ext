// Public facade over the Phase-1 core loop. Mirrors the user-facing surface of
// upstream `graphiti_core/graphiti.py::Graphiti` for the operations the Rust port
// supports today: add_episode, retrieve_episodes, search, and schema setup.

use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;
use crate::helpers::SEMAPHORE_LIMIT;
use crate::llm::LlmClient;
use crate::pipeline::add_episode::add_episode;
use crate::pipeline::clients::Clients;
use crate::search::{SearchConfig, edge_search};
use crate::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};

/// Request to ingest a single episode. All fields are explicit — there is no
/// `Default`, mirroring upstream `add_episode`'s required keyword arguments.
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
}

/// Result of [`Chronicle::add_episode`]. Mirrors upstream `AddEpisodeResults`
/// minus the community fields (Phase-2+).
pub struct AddEpisodeResults {
    pub episode: EpisodicNode,
    pub episodic_edges: Vec<EpisodicEdge>,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
}

/// Phase-1 Chronicle engine. Bundles the driver / LLM / embedder clients and the
/// shared concurrency semaphore.
pub struct Chronicle {
    clients: Clients,
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
        }
    }

    /// Ingest a single episode. See [`crate::pipeline::add_episode::add_episode`].
    pub async fn add_episode(
        &self,
        req: AddEpisodeRequest,
    ) -> Result<AddEpisodeResults, ChronicleError> {
        add_episode(&self.clients, req).await
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

    /// Hybrid edge search (BM25 + cosine, RRF-fused) over the given query.
    pub async fn search(
        &self,
        query: &str,
        group_ids: &[String],
        config: &SearchConfig,
    ) -> Result<Vec<EntityEdge>, ChronicleError> {
        edge_search(
            self.clients.driver.as_ref(),
            self.clients.embedder.as_ref(),
            query,
            group_ids,
            config,
        )
        .await
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
