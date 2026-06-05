// Ported from graphiti_core/nodes.py @ 34f56e65 (v0.29.1)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Mirrors `EpisodeType` in upstream `nodes.py`.
///
/// NOTE: `fact_triple` exists in upstream but is excluded here — it is a
/// specialised LLM-pre-processing variant that has no distinct storage shape
/// and is not used in the Phase-1 core loop. It can be added in a later phase
/// if needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeType {
    Message,
    Text,
    Json,
}

/// Mirrors `EpisodicNode` in upstream `nodes.py`.
///
/// Omitted upstream fields (all Phase-4+ or driver-only):
/// - `episode_metadata: dict[str, Any] | None` — customer filter metadata;
///   deferred to Phase 4 (metadata-filtering slice).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub source: EpisodeType,
    pub source_description: String,
    pub content: String,
    /// UUIDs of EntityEdges derived from this episode.
    pub entity_edges: Vec<String>,
    pub created_at: DateTime<Utc>,
    /// When the original episode document was created/occurred (upstream: "datetime of when the original document was created").
    pub valid_at: DateTime<Utc>,
}

impl EpisodicNode {
    pub fn new(
        name: String,
        group_id: String,
        source: EpisodeType,
        source_description: String,
        content: String,
        created_at: DateTime<Utc>,
        valid_at: DateTime<Utc>,
    ) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            name,
            group_id,
            labels: Vec::new(),
            source,
            source_description,
            content,
            entity_edges: Vec::new(),
            created_at,
            valid_at,
        }
    }
}
