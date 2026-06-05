// Ported from graphiti_core/nodes.py @ 34f56e65 (v0.29.1)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Mirrors `EntityNode` in upstream `nodes.py`.
///
/// Omitted upstream fields (all Phase-4+ or driver-only):
/// - All DB query/driver methods (`save`, `delete`, `get_by_*`) — persistence
///   layer belongs in `chronicle-driver-*` crates, not the domain type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    /// Always contains "Entity"; specific entity-type labels appended.
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub summary: String,
    pub attributes: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_embedding: Option<Vec<f32>>,
}

impl EntityNode {
    pub fn new(name: String, group_id: String, created_at: DateTime<Utc>) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            name,
            group_id,
            labels: vec!["Entity".to_string()],
            created_at,
            summary: String::new(),
            attributes: Map::new(),
            name_embedding: None,
        }
    }
}
