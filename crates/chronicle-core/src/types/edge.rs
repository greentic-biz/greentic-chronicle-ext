// Ported from graphiti_core/edges.py @ 34f56e65 (v0.29.1)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Temporal entity-to-entity fact. Bi-temporal:
/// `created_at`/`expired_at` = transaction time, `valid_at`/`invalid_at` = valid time.
///
/// Mirrors `EntityEdge` in upstream `edges.py`.
///
/// Omitted upstream fields:
/// - `reference_time: datetime | None` — episode-level reference timestamp used
///   transiently during ingestion to seed `valid_at`; not stored as a separate
///   field in Phase 1. Will be addressed when the ingestion pipeline stores
///   provenance (Phase 3+).
/// - All DB driver methods — belong in `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    /// SCREAMING_SNAKE_CASE relation type (e.g. `WORKS_AT`).
    pub name: String,
    pub fact: String,
    pub group_id: String,
    /// Episode UUIDs in which this fact was mentioned.
    pub episodes: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expired_at: Option<DateTime<Utc>>,
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    pub attributes: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_embedding: Option<Vec<f32>>,
}

impl EntityEdge {
    pub fn new(
        source_node_uuid: String,
        target_node_uuid: String,
        name: String,
        fact: String,
        group_id: String,
    ) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            source_node_uuid,
            target_node_uuid,
            name,
            fact,
            group_id,
            episodes: Vec::new(),
            created_at: crate::helpers::utc_now(),
            expired_at: None,
            valid_at: None,
            invalid_at: None,
            attributes: Map::new(),
            fact_embedding: None,
        }
    }
}

/// MENTIONS edge: episode → entity.
///
/// Mirrors `EpisodicEdge` in upstream `edges.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
}

impl EpisodicEdge {
    pub fn new(
        source_node_uuid: String,
        target_node_uuid: String,
        group_id: String,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            source_node_uuid,
            target_node_uuid,
            group_id,
            created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_edge_roundtrips_with_bitemporal_fields() {
        let edge = EntityEdge::new(
            "u1".into(),
            "u2".into(),
            "WORKS_AT".into(),
            "Alice works at Acme".into(),
            "g1".into(),
        );
        assert!(edge.valid_at.is_none() && edge.invalid_at.is_none() && edge.expired_at.is_none());
        let json = serde_json::to_string(&edge).unwrap();
        let back: EntityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fact, "Alice works at Acme");
    }
}
