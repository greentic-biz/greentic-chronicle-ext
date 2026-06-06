//! Row structs and chronicle <-> SurrealDB conversion for the embedded backend.
//!
//! Each chronicle node/edge type has a `*Row` struct that derives
//! [`surrealdb::types::SurrealValue`] so it can be `bind()`-ed into a query and
//! `take()`-n back out. The rows mirror the SurrealDB column shape:
//!
//! - **Datetimes** use [`surrealdb::types::Datetime`], never `chrono` directly
//!   (locked decision #1; chrono-at-the-boundary historically serialised as a
//!   string and broke datetime comparisons — issues #2753/#2804). chrono ↔
//!   surreal `Datetime` conversion is via `From`/`into_inner` at this boundary.
//! - **Embeddings** are `Option<Vec<f32>>` → native `array<float>`.
//! - **Attributes** are carried as `serde_json::Value` (an object), for which
//!   surrealdb 3.1.3 provides a `SurrealValue` impl directly.
//! - The **record id** is keyed by the chronicle uuid (`entity:⟨uuid⟩`), but we
//!   ALSO store `uuid` as a plain field and only ever read THAT back, so we
//!   never have to parse a `RecordId` key into a String.

use chrono::{DateTime, Utc};
use serde_json::{Map, Value};
use surrealdb::types::{Datetime, RecordId, SurrealValue};

use chronicle_core::driver::DriverError;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

// ─────────────────────────────────────────────────────────────────────────
// Scalar helpers
// ─────────────────────────────────────────────────────────────────────────

/// chrono `DateTime<Utc>` → surreal `Datetime` (boundary conversion).
pub fn to_dt(dt: DateTime<Utc>) -> Datetime {
    Datetime::from(dt)
}

/// surreal `Datetime` → chrono `DateTime<Utc>`.
pub fn from_dt(dt: Datetime) -> DateTime<Utc> {
    dt.into_inner()
}

/// `Option<DateTime<Utc>>` → `Option<Datetime>`.
fn opt_to_dt(dt: Option<DateTime<Utc>>) -> Option<Datetime> {
    dt.map(to_dt)
}

/// `Option<Datetime>` → `Option<DateTime<Utc>>`.
fn opt_from_dt(dt: Option<Datetime>) -> Option<DateTime<Utc>> {
    dt.map(from_dt)
}

/// A chronicle attributes map → a `serde_json::Value::Object` for storage.
fn attrs_to_value(attrs: &Map<String, Value>) -> Value {
    Value::Object(attrs.clone())
}

/// A stored attributes `Value` → a chronicle attributes map (non-objects and
/// null map to an empty map, matching the "flat attributes only" contract).
fn attrs_from_value(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// The record id for a node/edge table keyed by the chronicle uuid.
pub fn record_id(table: &str, uuid: &str) -> RecordId {
    RecordId::new(table, uuid)
}

// ─────────────────────────────────────────────────────────────────────────
// EpisodeType <-> stored string (lowercase value, mirrors the Neo4j driver)
// ─────────────────────────────────────────────────────────────────────────

/// Map `EpisodeType` to its stored string (lowercase, matching upstream's
/// `source.value`).
pub fn episode_type_to_str(t: EpisodeType) -> &'static str {
    match t {
        EpisodeType::Message => "message",
        EpisodeType::Text => "text",
        EpisodeType::Json => "json",
    }
}

fn episode_type_from_str(s: &str) -> Result<EpisodeType, DriverError> {
    match s {
        "message" => Ok(EpisodeType::Message),
        "text" => Ok(EpisodeType::Text),
        "json" => Ok(EpisodeType::Json),
        other => Err(DriverError::Decode(format!(
            "unknown episode source '{other}'"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Row structs (derive SurrealValue for bind/take)
// ─────────────────────────────────────────────────────────────────────────

/// Stored shape of an entity node.
#[derive(SurrealValue)]
pub struct EntityNodeRow {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: Datetime,
    pub summary: String,
    pub attributes: Value,
    pub name_embedding: Option<Vec<f32>>,
}

/// Stored shape of an episodic node.
#[derive(SurrealValue)]
pub struct EpisodicNodeRow {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub source: String,
    pub source_description: String,
    pub content: String,
    pub entity_edges: Vec<String>,
    pub created_at: Datetime,
    pub valid_at: Datetime,
}

/// Stored shape of a community node.
#[derive(SurrealValue)]
pub struct CommunityNodeRow {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub created_at: Datetime,
    pub summary: String,
    pub name_embedding: Option<Vec<f32>>,
}

/// Stored shape of a saga node.
#[derive(SurrealValue)]
pub struct SagaNodeRow {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub created_at: Datetime,
    pub summary: String,
    pub first_episode_uuid: Option<String>,
    pub last_episode_uuid: Option<String>,
    pub last_summarized_at: Option<Datetime>,
    pub last_summarized_episode_valid_at: Option<Datetime>,
}

/// Stored property shape of a `relates_to` edge. The graph endpoints (`in`/`out`)
/// are set by `RELATE`; these are the user-facing fields plus the source/target
/// uuids (carried explicitly so reads never need to resolve the record links).
#[derive(SurrealValue)]
pub struct EntityEdgeRow {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub name: String,
    pub fact: String,
    pub group_id: String,
    pub episodes: Vec<String>,
    pub created_at: Datetime,
    pub expired_at: Option<Datetime>,
    pub valid_at: Option<Datetime>,
    pub invalid_at: Option<Datetime>,
    pub attributes: Value,
    pub fact_embedding: Option<Vec<f32>>,
}

/// Stored property shape of a simple link edge (`mentions` / `has_member` /
/// `has_episode` / `next_episode`). All four share the same field set.
#[derive(SurrealValue)]
pub struct LinkEdgeRow {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: Datetime,
}

// ─────────────────────────────────────────────────────────────────────────
// chronicle -> row
// ─────────────────────────────────────────────────────────────────────────

pub fn entity_node_to_row(n: &EntityNode) -> EntityNodeRow {
    let mut labels = n.labels.clone();
    if !labels.iter().any(|l| l == "Entity") {
        labels.push("Entity".to_string());
    }
    EntityNodeRow {
        uuid: n.uuid.clone(),
        name: n.name.clone(),
        group_id: n.group_id.clone(),
        labels,
        created_at: to_dt(n.created_at),
        summary: n.summary.clone(),
        attributes: attrs_to_value(&n.attributes),
        name_embedding: n.name_embedding.clone(),
    }
}

pub fn episode_to_row(e: &EpisodicNode) -> EpisodicNodeRow {
    EpisodicNodeRow {
        uuid: e.uuid.clone(),
        name: e.name.clone(),
        group_id: e.group_id.clone(),
        source: episode_type_to_str(e.source).to_string(),
        source_description: e.source_description.clone(),
        content: e.content.clone(),
        entity_edges: e.entity_edges.clone(),
        created_at: to_dt(e.created_at),
        valid_at: to_dt(e.valid_at),
    }
}

pub fn community_node_to_row(n: &CommunityNode) -> CommunityNodeRow {
    CommunityNodeRow {
        uuid: n.uuid.clone(),
        name: n.name.clone(),
        group_id: n.group_id.clone(),
        created_at: to_dt(n.created_at),
        summary: n.summary.clone(),
        name_embedding: n.name_embedding.clone(),
    }
}

pub fn saga_node_to_row(n: &SagaNode) -> SagaNodeRow {
    SagaNodeRow {
        uuid: n.uuid.clone(),
        name: n.name.clone(),
        group_id: n.group_id.clone(),
        created_at: to_dt(n.created_at),
        summary: n.summary.clone(),
        first_episode_uuid: n.first_episode_uuid.clone(),
        last_episode_uuid: n.last_episode_uuid.clone(),
        last_summarized_at: opt_to_dt(n.last_summarized_at),
        last_summarized_episode_valid_at: opt_to_dt(n.last_summarized_episode_valid_at),
    }
}

pub fn entity_edge_to_row(e: &EntityEdge) -> EntityEdgeRow {
    EntityEdgeRow {
        uuid: e.uuid.clone(),
        source_node_uuid: e.source_node_uuid.clone(),
        target_node_uuid: e.target_node_uuid.clone(),
        name: e.name.clone(),
        fact: e.fact.clone(),
        group_id: e.group_id.clone(),
        episodes: e.episodes.clone(),
        created_at: to_dt(e.created_at),
        expired_at: opt_to_dt(e.expired_at),
        valid_at: opt_to_dt(e.valid_at),
        invalid_at: opt_to_dt(e.invalid_at),
        attributes: attrs_to_value(&e.attributes),
        fact_embedding: e.fact_embedding.clone(),
    }
}

pub fn episodic_edge_to_row(e: &EpisodicEdge) -> LinkEdgeRow {
    LinkEdgeRow {
        uuid: e.uuid.clone(),
        source_node_uuid: e.source_node_uuid.clone(),
        target_node_uuid: e.target_node_uuid.clone(),
        group_id: e.group_id.clone(),
        created_at: to_dt(e.created_at),
    }
}

pub fn community_edge_to_row(e: &CommunityEdge) -> LinkEdgeRow {
    LinkEdgeRow {
        uuid: e.uuid.clone(),
        source_node_uuid: e.source_node_uuid.clone(),
        target_node_uuid: e.target_node_uuid.clone(),
        group_id: e.group_id.clone(),
        created_at: to_dt(e.created_at),
    }
}

pub fn has_episode_edge_to_row(e: &HasEpisodeEdge) -> LinkEdgeRow {
    LinkEdgeRow {
        uuid: e.uuid.clone(),
        source_node_uuid: e.source_node_uuid.clone(),
        target_node_uuid: e.target_node_uuid.clone(),
        group_id: e.group_id.clone(),
        created_at: to_dt(e.created_at),
    }
}

pub fn next_episode_edge_to_row(e: &NextEpisodeEdge) -> LinkEdgeRow {
    LinkEdgeRow {
        uuid: e.uuid.clone(),
        source_node_uuid: e.source_node_uuid.clone(),
        target_node_uuid: e.target_node_uuid.clone(),
        group_id: e.group_id.clone(),
        created_at: to_dt(e.created_at),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// row -> chronicle
// ─────────────────────────────────────────────────────────────────────────

pub fn entity_node_from_row(r: EntityNodeRow) -> EntityNode {
    EntityNode {
        uuid: r.uuid,
        name: r.name,
        group_id: r.group_id,
        labels: r.labels,
        created_at: from_dt(r.created_at),
        summary: r.summary,
        attributes: attrs_from_value(r.attributes),
        name_embedding: r.name_embedding,
    }
}

pub fn episode_from_row(r: EpisodicNodeRow) -> Result<EpisodicNode, DriverError> {
    Ok(EpisodicNode {
        uuid: r.uuid,
        name: r.name,
        group_id: r.group_id,
        labels: Vec::new(),
        source: episode_type_from_str(&r.source)?,
        source_description: r.source_description,
        content: r.content,
        entity_edges: r.entity_edges,
        created_at: from_dt(r.created_at),
        valid_at: from_dt(r.valid_at),
    })
}

pub fn community_node_from_row(r: CommunityNodeRow) -> CommunityNode {
    CommunityNode {
        uuid: r.uuid,
        name: r.name,
        group_id: r.group_id,
        labels: vec!["Community".to_string()],
        created_at: from_dt(r.created_at),
        summary: r.summary,
        name_embedding: r.name_embedding,
    }
}

pub fn saga_node_from_row(r: SagaNodeRow) -> SagaNode {
    SagaNode {
        uuid: r.uuid,
        name: r.name,
        group_id: r.group_id,
        labels: vec!["Saga".to_string()],
        created_at: from_dt(r.created_at),
        summary: r.summary,
        first_episode_uuid: r.first_episode_uuid,
        last_episode_uuid: r.last_episode_uuid,
        last_summarized_at: opt_from_dt(r.last_summarized_at),
        last_summarized_episode_valid_at: opt_from_dt(r.last_summarized_episode_valid_at),
    }
}

pub fn entity_edge_from_row(r: EntityEdgeRow) -> EntityEdge {
    EntityEdge {
        uuid: r.uuid,
        source_node_uuid: r.source_node_uuid,
        target_node_uuid: r.target_node_uuid,
        name: r.name,
        fact: r.fact,
        group_id: r.group_id,
        episodes: r.episodes,
        created_at: from_dt(r.created_at),
        expired_at: opt_from_dt(r.expired_at),
        valid_at: opt_from_dt(r.valid_at),
        invalid_at: opt_from_dt(r.invalid_at),
        attributes: attrs_from_value(r.attributes),
        fact_embedding: r.fact_embedding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_core::types::EntityNode;
    use chrono::Utc;

    #[test]
    fn entity_node_row_roundtrip_preserves_fields() {
        let mut node = EntityNode::new("Alice".into(), "g1".into(), Utc::now());
        node.summary = "summary".into();
        node.name_embedding = Some(vec![0.1, 0.2, 0.3]);
        node.attributes
            .insert("role".into(), Value::String("admin".into()));

        let row = entity_node_to_row(&node);
        let back = entity_node_from_row(row);

        assert_eq!(back.uuid, node.uuid);
        assert_eq!(back.name, "Alice");
        assert_eq!(back.summary, "summary");
        assert_eq!(back.name_embedding, Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(
            back.attributes.get("role"),
            Some(&Value::String("admin".into()))
        );
        assert!(back.labels.contains(&"Entity".to_string()));
    }

    #[test]
    fn datetime_boundary_roundtrip_is_lossless() {
        let now = Utc::now();
        assert_eq!(from_dt(to_dt(now)), now);
    }

    #[test]
    fn episode_type_str_roundtrip() {
        for t in [EpisodeType::Message, EpisodeType::Text, EpisodeType::Json] {
            assert_eq!(episode_type_from_str(episode_type_to_str(t)).unwrap(), t);
        }
        assert!(episode_type_from_str("bogus").is_err());
    }
}
