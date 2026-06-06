// Ported from graphiti_core/nodes.py @ 34f56e65 (SagaNode)
//              graphiti_core/edges.py @ 34f56e65 (HasEpisodeEdge, NextEpisodeEdge)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Mirrors `SagaNode` in upstream `nodes.py`.
///
/// Groups a sequence of `EpisodicNode` instances into a narrative thread.
/// Two watermark fields track the incremental summary state:
/// - `last_summarized_at`: wall-clock time of the last `summarize_saga` run.
/// - `last_summarized_episode_valid_at`: maximum episode `valid_at` across the
///   episodes covered by the most recent summary (episode-time semantics).
///
/// Upstream fields: uuid / name / group_id / labels (["Saga"]) / created_at /
/// summary / first_episode_uuid / last_episode_uuid / last_summarized_at /
/// last_summarized_episode_valid_at.
///
/// Omitted upstream fields (driver-only):
/// - All DB query/driver methods — persistence layer belongs in
///   `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    /// Always `["Saga"]`.
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    /// Incremental narrative summary. Defaults to empty string.
    #[serde(default)]
    pub summary: String,
    /// UUID of the first episode in this saga's chain.
    #[serde(default)]
    pub first_episode_uuid: Option<String>,
    /// UUID of the most recently appended episode.
    #[serde(default)]
    pub last_episode_uuid: Option<String>,
    /// Wall-clock timestamp of the last `summarize_saga` run.
    #[serde(default)]
    pub last_summarized_at: Option<DateTime<Utc>>,
    /// Maximum `valid_at` across episodes covered by the last summary.
    #[serde(default)]
    pub last_summarized_episode_valid_at: Option<DateTime<Utc>>,
}

impl SagaNode {
    pub fn new(name: String, group_id: String, created_at: DateTime<Utc>) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            name,
            group_id,
            labels: vec!["Saga".to_string()],
            created_at,
            summary: String::new(),
            first_episode_uuid: None,
            last_episode_uuid: None,
            last_summarized_at: None,
            last_summarized_episode_valid_at: None,
        }
    }
}

/// HAS_EPISODE edge: Saga → Episodic.
///
/// Mirrors `HasEpisodeEdge` in upstream `edges.py`.
/// Links a saga node to each of its episodic nodes.
///
/// Omitted upstream fields (driver-only):
/// - All DB query/driver methods — belong in `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasEpisodeEdge {
    pub uuid: String,
    /// Saga node UUID (source of the HAS_EPISODE relationship).
    pub source_node_uuid: String,
    /// Episodic node UUID (target of the HAS_EPISODE relationship).
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
}

impl HasEpisodeEdge {
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

/// NEXT_EPISODE edge: Episodic → Episodic.
///
/// Mirrors `NextEpisodeEdge` in upstream `edges.py`.
/// Forms the sequential chain of episodes within a saga.
///
/// Omitted upstream fields (driver-only):
/// - All DB query/driver methods — belong in `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextEpisodeEdge {
    pub uuid: String,
    /// Previous episodic node UUID (source of the NEXT_EPISODE relationship).
    pub source_node_uuid: String,
    /// Current episodic node UUID (target of the NEXT_EPISODE relationship).
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
}

impl NextEpisodeEdge {
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
    use chrono::Utc;

    #[test]
    fn saga_node_roundtrip_all_options_none() {
        let node = SagaNode::new("customer-onboarding".into(), "g1".into(), Utc::now());
        assert_eq!(node.labels, vec!["Saga"]);
        assert!(node.summary.is_empty());
        assert!(node.first_episode_uuid.is_none());
        assert!(node.last_episode_uuid.is_none());
        assert!(node.last_summarized_at.is_none());
        assert!(node.last_summarized_episode_valid_at.is_none());

        let json = serde_json::to_string(&node).unwrap();
        let back: SagaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uuid, node.uuid);
        assert_eq!(back.name, "customer-onboarding");
        assert!(back.last_summarized_at.is_none());
        assert!(back.last_summarized_episode_valid_at.is_none());
    }

    #[test]
    fn saga_node_roundtrip_with_watermarks() {
        let now = Utc::now();
        let mut node = SagaNode::new("support-tickets".into(), "g2".into(), now);
        node.summary = "User reported issues with login".into();
        node.first_episode_uuid = Some("ep-001".into());
        node.last_episode_uuid = Some("ep-005".into());
        node.last_summarized_at = Some(now);
        node.last_summarized_episode_valid_at = Some(now);

        let json = serde_json::to_string(&node).unwrap();
        let back: SagaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.summary, "User reported issues with login");
        assert_eq!(back.first_episode_uuid.as_deref(), Some("ep-001"));
        assert_eq!(back.last_episode_uuid.as_deref(), Some("ep-005"));
        assert!(back.last_summarized_at.is_some());
        assert!(back.last_summarized_episode_valid_at.is_some());
    }

    #[test]
    fn saga_node_deserializes_without_optional_fields() {
        // Driver-boundary: absent optional fields must default to None/empty.
        let json = r#"{
            "uuid":"s1","name":"my-saga","group_id":"g1",
            "labels":["Saga"],
            "created_at":"2026-01-01T00:00:00Z"
        }"#;
        let node: SagaNode = serde_json::from_str(json).unwrap();
        assert!(node.summary.is_empty());
        assert!(node.first_episode_uuid.is_none());
        assert!(node.last_episode_uuid.is_none());
        assert!(node.last_summarized_at.is_none());
        assert!(node.last_summarized_episode_valid_at.is_none());
    }

    #[test]
    fn has_episode_edge_roundtrip() {
        let edge = HasEpisodeEdge::new("saga-1".into(), "ep-1".into(), "g1".into(), Utc::now());
        let json = serde_json::to_string(&edge).unwrap();
        let back: HasEpisodeEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uuid, edge.uuid);
        assert_eq!(back.source_node_uuid, "saga-1");
        assert_eq!(back.target_node_uuid, "ep-1");
        assert_eq!(back.group_id, "g1");
    }

    #[test]
    fn next_episode_edge_roundtrip() {
        let edge = NextEpisodeEdge::new("ep-1".into(), "ep-2".into(), "g1".into(), Utc::now());
        let json = serde_json::to_string(&edge).unwrap();
        let back: NextEpisodeEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uuid, edge.uuid);
        assert_eq!(back.source_node_uuid, "ep-1");
        assert_eq!(back.target_node_uuid, "ep-2");
        assert_eq!(back.group_id, "g1");
    }
}
