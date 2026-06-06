// Ported from graphiti_core/nodes.py @ 34f56e65 (CommunityNode)
//              graphiti_core/edges.py @ 34f56e65 (CommunityEdge / HAS_MEMBER)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Mirrors `CommunityNode` in upstream `nodes.py`.
///
/// Carries the label-propagation cluster identity for a set of `Entity` nodes.
/// No bi-temporal fields — only `created_at` (transaction time).
///
/// Upstream fields: uuid / name / group_id / labels (["Community"]) /
/// created_at / name_embedding / summary.
///
/// Omitted upstream fields (driver-only):
/// - All DB query/driver methods — persistence layer belongs in
///   `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    /// Always `["Community"]`.
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    /// Aggregate summary of community members. Defaults to empty string.
    #[serde(default)]
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_embedding: Option<Vec<f32>>,
}

impl CommunityNode {
    pub fn new(name: String, group_id: String, created_at: DateTime<Utc>) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            name,
            group_id,
            labels: vec!["Community".to_string()],
            created_at,
            summary: String::new(),
            name_embedding: None,
        }
    }
}

/// HAS_MEMBER edge: Community → Entity|Community.
///
/// Mirrors `CommunityEdge` in upstream `edges.py`.
///
/// Omitted upstream fields (driver-only):
/// - All DB query/driver methods — belong in `chronicle-driver-*` crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityEdge {
    pub uuid: String,
    /// Community node UUID (source of the HAS_MEMBER relationship).
    pub source_node_uuid: String,
    /// Entity or Community node UUID (target of the HAS_MEMBER relationship).
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
}

impl CommunityEdge {
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
    fn community_node_roundtrip_without_embedding() {
        let node = CommunityNode::new("Tech Companies".into(), "g1".into(), Utc::now());
        assert_eq!(node.labels, vec!["Community"]);
        assert!(node.summary.is_empty());
        assert!(node.name_embedding.is_none());

        let json = serde_json::to_string(&node).unwrap();
        // name_embedding must be absent when None (skip_serializing_if)
        assert!(
            !json.contains("name_embedding"),
            "embedding should be absent in serialized form"
        );
        let back: CommunityNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uuid, node.uuid);
        assert_eq!(back.name, "Tech Companies");
        assert_eq!(back.group_id, "g1");
        assert_eq!(back.labels, vec!["Community"]);
        assert!(back.name_embedding.is_none());
    }

    #[test]
    fn community_node_roundtrip_with_embedding() {
        let mut node = CommunityNode::new("Finance".into(), "g2".into(), Utc::now());
        node.name_embedding = Some(vec![0.1_f32, 0.2, 0.3]);
        node.summary = "Finance sector community".into();

        let json = serde_json::to_string(&node).unwrap();
        let back: CommunityNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.summary, "Finance sector community");
        let emb = back.name_embedding.unwrap();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 0.1_f32).abs() < 1e-6);
    }

    #[test]
    fn community_node_deserializes_without_summary_field() {
        // Driver-boundary: absent `summary` must default to empty string.
        let json = r#"{
            "uuid":"c1","name":"N","group_id":"g1",
            "labels":["Community"],
            "created_at":"2026-01-01T00:00:00Z"
        }"#;
        let node: CommunityNode = serde_json::from_str(json).unwrap();
        assert!(node.summary.is_empty());
        assert!(node.name_embedding.is_none());
    }

    #[test]
    fn community_edge_roundtrip() {
        let edge = CommunityEdge::new("src".into(), "tgt".into(), "g1".into(), Utc::now());
        let json = serde_json::to_string(&edge).unwrap();
        let back: CommunityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uuid, edge.uuid);
        assert_eq!(back.source_node_uuid, "src");
        assert_eq!(back.target_node_uuid, "tgt");
        assert_eq!(back.group_id, "g1");
    }
}
