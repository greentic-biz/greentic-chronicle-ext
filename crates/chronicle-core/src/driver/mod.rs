// Operation-level port of graphiti_core/driver/driver.py @ 34f56e65 (v0.29.1).
// Deliberate design divergence from upstream: backends implement typed
// operations instead of receiving raw Cypher strings (execute_query). This
// keeps non-Cypher/embedded backends honest and prevents dialect lock-in.
// SearchFilters/BFS params join in Phase 2 — extend, don't redesign.
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};

#[derive(Debug, Error)]
pub enum DriverError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("query error: {0}")]
    Query(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("decode error: {0}")]
    Decode(String),
}

#[async_trait]
pub trait EntityNodeOps: Send + Sync {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError>;
    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError>;
    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError>;
}

#[async_trait]
pub trait EntityEdgeOps: Send + Sync {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError>;
    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError>;
    /// Edges from source_uuid -> target_uuid (single direction, mirrors upstream
    /// EntityEdge.get_between_nodes / Neo4j MATCH (n)-[e:RELATES_TO]->(m)).
    /// Used as the node-pair duplicate-candidate pool in resolve_extracted_edges;
    /// upstream additionally re-ranks this pool via hybrid search
    /// (EDGE_HYBRID_SEARCH_RRF + SearchFilters(edge_uuids=...)) — that re-ranking
    /// happens in pipeline code, not in the driver.
    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError>;
}

#[async_trait]
pub trait EpisodeOps: Send + Sync {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError>;
    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError>;
    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError>;
    /// Last-n episodes with valid_at <= reference_time, returned in
    /// chronological order (upstream retrieve_episodes).
    /// Tie order for equal valid_at values is backend-dependent (upstream has no secondary sort key).
    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError>;
}

#[async_trait]
pub trait EpisodicEdgeOps: Send + Sync {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError>;
}

/// Vector / fulltext search primitives the backend must provide.
/// BFS traversal joins in Phase 2.
#[async_trait]
pub trait SearchOps: Send + Sync {
    async fn edge_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn node_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError>;
    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError>;
}

#[async_trait]
pub trait SchemaOps: Send + Sync {
    async fn build_indices_and_constraints(&self, delete_existing: bool)
    -> Result<(), DriverError>;
}

/// Composite storage backend contract (upstream GraphDriver, operation-level).
pub trait GraphDriver:
    EntityNodeOps + EntityEdgeOps + EpisodeOps + EpisodicEdgeOps + SearchOps + SchemaOps
{
    fn provider(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct NullDriver;

    #[async_trait]
    impl EntityNodeOps for NullDriver {
        async fn save_entity_nodes(&self, _nodes: &[EntityNode]) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_entity_node(&self, _uuid: &str) -> Result<Option<EntityNode>, DriverError> {
            Ok(None)
        }

        async fn get_entity_nodes_by_uuids(
            &self,
            _uuids: &[String],
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl EntityEdgeOps for NullDriver {
        async fn save_entity_edges(&self, _edges: &[EntityEdge]) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_entity_edge(&self, _uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
            Ok(None)
        }

        async fn get_edges_between_nodes(
            &self,
            _source_uuid: &str,
            _target_uuid: &str,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl EpisodeOps for NullDriver {
        async fn save_episode(&self, _episode: &EpisodicNode) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_episode(&self, _uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
            Ok(None)
        }

        async fn get_episodes_by_uuids(
            &self,
            _uuids: &[String],
        ) -> Result<Vec<EpisodicNode>, DriverError> {
            Ok(vec![])
        }

        async fn retrieve_episodes(
            &self,
            _reference_time: DateTime<Utc>,
            _last_n: usize,
            _group_ids: &[String],
            _source: Option<EpisodeType>,
        ) -> Result<Vec<EpisodicNode>, DriverError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl EpisodicEdgeOps for NullDriver {
        async fn save_episodic_edges(&self, _edges: &[EpisodicEdge]) -> Result<(), DriverError> {
            Ok(())
        }
    }

    #[async_trait]
    impl SearchOps for NullDriver {
        async fn edge_fulltext_search(
            &self,
            _query: &str,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn edge_similarity_search(
            &self,
            _search_vector: &[f32],
            _group_ids: &[String],
            _limit: usize,
            _min_score: f32,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn node_fulltext_search(
            &self,
            _query: &str,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn node_similarity_search(
            &self,
            _search_vector: &[f32],
            _group_ids: &[String],
            _limit: usize,
            _min_score: f32,
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl SchemaOps for NullDriver {
        async fn build_indices_and_constraints(
            &self,
            _delete_existing: bool,
        ) -> Result<(), DriverError> {
            Ok(())
        }
    }

    impl GraphDriver for NullDriver {
        fn provider(&self) -> &'static str {
            "null"
        }
    }

    #[tokio::test]
    async fn graph_driver_supertrait_is_object_safe() {
        let d: Arc<dyn GraphDriver> = Arc::new(NullDriver);
        assert_eq!(d.provider(), "null");
        assert!(d.get_entity_node("x").await.unwrap().is_none());
    }
}
