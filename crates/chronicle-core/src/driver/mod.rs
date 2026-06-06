// Operation-level port of graphiti_core/driver/driver.py @ 34f56e65 (v0.29.1).
// Deliberate design divergence from upstream: backends implement typed
// operations instead of receiving raw Cypher strings (execute_query). This
// keeps non-Cypher/embedded backends honest and prevents dialect lock-in.
// SearchFilters/BFS params join in Phase 2 — extend, don't redesign.
use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::search::filters::SearchFilters;
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

/// Vector / fulltext / BFS / rerank-support search primitives the backend must
/// provide.
///
/// Phase 2 extends this trait additively:
///   - The four original search methods gain a `filters: &SearchFilters` param
///     (upstream threads SearchFilters into every scope query — see
///     `graphiti_core/search/search_utils.py` edge/node fulltext+similarity).
///   - BFS traversal (`node_bfs_search`, `edge_bfs_search`) — plan R9.
///   - Embedding loaders for MMR reranking — plan R3/R4.
///   - Reranker-support primitives (`nodes_connected_to_center`,
///     `episode_mention_counts`) — plan R7/R8.
///   - Episode fulltext (`episode_fulltext_search`) — plan R5.
#[async_trait]
pub trait SearchOps: Send + Sync {
    async fn edge_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn node_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError>;
    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError>;

    /// Breadth-first traversal returning Entity nodes reachable from `origins`
    /// within `1..=max_depth` directed `RELATES_TO`/`MENTIONS` hops.
    ///
    /// Upstream (plan R9, `search_utils.py::node_bfs_search`):
    /// `MATCH (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..N]->(n:Entity)
    ///  WHERE n.group_id = origin.group_id {filters} RETURN ... LIMIT $limit`.
    /// Origins are label-free (Entity or Episodic); when `group_ids` is
    /// non-empty, both `n.group_id` and `origin.group_id` must be in the list.
    /// Returns empty when `origins` is empty or `max_depth < 1`.
    async fn node_bfs_search(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError>;

    /// Breadth-first traversal returning the `RELATES_TO` edges traversed along
    /// paths from `origins` within `1..=max_depth` hops.
    ///
    /// Upstream (plan R9, `search_utils.py::edge_bfs_search`): path expansion
    /// `MATCH path = (origin {uuid})-[:RELATES_TO|MENTIONS*1..N]->(:Entity)
    ///  UNWIND relationships(path) AS rel
    ///  MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity) {filters}
    ///  RETURN DISTINCT ... LIMIT $limit`. MENTIONS hops extend reach but only
    /// `RELATES_TO` edges are returned (the re-MATCH only resolves them).
    /// Returns empty when `origins` is empty or `max_depth < 1`.
    async fn edge_bfs_search(
        &self,
        origins: &[String],
        max_depth: usize,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError>;

    /// Fulltext (BM25) search over Episodic nodes — plan R5.
    /// Upstream uses the `episode_content` fulltext index; backends without a
    /// real BM25 index may approximate (document the approximation).
    async fn episode_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError>;

    /// Load `name_embedding` vectors for the given Entity node UUIDs.
    /// Used by the MMR reranker (plan R4). UUIDs without a stored embedding are
    /// omitted from the returned map.
    async fn get_embeddings_for_nodes(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError>;

    /// Load `fact_embedding` vectors for the given Entity edge UUIDs.
    /// Used by the MMR reranker (plan R3). UUIDs without a stored embedding are
    /// omitted from the returned map.
    async fn get_embeddings_for_edges(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError>;

    /// 1-hop **undirected** `RELATES_TO` adjacency to `center_uuid` (plan R7).
    ///
    /// Upstream `node_distance_reranker` Cypher:
    /// `MATCH (center:Entity {uuid:$center_uuid})-[:RELATES_TO]-(n:Entity {uuid:node_uuid})`
    /// (single undirected hop). Returns the subset of `node_uuids` that are
    /// adjacent to `center_uuid`.
    async fn nodes_connected_to_center(
        &self,
        node_uuids: &[String],
        center_uuid: &str,
    ) -> Result<Vec<String>, DriverError>;

    /// `MENTIONS` in-degree per Entity node UUID (plan R8).
    ///
    /// Upstream `episode_mentions_reranker` Cypher:
    /// `MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid:node_uuid})
    ///  RETURN count(*) AS score`. UUIDs with no mentions are omitted from the
    /// returned map (the reranker treats absent UUIDs as count 0 / `inf` rank).
    async fn episode_mention_counts(
        &self,
        node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError>;
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
            _filters: &SearchFilters,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn edge_similarity_search(
            &self,
            _search_vector: &[f32],
            _filters: &SearchFilters,
            _group_ids: &[String],
            _limit: usize,
            _min_score: f32,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn node_fulltext_search(
            &self,
            _query: &str,
            _filters: &SearchFilters,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn node_similarity_search(
            &self,
            _search_vector: &[f32],
            _filters: &SearchFilters,
            _group_ids: &[String],
            _limit: usize,
            _min_score: f32,
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn node_bfs_search(
            &self,
            _origins: &[String],
            _filters: &SearchFilters,
            _max_depth: usize,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn edge_bfs_search(
            &self,
            _origins: &[String],
            _max_depth: usize,
            _filters: &SearchFilters,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn episode_fulltext_search(
            &self,
            _query: &str,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<EpisodicNode>, DriverError> {
            Ok(vec![])
        }

        async fn get_embeddings_for_nodes(
            &self,
            _uuids: &[String],
        ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
            Ok(HashMap::new())
        }

        async fn get_embeddings_for_edges(
            &self,
            _uuids: &[String],
        ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
            Ok(HashMap::new())
        }

        async fn nodes_connected_to_center(
            &self,
            _node_uuids: &[String],
            _center_uuid: &str,
        ) -> Result<Vec<String>, DriverError> {
            Ok(vec![])
        }

        async fn episode_mention_counts(
            &self,
            _node_uuids: &[String],
        ) -> Result<HashMap<String, u64>, DriverError> {
            Ok(HashMap::new())
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
