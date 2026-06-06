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
use crate::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

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

    /// Entity nodes scoped to the given `group_ids` (upstream
    /// `EntityNode.get_by_group_ids`, Neo4j branch). Empty `group_ids` returns
    /// empty. Used by community-cluster projection (plan R2).
    async fn get_entity_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError>;

    /// Entity nodes `MENTIONS`-targeted by any of the given episodes (upstream
    /// `get_mentioned_nodes`, plan R5/R10):
    /// `MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) WHERE episode.uuid IN
    /// $uuids RETURN DISTINCT ...`. Empty `episode_uuids` returns empty.
    async fn get_mentioned_nodes(
        &self,
        episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError>;

    /// `DETACH DELETE` Entity nodes by UUID (upstream `Node.delete_by_uuids`,
    /// plan R5 cascade). Empty `uuids` is a no-op.
    async fn delete_entity_nodes_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError>;
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

    /// Entity edges by UUID (upstream `EntityEdge.get_by_uuids`, Neo4j branch):
    /// `MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.uuid IN $uuids
    /// RETURN ...`. Used by `remove_episode` (plan R5) and
    /// `get_nodes_and_edges_by_episode` (plan R10). Empty `uuids` returns empty.
    async fn get_entity_edges_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityEdge>, DriverError>;

    /// `DELETE` `RELATES_TO` edges by UUID (upstream `Edge.delete_by_uuids`,
    /// plan R5 cascade). Empty `uuids` is a no-op.
    async fn delete_entity_edges_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError>;
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

    /// `DETACH DELETE` a single Episodic node by UUID (upstream
    /// `EpisodicNode.delete`, plan R5 cascade). Missing uuid is a no-op.
    async fn delete_episode(&self, uuid: &str) -> Result<(), DriverError>;
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

/// One entity neighbour in a community-cluster projection row (plan R2).
///
/// Mirrors upstream `Neighbor` (community_operations.py): a neighbour entity
/// UUID and the count of `RELATES_TO` edges connecting it to the projected node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighbor {
    pub node_uuid: String,
    pub edge_count: u64,
}

/// The per-node adjacency projection for one node in a group (plan R2).
///
/// `node_uuid` is the projected Entity node; `neighbors` is its `RELATES_TO`
/// adjacency with edge counts (upstream `projection[node.uuid] = [Neighbor...]`).
#[derive(Debug, Clone)]
pub struct NodeNeighbors {
    pub node_uuid: String,
    pub neighbors: Vec<Neighbor>,
}

/// One group's complete cluster projection (plan R2).
///
/// `group_id` + the full per-node adjacency list. `label_propagation` (Task 3)
/// consumes the `nodes` list to assign each node an integer community and groups
/// by the converged label. Keeping the projection grouped by `group_id` mirrors
/// upstream `get_community_clusters`, which iterates groups and runs label
/// propagation independently per group.
#[derive(Debug, Clone)]
pub struct GroupClusterProjection {
    pub group_id: String,
    pub nodes: Vec<NodeNeighbors>,
}

/// Community persistence + search + membership primitives (plan R1/R2/R3/R4/R8).
///
/// Ported from `graphiti_core/utils/maintenance/community_operations.py` and the
/// community branches of `graphiti_core/search/search_utils.py` @ 34f56e65, plus
/// `node_db_queries.py` / `edge_db_queries.py` community save/return queries.
#[async_trait]
pub trait CommunityOps: Send + Sync {
    /// Upsert community nodes (upstream `get_community_node_save_query`, Neo4j
    /// branch — MERGE by uuid, SET scalar props, set `name_embedding` vector).
    async fn save_community_nodes(&self, nodes: &[CommunityNode]) -> Result<(), DriverError>;

    /// Upsert HAS_MEMBER edges (upstream `get_community_edge_save_query`, Neo4j
    /// branch — `MATCH (community:Community) MATCH (node:Entity|Community) MERGE
    /// (community)-[e:HAS_MEMBER {uuid}]->(node)`).
    async fn save_community_edges(&self, edges: &[CommunityEdge]) -> Result<(), DriverError>;

    /// Community nodes scoped to `group_ids` (upstream
    /// `CommunityNode.get_by_group_ids`). Empty `group_ids` returns empty.
    async fn get_community_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError>;

    /// Community nodes by UUID (upstream `CommunityNode.get_by_uuids`). Empty
    /// `uuids` returns empty.
    async fn get_community_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError>;

    /// Fulltext (BM25) community search (plan R8 `community_fulltext_search`,
    /// `community_name` index, group filter, ORDER BY score DESC LIMIT). Empty
    /// query short-circuits to empty.
    async fn community_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError>;

    /// Cosine similarity community search over `name_embedding` (plan R8
    /// `community_similarity_search`, `score > min_score`, ORDER BY score DESC
    /// LIMIT).
    async fn community_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError>;

    /// Load `name_embedding` vectors for the given Community UUIDs (plan R8
    /// `get_embeddings_for_communities`, MMR reranker support). UUIDs without a
    /// stored embedding are omitted.
    async fn get_embeddings_for_communities(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError>;

    /// `DETACH DELETE` all Community nodes in scope (plan R3 `remove_communities`:
    /// `MATCH (c:Community) DETACH DELETE c`). Always full-graph (no group scope)
    /// per upstream.
    async fn remove_communities(&self) -> Result<(), DriverError>;

    /// Per-group entity adjacency projection driving label propagation (plan R2
    /// `get_community_clusters`). For each `group_id` (or all distinct entity
    /// group_ids when `group_ids` is empty), returns each node's `RELATES_TO`
    /// neighbour list with per-neighbour edge counts.
    async fn get_community_clusters(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<GroupClusterProjection>, DriverError>;

    /// Already-member lookup (plan R4 `determine_entity_community` step a):
    /// `MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid}) RETURN <community>`.
    /// Returns the first community the entity already belongs to, if any.
    async fn community_of_member(
        &self,
        entity_uuid: &str,
    ) -> Result<Option<CommunityNode>, DriverError>;

    /// Neighbour-vote lookup (plan R4 `determine_entity_community` step b):
    /// `MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity
    /// {uuid}) RETURN <community>`. Returns ONE row per neighbour's community
    /// (NOT deduplicated); the caller (Task 3) does the mode/plurality count.
    async fn neighbor_communities(
        &self,
        entity_uuid: &str,
    ) -> Result<Vec<CommunityNode>, DriverError>;
}

/// Saga narrative-thread persistence + threading queries (plan R9).
///
/// Ported from `graphiti_core/graphiti.py` saga helpers (`get_or_create_saga`,
/// `_saga_get_previous_episode_uuid`, `_saga_get_episode_contents`) and the saga
/// save/return queries in `node_db_queries.py` / `edge_db_queries.py` @ 34f56e65.
#[async_trait]
pub trait SagaOps: Send + Sync {
    /// Upsert a saga node (upstream `get_saga_node_save_query`, Neo4j branch).
    async fn save_saga_node(&self, node: &SagaNode) -> Result<(), DriverError>;

    /// Upsert a HAS_EPISODE edge Saga→Episodic (upstream `HAS_EPISODE_EDGE_SAVE`).
    async fn save_has_episode_edge(&self, edge: &HasEpisodeEdge) -> Result<(), DriverError>;

    /// Upsert a NEXT_EPISODE edge Episodic→Episodic (upstream
    /// `NEXT_EPISODE_EDGE_SAVE`).
    async fn save_next_episode_edge(&self, edge: &NextEpisodeEdge) -> Result<(), DriverError>;

    /// Get-or-create lookup by `(name, group_id)` (plan R9 `get_or_create_saga`):
    /// `MATCH (s:Saga {name, group_id}) RETURN ...`. Returns the existing saga
    /// when present; creation is the caller's responsibility (this is the lookup
    /// half only).
    async fn get_saga_by_name(
        &self,
        name: &str,
        group_id: &str,
    ) -> Result<Option<SagaNode>, DriverError>;

    /// Saga by UUID (upstream `SagaNode.get_by_uuid`, used by `summarize_saga`):
    /// `MATCH (s:Saga {uuid}) RETURN <saga>`. Returns `None` when absent — the
    /// caller (`pipeline::saga::summarize_saga`) maps that to
    /// [`crate::errors::ChronicleError::NodeNotFound`].
    async fn get_saga_by_uuid(&self, uuid: &str) -> Result<Option<SagaNode>, DriverError>;

    /// Most-recent prior episode in a saga (plan R9 `_saga_get_previous_episode_uuid`):
    /// `MATCH (s:Saga {uuid})-[:HAS_EPISODE]->(e:Episodic) WHERE e.uuid <>
    /// $current ORDER BY e.valid_at DESC, e.created_at DESC LIMIT 1`. Returns the
    /// previous episode UUID for chaining the NEXT_EPISODE edge.
    async fn saga_previous_episode_uuid(
        &self,
        saga_uuid: &str,
        current_episode_uuid: &str,
    ) -> Result<Option<String>, DriverError>;

    /// `(content, valid_at)` per saga episode for summarization (plan R9
    /// `_saga_get_episode_contents` / `summarize_saga` fetch). When `since` is
    /// `Some`, filters `e.created_at > $since` and returns chronological
    /// (`valid_at ASC`); when `None`, returns all (chronological). `limit` caps
    /// the row count.
    async fn saga_episode_contents(
        &self,
        saga_uuid: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError>;
}

#[async_trait]
pub trait SchemaOps: Send + Sync {
    async fn build_indices_and_constraints(&self, delete_existing: bool)
    -> Result<(), DriverError>;
}

/// Composite storage backend contract (upstream GraphDriver, operation-level).
pub trait GraphDriver:
    EntityNodeOps
    + EntityEdgeOps
    + EpisodeOps
    + EpisodicEdgeOps
    + SearchOps
    + CommunityOps
    + SagaOps
    + SchemaOps
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

        async fn get_entity_nodes_by_group_ids(
            &self,
            _group_ids: &[String],
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn get_mentioned_nodes(
            &self,
            _episode_uuids: &[String],
        ) -> Result<Vec<EntityNode>, DriverError> {
            Ok(vec![])
        }

        async fn delete_entity_nodes_by_uuids(&self, _uuids: &[String]) -> Result<(), DriverError> {
            Ok(())
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

        async fn get_entity_edges_by_uuids(
            &self,
            _uuids: &[String],
        ) -> Result<Vec<EntityEdge>, DriverError> {
            Ok(vec![])
        }

        async fn delete_entity_edges_by_uuids(&self, _uuids: &[String]) -> Result<(), DriverError> {
            Ok(())
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

        async fn delete_episode(&self, _uuid: &str) -> Result<(), DriverError> {
            Ok(())
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
    impl CommunityOps for NullDriver {
        async fn save_community_nodes(&self, _nodes: &[CommunityNode]) -> Result<(), DriverError> {
            Ok(())
        }

        async fn save_community_edges(&self, _edges: &[CommunityEdge]) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_community_nodes_by_group_ids(
            &self,
            _group_ids: &[String],
        ) -> Result<Vec<CommunityNode>, DriverError> {
            Ok(vec![])
        }

        async fn get_community_nodes_by_uuids(
            &self,
            _uuids: &[String],
        ) -> Result<Vec<CommunityNode>, DriverError> {
            Ok(vec![])
        }

        async fn community_fulltext_search(
            &self,
            _query: &str,
            _group_ids: &[String],
            _limit: usize,
        ) -> Result<Vec<CommunityNode>, DriverError> {
            Ok(vec![])
        }

        async fn community_similarity_search(
            &self,
            _search_vector: &[f32],
            _group_ids: &[String],
            _limit: usize,
            _min_score: f32,
        ) -> Result<Vec<CommunityNode>, DriverError> {
            Ok(vec![])
        }

        async fn get_embeddings_for_communities(
            &self,
            _uuids: &[String],
        ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
            Ok(HashMap::new())
        }

        async fn remove_communities(&self) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_community_clusters(
            &self,
            _group_ids: &[String],
        ) -> Result<Vec<GroupClusterProjection>, DriverError> {
            Ok(vec![])
        }

        async fn community_of_member(
            &self,
            _entity_uuid: &str,
        ) -> Result<Option<CommunityNode>, DriverError> {
            Ok(None)
        }

        async fn neighbor_communities(
            &self,
            _entity_uuid: &str,
        ) -> Result<Vec<CommunityNode>, DriverError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl SagaOps for NullDriver {
        async fn save_saga_node(&self, _node: &SagaNode) -> Result<(), DriverError> {
            Ok(())
        }

        async fn save_has_episode_edge(&self, _edge: &HasEpisodeEdge) -> Result<(), DriverError> {
            Ok(())
        }

        async fn save_next_episode_edge(&self, _edge: &NextEpisodeEdge) -> Result<(), DriverError> {
            Ok(())
        }

        async fn get_saga_by_name(
            &self,
            _name: &str,
            _group_id: &str,
        ) -> Result<Option<SagaNode>, DriverError> {
            Ok(None)
        }

        async fn get_saga_by_uuid(&self, _uuid: &str) -> Result<Option<SagaNode>, DriverError> {
            Ok(None)
        }

        async fn saga_previous_episode_uuid(
            &self,
            _saga_uuid: &str,
            _current_episode_uuid: &str,
        ) -> Result<Option<String>, DriverError> {
            Ok(None)
        }

        async fn saga_episode_contents(
            &self,
            _saga_uuid: &str,
            _since: Option<DateTime<Utc>>,
            _limit: usize,
        ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError> {
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
