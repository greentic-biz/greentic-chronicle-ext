#![forbid(unsafe_code)]

//! Neo4j `GraphDriver` backend for chronicle, over neo4rs 0.8 (Bolt).
//!
//! Port of `graphiti_core/driver/neo4j_driver.py` (and the operation files under
//! `graphiti_core/driver/neo4j/operations/`) @ 34f56e65 (v0.29.1), adapted to the
//! typed operation-level driver traits in `chronicle-core::driver`.
//!
//! ## Cross-call atomicity (CLOSED in Phase 4)
//!
//! Upstream `add_episode` / `add_nodes_and_edges_bulk` persist episode + entity
//! nodes + entity edges + episodic edges inside a single Neo4j transaction. The
//! chronicle driver trait still exposes four independent save operations
//! (`save_episode`, `save_entity_nodes`, `save_entity_edges`,
//! `save_episodic_edges`) — each its OWN transaction, atomic only *within* the
//! op — for granular callers.
//!
//! Phase 4 adds [`BulkSaveOps::save_all`], which the `add_episode` and bulk
//! persist tails now call: it wraps all four writes in ONE `start_txn → run all →
//! commit` transaction (rollback on any error), so a mid-batch failure leaves the
//! graph unchanged. This closes the carried-forward group-atomicity gap for the
//! Neo4j backend; the in-memory `FakeDriver` inherits the default sequential
//! `save_all` (nothing can partially fail in memory).

mod convert;
mod queries;

pub use queries::{MAX_QUERY_LENGTH, build_fulltext_query, validate_group_ids};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use neo4rs::{Graph, Query, query};
use tracing::debug;

use std::collections::HashMap;

use chronicle_core::driver::{
    BulkSaveOps, CommunityOps, DriverError, EntityEdgeOps, EntityNodeOps, EpisodeOps,
    EpisodicEdgeOps, GraphDriver, GroupClusterProjection, Neighbor, NodeNeighbors, SagaOps,
    SchemaOps, SearchOps,
};
use chronicle_core::search::filters::SearchFilters;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

/// Neo4j-backed `GraphDriver`.
pub struct Neo4jDriver {
    graph: Graph,
    database: String,
}

impl Neo4jDriver {
    /// Connect to Neo4j over Bolt. Maps neo4rs connection failures to
    /// `DriverError::Connection`.
    pub async fn connect(
        uri: &str,
        user: &str,
        password: &str,
        database: impl Into<String>,
    ) -> Result<Self, DriverError> {
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(|e| DriverError::Connection(e.to_string()))?;
        Ok(Self {
            graph,
            database: database.into(),
        })
    }

    /// Run a write query (no result rows) on the configured database.
    async fn run(&self, q: Query, ctx: &str) -> Result<(), DriverError> {
        self.graph
            .run_on(&self.database, q)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))
    }

    /// Run a write query inside its own transaction, draining the result stream
    /// so the server completes the write before commit.
    async fn run_in_txn(&self, q: Query, ctx: &str) -> Result<(), DriverError> {
        let mut txn = self
            .graph
            .start_txn_on(self.database.as_str())
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: begin txn: {e}")))?;
        let mut stream = txn
            .execute(q)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: execute: {e}")))?;
        // Drain so the write is fully applied.
        while stream
            .next(txn.handle())
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: drain: {e}")))?
            .is_some()
        {}
        txn.commit()
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: commit: {e}")))
    }

    /// Run several write queries inside ONE transaction, committing only after
    /// every statement has been applied. If any statement fails the transaction
    /// is rolled back (best-effort) and the error is returned, leaving the graph
    /// unchanged — this is the atomic group-save primitive behind `save_all`.
    async fn run_all_in_txn(&self, queries: Vec<Query>, ctx: &str) -> Result<(), DriverError> {
        if queries.is_empty() {
            return Ok(());
        }
        let mut txn = self
            .graph
            .start_txn_on(self.database.as_str())
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: begin txn: {e}")))?;
        for q in queries {
            let run = async {
                let mut stream = txn
                    .execute(q)
                    .await
                    .map_err(|e| DriverError::Query(format!("{ctx}: execute: {e}")))?;
                while stream
                    .next(txn.handle())
                    .await
                    .map_err(|e| DriverError::Query(format!("{ctx}: drain: {e}")))?
                    .is_some()
                {}
                Ok::<(), DriverError>(())
            }
            .await;
            if let Err(e) = run {
                // Roll back so a mid-batch failure leaves no partial state.
                let _ = txn.rollback().await;
                return Err(e);
            }
        }
        txn.commit()
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: commit: {e}")))
    }

    /// Execute a read query and collect all rows.
    async fn fetch_rows(&self, q: Query, ctx: &str) -> Result<Vec<neo4rs::Row>, DriverError> {
        let mut stream = self
            .graph
            .execute_on(&self.database, q)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: execute: {e}")))?;
        let mut rows = Vec::new();
        while let Some(row) = stream
            .next()
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: fetch: {e}")))?
        {
            rows.push(row);
        }
        Ok(rows)
    }
}

#[async_trait]
impl EntityNodeOps for Neo4jDriver {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "neo4j save_entity_nodes");
        if nodes.is_empty() {
            return Ok(());
        }
        let mut payload = Vec::with_capacity(nodes.len());
        for n in nodes {
            payload.push(convert::entity_node_to_bolt(n)?);
        }
        let q = query(queries::SAVE_ENTITY_NODES).param("nodes", payload);
        self.run_in_txn(q, "save_entity_nodes").await
    }

    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError> {
        debug!(uuid, "neo4j get_entity_node");
        let cypher = format!(
            "{}{}",
            queries::GET_ENTITY_NODE,
            queries::ENTITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(query(&cypher).param("uuid", uuid), "get_entity_node")
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::entity_node_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_entity_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_ENTITY_NODES_BY_UUIDS,
            queries::ENTITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("uuids", uuids.to_vec()),
                "get_entity_nodes_by_uuids",
            )
            .await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn get_entity_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(
            count = group_ids.len(),
            "neo4j get_entity_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "MATCH (n:Entity) WHERE n.group_id IN $group_ids RETURN{}",
            queries::ENTITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("group_ids", group_ids.to_vec()),
                "get_entity_nodes_by_group_ids",
            )
            .await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn get_mentioned_nodes(
        &self,
        episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(count = episode_uuids.len(), "neo4j get_mentioned_nodes");
        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_MENTIONED_NODES,
            queries::ENTITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("uuids", episode_uuids.to_vec()),
                "get_mentioned_nodes",
            )
            .await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn delete_entity_nodes_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        debug!(count = uuids.len(), "neo4j delete_entity_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(());
        }
        let q = query(queries::DELETE_ENTITY_NODES_BY_UUIDS).param("uuids", uuids.to_vec());
        self.run_in_txn(q, "delete_entity_nodes_by_uuids").await
    }
}

#[async_trait]
impl EntityEdgeOps for Neo4jDriver {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "neo4j save_entity_edges");
        if edges.is_empty() {
            return Ok(());
        }
        let mut payload = Vec::with_capacity(edges.len());
        for e in edges {
            payload.push(convert::entity_edge_to_bolt(e)?);
        }
        let q = query(queries::SAVE_ENTITY_EDGES).param("edges", payload);
        self.run_in_txn(q, "save_entity_edges").await
    }

    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
        debug!(uuid, "neo4j get_entity_edge");
        let cypher = format!(
            "{}{}",
            queries::GET_ENTITY_EDGE,
            queries::ENTITY_EDGE_RETURN
        );
        let rows = self
            .fetch_rows(query(&cypher).param("uuid", uuid), "get_entity_edge")
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::entity_edge_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(source_uuid, target_uuid, "neo4j get_edges_between_nodes");
        let cypher = format!(
            "{}{}",
            queries::GET_EDGES_BETWEEN_NODES,
            queries::ENTITY_EDGE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher)
                    .param("source_node_uuid", source_uuid)
                    .param("target_node_uuid", target_uuid),
                "get_edges_between_nodes",
            )
            .await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn get_entity_edges_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_entity_edges_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_ENTITY_EDGES_BY_UUIDS,
            queries::ENTITY_EDGE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("uuids", uuids.to_vec()),
                "get_entity_edges_by_uuids",
            )
            .await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn delete_entity_edges_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        debug!(count = uuids.len(), "neo4j delete_entity_edges_by_uuids");
        if uuids.is_empty() {
            return Ok(());
        }
        let q = query(queries::DELETE_ENTITY_EDGES_BY_UUIDS).param("uuids", uuids.to_vec());
        self.run_in_txn(q, "delete_entity_edges_by_uuids").await
    }
}

#[async_trait]
impl EpisodeOps for Neo4jDriver {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError> {
        debug!(uuid = episode.uuid, "neo4j save_episode");
        let payload = vec![convert::episode_to_bolt(episode)];
        let q = query(queries::SAVE_EPISODES).param("episodes", payload);
        self.run_in_txn(q, "save_episode").await
    }

    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
        debug!(uuid, "neo4j get_episode");
        let cypher = format!("{}{}", queries::GET_EPISODE, queries::EPISODIC_NODE_RETURN);
        let rows = self
            .fetch_rows(query(&cypher).param("uuid", uuid), "get_episode")
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::episodic_node_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_episodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_EPISODES_BY_UUIDS,
            queries::EPISODIC_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("uuids", uuids.to_vec()),
                "get_episodes_by_uuids",
            )
            .await?;
        rows.iter().map(convert::episodic_node_from_row).collect()
    }

    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(
            last_n,
            group_count = group_ids.len(),
            "neo4j retrieve_episodes"
        );
        // Assemble the query the same way upstream does: base + optional group
        // filter + optional source filter + RETURN <cols> + tail.
        let mut cypher = String::from(queries::RETRIEVE_EPISODES_BASE);
        if !group_ids.is_empty() {
            cypher.push_str(queries::RETRIEVE_EPISODES_GROUP_FILTER);
        }
        if source.is_some() {
            cypher.push_str(queries::RETRIEVE_EPISODES_SOURCE_FILTER);
        }
        cypher.push_str("\nRETURN\n");
        cypher.push_str(queries::EPISODIC_NODE_RETURN);
        cypher.push_str(queries::RETRIEVE_EPISODES_TAIL);

        let mut q = query(&cypher)
            .param("reference_time", reference_time.fixed_offset())
            .param("num_episodes", last_n as i64);
        if !group_ids.is_empty() {
            q = q.param("group_ids", group_ids.to_vec());
        }
        if let Some(s) = source {
            q = q.param("source", convert::episode_type_to_str(s));
        }

        let rows = self.fetch_rows(q, "retrieve_episodes").await?;
        let mut episodes: Vec<EpisodicNode> = rows
            .iter()
            .map(convert::episodic_node_from_row)
            .collect::<Result<_, _>>()?;
        // Upstream returns ORDER BY valid_at DESC then reverses to chronological.
        episodes.reverse();
        Ok(episodes)
    }

    async fn delete_episode(&self, uuid: &str) -> Result<(), DriverError> {
        debug!(uuid, "neo4j delete_episode");
        let q = query(queries::DELETE_EPISODE).param("uuid", uuid);
        self.run_in_txn(q, "delete_episode").await
    }
}

#[async_trait]
impl EpisodicEdgeOps for Neo4jDriver {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "neo4j save_episodic_edges");
        if edges.is_empty() {
            return Ok(());
        }
        let payload: Vec<_> = edges.iter().map(convert::episodic_edge_to_bolt).collect();
        let q = query(queries::SAVE_EPISODIC_EDGES).param("episodic_edges", payload);
        self.run_in_txn(q, "save_episodic_edges").await
    }
}

#[async_trait]
impl BulkSaveOps for Neo4jDriver {
    /// Atomic group-save: episodes, entity nodes, entity edges and episodic edges
    /// in a SINGLE Neo4j transaction (closes the carried-forward atomicity gap).
    ///
    /// Each non-empty collection contributes one `UNWIND ... MERGE` statement,
    /// built identically to the standalone save ops, but they share one
    /// `start_txn → run all → commit` so a mid-batch failure rolls the whole
    /// batch back. Empty collections are skipped; an entirely-empty batch is a
    /// no-op.
    async fn save_all(
        &self,
        episodes: &[EpisodicNode],
        episodic_edges: &[EpisodicEdge],
        entity_nodes: &[EntityNode],
        entity_edges: &[EntityEdge],
    ) -> Result<(), DriverError> {
        debug!(
            episodes = episodes.len(),
            episodic_edges = episodic_edges.len(),
            entity_nodes = entity_nodes.len(),
            entity_edges = entity_edges.len(),
            "neo4j save_all (transactional)"
        );

        let mut statements: Vec<Query> = Vec::new();

        if !episodes.is_empty() {
            let payload: Vec<_> = episodes.iter().map(convert::episode_to_bolt).collect();
            statements.push(query(queries::SAVE_EPISODES).param("episodes", payload));
        }
        if !entity_nodes.is_empty() {
            let mut payload = Vec::with_capacity(entity_nodes.len());
            for n in entity_nodes {
                payload.push(convert::entity_node_to_bolt(n)?);
            }
            statements.push(query(queries::SAVE_ENTITY_NODES).param("nodes", payload));
        }
        if !entity_edges.is_empty() {
            let mut payload = Vec::with_capacity(entity_edges.len());
            for e in entity_edges {
                payload.push(convert::entity_edge_to_bolt(e)?);
            }
            statements.push(query(queries::SAVE_ENTITY_EDGES).param("edges", payload));
        }
        if !episodic_edges.is_empty() {
            let payload: Vec<_> = episodic_edges
                .iter()
                .map(convert::episodic_edge_to_bolt)
                .collect();
            statements.push(query(queries::SAVE_EPISODIC_EDGES).param("episodic_edges", payload));
        }

        self.run_all_in_txn(statements, "save_all").await
    }
}

impl Neo4jDriver {
    /// Bind a `[(name, BoltType)]` param list (produced by the filter-fragment
    /// builders in `queries.rs`) onto a [`Query`]. Centralised so every search
    /// method threads filter params identically.
    fn bind_filter_params(mut q: Query, params: Vec<(String, neo4rs::BoltType)>) -> Query {
        for (name, value) in params {
            q = q.param(&name, value);
        }
        q
    }
}

#[async_trait]
impl SearchOps for Neo4jDriver {
    // Phase-2 Task 6: the four existing search methods now apply the FULL
    // SearchFilters surface (edge_types / edge_uuids / node_labels / date
    // OR-of-ANDs groups) via the WHERE-fragment builders in `queries.rs`. Fulltext
    // appends fragments to the post-YIELD WHERE; similarity appends them into the
    // WHERE alongside the group/score conditions. Node scopes apply node_labels
    // only (matching upstream `node_search_filter_query_constructor`).
    async fn edge_fulltext_search(
        &self,
        query_text: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(query_text, limit, "neo4j edge_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        let (fragments, filter_params) = queries::edge_filter_fragments(filters)?;
        // Base post-YIELD WHERE is `e.group_id IN $group_ids`; append filter
        // fragments with AND so they compose with the group scope.
        let mut where_extra = String::new();
        for frag in &fragments {
            where_extra.push_str("\n    AND ");
            where_extra.push_str(frag);
        }
        let cypher = queries::EDGE_FULLTEXT_SEARCH.replace("{filters}", &where_extra);
        let q = query(&cypher)
            .param("query", fuzzy)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64);
        let q = Self::bind_filter_params(q, filter_params);
        let rows = self.fetch_rows(q, "edge_fulltext_search").await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(limit, min_score, "neo4j edge_similarity_search");
        let (fragments, filter_params) = queries::edge_filter_fragments(filters)?;
        // The first WHERE gates on group_id + non-null embedding; append the
        // filter fragments there so they prune candidates BEFORE the cosine call.
        let mut where_extra = String::new();
        for frag in &fragments {
            where_extra.push_str("\n    AND ");
            where_extra.push_str(frag);
        }
        let cypher = queries::EDGE_SIMILARITY_SEARCH.replace("{filters}", &where_extra);
        let vector: Vec<f64> = search_vector.iter().map(|f| *f as f64).collect();
        let q = query(&cypher)
            .param("search_vector", vector)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64)
            .param("min_score", min_score as f64);
        let q = Self::bind_filter_params(q, filter_params);
        let rows = self.fetch_rows(q, "edge_similarity_search").await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn node_fulltext_search(
        &self,
        query_text: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(query_text, limit, "neo4j node_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        let (fragments, filter_params) = queries::node_filter_fragments(filters)?;
        let mut where_extra = String::new();
        for frag in &fragments {
            where_extra.push_str("\n    AND ");
            where_extra.push_str(frag);
        }
        let cypher = queries::NODE_FULLTEXT_SEARCH.replace("{filters}", &where_extra);
        let q = query(&cypher)
            .param("query", fuzzy)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64);
        let q = Self::bind_filter_params(q, filter_params);
        let rows = self.fetch_rows(q, "node_fulltext_search").await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(limit, min_score, "neo4j node_similarity_search");
        let (fragments, filter_params) = queries::node_filter_fragments(filters)?;
        let mut where_extra = String::new();
        for frag in &fragments {
            where_extra.push_str("\n    AND ");
            where_extra.push_str(frag);
        }
        let cypher = queries::NODE_SIMILARITY_SEARCH.replace("{filters}", &where_extra);
        let vector: Vec<f64> = search_vector.iter().map(|f| *f as f64).collect();
        let q = query(&cypher)
            .param("search_vector", vector)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64)
            .param("min_score", min_score as f64);
        let q = Self::bind_filter_params(q, filter_params);
        let rows = self.fetch_rows(q, "node_similarity_search").await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    // ── Phase-2 search primitives (R5/R7/R8/R9) ──────────────────────────────

    async fn node_bfs_search(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(
            origins = origins.len(),
            max_depth, limit, "neo4j node_bfs_search"
        );
        // Upstream early-return: no origins or depth < 1 → empty.
        if origins.is_empty() || max_depth < 1 {
            return Ok(Vec::new());
        }
        let (fragments, filter_params) = queries::node_filter_fragments(filters)?;
        let with_group_ids = !group_ids.is_empty();
        let cypher = queries::node_bfs_query(max_depth, &fragments, with_group_ids);
        let mut q = query(&cypher)
            .param("bfs_origin_node_uuids", origins.to_vec())
            .param("limit", limit as i64);
        if with_group_ids {
            q = q.param("group_ids", group_ids.to_vec());
        }
        let q = Self::bind_filter_params(q, filter_params);
        let rows = self.fetch_rows(q, "node_bfs_search").await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn edge_bfs_search(
        &self,
        origins: &[String],
        max_depth: usize,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(
            origins = origins.len(),
            max_depth, limit, "neo4j edge_bfs_search"
        );
        // Upstream early-return: no origins → empty.
        if origins.is_empty() {
            return Ok(Vec::new());
        }
        let (mut fragments, mut filter_params) = queries::edge_filter_fragments(filters)?;
        // Upstream appends `e.group_id IN $group_ids` to the filter list when
        // group_ids is provided (NOT a base WHERE — edge BFS has no base WHERE).
        if !group_ids.is_empty() {
            fragments.push("e.group_id IN $group_ids".to_string());
        }
        let cypher = queries::edge_bfs_query(max_depth, &fragments);
        let mut q = query(&cypher)
            .param("bfs_origin_node_uuids", origins.to_vec())
            .param("limit", limit as i64);
        if !group_ids.is_empty() {
            q = q.param("group_ids", group_ids.to_vec());
        }
        // bind_filter_params consumes the vec; bind here.
        for (name, value) in std::mem::take(&mut filter_params) {
            q = q.param(&name, value);
        }
        let rows = self.fetch_rows(q, "edge_bfs_search").await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn episode_fulltext_search(
        &self,
        query_text: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(query_text, limit, "neo4j episode_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        // Assemble head + optional group filter + tail, mirroring upstream's
        // `group_filter_query` concatenation.
        let mut cypher = String::from(queries::EPISODE_FULLTEXT_SEARCH_HEAD);
        if !group_ids.is_empty() {
            cypher.push_str(queries::EPISODE_FULLTEXT_GROUP_FILTER);
        }
        cypher.push_str(queries::EPISODE_FULLTEXT_SEARCH_TAIL);
        let mut q = query(&cypher)
            .param("query", fuzzy)
            .param("limit", limit as i64);
        if !group_ids.is_empty() {
            q = q.param("group_ids", group_ids.to_vec());
        }
        let rows = self.fetch_rows(q, "episode_fulltext_search").await?;
        rows.iter().map(convert::episodic_node_from_row).collect()
    }

    async fn get_embeddings_for_nodes(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_embeddings_for_nodes");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let q = query(queries::GET_NODE_EMBEDDINGS).param("uuids", uuids.to_vec());
        let rows = self.fetch_rows(q, "get_embeddings_for_nodes").await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in &rows {
            if let Some((uuid, emb)) = convert::embedding_row(row)? {
                out.insert(uuid, emb);
            }
        }
        Ok(out)
    }

    async fn get_embeddings_for_edges(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_embeddings_for_edges");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let q = query(queries::GET_EDGE_EMBEDDINGS).param("uuids", uuids.to_vec());
        let rows = self.fetch_rows(q, "get_embeddings_for_edges").await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in &rows {
            if let Some((uuid, emb)) = convert::embedding_row(row)? {
                out.insert(uuid, emb);
            }
        }
        Ok(out)
    }

    async fn nodes_connected_to_center(
        &self,
        node_uuids: &[String],
        center_uuid: &str,
    ) -> Result<Vec<String>, DriverError> {
        debug!(count = node_uuids.len(), "neo4j nodes_connected_to_center");
        if node_uuids.is_empty() {
            return Ok(Vec::new());
        }
        let q = query(queries::NODES_CONNECTED_TO_CENTER)
            .param("node_uuids", node_uuids.to_vec())
            .param("center_uuid", center_uuid);
        let rows = self.fetch_rows(q, "nodes_connected_to_center").await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let uuid: String = row
                .get("uuid")
                .map_err(|e| DriverError::Decode(format!("adjacency row missing uuid: {e}")))?;
            out.push(uuid);
        }
        Ok(out)
    }

    async fn episode_mention_counts(
        &self,
        node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError> {
        debug!(count = node_uuids.len(), "neo4j episode_mention_counts");
        if node_uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let q = query(queries::EPISODE_MENTION_COUNTS).param("node_uuids", node_uuids.to_vec());
        let rows = self.fetch_rows(q, "episode_mention_counts").await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in &rows {
            let uuid: String = row
                .get("uuid")
                .map_err(|e| DriverError::Decode(format!("mention-count row missing uuid: {e}")))?;
            let score: i64 = row.get("score").map_err(|e| {
                DriverError::Decode(format!("mention-count row missing score: {e}"))
            })?;
            // count(*) is non-negative; clamp defensively for the u64 cast.
            out.insert(uuid, score.max(0) as u64);
        }
        Ok(out)
    }
}

#[async_trait]
impl CommunityOps for Neo4jDriver {
    async fn save_community_nodes(&self, nodes: &[CommunityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "neo4j save_community_nodes");
        if nodes.is_empty() {
            return Ok(());
        }
        let payload: Vec<_> = nodes.iter().map(convert::community_node_to_bolt).collect();
        let q = query(queries::SAVE_COMMUNITY_NODES).param("nodes", payload);
        self.run_in_txn(q, "save_community_nodes").await
    }

    async fn save_community_edges(&self, edges: &[CommunityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "neo4j save_community_edges");
        if edges.is_empty() {
            return Ok(());
        }
        let payload: Vec<_> = edges.iter().map(convert::community_edge_to_bolt).collect();
        let q = query(queries::SAVE_COMMUNITY_EDGES).param("edges", payload);
        self.run_in_txn(q, "save_community_edges").await
    }

    async fn get_community_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(
            count = group_ids.len(),
            "neo4j get_community_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_COMMUNITY_NODES_BY_GROUP_IDS,
            queries::COMMUNITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("group_ids", group_ids.to_vec()),
                "get_community_nodes_by_group_ids",
            )
            .await?;
        rows.iter().map(convert::community_node_from_row).collect()
    }

    async fn get_community_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_community_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "{}{}",
            queries::GET_COMMUNITY_NODES_BY_UUIDS,
            queries::COMMUNITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("uuids", uuids.to_vec()),
                "get_community_nodes_by_uuids",
            )
            .await?;
        rows.iter().map(convert::community_node_from_row).collect()
    }

    async fn community_fulltext_search(
        &self,
        query_text: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(query_text, limit, "neo4j community_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        let mut cypher = String::from(queries::COMMUNITY_FULLTEXT_SEARCH_HEAD);
        if !group_ids.is_empty() {
            cypher.push_str(queries::COMMUNITY_FULLTEXT_GROUP_FILTER);
        }
        cypher.push_str(queries::COMMUNITY_FULLTEXT_SEARCH_TAIL);
        let mut q = query(&cypher)
            .param("query", fuzzy)
            .param("limit", limit as i64);
        if !group_ids.is_empty() {
            q = q.param("group_ids", group_ids.to_vec());
        }
        let rows = self.fetch_rows(q, "community_fulltext_search").await?;
        rows.iter().map(convert::community_node_from_row).collect()
    }

    async fn community_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(limit, min_score, "neo4j community_similarity_search");
        let mut cypher = String::from(queries::COMMUNITY_SIMILARITY_SEARCH_HEAD);
        if !group_ids.is_empty() {
            cypher.push_str(queries::COMMUNITY_SIMILARITY_GROUP_FILTER);
        }
        cypher.push_str(queries::COMMUNITY_SIMILARITY_SEARCH_TAIL);
        let vector: Vec<f64> = search_vector.iter().map(|f| *f as f64).collect();
        let mut q = query(&cypher)
            .param("search_vector", vector)
            .param("limit", limit as i64)
            .param("min_score", min_score as f64);
        if !group_ids.is_empty() {
            q = q.param("group_ids", group_ids.to_vec());
        }
        let rows = self.fetch_rows(q, "community_similarity_search").await?;
        rows.iter().map(convert::community_node_from_row).collect()
    }

    async fn get_embeddings_for_communities(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "neo4j get_embeddings_for_communities");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let q = query(queries::GET_COMMUNITY_EMBEDDINGS).param("uuids", uuids.to_vec());
        let rows = self.fetch_rows(q, "get_embeddings_for_communities").await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in &rows {
            if let Some((uuid, emb)) = convert::embedding_row(row)? {
                out.insert(uuid, emb);
            }
        }
        Ok(out)
    }

    async fn remove_communities(&self) -> Result<(), DriverError> {
        debug!("neo4j remove_communities");
        self.run_in_txn(query(queries::REMOVE_COMMUNITIES), "remove_communities")
            .await
    }

    async fn get_community_clusters(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<GroupClusterProjection>, DriverError> {
        debug!(count = group_ids.len(), "neo4j get_community_clusters");

        // Resolve the group set: explicit param, or all distinct entity group_ids.
        let groups: Vec<String> = if group_ids.is_empty() {
            let rows = self
                .fetch_rows(
                    query(queries::DISTINCT_ENTITY_GROUP_IDS),
                    "distinct_entity_group_ids",
                )
                .await?;
            match rows.first() {
                Some(row) => row.get("group_ids").map_err(|e| {
                    DriverError::Decode(format!("distinct group_ids row missing column: {e}"))
                })?,
                None => Vec::new(),
            }
        } else {
            group_ids.to_vec()
        };

        let mut out: Vec<GroupClusterProjection> = Vec::new();
        for group_id in groups {
            // Nodes in this group (drives the per-node neighbour projection).
            let nodes = self
                .get_entity_nodes_by_group_ids(std::slice::from_ref(&group_id))
                .await?;
            let mut node_neighbors: Vec<NodeNeighbors> = Vec::with_capacity(nodes.len());
            for node in &nodes {
                let rows = self
                    .fetch_rows(
                        query(queries::COMMUNITY_CLUSTER_NODE_NEIGHBORS)
                            .param("group_id", group_id.as_str())
                            .param("uuid", node.uuid.as_str()),
                        "community_cluster_node_neighbors",
                    )
                    .await?;
                let mut neighbors: Vec<Neighbor> = Vec::with_capacity(rows.len());
                for row in &rows {
                    let node_uuid: String = row.get("uuid").map_err(|e| {
                        DriverError::Decode(format!("cluster neighbour row missing uuid: {e}"))
                    })?;
                    let count: i64 = row.get("count").map_err(|e| {
                        DriverError::Decode(format!("cluster neighbour row missing count: {e}"))
                    })?;
                    neighbors.push(Neighbor {
                        node_uuid,
                        edge_count: count.max(0) as u64,
                    });
                }
                node_neighbors.push(NodeNeighbors {
                    node_uuid: node.uuid.clone(),
                    neighbors,
                });
            }
            out.push(GroupClusterProjection {
                group_id,
                nodes: node_neighbors,
            });
        }
        Ok(out)
    }

    async fn community_of_member(
        &self,
        entity_uuid: &str,
    ) -> Result<Option<CommunityNode>, DriverError> {
        debug!(entity_uuid, "neo4j community_of_member");
        let cypher = format!(
            "{}{}",
            queries::COMMUNITY_OF_MEMBER,
            queries::COMMUNITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("entity_uuid", entity_uuid),
                "community_of_member",
            )
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::community_node_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn neighbor_communities(
        &self,
        entity_uuid: &str,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(entity_uuid, "neo4j neighbor_communities");
        let cypher = format!(
            "{}{}",
            queries::NEIGHBOR_COMMUNITIES,
            queries::COMMUNITY_NODE_RETURN
        );
        let rows = self
            .fetch_rows(
                query(&cypher).param("entity_uuid", entity_uuid),
                "neighbor_communities",
            )
            .await?;
        rows.iter().map(convert::community_node_from_row).collect()
    }
}

#[async_trait]
impl SagaOps for Neo4jDriver {
    async fn save_saga_node(&self, node: &SagaNode) -> Result<(), DriverError> {
        debug!(uuid = node.uuid, "neo4j save_saga_node");
        let q = query(queries::SAVE_SAGA_NODE)
            .param("uuid", node.uuid.as_str())
            .param("name", node.name.as_str())
            .param("group_id", node.group_id.as_str())
            .param("created_at", node.created_at.fixed_offset())
            .param("summary", node.summary.as_str())
            .param(
                "first_episode_uuid",
                convert::opt_str_param(node.first_episode_uuid.as_deref()),
            )
            .param(
                "last_episode_uuid",
                convert::opt_str_param(node.last_episode_uuid.as_deref()),
            )
            .param(
                "last_summarized_at",
                convert::opt_datetime_param(node.last_summarized_at),
            )
            .param(
                "last_summarized_episode_valid_at",
                convert::opt_datetime_param(node.last_summarized_episode_valid_at),
            );
        self.run_in_txn(q, "save_saga_node").await
    }

    async fn save_has_episode_edge(&self, edge: &HasEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "neo4j save_has_episode_edge");
        let q = query(queries::SAVE_HAS_EPISODE_EDGE)
            .param("saga_uuid", edge.source_node_uuid.as_str())
            .param("episode_uuid", edge.target_node_uuid.as_str())
            .param("uuid", edge.uuid.as_str())
            .param("group_id", edge.group_id.as_str())
            .param("created_at", edge.created_at.fixed_offset());
        self.run_in_txn(q, "save_has_episode_edge").await
    }

    async fn save_next_episode_edge(&self, edge: &NextEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "neo4j save_next_episode_edge");
        let q = query(queries::SAVE_NEXT_EPISODE_EDGE)
            .param("source_episode_uuid", edge.source_node_uuid.as_str())
            .param("target_episode_uuid", edge.target_node_uuid.as_str())
            .param("uuid", edge.uuid.as_str())
            .param("group_id", edge.group_id.as_str())
            .param("created_at", edge.created_at.fixed_offset());
        self.run_in_txn(q, "save_next_episode_edge").await
    }

    async fn get_saga_by_name(
        &self,
        name: &str,
        group_id: &str,
    ) -> Result<Option<SagaNode>, DriverError> {
        debug!(name, group_id, "neo4j get_saga_by_name");
        let cypher = format!("{}{}", queries::GET_SAGA_BY_NAME, queries::SAGA_NODE_RETURN);
        let rows = self
            .fetch_rows(
                query(&cypher)
                    .param("name", name)
                    .param("group_id", group_id),
                "get_saga_by_name",
            )
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::saga_node_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn get_saga_by_uuid(&self, uuid: &str) -> Result<Option<SagaNode>, DriverError> {
        debug!(uuid, "neo4j get_saga_by_uuid");
        let cypher = format!("{}{}", queries::GET_SAGA_BY_UUID, queries::SAGA_NODE_RETURN);
        let rows = self
            .fetch_rows(query(&cypher).param("uuid", uuid), "get_saga_by_uuid")
            .await?;
        match rows.first() {
            Some(row) => Ok(Some(convert::saga_node_from_row(row)?)),
            None => Ok(None),
        }
    }

    async fn saga_previous_episode_uuid(
        &self,
        saga_uuid: &str,
        current_episode_uuid: &str,
    ) -> Result<Option<String>, DriverError> {
        debug!(saga_uuid, "neo4j saga_previous_episode_uuid");
        let rows = self
            .fetch_rows(
                query(queries::SAGA_PREVIOUS_EPISODE_UUID)
                    .param("saga_uuid", saga_uuid)
                    .param("current_episode_uuid", current_episode_uuid),
                "saga_previous_episode_uuid",
            )
            .await?;
        match rows.first() {
            Some(row) => {
                let uuid: String = row.get("uuid").map_err(|e| {
                    DriverError::Decode(format!("saga previous-episode row missing uuid: {e}"))
                })?;
                Ok(Some(uuid))
            }
            None => Ok(None),
        }
    }

    async fn saga_episode_contents(
        &self,
        saga_uuid: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError> {
        debug!(saga_uuid, limit, "neo4j saga_episode_contents");
        let (cypher, reverse) = match since {
            Some(_) => (queries::SAGA_EPISODE_CONTENTS_SINCE, false),
            None => (queries::SAGA_EPISODE_CONTENTS_ALL, true),
        };
        let mut q = query(cypher)
            .param("saga_uuid", saga_uuid)
            .param("limit", limit as i64);
        if let Some(s) = since {
            q = q.param("since", s.fixed_offset());
        }
        let rows = self.fetch_rows(q, "saga_episode_contents").await?;
        let mut out: Vec<(String, DateTime<Utc>)> = Vec::with_capacity(rows.len());
        for row in &rows {
            out.push(convert::saga_episode_content_row(row)?);
        }
        // The no-watermark query orders DESC LIMIT to keep the latest episodes,
        // matching upstream; reverse to chronological order before returning.
        if reverse {
            out.reverse();
        }
        Ok(out)
    }
}

#[async_trait]
impl SchemaOps for Neo4jDriver {
    async fn build_indices_and_constraints(
        &self,
        delete_existing: bool,
    ) -> Result<(), DriverError> {
        debug!(delete_existing, "neo4j build_indices_and_constraints");
        if delete_existing {
            for stmt in queries::drop_index_statements() {
                self.run(query(&stmt), "drop_index").await?;
            }
        }
        for stmt in queries::RANGE_INDICES {
            self.run(query(stmt), "create_range_index").await?;
        }
        for stmt in queries::FULLTEXT_INDICES {
            self.run(query(stmt), "create_fulltext_index").await?;
        }
        // Fulltext indices in Neo4j are populated asynchronously; block until all
        // indices are ONLINE so a search immediately after build is consistent.
        self.run(query("CALL db.awaitIndexes(300)"), "await_indexes")
            .await?;
        Ok(())
    }
}

impl GraphDriver for Neo4jDriver {
    fn provider(&self) -> &'static str {
        "neo4j"
    }
}
