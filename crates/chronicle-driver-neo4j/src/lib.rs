#![forbid(unsafe_code)]

//! Neo4j `GraphDriver` backend for chronicle, over neo4rs 0.8 (Bolt).
//!
//! Port of `graphiti_core/driver/neo4j_driver.py` (and the operation files under
//! `graphiti_core/driver/neo4j/operations/`) @ 34f56e65 (v0.29.1), adapted to the
//! typed operation-level driver traits in `chronicle-core::driver`.
//!
//! ## Cross-call atomicity gap (CARRIED-FORWARD NOTE)
//!
//! Upstream `add_episode` persists episode + entity nodes + entity edges +
//! episodic edges inside a single Neo4j transaction. The chronicle driver trait
//! splits persistence into four independent save operations
//! (`save_episode`, `save_entity_nodes`, `save_entity_edges`,
//! `save_episodic_edges`), each invoked sequentially by the pipeline. Here, each
//! save op runs as its OWN single transaction (`Graph::start_txn` →
//! `UNWIND ... RETURN` → `commit`), so it is atomic *within* the op, but the four
//! ops are NOT atomic *as a group*. A crash between calls can leave a partially
//! persisted episode.
//!
//! Phase-2 improvement: add a transactional `save_all` operation to the driver
//! trait that wraps all four writes in one transaction. Until then, callers must
//! treat `add_episode` as best-effort-with-retry, not all-or-nothing.

mod convert;
mod queries;

pub use queries::{MAX_QUERY_LENGTH, build_fulltext_query, validate_group_ids};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use neo4rs::{Graph, Query, query};
use tracing::debug;

use chronicle_core::driver::{
    DriverError, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SchemaOps,
    SearchOps,
};
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};

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
impl SearchOps for Neo4jDriver {
    async fn edge_fulltext_search(
        &self,
        query_text: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(query_text, limit, "neo4j edge_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        let q = query(queries::EDGE_FULLTEXT_SEARCH)
            .param("query", fuzzy)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64);
        let rows = self.fetch_rows(q, "edge_fulltext_search").await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(limit, min_score, "neo4j edge_similarity_search");
        let vector: Vec<f64> = search_vector.iter().map(|f| *f as f64).collect();
        let q = query(queries::EDGE_SIMILARITY_SEARCH)
            .param("search_vector", vector)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64)
            .param("min_score", min_score as f64);
        let rows = self.fetch_rows(q, "edge_similarity_search").await?;
        rows.iter().map(convert::entity_edge_from_row).collect()
    }

    async fn node_fulltext_search(
        &self,
        query_text: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(query_text, limit, "neo4j node_fulltext_search");
        let Some(fuzzy) = build_fulltext_query(query_text, group_ids)? else {
            return Ok(Vec::new());
        };
        let q = query(queries::NODE_FULLTEXT_SEARCH)
            .param("query", fuzzy)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64);
        let rows = self.fetch_rows(q, "node_fulltext_search").await?;
        rows.iter().map(convert::entity_node_from_row).collect()
    }

    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(limit, min_score, "neo4j node_similarity_search");
        let vector: Vec<f64> = search_vector.iter().map(|f| *f as f64).collect();
        let q = query(queries::NODE_SIMILARITY_SEARCH)
            .param("search_vector", vector)
            .param("group_ids", group_ids.to_vec())
            .param("limit", limit as i64)
            .param("min_score", min_score as f64);
        let rows = self.fetch_rows(q, "node_similarity_search").await?;
        rows.iter().map(convert::entity_node_from_row).collect()
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
