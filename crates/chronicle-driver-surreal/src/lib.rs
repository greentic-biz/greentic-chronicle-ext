#![forbid(unsafe_code)]

//! Embedded SurrealDB `GraphDriver` backend for chronicle.
//!
//! A server-less graph-memory backend built on the `surrealdb` crate's embedded
//! engines (`kv-rocksdb` for on-disk, `kv-mem` for tests). It plugs into the
//! SAME operation-level driver traits as the Neo4j backend
//! (`chronicle_core::driver`); SurrealQL is a from-scratch query layer, not a
//! Cypher port — behaviour parity with the Neo4j driver / FakeDriver is the
//! correctness oracle.
//!
//! ## Phasing
//!
//! This crate lands in three implementation tasks. **Task 1** (this file as
//! shipped here) covers: connect (embedded + in-memory), idempotent schema DDL,
//! node/edge persistence (save + get + get-by-uuids + by-group-ids), episode
//! retrieval, community/saga save+get, and `SchemaOps`. Search, BFS, embeddings
//! loaders, adjacency/mention rerank support, cluster projection, transactional
//! bulk save, and the deletion/maintenance ops land in **Tasks 2 and 3**; until
//! then they return a loud [`DriverError::Query`] (never a silent empty success)
//! so an accidental caller fails fast.
//!
//! ## Datetime boundary (locked decision #1)
//!
//! All datetimes cross the boundary as [`surrealdb::types::Datetime`], never as
//! `chrono` values directly (see `convert.rs`), so stored values are real
//! `datetime`s — proven by the `datetime_is_stored_as_datetime` test.

mod convert;
mod schema;

pub use schema::{ANALYZER, DATABASE, NAMESPACE, schema_ddl};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::SurrealValue;
use tracing::debug;

use chronicle_core::driver::{
    BulkSaveOps, CommunityOps, DriverError, EntityEdgeOps, EntityNodeOps, EpisodeOps,
    EpisodicEdgeOps, GraphDriver, GroupClusterProjection, SagaOps, SchemaOps, SearchOps,
};
use chronicle_core::search::filters::SearchFilters;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

use convert::{
    CommunityNodeRow, EntityEdgeRow, EntityNodeRow, EpisodicNodeRow, SagaNodeRow,
    community_node_from_row, community_node_to_row, entity_edge_from_row, entity_edge_to_row,
    entity_node_from_row, entity_node_to_row, episode_from_row, episode_to_row, record_id,
    saga_node_from_row, saga_node_to_row,
};

/// Loud placeholder for ops deferred to Phase-3 Task 2 / Task 3. Returns an
/// error (never a silent empty success) so a premature caller fails fast.
fn pending(method: &str) -> DriverError {
    DriverError::Query(format!("phase-3 task-2/3 pending: {method}"))
}

/// Embedded SurrealDB-backed `GraphDriver`.
pub struct SurrealDriver {
    db: Surreal<Db>,
    #[allow(dead_code)] // consumed by HNSW DDL on connect; retained for rebuilds.
    embedding_dim: usize,
}

impl SurrealDriver {
    /// Open an on-disk embedded store at `path` (RocksDB), select the chronicle
    /// namespace/database, and run the idempotent schema DDL.
    pub async fn connect_embedded(path: &str, embedding_dim: usize) -> Result<Self, DriverError> {
        let db = Surreal::new::<surrealdb::engine::local::RocksDb>(path)
            .await
            .map_err(|e| DriverError::Connection(format!("open rocksdb at {path}: {e}")))?;
        Self::init(db, embedding_dim).await
    }

    /// Open an in-memory embedded store (for tests). Same schema + selection.
    pub async fn connect_memory(embedding_dim: usize) -> Result<Self, DriverError> {
        let db = Surreal::new::<surrealdb::engine::local::Mem>(())
            .await
            .map_err(|e| DriverError::Connection(format!("open in-memory store: {e}")))?;
        Self::init(db, embedding_dim).await
    }

    async fn init(db: Surreal<Db>, embedding_dim: usize) -> Result<Self, DriverError> {
        db.use_ns(NAMESPACE)
            .use_db(DATABASE)
            .await
            .map_err(|e| DriverError::Connection(format!("use_ns/use_db: {e}")))?;
        let driver = Self { db, embedding_dim };
        driver
            .run_ddl(&schema_ddl(embedding_dim), "schema_ddl")
            .await?;
        Ok(driver)
    }

    /// Run a multi-statement DDL string, surfacing any statement error.
    async fn run_ddl(&self, ddl: &str, ctx: &str) -> Result<(), DriverError> {
        self.db
            .query(ddl)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?
            .check()
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?;
        Ok(())
    }

    /// Build a query from `sql` and apply `bind` (a closure chaining `.bind()`
    /// calls — each variable bound as its own `(name, value)` pair, the form
    /// `IntoVariables` parses unambiguously).
    fn build<'a>(
        &'a self,
        sql: &'a str,
        bind: impl FnOnce(
            surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
        ) -> surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
    ) -> surrealdb::method::Query<'a, surrealdb::engine::local::Db> {
        bind(self.db.query(sql))
    }

    /// Run a write query and drain/check the response.
    async fn run_write<'a>(
        &'a self,
        sql: &'a str,
        bind: impl FnOnce(
            surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
        ) -> surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
        ctx: &str,
    ) -> Result<(), DriverError> {
        self.build(sql, bind)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?
            .check()
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?;
        Ok(())
    }

    /// Run a read query that returns rows of `R` from result-set `idx`.
    async fn fetch<'a, R>(
        &'a self,
        sql: &'a str,
        bind: impl FnOnce(
            surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
        ) -> surrealdb::method::Query<'a, surrealdb::engine::local::Db>,
        idx: usize,
        ctx: &str,
    ) -> Result<Vec<R>, DriverError>
    where
        R: SurrealValue,
    {
        let mut res = self
            .build(sql, bind)
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?;
        res.take::<Vec<R>>(idx)
            .map_err(|e| DriverError::Decode(format!("{ctx}: take: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// EntityNodeOps
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl EntityNodeOps for SurrealDriver {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "surreal save_entity_nodes");
        for n in nodes {
            let row = entity_node_to_row(n);
            let id = record_id("entity", &n.uuid);
            self.run_write(
                "UPSERT $id CONTENT $row",
                |q| q.bind(("id", id)).bind(("row", row)),
                "save_entity_nodes",
            )
            .await?;
        }
        Ok(())
    }

    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError> {
        debug!(uuid, "surreal get_entity_node");
        let rows: Vec<EntityNodeRow> = self
            .fetch(
                "SELECT * FROM $id",
                |q| q.bind(("id", record_id("entity", uuid))),
                0,
                "get_entity_node",
            )
            .await?;
        Ok(rows.into_iter().next().map(entity_node_from_row))
    }

    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(count = uuids.len(), "surreal get_entity_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<EntityNodeRow> = self
            .fetch(
                "SELECT * FROM entity WHERE uuid IN $uuids",
                |q| q.bind(("uuids", uuids.to_vec())),
                0,
                "get_entity_nodes_by_uuids",
            )
            .await?;
        Ok(order_by_uuids(rows, uuids, |r| &r.uuid)
            .into_iter()
            .map(entity_node_from_row)
            .collect())
    }

    async fn get_entity_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(
            count = group_ids.len(),
            "surreal get_entity_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<EntityNodeRow> = self
            .fetch(
                "SELECT * FROM entity WHERE group_id IN $group_ids",
                |q| q.bind(("group_ids", group_ids.to_vec())),
                0,
                "get_entity_nodes_by_group_ids",
            )
            .await?;
        Ok(rows.into_iter().map(entity_node_from_row).collect())
    }

    async fn get_mentioned_nodes(
        &self,
        _episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        // Graph traversal over `mentions` — Task 3.
        Err(pending("get_mentioned_nodes"))
    }

    async fn delete_entity_nodes_by_uuids(&self, _uuids: &[String]) -> Result<(), DriverError> {
        // Cascade delete (DETACH-equivalent) — Task 3.
        Err(pending("delete_entity_nodes_by_uuids"))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// EntityEdgeOps
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl EntityEdgeOps for SurrealDriver {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "surreal save_entity_edges");
        for e in edges {
            let row = entity_edge_to_row(e);
            // RELATE creates the typed relation; the relation record id is keyed
            // by the edge uuid so re-saving the same fact upserts in place.
            let src = record_id("entity", &e.source_node_uuid);
            let dst = record_id("entity", &e.target_node_uuid);
            let rel = record_id("relates_to", &e.uuid);
            self.run_write(
                "RELATE $src->$rel->$dst CONTENT $row",
                |q| {
                    q.bind(("src", src))
                        .bind(("rel", rel))
                        .bind(("dst", dst))
                        .bind(("row", row))
                },
                "save_entity_edges",
            )
            .await?;
        }
        Ok(())
    }

    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
        debug!(uuid, "surreal get_entity_edge");
        let rows: Vec<EntityEdgeRow> = self
            .fetch(
                "SELECT * FROM relates_to WHERE uuid = $uuid",
                |q| q.bind(("uuid", uuid.to_string())),
                0,
                "get_entity_edge",
            )
            .await?;
        Ok(rows.into_iter().next().map(entity_edge_from_row))
    }

    async fn get_edges_between_nodes(
        &self,
        _source_uuid: &str,
        _target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        // Directed RELATE traversal — Task 2.
        Err(pending("get_edges_between_nodes"))
    }

    async fn get_entity_edges_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(count = uuids.len(), "surreal get_entity_edges_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<EntityEdgeRow> = self
            .fetch(
                "SELECT * FROM relates_to WHERE uuid IN $uuids",
                |q| q.bind(("uuids", uuids.to_vec())),
                0,
                "get_entity_edges_by_uuids",
            )
            .await?;
        Ok(order_by_uuids(rows, uuids, |r| &r.uuid)
            .into_iter()
            .map(entity_edge_from_row)
            .collect())
    }

    async fn delete_entity_edges_by_uuids(&self, _uuids: &[String]) -> Result<(), DriverError> {
        Err(pending("delete_entity_edges_by_uuids"))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// EpisodeOps
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl EpisodeOps for SurrealDriver {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError> {
        debug!(uuid = episode.uuid, "surreal save_episode");
        let row = episode_to_row(episode);
        let id = record_id("episodic", &episode.uuid);
        self.run_write(
            "UPSERT $id CONTENT $row",
            |q| q.bind(("id", id)).bind(("row", row)),
            "save_episode",
        )
        .await
    }

    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
        debug!(uuid, "surreal get_episode");
        let rows: Vec<EpisodicNodeRow> = self
            .fetch(
                "SELECT * FROM $id",
                |q| q.bind(("id", record_id("episodic", uuid))),
                0,
                "get_episode",
            )
            .await?;
        match rows.into_iter().next() {
            Some(r) => Ok(Some(episode_from_row(r)?)),
            None => Ok(None),
        }
    }

    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(count = uuids.len(), "surreal get_episodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<EpisodicNodeRow> = self
            .fetch(
                "SELECT * FROM episodic WHERE uuid IN $uuids",
                |q| q.bind(("uuids", uuids.to_vec())),
                0,
                "get_episodes_by_uuids",
            )
            .await?;
        order_by_uuids(rows, uuids, |r| &r.uuid)
            .into_iter()
            .map(episode_from_row)
            .collect()
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
            "surreal retrieve_episodes"
        );
        // Assemble: valid_at <= ref [+ group filter] [+ source filter],
        // ORDER valid_at DESC LIMIT n, then reverse to chronological — mirrors
        // the Neo4j driver / upstream retrieve_episodes.
        let mut sql = String::from("SELECT * FROM episodic WHERE valid_at <= $ref");
        if !group_ids.is_empty() {
            sql.push_str(" AND group_id IN $group_ids");
        }
        if source.is_some() {
            sql.push_str(" AND source = $source");
        }
        sql.push_str(" ORDER BY valid_at DESC LIMIT $limit");

        // Bind explicitly (variable set differs per call).
        let mut q = self
            .db
            .query(&sql)
            .bind(("ref", convert::to_dt(reference_time)))
            .bind(("limit", last_n as i64));
        if !group_ids.is_empty() {
            q = q.bind(("group_ids", group_ids.to_vec()));
        }
        if let Some(s) = source {
            q = q.bind(("source", convert::episode_type_to_str(s).to_string()));
        }
        let mut res = q
            .await
            .map_err(|e| DriverError::Query(format!("retrieve_episodes: {e}")))?;
        let rows: Vec<EpisodicNodeRow> = res
            .take(0)
            .map_err(|e| DriverError::Decode(format!("retrieve_episodes: take: {e}")))?;
        let mut episodes: Vec<EpisodicNode> = rows
            .into_iter()
            .map(episode_from_row)
            .collect::<Result<_, _>>()?;
        episodes.reverse();
        Ok(episodes)
    }

    async fn delete_episode(&self, _uuid: &str) -> Result<(), DriverError> {
        Err(pending("delete_episode"))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// EpisodicEdgeOps
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl EpisodicEdgeOps for SurrealDriver {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "surreal save_episodic_edges");
        for e in edges {
            let row = convert::episodic_edge_to_row(e);
            let src = record_id("episodic", &e.source_node_uuid);
            let dst = record_id("entity", &e.target_node_uuid);
            let rel = record_id("mentions", &e.uuid);
            self.run_write(
                "RELATE $src->$rel->$dst CONTENT $row",
                |q| {
                    q.bind(("src", src))
                        .bind(("rel", rel))
                        .bind(("dst", dst))
                        .bind(("row", row))
                },
                "save_episodic_edges",
            )
            .await?;
        }
        Ok(())
    }
}

#[async_trait]
impl BulkSaveOps for SurrealDriver {
    // Task 3 supplies a single-transaction override. Until then the default
    // sequential `save_all` (four independent saves) is inherited; it is
    // correct, just not atomic across the four writes.
}

// ─────────────────────────────────────────────────────────────────────────
// SearchOps — all of this trait is Task 2.
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl SearchOps for SurrealDriver {
    async fn edge_fulltext_search(
        &self,
        _query: &str,
        _filters: &SearchFilters,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        Err(pending("edge_fulltext_search"))
    }

    async fn edge_similarity_search(
        &self,
        _search_vector: &[f32],
        _filters: &SearchFilters,
        _group_ids: &[String],
        _limit: usize,
        _min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        Err(pending("edge_similarity_search"))
    }

    async fn node_fulltext_search(
        &self,
        _query: &str,
        _filters: &SearchFilters,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        Err(pending("node_fulltext_search"))
    }

    async fn node_similarity_search(
        &self,
        _search_vector: &[f32],
        _filters: &SearchFilters,
        _group_ids: &[String],
        _limit: usize,
        _min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        Err(pending("node_similarity_search"))
    }

    async fn node_bfs_search(
        &self,
        _origins: &[String],
        _filters: &SearchFilters,
        _max_depth: usize,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        Err(pending("node_bfs_search"))
    }

    async fn edge_bfs_search(
        &self,
        _origins: &[String],
        _max_depth: usize,
        _filters: &SearchFilters,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        Err(pending("edge_bfs_search"))
    }

    async fn episode_fulltext_search(
        &self,
        _query: &str,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        Err(pending("episode_fulltext_search"))
    }

    async fn get_embeddings_for_nodes(
        &self,
        _uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        Err(pending("get_embeddings_for_nodes"))
    }

    async fn get_embeddings_for_edges(
        &self,
        _uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        Err(pending("get_embeddings_for_edges"))
    }

    async fn nodes_connected_to_center(
        &self,
        _node_uuids: &[String],
        _center_uuid: &str,
    ) -> Result<Vec<String>, DriverError> {
        Err(pending("nodes_connected_to_center"))
    }

    async fn episode_mention_counts(
        &self,
        _node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError> {
        Err(pending("episode_mention_counts"))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CommunityOps — save/get implemented (Task 1); search/cluster/membership Task 2/3.
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl CommunityOps for SurrealDriver {
    async fn save_community_nodes(&self, nodes: &[CommunityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "surreal save_community_nodes");
        for n in nodes {
            let row = community_node_to_row(n);
            let id = record_id("community", &n.uuid);
            self.run_write(
                "UPSERT $id CONTENT $row",
                |q| q.bind(("id", id)).bind(("row", row)),
                "save_community_nodes",
            )
            .await?;
        }
        Ok(())
    }

    async fn save_community_edges(&self, edges: &[CommunityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "surreal save_community_edges");
        for e in edges {
            let row = convert::community_edge_to_row(e);
            // Target may be an Entity or another Community; the relation table
            // permits both (`OUT entity|community`). We pick the table by the
            // edge's intent — community membership targets entities by default,
            // sub-community nesting targets communities. We can only know the
            // target kind from existing data; probe both, preferring entity.
            let dst = self.resolve_member_target(&e.target_node_uuid).await?;
            let src = record_id("community", &e.source_node_uuid);
            let rel = record_id("has_member", &e.uuid);
            self.run_write(
                "RELATE $src->$rel->$dst CONTENT $row",
                |q| {
                    q.bind(("src", src))
                        .bind(("rel", rel))
                        .bind(("dst", dst))
                        .bind(("row", row))
                },
                "save_community_edges",
            )
            .await?;
        }
        Ok(())
    }

    async fn get_community_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(
            count = group_ids.len(),
            "surreal get_community_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<CommunityNodeRow> = self
            .fetch(
                "SELECT * FROM community WHERE group_id IN $group_ids",
                |q| q.bind(("group_ids", group_ids.to_vec())),
                0,
                "get_community_nodes_by_group_ids",
            )
            .await?;
        Ok(rows.into_iter().map(community_node_from_row).collect())
    }

    async fn get_community_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(count = uuids.len(), "surreal get_community_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<CommunityNodeRow> = self
            .fetch(
                "SELECT * FROM community WHERE uuid IN $uuids",
                |q| q.bind(("uuids", uuids.to_vec())),
                0,
                "get_community_nodes_by_uuids",
            )
            .await?;
        Ok(order_by_uuids(rows, uuids, |r| &r.uuid)
            .into_iter()
            .map(community_node_from_row)
            .collect())
    }

    async fn community_fulltext_search(
        &self,
        _query: &str,
        _group_ids: &[String],
        _limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        Err(pending("community_fulltext_search"))
    }

    async fn community_similarity_search(
        &self,
        _search_vector: &[f32],
        _group_ids: &[String],
        _limit: usize,
        _min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        Err(pending("community_similarity_search"))
    }

    async fn get_embeddings_for_communities(
        &self,
        _uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        Err(pending("get_embeddings_for_communities"))
    }

    async fn remove_communities(&self) -> Result<(), DriverError> {
        Err(pending("remove_communities"))
    }

    async fn get_community_clusters(
        &self,
        _group_ids: &[String],
    ) -> Result<Vec<GroupClusterProjection>, DriverError> {
        Err(pending("get_community_clusters"))
    }

    async fn community_of_member(
        &self,
        _entity_uuid: &str,
    ) -> Result<Option<CommunityNode>, DriverError> {
        Err(pending("community_of_member"))
    }

    async fn neighbor_communities(
        &self,
        _entity_uuid: &str,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        Err(pending("neighbor_communities"))
    }
}

impl SurrealDriver {
    /// Resolve a HAS_MEMBER target uuid to its record id, preferring an existing
    /// `entity` record and falling back to `community` (sub-community nesting).
    /// A `has_member` edge's target is `entity|community`; we must point at the
    /// right table.
    async fn resolve_member_target(
        &self,
        target_uuid: &str,
    ) -> Result<surrealdb::types::RecordId, DriverError> {
        let entity_hit: Vec<EntityNodeRow> = self
            .fetch(
                "SELECT * FROM entity WHERE uuid = $uuid LIMIT 1",
                |q| q.bind(("uuid", target_uuid.to_string())),
                0,
                "resolve_member_target",
            )
            .await?;
        if !entity_hit.is_empty() {
            Ok(record_id("entity", target_uuid))
        } else {
            // Default to community when no entity exists (or for nesting).
            Ok(record_id("community", target_uuid))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// SagaOps — save + get-by-name/uuid implemented (Task 1); threading queries Task 3.
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl SagaOps for SurrealDriver {
    async fn save_saga_node(&self, node: &SagaNode) -> Result<(), DriverError> {
        debug!(uuid = node.uuid, "surreal save_saga_node");
        let row = saga_node_to_row(node);
        let id = record_id("saga", &node.uuid);
        self.run_write(
            "UPSERT $id CONTENT $row",
            |q| q.bind(("id", id)).bind(("row", row)),
            "save_saga_node",
        )
        .await
    }

    async fn save_has_episode_edge(&self, edge: &HasEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "surreal save_has_episode_edge");
        let row = convert::has_episode_edge_to_row(edge);
        let src = record_id("saga", &edge.source_node_uuid);
        let dst = record_id("episodic", &edge.target_node_uuid);
        let rel = record_id("has_episode", &edge.uuid);
        self.run_write(
            "RELATE $src->$rel->$dst CONTENT $row",
            |q| {
                q.bind(("src", src))
                    .bind(("rel", rel))
                    .bind(("dst", dst))
                    .bind(("row", row))
            },
            "save_has_episode_edge",
        )
        .await
    }

    async fn save_next_episode_edge(&self, edge: &NextEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "surreal save_next_episode_edge");
        let row = convert::next_episode_edge_to_row(edge);
        let src = record_id("episodic", &edge.source_node_uuid);
        let dst = record_id("episodic", &edge.target_node_uuid);
        let rel = record_id("next_episode", &edge.uuid);
        self.run_write(
            "RELATE $src->$rel->$dst CONTENT $row",
            |q| {
                q.bind(("src", src))
                    .bind(("rel", rel))
                    .bind(("dst", dst))
                    .bind(("row", row))
            },
            "save_next_episode_edge",
        )
        .await
    }

    async fn get_saga_by_name(
        &self,
        name: &str,
        group_id: &str,
    ) -> Result<Option<SagaNode>, DriverError> {
        debug!(name, group_id, "surreal get_saga_by_name");
        let rows: Vec<SagaNodeRow> = self
            .fetch(
                "SELECT * FROM saga WHERE name = $name AND group_id = $group_id LIMIT 1",
                |q| {
                    q.bind(("name", name.to_string()))
                        .bind(("group_id", group_id.to_string()))
                },
                0,
                "get_saga_by_name",
            )
            .await?;
        Ok(rows.into_iter().next().map(saga_node_from_row))
    }

    async fn get_saga_by_uuid(&self, uuid: &str) -> Result<Option<SagaNode>, DriverError> {
        debug!(uuid, "surreal get_saga_by_uuid");
        let rows: Vec<SagaNodeRow> = self
            .fetch(
                "SELECT * FROM $id",
                |q| q.bind(("id", record_id("saga", uuid))),
                0,
                "get_saga_by_uuid",
            )
            .await?;
        Ok(rows.into_iter().next().map(saga_node_from_row))
    }

    async fn saga_previous_episode_uuid(
        &self,
        _saga_uuid: &str,
        _current_episode_uuid: &str,
    ) -> Result<Option<String>, DriverError> {
        Err(pending("saga_previous_episode_uuid"))
    }

    async fn saga_episode_contents(
        &self,
        _saga_uuid: &str,
        _since: Option<DateTime<Utc>>,
        _limit: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError> {
        Err(pending("saga_episode_contents"))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// SchemaOps
// ─────────────────────────────────────────────────────────────────────────

#[async_trait]
impl SchemaOps for SurrealDriver {
    async fn build_indices_and_constraints(
        &self,
        delete_existing: bool,
    ) -> Result<(), DriverError> {
        debug!(delete_existing, "surreal build_indices_and_constraints");
        if delete_existing {
            self.run_ddl(&schema::drop_ddl(), "drop_ddl").await?;
        }
        self.run_ddl(&schema_ddl(self.embedding_dim), "schema_ddl")
            .await
    }
}

impl GraphDriver for SurrealDriver {
    fn provider(&self) -> &'static str {
        "surreal"
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

/// Reorder `rows` to follow the order of `uuids` (the `IN` query returns
/// storage order, not request order). Rows whose uuid is not in `uuids` are
/// dropped; missing uuids are simply absent — mirroring the Neo4j driver, which
/// preserves request order for `get_*_by_uuids`.
fn order_by_uuids<R>(rows: Vec<R>, uuids: &[String], key: impl Fn(&R) -> &String) -> Vec<R> {
    let mut by_uuid: HashMap<String, R> = rows.into_iter().map(|r| (key(&r).clone(), r)).collect();
    uuids.iter().filter_map(|u| by_uuid.remove(u)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_core::types::{CommunityEdge, EntityEdge};
    use chrono::Utc;

    async fn mem() -> SurrealDriver {
        SurrealDriver::connect_memory(8)
            .await
            .expect("connect_memory")
    }

    fn entity(name: &str, group: &str) -> EntityNode {
        let mut n = EntityNode::new(name.into(), group.into(), Utc::now());
        n.summary = format!("{name} summary");
        n.name_embedding = Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        n.attributes
            .insert("role".into(), serde_json::Value::String("admin".into()));
        n
    }

    #[tokio::test]
    async fn entity_node_save_get_roundtrip() {
        let d = mem().await;
        let n = entity("Alice", "g1");
        d.save_entity_nodes(std::slice::from_ref(&n)).await.unwrap();

        let got = d.get_entity_node(&n.uuid).await.unwrap().unwrap();
        assert_eq!(got.uuid, n.uuid);
        assert_eq!(got.name, "Alice");
        assert_eq!(got.summary, "Alice summary");
        assert_eq!(got.name_embedding, n.name_embedding);
        assert_eq!(
            got.attributes.get("role"),
            Some(&serde_json::Value::String("admin".into()))
        );
        assert_eq!(got.created_at, n.created_at);
    }

    #[tokio::test]
    async fn entity_get_by_uuids_preserves_order() {
        let d = mem().await;
        let a = entity("A", "g1");
        let b = entity("B", "g1");
        let c = entity("C", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone(), c.clone()])
            .await
            .unwrap();

        // Request in a scrambled order; result must match request order.
        let order = vec![c.uuid.clone(), a.uuid.clone(), b.uuid.clone()];
        let got = d.get_entity_nodes_by_uuids(&order).await.unwrap();
        let got_uuids: Vec<String> = got.iter().map(|n| n.uuid.clone()).collect();
        assert_eq!(got_uuids, order);
    }

    #[tokio::test]
    async fn entity_by_group_ids() {
        let d = mem().await;
        d.save_entity_nodes(&[entity("A", "g1"), entity("B", "g2")])
            .await
            .unwrap();
        let g1 = d
            .get_entity_nodes_by_group_ids(&["g1".to_string()])
            .await
            .unwrap();
        assert_eq!(g1.len(), 1);
        assert_eq!(g1[0].name, "A");
    }

    #[tokio::test]
    async fn entity_edge_save_get_roundtrip() {
        let d = mem().await;
        let a = entity("A", "g1");
        let b = entity("B", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let mut e = EntityEdge::new(
            a.uuid.clone(),
            b.uuid.clone(),
            "WORKS_WITH".into(),
            "A works with B".into(),
            "g1".into(),
        );
        e.valid_at = Some(Utc::now());
        e.fact_embedding = Some(vec![0.5; 8]);
        e.episodes = vec!["ep1".into()];
        e.attributes.insert("weight".into(), serde_json::json!(0.9));
        d.save_entity_edges(std::slice::from_ref(&e)).await.unwrap();

        let got = d.get_entity_edge(&e.uuid).await.unwrap().unwrap();
        assert_eq!(got.uuid, e.uuid);
        assert_eq!(got.source_node_uuid, a.uuid);
        assert_eq!(got.target_node_uuid, b.uuid);
        assert_eq!(got.fact, "A works with B");
        assert_eq!(got.valid_at, e.valid_at);
        assert_eq!(got.fact_embedding, e.fact_embedding);
        assert_eq!(got.episodes, vec!["ep1".to_string()]);
        assert_eq!(got.attributes.get("weight"), Some(&serde_json::json!(0.9)));

        let by_uuids = d
            .get_entity_edges_by_uuids(&[e.uuid.clone()])
            .await
            .unwrap();
        assert_eq!(by_uuids.len(), 1);
        assert_eq!(by_uuids[0].uuid, e.uuid);
    }

    fn episode(name: &str, group: &str, valid: DateTime<Utc>) -> EpisodicNode {
        EpisodicNode::new(
            name.into(),
            group.into(),
            EpisodeType::Message,
            "src".into(),
            format!("content of {name}"),
            Utc::now(),
            valid,
        )
    }

    #[tokio::test]
    async fn episode_save_get_roundtrip() {
        let d = mem().await;
        let ep = episode("ep1", "g1", Utc::now());
        d.save_episode(&ep).await.unwrap();
        let got = d.get_episode(&ep.uuid).await.unwrap().unwrap();
        assert_eq!(got.uuid, ep.uuid);
        assert_eq!(got.content, "content of ep1");
        assert_eq!(got.source, EpisodeType::Message);
        assert_eq!(got.valid_at, ep.valid_at);

        let by_uuids = d
            .get_episodes_by_uuids(std::slice::from_ref(&ep.uuid))
            .await
            .unwrap();
        assert_eq!(by_uuids.len(), 1);
    }

    #[tokio::test]
    async fn retrieve_episodes_orders_and_cuts_off() {
        let d = mem().await;
        let base = Utc::now();
        // Three episodes at t-3, t-2, t-1; one in the future (t+1).
        let e1 = episode("e1", "g1", base - chrono::Duration::hours(3));
        let e2 = episode("e2", "g1", base - chrono::Duration::hours(2));
        let e3 = episode("e3", "g1", base - chrono::Duration::hours(1));
        let future = episode("future", "g1", base + chrono::Duration::hours(1));
        for ep in [&e1, &e2, &e3, &future] {
            d.save_episode(ep).await.unwrap();
        }

        // last 2 with valid_at <= base → e2, e3 in chronological order.
        let got = d
            .retrieve_episodes(base, 2, &["g1".to_string()], None)
            .await
            .unwrap();
        let names: Vec<String> = got.iter().map(|e| e.name.clone()).collect();
        assert_eq!(names, vec!["e2".to_string(), "e3".to_string()]);

        // group filter excludes other groups.
        let none = d
            .retrieve_episodes(base, 10, &["other".to_string()], None)
            .await
            .unwrap();
        assert!(none.is_empty());

        // source filter.
        let msgs = d
            .retrieve_episodes(base, 10, &["g1".to_string()], Some(EpisodeType::Message))
            .await
            .unwrap();
        assert_eq!(msgs.len(), 3);
        let texts = d
            .retrieve_episodes(base, 10, &["g1".to_string()], Some(EpisodeType::Text))
            .await
            .unwrap();
        assert!(texts.is_empty());
    }

    #[tokio::test]
    async fn community_save_get_roundtrip() {
        let d = mem().await;
        let mut c = CommunityNode::new("Tech".into(), "g1".into(), Utc::now());
        c.summary = "tech community".into();
        c.name_embedding = Some(vec![0.2; 8]);
        d.save_community_nodes(std::slice::from_ref(&c))
            .await
            .unwrap();

        let by_uuid = d
            .get_community_nodes_by_uuids(&[c.uuid.clone()])
            .await
            .unwrap();
        assert_eq!(by_uuid.len(), 1);
        assert_eq!(by_uuid[0].summary, "tech community");
        assert_eq!(by_uuid[0].name_embedding, c.name_embedding);

        let by_group = d
            .get_community_nodes_by_group_ids(&["g1".to_string()])
            .await
            .unwrap();
        assert_eq!(by_group.len(), 1);
    }

    #[tokio::test]
    async fn community_edge_to_entity_member() {
        let d = mem().await;
        let ent = entity("member", "g1");
        d.save_entity_nodes(std::slice::from_ref(&ent))
            .await
            .unwrap();
        let comm = CommunityNode::new("C".into(), "g1".into(), Utc::now());
        d.save_community_nodes(std::slice::from_ref(&comm))
            .await
            .unwrap();
        let edge = CommunityEdge::new(comm.uuid.clone(), ent.uuid.clone(), "g1".into(), Utc::now());
        // Must succeed (target resolves to the existing entity record).
        d.save_community_edges(std::slice::from_ref(&edge))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn saga_save_get_roundtrip() {
        let d = mem().await;
        let mut s = SagaNode::new("onboarding".into(), "g1".into(), Utc::now());
        s.summary = "saga summary".into();
        s.first_episode_uuid = Some("ep1".into());
        s.last_summarized_at = Some(Utc::now());
        d.save_saga_node(&s).await.unwrap();

        let by_uuid = d.get_saga_by_uuid(&s.uuid).await.unwrap().unwrap();
        assert_eq!(by_uuid.name, "onboarding");
        assert_eq!(by_uuid.summary, "saga summary");
        assert_eq!(by_uuid.first_episode_uuid.as_deref(), Some("ep1"));
        assert!(by_uuid.last_summarized_at.is_some());

        let by_name = d
            .get_saga_by_name("onboarding", "g1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_name.uuid, s.uuid);

        // saga edges save.
        let ep = episode("ep", "g1", Utc::now());
        d.save_episode(&ep).await.unwrap();
        let he = HasEpisodeEdge::new(s.uuid.clone(), ep.uuid.clone(), "g1".into(), Utc::now());
        d.save_has_episode_edge(&he).await.unwrap();
    }

    #[tokio::test]
    async fn datetime_is_stored_as_datetime() {
        // Locked-decision #1 proof: stored created_at must be a real datetime,
        // not a string. type::is_datetime() returns true only for datetimes.
        let d = mem().await;
        let n = entity("A", "g1");
        d.save_entity_nodes(std::slice::from_ref(&n)).await.unwrap();

        #[derive(surrealdb::types::SurrealValue)]
        struct Guard {
            is_dt: bool,
        }
        let mut res =
            d.db.query("SELECT type::is_datetime(created_at) AS is_dt FROM entity")
                .await
                .unwrap();
        let rows: Vec<Guard> = res.take(0).unwrap();
        assert!(
            rows.first().map(|g| g.is_dt).unwrap_or(false),
            "created_at must be stored as a datetime, not a string"
        );
    }

    #[tokio::test]
    async fn build_indices_idempotent_and_rebuild() {
        let d = mem().await;
        // Re-run (idempotent).
        d.build_indices_and_constraints(false).await.unwrap();
        // Drop + redefine.
        d.build_indices_and_constraints(true).await.unwrap();
        // Driver still usable afterwards.
        let n = entity("A", "g1");
        d.save_entity_nodes(std::slice::from_ref(&n)).await.unwrap();
        assert!(d.get_entity_node(&n.uuid).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn pending_methods_error_loudly() {
        let d = mem().await;
        assert!(d.get_mentioned_nodes(&["x".into()]).await.is_err());
        assert!(d.delete_episode("x").await.is_err());
        assert!(
            d.node_fulltext_search("q", &SearchFilters::default(), &[], 10)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn provider_name() {
        let d = mem().await;
        assert_eq!(d.provider(), "surreal");
    }

    #[tokio::test]
    async fn embedded_rocksdb_smoke() {
        // connect_memory may be unavailable in some builds; the on-disk path is
        // the production one, so smoke-test it via a tempdir.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let d = SurrealDriver::connect_embedded(path.to_str().unwrap(), 8)
            .await
            .expect("connect_embedded");
        let n = entity("Persisted", "g1");
        d.save_entity_nodes(std::slice::from_ref(&n)).await.unwrap();
        assert_eq!(
            d.get_entity_node(&n.uuid).await.unwrap().unwrap().name,
            "Persisted"
        );
    }
}
