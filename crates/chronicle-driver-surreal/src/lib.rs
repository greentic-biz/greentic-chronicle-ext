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
mod filters;
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
    CommunityNodeRow, EmbeddingRow, EntityEdgeRow, EntityNodeRow, EpisodicNodeRow, MentionCountRow,
    SagaNodeRow, ScoredCommunityNodeRow, ScoredEntityEdgeRow, ScoredEntityNodeRow,
    community_node_from_row, community_node_to_row, entity_edge_from_row, entity_edge_to_row,
    entity_node_from_row, entity_node_to_row, episode_from_row, episode_to_row, record_id,
    saga_node_from_row, saga_node_to_row,
};
use filters::{FilterFragments, edge_filter_fragments, node_filter_fragments};

/// BFS depth hard cap (locked decision #4). A requested depth is clamped into
/// `[1, MAX_BFS_DEPTH]`; depth < 1 short-circuits to an empty result upstream.
const MAX_BFS_DEPTH: usize = 5;

/// KNN over-fetch factor (locked decision #3). The HNSW `<|K, EF|>` operator
/// selects K candidates from the index FIRST, THEN the `WHERE` post-filters
/// (verified against surrealdb 3.1.3: a group filter excluding the 2nd-nearest
/// drops it, shrinking the result below K). To survive post-filtering + the
/// `score >= min_score` cutoff we over-fetch `limit * OVERFETCH` (floor 30).
const KNN_OVERFETCH: usize = 3;
/// Floor for the over-fetched K so a tiny `limit` still pulls a usable candidate
/// pool through the post-filter.
const KNN_MIN_FETCH: usize = 30;
/// HNSW search-expansion factor (`EF` in `<|K, EF|>`); larger = better recall,
/// more work. 64 is a conservative default for chronicle's small node-local graphs.
const HNSW_EF: usize = 64;

/// Compute the over-fetch K for a similarity search.
fn overfetch_k(limit: usize) -> usize {
    (limit.saturating_mul(KNN_OVERFETCH)).max(KNN_MIN_FETCH)
}

/// Clamp a requested BFS depth into the inline-safe range `[1, MAX_BFS_DEPTH]`.
/// The clamp is the injection guard: a usize through `1..=MAX` renders as digits.
fn clamp_bfs_depth(depth: usize) -> usize {
    depth.clamp(1, MAX_BFS_DEPTH)
}

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

    /// Run an OWNED SQL string with a list of OWNED `(name, value)` params (the
    /// shape the dynamic search/filter builders produce) and decode rows of `R`
    /// from result-set 0.
    ///
    /// Unlike [`fetch`], the SQL and params are owned, so callers can build them
    /// at runtime (filter fragments, per-depth BFS chains) without lifetime
    /// gymnastics. Every value is bound via `$param` — no user value is ever
    /// string-interpolated.
    async fn fetch_dyn<R>(
        &self,
        sql: String,
        params: Vec<(String, surrealdb::types::Value)>,
        ctx: &str,
    ) -> Result<Vec<R>, DriverError>
    where
        R: surrealdb::types::SurrealValue,
    {
        let mut q = self.db.query(sql);
        for (name, value) in params {
            q = q.bind((name, value));
        }
        let mut res = q
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?;
        res.take::<Vec<R>>(0)
            .map_err(|e| DriverError::Decode(format!("{ctx}: take: {e}")))
    }
}

/// Append a filter block to `where_parts` (each fragment a separate `AND` term)
/// and collect its params. The caller assembles the final `WHERE` from
/// `where_parts.join(" AND ")`.
fn push_filter(
    built: FilterFragments,
    where_parts: &mut Vec<String>,
    params: &mut Vec<(String, surrealdb::types::Value)>,
) {
    where_parts.extend(built.fragments);
    params.extend(built.params);
}

/// Bind a `group_ids` list param when non-empty, pushing the WHERE fragment.
fn push_group_filter(
    field: &str,
    group_ids: &[String],
    where_parts: &mut Vec<String>,
    params: &mut Vec<(String, surrealdb::types::Value)>,
) {
    if !group_ids.is_empty() {
        where_parts.push(format!("{field} IN $group_ids"));
        params.push(("group_ids".to_string(), group_ids.to_vec().into_value()));
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
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(source_uuid, target_uuid, "surreal get_edges_between_nodes");
        // Directed: source -> relates_to -> target. The relation row carries
        // `in`/`out` record links; we match them by the endpoint record ids.
        let src = record_id("entity", source_uuid);
        let dst = record_id("entity", target_uuid);
        let rows: Vec<EntityEdgeRow> = self
            .fetch(
                "SELECT * FROM relates_to WHERE in = $src AND out = $dst",
                |q| q.bind(("src", src)).bind(("dst", dst)),
                0,
                "get_edges_between_nodes",
            )
            .await?;
        Ok(rows.into_iter().map(entity_edge_from_row).collect())
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
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(limit, "surreal edge_fulltext_search");
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // BM25 over the `fact` index (@0@), scored by search::score(0). Group +
        // SearchFilters in the WHERE; ORDER score DESC LIMIT.
        let mut where_parts: Vec<String> = vec!["fact @0@ $query".to_string()];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("query".to_string(), query.to_string().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        push_filter(
            edge_filter_fragments(filters)?,
            &mut where_parts,
            &mut params,
        );
        params.push(("limit".to_string(), (limit as i64).into_value()));

        let sql = format!(
            "SELECT *, search::score(0) AS score FROM relates_to WHERE {} \
             ORDER BY score DESC LIMIT $limit",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredEntityEdgeRow> =
            self.fetch_dyn(sql, params, "edge_fulltext_search").await?;
        Ok(rows
            .into_iter()
            .map(|r| entity_edge_from_row(r.into_parts().0))
            .collect())
    }

    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(limit, min_score, "surreal edge_similarity_search");
        // HNSW KNN over-fetch (locked decision #3): the `<|K, EF|>` operator
        // selects K nearest FIRST, then the WHERE post-filters — so we over-fetch
        // and re-cut to `limit` after the group/SearchFilters + min_score pass.
        let k = overfetch_k(limit);
        let mut where_parts: Vec<String> = vec![format!("fact_embedding <|{k},{HNSW_EF}|> $vec")];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("vec".to_string(), search_vector.to_vec().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        push_filter(
            edge_filter_fragments(filters)?,
            &mut where_parts,
            &mut params,
        );

        // cosine distance ∈ [0, 2]; similarity = 1 - distance ∈ [-1, 1].
        let sql = format!(
            "SELECT *, (1 - vector::distance::knn()) AS score FROM relates_to WHERE {} \
             ORDER BY score DESC",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredEntityEdgeRow> = self
            .fetch_dyn(sql, params, "edge_similarity_search")
            .await?;
        Ok(rows
            .into_iter()
            .map(ScoredEntityEdgeRow::into_parts)
            .filter(|(_, score)| *score >= min_score as f64)
            .take(limit)
            .map(|(row, _)| entity_edge_from_row(row))
            .collect())
    }

    async fn node_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(limit, "surreal node_fulltext_search");
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // Multi-field per locked decision #5: separate name (@0@) + summary (@1@)
        // indexes, OR-ed; score = sum of the two field scores (a field that does
        // not match contributes 0).
        let mut where_parts: Vec<String> =
            vec!["(name @0@ $query OR summary @1@ $query)".to_string()];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("query".to_string(), query.to_string().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        push_filter(
            node_filter_fragments(filters)?,
            &mut where_parts,
            &mut params,
        );
        params.push(("limit".to_string(), (limit as i64).into_value()));

        let sql = format!(
            "SELECT *, (search::score(0) + search::score(1)) AS score FROM entity WHERE {} \
             ORDER BY score DESC LIMIT $limit",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredEntityNodeRow> =
            self.fetch_dyn(sql, params, "node_fulltext_search").await?;
        Ok(rows
            .into_iter()
            .map(|r| entity_node_from_row(r.into_parts().0))
            .collect())
    }

    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(limit, min_score, "surreal node_similarity_search");
        let k = overfetch_k(limit);
        let mut where_parts: Vec<String> = vec![format!("name_embedding <|{k},{HNSW_EF}|> $vec")];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("vec".to_string(), search_vector.to_vec().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        push_filter(
            node_filter_fragments(filters)?,
            &mut where_parts,
            &mut params,
        );

        let sql = format!(
            "SELECT *, (1 - vector::distance::knn()) AS score FROM entity WHERE {} \
             ORDER BY score DESC",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredEntityNodeRow> = self
            .fetch_dyn(sql, params, "node_similarity_search")
            .await?;
        Ok(rows
            .into_iter()
            .map(ScoredEntityNodeRow::into_parts)
            .filter(|(_, score)| *score >= min_score as f64)
            .take(limit)
            .map(|(row, _)| entity_node_from_row(row))
            .collect())
    }

    async fn node_bfs_search(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(max_depth, limit, "surreal node_bfs_search");
        if origins.is_empty() || max_depth < 1 {
            return Ok(Vec::new());
        }
        let rows = self
            .bfs_node_rows(origins, filters, max_depth, group_ids)
            .await?;
        // DISTINCT by uuid, preserving first-seen order; cap at `limit`.
        Ok(dedup_by_uuid(rows, |r| r.uuid.clone())
            .into_iter()
            .take(limit)
            .map(entity_node_from_row)
            .collect())
    }

    async fn edge_bfs_search(
        &self,
        origins: &[String],
        max_depth: usize,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(max_depth, limit, "surreal edge_bfs_search");
        if origins.is_empty() || max_depth < 1 {
            return Ok(Vec::new());
        }
        // MENTIONS hops extend reach but only RELATES_TO edges are returned
        // (mirrors Neo4j semantics). Collect the relates_to uuids traversed along
        // 1..=depth paths, then re-load those edge rows applying the SearchFilters.
        let edge_uuids = self.bfs_edge_uuids(origins, max_depth, group_ids).await?;
        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        let mut where_parts: Vec<String> = vec!["uuid IN $edge_uuids".to_string()];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("edge_uuids".to_string(), edge_uuids.into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        push_filter(
            edge_filter_fragments(filters)?,
            &mut where_parts,
            &mut params,
        );
        params.push(("limit".to_string(), (limit as i64).into_value()));

        let sql = format!(
            "SELECT * FROM relates_to WHERE {} LIMIT $limit",
            where_parts.join(" AND ")
        );
        let rows: Vec<EntityEdgeRow> = self.fetch_dyn(sql, params, "edge_bfs_search").await?;
        Ok(dedup_by_uuid(rows, |r| r.uuid.clone())
            .into_iter()
            .map(entity_edge_from_row)
            .collect())
    }

    async fn episode_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(limit, "surreal episode_fulltext_search");
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut where_parts: Vec<String> = vec!["content @0@ $query".to_string()];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("query".to_string(), query.to_string().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        params.push(("limit".to_string(), (limit as i64).into_value()));

        let sql = format!(
            "SELECT * FROM episodic WHERE {} \
             ORDER BY search::score(0) DESC LIMIT $limit",
            where_parts.join(" AND ")
        );
        let rows: Vec<EpisodicNodeRow> = self
            .fetch_dyn(sql, params, "episode_fulltext_search")
            .await?;
        rows.into_iter().map(episode_from_row).collect()
    }

    async fn get_embeddings_for_nodes(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        self.load_embeddings(
            "entity",
            "name_embedding",
            uuids,
            "get_embeddings_for_nodes",
        )
        .await
    }

    async fn get_embeddings_for_edges(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        self.load_embeddings(
            "relates_to",
            "fact_embedding",
            uuids,
            "get_embeddings_for_edges",
        )
        .await
    }

    async fn nodes_connected_to_center(
        &self,
        node_uuids: &[String],
        center_uuid: &str,
    ) -> Result<Vec<String>, DriverError> {
        debug!(center_uuid, "surreal nodes_connected_to_center");
        if node_uuids.is_empty() {
            return Ok(Vec::new());
        }
        // UNDIRECTED 1-hop relates_to adjacency to center: a candidate is adjacent
        // if a relates_to edge connects it to center in EITHER direction.
        let center = record_id("entity", center_uuid);
        let candidate_ids: Vec<surrealdb::types::Value> = node_uuids
            .iter()
            .map(|u| record_id("entity", u).into_value())
            .collect();
        let sql = "SELECT VALUE uuid FROM entity \
                   WHERE id IN $candidates \
                   AND (id IN (SELECT VALUE out FROM relates_to WHERE in = $center) \
                     OR id IN (SELECT VALUE in FROM relates_to WHERE out = $center))"
            .to_string();
        let params = vec![
            ("candidates".to_string(), candidate_ids.into_value()),
            ("center".to_string(), center.into_value()),
        ];
        let adjacent: Vec<String> = self
            .fetch_dyn(sql, params, "nodes_connected_to_center")
            .await?;
        Ok(adjacent)
    }

    async fn episode_mention_counts(
        &self,
        node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError> {
        debug!(count = node_uuids.len(), "surreal episode_mention_counts");
        if node_uuids.is_empty() {
            return Ok(HashMap::new());
        }
        // MENTIONS in-degree per target entity. Group the mentions relation by its
        // out endpoint, count per group, restricted to the requested targets.
        let targets: Vec<surrealdb::types::Value> = node_uuids
            .iter()
            .map(|u| record_id("entity", u).into_value())
            .collect();
        let sql = "SELECT out.uuid AS uuid, count() AS count FROM mentions \
                   WHERE out IN $targets GROUP BY uuid"
            .to_string();
        let params = vec![("targets".to_string(), targets.into_value())];
        let rows: Vec<MentionCountRow> = self
            .fetch_dyn(sql, params, "episode_mention_counts")
            .await?;
        Ok(rows
            .into_iter()
            .map(|r| (r.uuid, r.count.max(0) as u64))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Search helpers (BFS, embedding loading)
// ─────────────────────────────────────────────────────────────────────────

impl SurrealDriver {
    /// Iterative BFS node collection (locked-decision fallback: surrealdb 3.1.3's
    /// recursive `.{1..N}(...)` idiom returns only the deepest endpoint, not every
    /// node along 1..=N, so we expand depth-by-depth and union in Rust).
    ///
    /// At each depth `d` we follow `d` chained `->(relates_to,mentions)->entity`
    /// hops from the origins (MENTIONS hops extend reach exactly like Neo4j's
    /// `[:RELATES_TO|MENTIONS*1..N]`), apply the node SearchFilters + group filter,
    /// and accumulate the full entity rows. `depth` is clamped to `[1, 5]`.
    async fn bfs_node_rows(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
    ) -> Result<Vec<EntityNodeRow>, DriverError> {
        let depth = clamp_bfs_depth(max_depth);
        let origin_ids = bfs_origin_ids(origins);

        let mut out: Vec<EntityNodeRow> = Vec::new();
        for d in 1..=depth {
            // d chained hops: ->(relates_to,mentions)->entity repeated d times.
            let hop = "->(relates_to,mentions)->entity";
            let chain = hop.repeat(d);
            let mut where_parts: Vec<String> = Vec::new();
            let mut params: Vec<(String, surrealdb::types::Value)> =
                vec![("origins".to_string(), origin_ids.clone().into_value())];
            push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
            push_filter(
                node_filter_fragments(filters)?,
                &mut where_parts,
                &mut params,
            );
            let where_clause = if where_parts.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", where_parts.join(" AND "))
            };
            // Flatten the per-origin reachable set into entity rows.
            let sql = format!(
                "SELECT * FROM (SELECT VALUE {chain} FROM $origins).flatten(){where_clause}"
            );
            let mut rows: Vec<EntityNodeRow> =
                self.fetch_dyn(sql, params, "node_bfs_search").await?;
            out.append(&mut rows);
        }
        Ok(out)
    }

    /// Iterative BFS edge-uuid collection: the relates_to uuids traversed along
    /// 1..=depth paths from the origins. MENTIONS hops extend reach (so deeper
    /// relates_to edges are discoverable) but only relates_to uuids are collected.
    async fn bfs_edge_uuids(
        &self,
        origins: &[String],
        max_depth: usize,
        group_ids: &[String],
    ) -> Result<Vec<String>, DriverError> {
        let depth = clamp_bfs_depth(max_depth);
        let origin_ids = bfs_origin_ids(origins);

        let mut seen: Vec<String> = Vec::new();
        for d in 1..=depth {
            // Reach the node set at depth d-1 via mixed hops, then take the
            // relates_to edge uuids leaving it (the d-th hop's relates_to edges).
            // At d==1 the source set is the origins themselves.
            let mut where_parts: Vec<String> = Vec::new();
            let mut params: Vec<(String, surrealdb::types::Value)> =
                vec![("origins".to_string(), origin_ids.clone().into_value())];
            push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
            let where_clause = if where_parts.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", where_parts.join(" AND "))
            };
            // Build the d-1-hop source node-id set leaving the origins.
            let source_set = if d == 1 {
                "$origins".to_string()
            } else {
                let prefix = "->(relates_to,mentions)->entity".repeat(d - 1);
                format!(
                    "(SELECT VALUE id FROM (SELECT VALUE {prefix} FROM $origins).flatten(){where_clause})"
                )
            };
            // From each source node, the relates_to relation rows leaving it.
            let sql = format!("SELECT VALUE uuid FROM relates_to WHERE in IN {source_set}");
            let mut uuids: Vec<String> = self.fetch_dyn(sql, params, "edge_bfs_search").await?;
            seen.append(&mut uuids);
        }
        // Dedup, preserve order.
        let mut uniq: Vec<String> = Vec::new();
        for u in seen {
            if !uniq.contains(&u) {
                uniq.push(u);
            }
        }
        Ok(uniq)
    }

    /// Load a table's embedding column for the given uuids into a uuid→vector map,
    /// omitting uuids without a stored embedding (mirrors the Neo4j loaders).
    async fn load_embeddings(
        &self,
        table: &str,
        field: &str,
        uuids: &[String],
        ctx: &str,
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let sql = format!(
            "SELECT uuid, {field} AS embedding FROM {table} \
             WHERE uuid IN $uuids AND {field} != NONE"
        );
        let params = vec![("uuids".to_string(), uuids.to_vec().into_value())];
        let rows: Vec<EmbeddingRow> = self.fetch_dyn(sql, params, ctx).await?;
        Ok(rows.into_iter().map(|r| (r.uuid, r.embedding)).collect())
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
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(limit, "surreal community_fulltext_search");
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut where_parts: Vec<String> = vec!["name @0@ $query".to_string()];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("query".to_string(), query.to_string().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);
        params.push(("limit".to_string(), (limit as i64).into_value()));

        let sql = format!(
            "SELECT *, search::score(0) AS score FROM community WHERE {} \
             ORDER BY score DESC LIMIT $limit",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredCommunityNodeRow> = self
            .fetch_dyn(sql, params, "community_fulltext_search")
            .await?;
        Ok(rows
            .into_iter()
            .map(|r| community_node_from_row(r.into_parts().0))
            .collect())
    }

    async fn community_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(limit, min_score, "surreal community_similarity_search");
        let k = overfetch_k(limit);
        let mut where_parts: Vec<String> = vec![format!("name_embedding <|{k},{HNSW_EF}|> $vec")];
        let mut params: Vec<(String, surrealdb::types::Value)> =
            vec![("vec".to_string(), search_vector.to_vec().into_value())];
        push_group_filter("group_id", group_ids, &mut where_parts, &mut params);

        let sql = format!(
            "SELECT *, (1 - vector::distance::knn()) AS score FROM community WHERE {} \
             ORDER BY score DESC",
            where_parts.join(" AND ")
        );
        let rows: Vec<ScoredCommunityNodeRow> = self
            .fetch_dyn(sql, params, "community_similarity_search")
            .await?;
        Ok(rows
            .into_iter()
            .map(ScoredCommunityNodeRow::into_parts)
            .filter(|(_, score)| *score >= min_score as f64)
            .take(limit)
            .map(|(row, _)| community_node_from_row(row))
            .collect())
    }

    async fn get_embeddings_for_communities(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        self.load_embeddings(
            "community",
            "name_embedding",
            uuids,
            "get_embeddings_for_communities",
        )
        .await
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

/// Build the BFS origin record-id set. An origin is label-free in the trait
/// contract (it may be an Entity or an Episodic — a MENTIONS hop from an episode
/// origin must be reachable), so we emit BOTH `entity:<uuid>` and
/// `episodic:<uuid>` ids for each origin. A non-existent record simply yields no
/// out-edges, so over-emitting is harmless and lets a single traversal cover both
/// origin kinds (mirrors Neo4j's label-free `MATCH (origin {uuid})`).
fn bfs_origin_ids(origins: &[String]) -> Vec<surrealdb::types::Value> {
    let mut ids = Vec::with_capacity(origins.len() * 2);
    for u in origins {
        ids.push(record_id("entity", u).into_value());
        ids.push(record_id("episodic", u).into_value());
    }
    ids
}

/// DISTINCT-by-uuid preserving first-seen order (BFS unions the same node/edge
/// across depths). `key` extracts the dedup uuid for each row.
fn dedup_by_uuid<R>(rows: Vec<R>, key: impl Fn(&R) -> String) -> Vec<R> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<R> = Vec::with_capacity(rows.len());
    for r in rows {
        if seen.insert(key(&r)) {
            out.push(r);
        }
    }
    out
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
        // Only Task-3 ops remain pending; Task-2 search ops are implemented.
        let d = mem().await;
        assert!(d.get_mentioned_nodes(&["x".into()]).await.is_err());
        assert!(d.delete_episode("x").await.is_err());
        assert!(d.remove_communities().await.is_err());
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

    // ─────────────────────────────────────────────────────────────────────
    // Task-2: search / BFS / embeddings / adjacency / mentions / filters
    // ─────────────────────────────────────────────────────────────────────

    use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};
    use chronicle_core::types::EpisodicEdge;

    /// Entity with a controllable embedding (length must match the driver dim 8).
    fn entity_vec(name: &str, group: &str, emb: Vec<f32>) -> EntityNode {
        let mut n = EntityNode::new(name.into(), group.into(), Utc::now());
        n.summary = format!("{name} summary");
        n.name_embedding = Some(emb);
        n
    }

    fn edge(src: &str, dst: &str, name: &str, fact: &str, group: &str) -> EntityEdge {
        EntityEdge::new(
            src.into(),
            dst.into(),
            name.into(),
            fact.into(),
            group.into(),
        )
    }

    #[tokio::test]
    async fn node_similarity_ranks_exact_match_first_and_cuts_min_score() {
        let d = mem().await;
        // a = query direction (cos 1.0); c near (cos ~0.99); b orthogonal (cos 0).
        let a = entity_vec("alice", "g1", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = entity_vec("bob", "g1", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let c = entity_vec("carol", "g1", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        d.save_entity_nodes(&[a.clone(), b.clone(), c.clone()])
            .await
            .unwrap();

        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let got = d
            .node_similarity_search(&q, &SearchFilters::default(), &["g1".into()], 10, 0.5)
            .await
            .unwrap();
        let names: Vec<&str> = got.iter().map(|n| n.name.as_str()).collect();
        // exact match ranks first; orthogonal `bob` (cos 0 < 0.5) is cut.
        assert_eq!(names.first(), Some(&"alice"));
        assert!(names.contains(&"carol"));
        assert!(
            !names.contains(&"bob"),
            "min_score must drop orthogonal node"
        );
    }

    #[tokio::test]
    async fn node_similarity_limit_truncates_after_overfetch() {
        let d = mem().await;
        for i in 0..5 {
            let mut e = vec![0.0f32; 8];
            e[0] = 1.0 - (i as f32) * 0.01;
            e[1] = (i as f32) * 0.01;
            d.save_entity_nodes(&[entity_vec(&format!("n{i}"), "g1", e)])
                .await
                .unwrap();
        }
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let got = d
            .node_similarity_search(&q, &SearchFilters::default(), &["g1".into()], 2, 0.0)
            .await
            .unwrap();
        assert_eq!(got.len(), 2, "limit truncates to 2 after over-fetch");
        assert_eq!(got[0].name, "n0", "closest first");
    }

    #[tokio::test]
    async fn node_fulltext_recall_and_multifield() {
        let d = mem().await;
        let mut a = EntityNode::new("alpha project".into(), "g1".into(), Utc::now());
        a.summary = "unrelated".into();
        let mut b = EntityNode::new("beta".into(), "g1".into(), Utc::now());
        b.summary = "the alpha initiative".into();
        let mut c = EntityNode::new("gamma".into(), "g1".into(), Utc::now());
        c.summary = "nothing here".into();
        // Extra docs so BM25 IDF for "alpha" is > 0.
        let d1 = EntityNode::new("delta".into(), "g1".into(), Utc::now());
        let e1 = EntityNode::new("epsilon".into(), "g1".into(), Utc::now());
        d.save_entity_nodes(&[a, b, c, d1, e1]).await.unwrap();

        let got = d
            .node_fulltext_search("alpha", &SearchFilters::default(), &["g1".into()], 10)
            .await
            .unwrap();
        let names: Vec<String> = got.iter().map(|n| n.name.clone()).collect();
        // name-match and summary-match both recalled; non-match excluded.
        assert!(names.contains(&"alpha project".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert!(!names.contains(&"gamma".to_string()));
    }

    #[tokio::test]
    async fn edge_fulltext_and_similarity() {
        let d = mem().await;
        let a = entity_vec("A", "g1", vec![1.0; 8]);
        let b = entity_vec("B", "g1", vec![0.5; 8]);
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();
        let mut e1 = edge(&a.uuid, &b.uuid, "KNOWS", "alice founded acme corp", "g1");
        e1.fact_embedding = Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut e2 = edge(&b.uuid, &a.uuid, "LIKES", "bob enjoys tea", "g1");
        e2.fact_embedding = Some(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        d.save_entity_edges(&[e1.clone(), e2.clone()])
            .await
            .unwrap();

        let ft = d
            .edge_fulltext_search("acme", &SearchFilters::default(), &["g1".into()], 10)
            .await
            .unwrap();
        assert!(ft.iter().any(|e| e.uuid == e1.uuid));
        assert!(!ft.iter().any(|e| e.uuid == e2.uuid));

        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sim = d
            .edge_similarity_search(&q, &SearchFilters::default(), &["g1".into()], 10, 0.5)
            .await
            .unwrap();
        assert_eq!(sim.first().map(|e| e.uuid.clone()), Some(e1.uuid.clone()));
        assert!(!sim.iter().any(|e| e.uuid == e2.uuid));
    }

    #[tokio::test]
    async fn get_edges_between_nodes_is_directed() {
        let d = mem().await;
        let a = entity("A", "g1");
        let b = entity("B", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();
        let e = edge(&a.uuid, &b.uuid, "KNOWS", "a knows b", "g1");
        d.save_entity_edges(std::slice::from_ref(&e)).await.unwrap();

        let ab = d.get_edges_between_nodes(&a.uuid, &b.uuid).await.unwrap();
        assert_eq!(ab.len(), 1);
        assert_eq!(ab[0].uuid, e.uuid);
        // Reverse direction returns nothing (directed).
        let ba = d.get_edges_between_nodes(&b.uuid, &a.uuid).await.unwrap();
        assert!(ba.is_empty());
    }

    #[tokio::test]
    async fn embeddings_loaders_omit_missing() {
        let d = mem().await;
        let a = entity_vec("A", "g1", vec![0.1; 8]);
        let mut b = EntityNode::new("B".into(), "g1".into(), Utc::now()); // no embedding
        b.name_embedding = None;
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let map = d
            .get_embeddings_for_nodes(&[a.uuid.clone(), b.uuid.clone()])
            .await
            .unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&a.uuid), Some(&vec![0.1f32; 8]));
        assert!(!map.contains_key(&b.uuid));

        let mut e = edge(&a.uuid, &b.uuid, "R", "fact", "g1");
        e.fact_embedding = Some(vec![0.2; 8]);
        d.save_entity_edges(std::slice::from_ref(&e)).await.unwrap();
        let emap = d.get_embeddings_for_edges(&[e.uuid.clone()]).await.unwrap();
        assert_eq!(emap.get(&e.uuid), Some(&vec![0.2f32; 8]));
    }

    /// Build a small directed chain a->b->c->dd plus an episode mentioning a.
    async fn bfs_fixture(d: &SurrealDriver) -> (EntityNode, EntityNode, EntityNode, EntityNode) {
        let a = entity("a", "g1");
        let b = entity("b", "g1");
        let c = entity("c", "g1");
        let dd = entity("dd", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone(), c.clone(), dd.clone()])
            .await
            .unwrap();
        d.save_entity_edges(&[
            edge(&a.uuid, &b.uuid, "KNOWS", "a knows b", "g1"),
            edge(&b.uuid, &c.uuid, "LIKES", "b likes c", "g1"),
            edge(&c.uuid, &dd.uuid, "KNOWS", "c knows dd", "g1"),
        ])
        .await
        .unwrap();
        (a, b, c, dd)
    }

    #[tokio::test]
    async fn node_bfs_depth_1_vs_3() {
        let d = mem().await;
        let (a, b, c, dd) = bfs_fixture(&d).await;

        let depth1 = d
            .node_bfs_search(
                std::slice::from_ref(&a.uuid),
                &SearchFilters::default(),
                1,
                &["g1".into()],
                50,
            )
            .await
            .unwrap();
        let n1: Vec<String> = depth1.iter().map(|n| n.name.clone()).collect();
        assert!(n1.contains(&"b".to_string()));
        assert!(!n1.contains(&"c".to_string()), "depth 1 stops at b");

        let depth3 = d
            .node_bfs_search(
                std::slice::from_ref(&a.uuid),
                &SearchFilters::default(),
                3,
                &["g1".into()],
                50,
            )
            .await
            .unwrap();
        let n3: Vec<String> = depth3.iter().map(|n| n.name.clone()).collect();
        for want in ["b", "c", "dd"] {
            assert!(n3.contains(&want.to_string()), "depth 3 reaches {want}");
        }
        // DISTINCT: no duplicates across depths.
        let mut sorted = n3.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n3.len(), "results are distinct by uuid");
        let _ = (b, c, dd);
    }

    #[tokio::test]
    async fn node_bfs_edge_type_filter() {
        let d = mem().await;
        let (a, _b, _c, _dd) = bfs_fixture(&d).await;
        // Edge-type filter is an EDGE filter; for node BFS it does not constrain
        // intermediate hops (mirrors Neo4j node_bfs, which filters target nodes).
        // Here we assert depth-2 reach with a node-label filter instead.
        let f = SearchFilters {
            node_labels: Some(vec!["Entity".into()]),
            ..Default::default()
        };
        let got = d
            .node_bfs_search(std::slice::from_ref(&a.uuid), &f, 2, &["g1".into()], 50)
            .await
            .unwrap();
        let names: Vec<String> = got.iter().map(|n| n.name.clone()).collect();
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn node_bfs_mentions_origin_reach() {
        // An episode that MENTIONS `a` is a valid origin; the MENTIONS hop extends
        // reach so the BFS from the episode finds `a` (and its relates_to neighbours).
        let d = mem().await;
        let a = entity("a", "g1");
        let b = entity("b", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();
        d.save_entity_edges(&[edge(&a.uuid, &b.uuid, "KNOWS", "a knows b", "g1")])
            .await
            .unwrap();
        let ep = episode("ep", "g1", Utc::now());
        d.save_episode(&ep).await.unwrap();
        let me = EpisodicEdge::new(ep.uuid.clone(), a.uuid.clone(), "g1".into(), Utc::now());
        d.save_episodic_edges(std::slice::from_ref(&me))
            .await
            .unwrap();

        // Origin = the episode uuid; depth 2 = episode -MENTIONS-> a -RELATES_TO-> b.
        let got = d
            .node_bfs_search(
                std::slice::from_ref(&ep.uuid),
                &SearchFilters::default(),
                2,
                &["g1".into()],
                50,
            )
            .await
            .unwrap();
        let names: Vec<String> = got.iter().map(|n| n.name.clone()).collect();
        assert!(names.contains(&"a".to_string()), "MENTIONS hop reaches a");
        assert!(
            names.contains(&"b".to_string()),
            "then RELATES_TO reaches b"
        );
    }

    #[tokio::test]
    async fn edge_bfs_returns_only_relates_to() {
        let d = mem().await;
        let (a, _b, _c, _dd) = bfs_fixture(&d).await;
        let edges = d
            .edge_bfs_search(
                std::slice::from_ref(&a.uuid),
                3,
                &SearchFilters::default(),
                &["g1".into()],
                50,
            )
            .await
            .unwrap();
        // 3 relates_to edges along a->b->c->dd; all are RELATES_TO (no MENTIONS row).
        assert_eq!(edges.len(), 3);
        let facts: Vec<String> = edges.iter().map(|e| e.fact.clone()).collect();
        assert!(facts.iter().any(|f| f.contains("a knows b")));
        assert!(facts.iter().any(|f| f.contains("c knows dd")));
    }

    #[tokio::test]
    async fn center_adjacency_is_undirected() {
        let d = mem().await;
        let center = entity("center", "g1");
        let x = entity("x", "g1"); // center -> x
        let y = entity("y", "g1"); // y -> center
        let z = entity("z", "g1"); // not connected
        d.save_entity_nodes(&[center.clone(), x.clone(), y.clone(), z.clone()])
            .await
            .unwrap();
        d.save_entity_edges(&[
            edge(&center.uuid, &x.uuid, "R", "c->x", "g1"),
            edge(&y.uuid, &center.uuid, "R", "y->c", "g1"),
        ])
        .await
        .unwrap();

        let mut adj = d
            .nodes_connected_to_center(
                &[x.uuid.clone(), y.uuid.clone(), z.uuid.clone()],
                &center.uuid,
            )
            .await
            .unwrap();
        adj.sort();
        let mut want = vec![x.uuid.clone(), y.uuid.clone()];
        want.sort();
        assert_eq!(adj, want, "both directions count (undirected); z excluded");
    }

    #[tokio::test]
    async fn mention_counts_per_target() {
        let d = mem().await;
        let a = entity("a", "g1");
        let b = entity("b", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();
        let ep1 = episode("ep1", "g1", Utc::now());
        let ep2 = episode("ep2", "g1", Utc::now());
        d.save_episode(&ep1).await.unwrap();
        d.save_episode(&ep2).await.unwrap();
        // a mentioned twice, b once.
        d.save_episodic_edges(&[
            EpisodicEdge::new(ep1.uuid.clone(), a.uuid.clone(), "g1".into(), Utc::now()),
            EpisodicEdge::new(ep2.uuid.clone(), a.uuid.clone(), "g1".into(), Utc::now()),
            EpisodicEdge::new(ep1.uuid.clone(), b.uuid.clone(), "g1".into(), Utc::now()),
        ])
        .await
        .unwrap();

        let counts = d
            .episode_mention_counts(&[a.uuid.clone(), b.uuid.clone()])
            .await
            .unwrap();
        assert_eq!(counts.get(&a.uuid), Some(&2));
        assert_eq!(counts.get(&b.uuid), Some(&1));
    }

    #[tokio::test]
    async fn edge_filter_combo_types_dates_uuids() {
        let d = mem().await;
        let a = entity("A", "g1");
        let b = entity("B", "g1");
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();
        let base = Utc::now();
        let mut hit = edge(&a.uuid, &b.uuid, "KNOWS", "alice meets bob today", "g1");
        hit.valid_at = Some(base);
        hit.fact_embedding = Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut wrong_type = edge(&a.uuid, &b.uuid, "DISLIKES", "alice meets carol", "g1");
        wrong_type.valid_at = Some(base);
        wrong_type.fact_embedding = Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut out_of_window = edge(&a.uuid, &b.uuid, "KNOWS", "alice meets dave", "g1");
        out_of_window.valid_at = Some(base - chrono::Duration::days(10));
        out_of_window.fact_embedding = Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        d.save_entity_edges(&[hit.clone(), wrong_type.clone(), out_of_window.clone()])
            .await
            .unwrap();

        // edge_types=KNOWS + valid_at OR-of-ANDs window [base-1d, base+1d] + edge_uuids set.
        let f = SearchFilters {
            edge_types: Some(vec!["KNOWS".into()]),
            edge_uuids: Some(vec![hit.uuid.clone(), out_of_window.uuid.clone()]),
            valid_at: Some(vec![vec![
                DateFilter {
                    date: Some(base - chrono::Duration::days(1)),
                    comparison_operator: ComparisonOperator::Gte,
                },
                DateFilter {
                    date: Some(base + chrono::Duration::days(1)),
                    comparison_operator: ComparisonOperator::Lte,
                },
            ]]),
            ..Default::default()
        };
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let got = d
            .edge_similarity_search(&q, &f, &["g1".into()], 10, 0.0)
            .await
            .unwrap();
        let uuids: Vec<String> = got.iter().map(|e| e.uuid.clone()).collect();
        assert_eq!(
            uuids,
            vec![hit.uuid.clone()],
            "only the in-window KNOWS edge in the uuid set survives"
        );
    }

    #[tokio::test]
    async fn node_label_filter_excludes_non_matching() {
        let d = mem().await;
        let mut a = entity_vec("A", "g1", vec![1.0; 8]);
        a.labels = vec!["Person".into()];
        let mut b = entity_vec("B", "g1", vec![1.0; 8]);
        b.labels = vec!["Org".into()];
        d.save_entity_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let f = SearchFilters {
            node_labels: Some(vec!["Person".into()]),
            ..Default::default()
        };
        let q = vec![1.0f32; 8];
        let got = d
            .node_similarity_search(&q, &f, &["g1".into()], 10, 0.0)
            .await
            .unwrap();
        let names: Vec<String> = got.iter().map(|n| n.name.clone()).collect();
        assert!(names.contains(&"A".to_string()));
        assert!(
            !names.contains(&"B".to_string()),
            "Org node excluded by Person label filter"
        );
    }

    #[tokio::test]
    async fn community_search_fulltext_and_similarity() {
        let d = mem().await;
        let mut c1 = CommunityNode::new("alpha squad".into(), "g1".into(), Utc::now());
        c1.name_embedding = Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut c2 = CommunityNode::new("beta team".into(), "g1".into(), Utc::now());
        c2.name_embedding = Some(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Extra docs for IDF.
        let c3 = CommunityNode::new("gamma alpha".into(), "g1".into(), Utc::now());
        let c4 = CommunityNode::new("delta".into(), "g1".into(), Utc::now());
        d.save_community_nodes(&[c1.clone(), c2.clone(), c3.clone(), c4.clone()])
            .await
            .unwrap();

        let ft = d
            .community_fulltext_search("alpha", &["g1".into()], 10)
            .await
            .unwrap();
        let names: Vec<String> = ft.iter().map(|c| c.name.clone()).collect();
        assert!(names.contains(&"alpha squad".to_string()));
        assert!(!names.contains(&"beta team".to_string()));

        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sim = d
            .community_similarity_search(&q, &["g1".into()], 10, 0.5)
            .await
            .unwrap();
        assert_eq!(sim.first().map(|c| c.uuid.clone()), Some(c1.uuid.clone()));
        assert!(!sim.iter().any(|c| c.uuid == c2.uuid));

        let emap = d
            .get_embeddings_for_communities(&[c1.uuid.clone(), c4.uuid.clone()])
            .await
            .unwrap();
        assert!(emap.contains_key(&c1.uuid));
        assert!(!emap.contains_key(&c4.uuid), "no embedding → omitted");
    }

    #[tokio::test]
    async fn search_empty_inputs_short_circuit() {
        let d = mem().await;
        assert!(
            d.node_bfs_search(&[], &SearchFilters::default(), 3, &["g1".into()], 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            d.edge_bfs_search(
                &[String::from("x")],
                0,
                &SearchFilters::default(),
                &["g1".into()],
                10
            )
            .await
            .unwrap()
            .is_empty()
        );
        assert!(
            d.node_fulltext_search("", &SearchFilters::default(), &["g1".into()], 10)
                .await
                .unwrap()
                .is_empty()
        );
    }
}
