#![forbid(unsafe_code)]

//! FalkorDB `GraphDriver` backend for chronicle, over the `falkordb` 0.2 crate
//! (openCypher on a Redis module).
//!
//! This is a **Cypher-dialect adaptation of the Neo4j driver**, not a
//! from-scratch backend. The structural template is
//! [`chronicle_driver_neo4j::Neo4jDriver`]; the behavioural oracle is parity with
//! the `FakeDriver` / Neo4j conformance + e2e gate. The dialect deltas (verified
//! against live `falkordb/falkordb:latest`) live in [`convert`] and [`schema`]:
//!
//! - **Datetime → epoch millis `i64`** (FalkorDB has no native datetime type).
//! - **Embeddings → `vecf32([...])`** (written inline, read back as `Vec32`).
//! - **Attributes → a single JSON-string property** (`attrs_json`; nested-map
//!   properties are rejected by FalkorDB).
//! - **Parameters are textual Cypher literals** (`CYPHER key=<literal> <query>`).
//!   Every dynamic value crosses the boundary via the escaping encoders in
//!   [`convert`] — no raw string interpolation of user data.
//! - **No multi-statement / cross-statement transactions** (`GRAPH.QUERY` rejects
//!   `;`-separated statements). `save_all` is therefore best-effort sequential
//!   (Task 3); the atomicity gap is documented there.
//!
//! ## Phasing
//!
//! The full `GraphDriver` supertrait is implemented: persistence (Task 1),
//! search / BFS / embeddings / filters (Task 2), and bulk / maintenance /
//! community-cluster / saga-threading ops (Task 3). The conformance + e2e gate
//! exercises this backend through the [`chronicle_core::chronicle::Chronicle`]
//! facade against live `falkordb/falkordb:latest`.
//!
//! `save_all` is the one deliberate behavioural deviation: FalkorDB has no atomic
//! multi-statement batch, so it is **best-effort sequential** rather than
//! transactional (the Neo4j backend's `save_all` is atomic). See the
//! `BulkSaveOps` impl and `docs/port-fidelity.md`.

mod convert;
mod filters;
mod schema;

pub use schema::{
    NODE_FULLTEXT_INDICES, RANGE_INDICES, RELATIONSHIP_FULLTEXT, VECTOR_INDICES, already_exists,
};

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use falkordb::{FalkorAsyncClient, FalkorClientBuilder, FalkorValue};
use tracing::debug;

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

/// BFS depth bounds (parity with the Neo4j driver). The depth is inlined into
/// the var-length pattern `*1..N` (Cypher does not allow a parameter there), so
/// clamping into `[MIN_BFS_DEPTH, MAX_BFS_DEPTH]` is the injection guard — a
/// usize that has passed through `clamp` can only render as digits.
const MIN_BFS_DEPTH: usize = 1;
const MAX_BFS_DEPTH: usize = 5;

/// Clamp a requested BFS depth into the inline-safe range `[1, 5]` (the plan caps
/// FalkorDB depth at 5).
fn clamp_bfs_depth(depth: usize) -> usize {
    depth.clamp(MIN_BFS_DEPTH, MAX_BFS_DEPTH)
}

/// Vector-procedure over-fetch: KNN candidate count = `limit * 3`, floor 30. The
/// vector procedure cannot post-filter inline on min_score / group / labels, so
/// we over-fetch then filter + truncate in Rust.
fn overfetch(limit: usize) -> usize {
    limit.saturating_mul(3).max(30)
}

/// Build the optional group-scope WHERE fragment for the node alias `n`.
fn group_scope_node(group_ids: &[String]) -> Vec<String> {
    if group_ids.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "n.group_id IN {}",
            convert::lit_string_list(group_ids)
        )]
    }
}

/// Build the optional group-scope WHERE fragment for the edge alias `e`.
fn group_scope_edge(group_ids: &[String]) -> Vec<String> {
    if group_ids.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "e.group_id IN {}",
            convert::lit_string_list(group_ids)
        )]
    }
}

/// Join WHERE fragments into a ` WHERE a AND b` block (empty → empty string).
fn where_clause(fragments: &[String]) -> String {
    if fragments.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", fragments.join(" AND "))
    }
}

/// Parse `(uuid, embedding)` rows from an embeddings-loader query into a map.
fn embeddings_from_rows(
    rows: Vec<Vec<FalkorValue>>,
) -> Result<HashMap<String, Vec<f32>>, DriverError> {
    let mut out = HashMap::with_capacity(rows.len());
    for row in &rows {
        let uuid = convert::string_from_value(
            row.first()
                .ok_or_else(|| DriverError::Decode("embeddings: empty row".into()))?,
        )?;
        let emb = convert::embedding_from_value(
            row.get(1)
                .ok_or_else(|| DriverError::Decode("embeddings: missing embedding".into()))?,
        )?;
        out.insert(uuid, emb);
    }
    Ok(out)
}

/// FalkorDB-backed `GraphDriver`.
pub struct FalkorDriver {
    client: FalkorAsyncClient,
    graph_name: String,
    embedding_dim: usize,
}

impl FalkorDriver {
    /// Connect to FalkorDB, select the graph and build indices.
    ///
    /// `conn_str` is a FalkorDB connection string (e.g.
    /// `falkor://127.0.0.1:6379`). `graph_name` is the logical graph key in
    /// Redis. `embedding_dim` is the vector dimension used for the vector
    /// indices (must match the embedder).
    ///
    /// **Runtime requirement:** the `falkordb` 0.2 crate refreshes the graph
    /// schema (label / property-key id → name maps, needed to decode `--compact`
    /// result rows) via an internal blocking Redis round-trip. Under a
    /// single-threaded Tokio runtime that blocking call aborts and rows decode as
    /// `Unparseable`. The driver MUST therefore run on a multi-threaded runtime
    /// (`#[tokio::test(flavor = "multi_thread")]` / `Runtime::new()` with
    /// `rt-multi-thread`). Greentic's runtime is multi-threaded, so this is only a
    /// constraint for tests.
    pub async fn connect(
        conn_str: &str,
        graph_name: &str,
        embedding_dim: usize,
    ) -> Result<Self, DriverError> {
        let conn_info = conn_str
            .try_into()
            .map_err(|e| DriverError::Connection(format!("invalid connection string: {e}")))?;
        let client = FalkorClientBuilder::new_async()
            .with_connection_info(conn_info)
            .build()
            .await
            .map_err(|e| DriverError::Connection(e.to_string()))?;
        let driver = Self {
            client,
            graph_name: graph_name.to_string(),
            embedding_dim,
        };
        driver.build_indices_and_constraints(false).await?;
        Ok(driver)
    }

    /// Run a write/DDL query with no result handling (the result set is drained
    /// by the crate). `params` are pre-encoded Cypher literals.
    async fn run(
        &self,
        cypher: &str,
        params: &HashMap<String, String>,
        ctx: &str,
    ) -> Result<(), DriverError> {
        let mut graph = self.client.select_graph(&self.graph_name);
        graph
            .query(cypher)
            .with_params(params)
            .execute()
            .await
            .map(|_| ())
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))
    }

    /// Run a DDL statement that may already exist; swallow the "already exists"
    /// class of errors so index builds are idempotent.
    async fn run_idempotent(&self, cypher: &str, ctx: &str) -> Result<(), DriverError> {
        let mut graph = self.client.select_graph(&self.graph_name);
        match graph.query(cypher).execute().await {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if already_exists(&msg) {
                    Ok(())
                } else {
                    Err(DriverError::Query(format!("{ctx}: {msg}")))
                }
            }
        }
    }

    /// Run a read query and collect all rows as `Vec<Vec<FalkorValue>>`,
    /// positionally aligned with the (returned) header.
    async fn fetch_rows(
        &self,
        cypher: &str,
        params: &HashMap<String, String>,
        ctx: &str,
    ) -> Result<Vec<Vec<FalkorValue>>, DriverError> {
        let mut graph = self.client.select_graph(&self.graph_name);
        let mut result = graph
            .query(cypher)
            .with_params(params)
            .execute()
            .await
            .map_err(|e| DriverError::Query(format!("{ctx}: {e}")))?;
        Ok(result.data.by_ref().collect())
    }

    /// Fetch the single first column of the first row as a Node's property map,
    /// returning the parsed value via `f`. Used by all single-node getters.
    async fn fetch_one_node<T>(
        &self,
        cypher: &str,
        params: &HashMap<String, String>,
        ctx: &str,
        f: impl Fn(&HashMap<String, FalkorValue>) -> Result<T, DriverError>,
    ) -> Result<Option<T>, DriverError> {
        let rows = self.fetch_rows(cypher, params, ctx).await?;
        match rows.first().and_then(|r| r.first()) {
            Some(value) => Ok(Some(f(convert::node_props(value)?)?)),
            None => Ok(None),
        }
    }

    /// Fetch all rows, parsing the first column of each as a Node property map.
    async fn fetch_nodes<T>(
        &self,
        cypher: &str,
        params: &HashMap<String, String>,
        ctx: &str,
        f: impl Fn(&HashMap<String, FalkorValue>) -> Result<T, DriverError>,
    ) -> Result<Vec<T>, DriverError> {
        let rows = self.fetch_rows(cypher, params, ctx).await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let value = row
                .first()
                .ok_or_else(|| DriverError::Decode(format!("{ctx}: empty row")))?;
            out.push(f(convert::node_props(value)?)?);
        }
        Ok(out)
    }

    /// Collect `(node, score)` rows from a vector-similarity query, converting the
    /// cosine DISTANCE score to a similarity (`1 - score`), post-filtering
    /// `sim >= min_score`, and truncating to `limit`. Rows arrive ordered by score
    /// ASC (closest first), so the order is preserved after truncation.
    fn collect_scored_nodes<T>(
        &self,
        rows: Vec<Vec<FalkorValue>>,
        min_score: f32,
        limit: usize,
        f: impl Fn(&HashMap<String, FalkorValue>) -> Result<T, DriverError>,
    ) -> Result<Vec<T>, DriverError> {
        let mut out = Vec::new();
        for row in &rows {
            let node = row
                .first()
                .ok_or_else(|| DriverError::Decode("scored node: empty row".into()))?;
            let score = convert::f64_from_value(
                row.get(1)
                    .ok_or_else(|| DriverError::Decode("scored node: missing score".into()))?,
            )?;
            let similarity = 1.0 - score as f32;
            if similarity >= min_score {
                out.push(f(convert::node_props(node)?)?);
                if out.len() == limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Edge analogue of [`Self::collect_scored_nodes`].
    fn collect_scored_edges(
        &self,
        rows: Vec<Vec<FalkorValue>>,
        min_score: f32,
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let mut out = Vec::new();
        for row in &rows {
            let edge = row
                .first()
                .ok_or_else(|| DriverError::Decode("scored edge: empty row".into()))?;
            let score = convert::f64_from_value(
                row.get(1)
                    .ok_or_else(|| DriverError::Decode("scored edge: missing score".into()))?,
            )?;
            let similarity = 1.0 - score as f32;
            if similarity >= min_score {
                out.push(convert::entity_edge_from_props(convert::edge_props(edge)?)?);
                if out.len() == limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Fetch all rows, parsing the first column of each as an Edge property map.
    async fn fetch_edges<T>(
        &self,
        cypher: &str,
        params: &HashMap<String, String>,
        ctx: &str,
        f: impl Fn(&HashMap<String, FalkorValue>) -> Result<T, DriverError>,
    ) -> Result<Vec<T>, DriverError> {
        let rows = self.fetch_rows(cypher, params, ctx).await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let value = row
                .first()
                .ok_or_else(|| DriverError::Decode(format!("{ctx}: empty row")))?;
            out.push(f(convert::edge_props(value)?)?);
        }
        Ok(out)
    }
}

// =====================================================================
// EntityNodeOps
// =====================================================================

#[async_trait]
impl EntityNodeOps for FalkorDriver {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "falkor save_entity_nodes");
        // No nested-map params / multi-statement: one MERGE per node.
        for node in nodes {
            let mut labels = node.labels.clone();
            if !labels.iter().any(|l| l == "Entity") {
                labels.push("Entity".to_string());
            }
            // Property assignments as Cypher literals (params would be string-only
            // and the embedding needs the vecf32 constructor inline anyway).
            let mut sets = vec![
                format!("n.name = {}", convert::lit_str(&node.name)),
                format!("n.group_id = {}", convert::lit_str(&node.group_id)),
                format!("n.summary = {}", convert::lit_str(&node.summary)),
                format!("n.created_at = {}", convert::lit_datetime(node.created_at)),
                format!("n.labels = {}", convert::lit_string_list(&labels)),
                format!("n.attrs_json = {}", convert::lit_attrs(&node.attributes)?),
            ];
            if let Some(emb) = &node.name_embedding {
                sets.push(format!("n.name_embedding = {}", convert::lit_vecf32(emb)));
            }
            let cypher = format!(
                "MERGE (n:Entity {{uuid: {}}}) SET {}",
                convert::lit_str(&node.uuid),
                sets.join(", ")
            );
            self.run(&cypher, &HashMap::new(), "save_entity_nodes")
                .await?;
        }
        Ok(())
    }

    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError> {
        debug!(uuid, "falkor get_entity_node");
        let cypher = format!(
            "MATCH (n:Entity {{uuid: {}}}) RETURN n",
            convert::lit_str(uuid)
        );
        self.fetch_one_node(
            &cypher,
            &HashMap::new(),
            "get_entity_node",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(count = uuids.len(), "falkor get_entity_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        // Order-preserving: MATCH all, then UNWIND the requested order and re-join.
        let cypher = format!(
            "UNWIND {} AS wanted \
             MATCH (n:Entity {{uuid: wanted}}) RETURN n",
            convert::lit_string_list(uuids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_entity_nodes_by_uuids",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn get_entity_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(
            count = group_ids.len(),
            "falkor get_entity_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "MATCH (n:Entity) WHERE n.group_id IN {} RETURN n",
            convert::lit_string_list(group_ids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_entity_nodes_by_group_ids",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn get_mentioned_nodes(
        &self,
        episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(count = episode_uuids.len(), "falkor get_mentioned_nodes");
        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }
        // Entities MENTIONS-targeted by any of the given episodes, DISTINCT
        // (parity with Neo4j GET_MENTIONED_NODES).
        let cypher = format!(
            "MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) \
             WHERE episode.uuid IN {} RETURN DISTINCT n",
            convert::lit_string_list(episode_uuids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_mentioned_nodes",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn delete_entity_nodes_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        debug!(count = uuids.len(), "falkor delete_entity_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(());
        }
        let cypher = format!(
            "MATCH (n:Entity) WHERE n.uuid IN {} DETACH DELETE n",
            convert::lit_string_list(uuids)
        );
        self.run(&cypher, &HashMap::new(), "delete_entity_nodes_by_uuids")
            .await
    }
}

// =====================================================================
// EntityEdgeOps
// =====================================================================

#[async_trait]
impl EntityEdgeOps for FalkorDriver {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "falkor save_entity_edges");
        for edge in edges {
            let mut sets = vec![
                format!("e.name = {}", convert::lit_str(&edge.name)),
                format!("e.fact = {}", convert::lit_str(&edge.fact)),
                format!("e.group_id = {}", convert::lit_str(&edge.group_id)),
                format!(
                    "e.source_node_uuid = {}",
                    convert::lit_str(&edge.source_node_uuid)
                ),
                format!(
                    "e.target_node_uuid = {}",
                    convert::lit_str(&edge.target_node_uuid)
                ),
                format!("e.episodes = {}", convert::lit_string_list(&edge.episodes)),
                format!("e.created_at = {}", convert::lit_datetime(edge.created_at)),
                format!(
                    "e.expired_at = {}",
                    convert::opt_lit(edge.expired_at.map(convert::lit_datetime))
                ),
                format!(
                    "e.valid_at = {}",
                    convert::opt_lit(edge.valid_at.map(convert::lit_datetime))
                ),
                format!(
                    "e.invalid_at = {}",
                    convert::opt_lit(edge.invalid_at.map(convert::lit_datetime))
                ),
                format!("e.attrs_json = {}", convert::lit_attrs(&edge.attributes)?),
            ];
            if let Some(emb) = &edge.fact_embedding {
                sets.push(format!("e.fact_embedding = {}", convert::lit_vecf32(emb)));
            }
            let cypher = format!(
                "MATCH (source:Entity {{uuid: {}}}), (target:Entity {{uuid: {}}}) \
                 MERGE (source)-[e:RELATES_TO {{uuid: {}}}]->(target) SET {}",
                convert::lit_str(&edge.source_node_uuid),
                convert::lit_str(&edge.target_node_uuid),
                convert::lit_str(&edge.uuid),
                sets.join(", ")
            );
            self.run(&cypher, &HashMap::new(), "save_entity_edges")
                .await?;
        }
        Ok(())
    }

    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
        debug!(uuid, "falkor get_entity_edge");
        let cypher = format!(
            "MATCH (n:Entity)-[e:RELATES_TO {{uuid: {}}}]->(m:Entity) RETURN e",
            convert::lit_str(uuid)
        );
        let edges = self
            .fetch_edges(
                &cypher,
                &HashMap::new(),
                "get_entity_edge",
                convert::entity_edge_from_props,
            )
            .await?;
        Ok(edges.into_iter().next())
    }

    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(source_uuid, target_uuid, "falkor get_edges_between_nodes");
        // Directed source -> target RELATES_TO (parity with Neo4j
        // EntityEdge.get_between_nodes).
        let cypher = format!(
            "MATCH (n:Entity {{uuid: {}}})-[e:RELATES_TO]->(m:Entity {{uuid: {}}}) RETURN e",
            convert::lit_str(source_uuid),
            convert::lit_str(target_uuid)
        );
        self.fetch_edges(
            &cypher,
            &HashMap::new(),
            "get_edges_between_nodes",
            convert::entity_edge_from_props,
        )
        .await
    }

    async fn get_entity_edges_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(count = uuids.len(), "falkor get_entity_edges_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.uuid IN {} RETURN e",
            convert::lit_string_list(uuids)
        );
        self.fetch_edges(
            &cypher,
            &HashMap::new(),
            "get_entity_edges_by_uuids",
            convert::entity_edge_from_props,
        )
        .await
    }

    async fn delete_entity_edges_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        debug!(count = uuids.len(), "falkor delete_entity_edges_by_uuids");
        if uuids.is_empty() {
            return Ok(());
        }
        let cypher = format!(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN {} DELETE e",
            convert::lit_string_list(uuids)
        );
        self.run(&cypher, &HashMap::new(), "delete_entity_edges_by_uuids")
            .await
    }
}

// =====================================================================
// EpisodeOps
// =====================================================================

#[async_trait]
impl EpisodeOps for FalkorDriver {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError> {
        debug!(uuid = episode.uuid, "falkor save_episode");
        let sets = [
            format!("n.name = {}", convert::lit_str(&episode.name)),
            format!("n.group_id = {}", convert::lit_str(&episode.group_id)),
            format!(
                "n.source_description = {}",
                convert::lit_str(&episode.source_description)
            ),
            format!(
                "n.source = {}",
                convert::lit_str(convert::episode_type_to_str(episode.source))
            ),
            format!("n.content = {}", convert::lit_str(&episode.content)),
            format!(
                "n.entity_edges = {}",
                convert::lit_string_list(&episode.entity_edges)
            ),
            format!(
                "n.created_at = {}",
                convert::lit_datetime(episode.created_at)
            ),
            format!("n.valid_at = {}", convert::lit_datetime(episode.valid_at)),
        ];
        let cypher = format!(
            "MERGE (n:Episodic {{uuid: {}}}) SET {}",
            convert::lit_str(&episode.uuid),
            sets.join(", ")
        );
        self.run(&cypher, &HashMap::new(), "save_episode").await
    }

    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
        debug!(uuid, "falkor get_episode");
        let cypher = format!(
            "MATCH (n:Episodic {{uuid: {}}}) RETURN n",
            convert::lit_str(uuid)
        );
        self.fetch_one_node(
            &cypher,
            &HashMap::new(),
            "get_episode",
            convert::episodic_node_from_props,
        )
        .await
    }

    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(count = uuids.len(), "falkor get_episodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "UNWIND {} AS wanted MATCH (n:Episodic {{uuid: wanted}}) RETURN n",
            convert::lit_string_list(uuids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_episodes_by_uuids",
            convert::episodic_node_from_props,
        )
        .await
    }

    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(last_n, "falkor retrieve_episodes");
        // valid_at stored as epoch millis → integer comparison.
        let mut wheres = vec![format!(
            "n.valid_at <= {}",
            convert::lit_int(convert::datetime_to_millis(reference_time))
        )];
        if !group_ids.is_empty() {
            wheres.push(format!(
                "n.group_id IN {}",
                convert::lit_string_list(group_ids)
            ));
        }
        if let Some(s) = source {
            wheres.push(format!(
                "n.source = {}",
                convert::lit_str(convert::episode_type_to_str(s))
            ));
        }
        let cypher = format!(
            "MATCH (n:Episodic) WHERE {} RETURN n ORDER BY n.valid_at DESC LIMIT {}",
            wheres.join(" AND "),
            convert::lit_int(last_n as i64)
        );
        let mut episodes = self
            .fetch_nodes(
                &cypher,
                &HashMap::new(),
                "retrieve_episodes",
                convert::episodic_node_from_props,
            )
            .await?;
        // Query yields DESC; reverse to chronological order (parity with Neo4j).
        episodes.reverse();
        Ok(episodes)
    }

    async fn delete_episode(&self, uuid: &str) -> Result<(), DriverError> {
        debug!(uuid, "falkor delete_episode");
        let cypher = format!(
            "MATCH (n:Episodic {{uuid: {}}}) DETACH DELETE n",
            convert::lit_str(uuid)
        );
        self.run(&cypher, &HashMap::new(), "delete_episode").await
    }
}

// =====================================================================
// EpisodicEdgeOps (MENTIONS)
// =====================================================================

#[async_trait]
impl EpisodicEdgeOps for FalkorDriver {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "falkor save_episodic_edges");
        for edge in edges {
            let cypher = format!(
                "MATCH (episode:Episodic {{uuid: {}}}), (node:Entity {{uuid: {}}}) \
                 MERGE (episode)-[e:MENTIONS {{uuid: {}}}]->(node) \
                 SET e.group_id = {}, e.created_at = {}",
                convert::lit_str(&edge.source_node_uuid),
                convert::lit_str(&edge.target_node_uuid),
                convert::lit_str(&edge.uuid),
                convert::lit_str(&edge.group_id),
                convert::lit_datetime(edge.created_at),
            );
            self.run(&cypher, &HashMap::new(), "save_episodic_edges")
                .await?;
        }
        Ok(())
    }
}

// =====================================================================
// BulkSaveOps — best-effort sequential save_all (DEVIATION).
//
// FalkorDB's `GRAPH.QUERY` rejects `;`-separated multi-statement bodies and the
// `falkordb` 0.2 crate exposes no MULTI/EXEC transaction handle, so there is no
// way to wrap the four collection writes in a single atomic batch. Unlike the
// Neo4j backend (which OVERRIDES `save_all` with a real `start_txn → run all →
// commit`), this backend keeps the default-shaped four-call sequence: episodes,
// entity nodes, entity edges, episodic edges, in that order.
//
// We still OVERRIDE the trait method (rather than inherit the default) to make
// the atomicity deviation explicit at the call site and to document it here: a
// mid-batch failure leaves earlier collections persisted (no rollback). The
// ordering matches the default so a later failure (e.g. an edge whose endpoint
// node never saved) cannot succeed-then-orphan. This is an accepted FalkorDB
// limitation, recorded in docs/port-fidelity.md.
// =====================================================================

#[async_trait]
impl BulkSaveOps for FalkorDriver {
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
            "falkor save_all (best-effort sequential — no atomic batch)"
        );
        // DEVIATION: best-effort, non-atomic. See the module comment above.
        for episode in episodes {
            self.save_episode(episode).await?;
        }
        self.save_entity_nodes(entity_nodes).await?;
        self.save_entity_edges(entity_edges).await?;
        self.save_episodic_edges(episodic_edges).await?;
        Ok(())
    }
}

// =====================================================================
// SearchOps — all Task 2.
// =====================================================================

#[async_trait]
impl SearchOps for FalkorDriver {
    async fn edge_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(query, limit, "falkor edge_fulltext_search");
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        // Relationship fulltext via the DDL-form index on RELATES_TO.fact.
        let (rel, _prop) = schema::RELATIONSHIP_FULLTEXT;
        let mut wheres = group_scope_edge(group_ids);
        wheres.extend(filters::edge_filter_fragments(filters)?);
        let where_block = where_clause(&wheres);
        let cypher = format!(
            "CALL db.idx.fulltext.queryRelationships('{rel}', {q}) YIELD relationship AS e, score \
             MATCH (n:Entity)-[e2:RELATES_TO {{uuid: e.uuid}}]->(m:Entity) \
             WITH e2 AS e, n, m, score{where_block} \
             RETURN e ORDER BY score DESC LIMIT {lim}",
            q = convert::lit_str(query),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_edges(
            &cypher,
            &HashMap::new(),
            "edge_fulltext_search",
            convert::entity_edge_from_props,
        )
        .await
    }

    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(limit, min_score, "falkor edge_similarity_search");
        if limit == 0 {
            return Ok(Vec::new());
        }
        let k = overfetch(limit);
        let mut wheres = group_scope_edge(group_ids);
        wheres.extend(filters::edge_filter_fragments(filters)?);
        let where_block = where_clause(&wheres);
        // The vector procedure yields (relationship, score=cosine DISTANCE). We
        // re-MATCH the relationship by uuid so the endpoint aliases n/m exist for
        // the node-label / group / date filters, then carry score through.
        let cypher = format!(
            "CALL db.idx.vector.queryRelationships('RELATES_TO','fact_embedding',{k},{vec}) \
             YIELD relationship AS rel, score \
             MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]->(m:Entity){where_block} \
             RETURN e, score ORDER BY score ASC",
            vec = convert::lit_vecf32(search_vector),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "edge_similarity_search")
            .await?;
        self.collect_scored_edges(rows, min_score, limit)
    }

    async fn node_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(query, limit, "falkor node_fulltext_search");
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut wheres = group_scope_node(group_ids);
        wheres.extend(filters::node_filter_fragments(filters)?);
        let where_block = where_clause(&wheres);
        let cypher = format!(
            "CALL db.idx.fulltext.queryNodes('Entity', {q}) YIELD node AS n, score{where_block} \
             RETURN n ORDER BY score DESC LIMIT {lim}",
            q = convert::lit_str(query),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "node_fulltext_search",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(limit, min_score, "falkor node_similarity_search");
        if limit == 0 {
            return Ok(Vec::new());
        }
        let k = overfetch(limit);
        let mut wheres = group_scope_node(group_ids);
        wheres.extend(filters::node_filter_fragments(filters)?);
        let where_block = where_clause(&wheres);
        let cypher = format!(
            "CALL db.idx.vector.queryNodes('Entity','name_embedding',{k},{vec}) \
             YIELD node AS n, score{where_block} \
             RETURN n, score ORDER BY score ASC",
            vec = convert::lit_vecf32(search_vector),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "node_similarity_search")
            .await?;
        self.collect_scored_nodes(rows, min_score, limit, convert::entity_node_from_props)
    }

    async fn node_bfs_search(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        debug!(max_depth, limit, "falkor node_bfs_search");
        if origins.is_empty() || max_depth < 1 || limit == 0 {
            return Ok(Vec::new());
        }
        let depth = clamp_bfs_depth(max_depth);
        let mut wheres = vec!["n.group_id = origin.group_id".to_string()];
        if !group_ids.is_empty() {
            let list = convert::lit_string_list(group_ids);
            wheres.push(format!("n.group_id IN {list}"));
            wheres.push(format!("origin.group_id IN {list}"));
        }
        wheres.extend(filters::node_filter_fragments(filters)?);
        let cypher = format!(
            "UNWIND {origins} AS origin_uuid \
             MATCH (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{depth}]->(n:Entity) \
             WHERE {wheres} \
             RETURN DISTINCT n LIMIT {lim}",
            origins = convert::lit_string_list(origins),
            wheres = wheres.join(" AND "),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "node_bfs_search",
            convert::entity_node_from_props,
        )
        .await
    }

    async fn edge_bfs_search(
        &self,
        origins: &[String],
        max_depth: usize,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        debug!(max_depth, limit, "falkor edge_bfs_search");
        if origins.is_empty() || max_depth < 1 || limit == 0 {
            return Ok(Vec::new());
        }
        let depth = clamp_bfs_depth(max_depth);
        let mut wheres = group_scope_edge(group_ids);
        wheres.extend(filters::edge_filter_fragments(filters)?);
        let where_block = where_clause(&wheres);
        // Expand RELATES_TO|MENTIONS paths, then re-MATCH each traversed
        // RELATES_TO edge by uuid (UNDIRECTED, parity with Neo4j) so the endpoint
        // aliases exist for filters. Only RELATES_TO edges are returned.
        let cypher = format!(
            "UNWIND {origins} AS origin_uuid \
             MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{depth}]->(:Entity) \
             UNWIND relationships(path) AS rel \
             MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity){where_block} \
             RETURN DISTINCT e LIMIT {lim}",
            origins = convert::lit_string_list(origins),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_edges(
            &cypher,
            &HashMap::new(),
            "edge_bfs_search",
            convert::entity_edge_from_props,
        )
        .await
    }

    async fn episode_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        debug!(query, limit, "falkor episode_fulltext_search");
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let where_block = if group_ids.is_empty() {
            String::new()
        } else {
            format!(
                " WHERE n.group_id IN {}",
                convert::lit_string_list(group_ids)
            )
        };
        let cypher = format!(
            "CALL db.idx.fulltext.queryNodes('Episodic', {q}) YIELD node AS n, score{where_block} \
             RETURN n ORDER BY score DESC LIMIT {lim}",
            q = convert::lit_str(query),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "episode_fulltext_search",
            convert::episodic_node_from_props,
        )
        .await
    }

    async fn get_embeddings_for_nodes(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "falkor get_embeddings_for_nodes");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let cypher = format!(
            "MATCH (n:Entity) WHERE n.uuid IN {} AND n.name_embedding IS NOT NULL \
             RETURN DISTINCT n.uuid, n.name_embedding",
            convert::lit_string_list(uuids)
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "get_embeddings_for_nodes")
            .await?;
        embeddings_from_rows(rows)
    }

    async fn get_embeddings_for_edges(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "falkor get_embeddings_for_edges");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let cypher = format!(
            "MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity) \
             WHERE e.uuid IN {} AND e.fact_embedding IS NOT NULL \
             RETURN DISTINCT e.uuid, e.fact_embedding",
            convert::lit_string_list(uuids)
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "get_embeddings_for_edges")
            .await?;
        embeddings_from_rows(rows)
    }

    async fn nodes_connected_to_center(
        &self,
        node_uuids: &[String],
        center_uuid: &str,
    ) -> Result<Vec<String>, DriverError> {
        debug!(count = node_uuids.len(), "falkor nodes_connected_to_center");
        if node_uuids.is_empty() {
            return Ok(Vec::new());
        }
        // 1-hop UNDIRECTED RELATES_TO adjacency (parity with Neo4j).
        let cypher = format!(
            "UNWIND {nodes} AS node_uuid \
             MATCH (center:Entity {{uuid: {center}}})-[:RELATES_TO]-(n:Entity {{uuid: node_uuid}}) \
             RETURN node_uuid",
            nodes = convert::lit_string_list(node_uuids),
            center = convert::lit_str(center_uuid),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "nodes_connected_to_center")
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let value = row.first().ok_or_else(|| {
                DriverError::Decode("nodes_connected_to_center: empty row".into())
            })?;
            out.push(convert::string_from_value(value)?);
        }
        Ok(out)
    }

    async fn episode_mention_counts(
        &self,
        node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError> {
        debug!(count = node_uuids.len(), "falkor episode_mention_counts");
        if node_uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let cypher = format!(
            "UNWIND {nodes} AS node_uuid \
             MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {{uuid: node_uuid}}) \
             RETURN n.uuid, count(*)",
            nodes = convert::lit_string_list(node_uuids),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "episode_mention_counts")
            .await?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in &rows {
            let uuid =
                convert::string_from_value(row.first().ok_or_else(|| {
                    DriverError::Decode("episode_mention_counts: empty row".into())
                })?)?;
            let count = convert::i64_from_value(row.get(1).ok_or_else(|| {
                DriverError::Decode("episode_mention_counts: missing count".into())
            })?)?;
            out.insert(uuid, count.max(0) as u64);
        }
        Ok(out)
    }
}

// =====================================================================
// CommunityOps — save/get implemented (Task 1); search/cluster Task 2/3.
// =====================================================================

#[async_trait]
impl CommunityOps for FalkorDriver {
    async fn save_community_nodes(&self, nodes: &[CommunityNode]) -> Result<(), DriverError> {
        debug!(count = nodes.len(), "falkor save_community_nodes");
        for node in nodes {
            let mut sets = vec![
                format!("n.name = {}", convert::lit_str(&node.name)),
                format!("n.group_id = {}", convert::lit_str(&node.group_id)),
                format!("n.summary = {}", convert::lit_str(&node.summary)),
                format!("n.created_at = {}", convert::lit_datetime(node.created_at)),
            ];
            if let Some(emb) = &node.name_embedding {
                sets.push(format!("n.name_embedding = {}", convert::lit_vecf32(emb)));
            }
            let cypher = format!(
                "MERGE (n:Community {{uuid: {}}}) SET {}",
                convert::lit_str(&node.uuid),
                sets.join(", ")
            );
            self.run(&cypher, &HashMap::new(), "save_community_nodes")
                .await?;
        }
        Ok(())
    }

    async fn save_community_edges(&self, edges: &[CommunityEdge]) -> Result<(), DriverError> {
        debug!(count = edges.len(), "falkor save_community_edges");
        for edge in edges {
            // Target may be an Entity or a Community; match label-free by uuid.
            let cypher = format!(
                "MATCH (community:Community {{uuid: {}}}), (node {{uuid: {}}}) \
                 MERGE (community)-[e:HAS_MEMBER {{uuid: {}}}]->(node) \
                 SET e.group_id = {}, e.created_at = {}",
                convert::lit_str(&edge.source_node_uuid),
                convert::lit_str(&edge.target_node_uuid),
                convert::lit_str(&edge.uuid),
                convert::lit_str(&edge.group_id),
                convert::lit_datetime(edge.created_at),
            );
            self.run(&cypher, &HashMap::new(), "save_community_edges")
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
            "falkor get_community_nodes_by_group_ids"
        );
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "MATCH (n:Community) WHERE n.group_id IN {} RETURN n",
            convert::lit_string_list(group_ids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_community_nodes_by_group_ids",
            convert::community_node_from_props,
        )
        .await
    }

    async fn get_community_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(count = uuids.len(), "falkor get_community_nodes_by_uuids");
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let cypher = format!(
            "UNWIND {} AS wanted MATCH (n:Community {{uuid: wanted}}) RETURN n",
            convert::lit_string_list(uuids)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "get_community_nodes_by_uuids",
            convert::community_node_from_props,
        )
        .await
    }

    async fn community_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(query, limit, "falkor community_fulltext_search");
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let where_block = if group_ids.is_empty() {
            String::new()
        } else {
            format!(
                " WHERE n.group_id IN {}",
                convert::lit_string_list(group_ids)
            )
        };
        let cypher = format!(
            "CALL db.idx.fulltext.queryNodes('Community', {q}) YIELD node AS n, score{where_block} \
             RETURN n ORDER BY score DESC LIMIT {lim}",
            q = convert::lit_str(query),
            lim = convert::lit_int(limit as i64),
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "community_fulltext_search",
            convert::community_node_from_props,
        )
        .await
    }

    async fn community_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(limit, min_score, "falkor community_similarity_search");
        if limit == 0 {
            return Ok(Vec::new());
        }
        let k = overfetch(limit);
        let where_block = where_clause(&group_scope_node(group_ids));
        let cypher = format!(
            "CALL db.idx.vector.queryNodes('Community','name_embedding',{k},{vec}) \
             YIELD node AS n, score{where_block} \
             RETURN n, score ORDER BY score ASC",
            vec = convert::lit_vecf32(search_vector),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "community_similarity_search")
            .await?;
        self.collect_scored_nodes(rows, min_score, limit, convert::community_node_from_props)
    }

    async fn get_embeddings_for_communities(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        debug!(count = uuids.len(), "falkor get_embeddings_for_communities");
        if uuids.is_empty() {
            return Ok(HashMap::new());
        }
        let cypher = format!(
            "MATCH (n:Community) WHERE n.uuid IN {} AND n.name_embedding IS NOT NULL \
             RETURN DISTINCT n.uuid, n.name_embedding",
            convert::lit_string_list(uuids)
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "get_embeddings_for_communities")
            .await?;
        embeddings_from_rows(rows)
    }

    async fn remove_communities(&self) -> Result<(), DriverError> {
        debug!("falkor remove_communities");
        // DETACH DELETE all Community nodes (removes incident HAS_MEMBER edges).
        // Always full-graph (no group scope) per upstream.
        self.run(
            "MATCH (c:Community) DETACH DELETE c",
            &HashMap::new(),
            "remove_communities",
        )
        .await
    }

    async fn get_community_clusters(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<GroupClusterProjection>, DriverError> {
        debug!(count = group_ids.len(), "falkor get_community_clusters");

        // Resolve the group set: explicit param, or all distinct entity group_ids
        // (mirrors Neo4j's `collect(DISTINCT n.group_id)`).
        let groups: Vec<String> = if group_ids.is_empty() {
            let rows = self
                .fetch_rows(
                    "MATCH (n:Entity) WHERE n.group_id IS NOT NULL RETURN DISTINCT n.group_id",
                    &HashMap::new(),
                    "distinct_entity_group_ids",
                )
                .await?;
            let mut out = Vec::with_capacity(rows.len());
            for row in &rows {
                let value = row.first().ok_or_else(|| {
                    DriverError::Decode("distinct_entity_group_ids: empty row".into())
                })?;
                out.push(convert::string_from_value(value)?);
            }
            out
        } else {
            group_ids.to_vec()
        };

        let mut out: Vec<GroupClusterProjection> = Vec::new();
        for group_id in groups {
            // Nodes in this group drive the per-node neighbour projection.
            let nodes = self
                .get_entity_nodes_by_group_ids(std::slice::from_ref(&group_id))
                .await?;
            let mut node_neighbors: Vec<NodeNeighbors> = Vec::with_capacity(nodes.len());
            for node in &nodes {
                // Undirected RELATES_TO neighbours within the same group, with the
                // count of edges to each (parity with the Neo4j cluster query
                // `(n)-[e:RELATES_TO]-(m) WITH count(e) AS count, m.uuid`).
                let cypher = format!(
                    "MATCH (n:Entity {{group_id: {gid}, uuid: {uuid}}})\
                     -[e:RELATES_TO]-(m:Entity {{group_id: {gid}}}) \
                     RETURN m.uuid, count(e)",
                    gid = convert::lit_str(&group_id),
                    uuid = convert::lit_str(&node.uuid),
                );
                let rows = self
                    .fetch_rows(&cypher, &HashMap::new(), "community_cluster_node_neighbors")
                    .await?;
                let mut neighbors: Vec<Neighbor> = Vec::with_capacity(rows.len());
                for row in &rows {
                    let node_uuid = convert::string_from_value(row.first().ok_or_else(|| {
                        DriverError::Decode("cluster neighbour row missing uuid".into())
                    })?)?;
                    let count = convert::i64_from_value(row.get(1).ok_or_else(|| {
                        DriverError::Decode("cluster neighbour row missing count".into())
                    })?)?;
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
        debug!(entity_uuid, "falkor community_of_member");
        // First community with a HAS_MEMBER edge to this entity (parity with Neo4j
        // COMMUNITY_OF_MEMBER).
        let cypher = format!(
            "MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {{uuid: {}}}) RETURN c LIMIT 1",
            convert::lit_str(entity_uuid)
        );
        self.fetch_one_node(
            &cypher,
            &HashMap::new(),
            "community_of_member",
            convert::community_node_from_props,
        )
        .await
    }

    async fn neighbor_communities(
        &self,
        entity_uuid: &str,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        debug!(entity_uuid, "falkor neighbor_communities");
        // ONE row per neighbour's community membership — NOT deduplicated; the
        // caller does the mode/plurality count (parity with Neo4j
        // NEIGHBOR_COMMUNITIES).
        let cypher = format!(
            "MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {{uuid: {}}}) \
             RETURN c",
            convert::lit_str(entity_uuid)
        );
        self.fetch_nodes(
            &cypher,
            &HashMap::new(),
            "neighbor_communities",
            convert::community_node_from_props,
        )
        .await
    }
}

// =====================================================================
// SagaOps — save + get_by_name/uuid + has/next episode edges (Task 1);
// previous-episode + contents queries Task 3.
// =====================================================================

#[async_trait]
impl SagaOps for FalkorDriver {
    async fn save_saga_node(&self, node: &SagaNode) -> Result<(), DriverError> {
        debug!(uuid = node.uuid, "falkor save_saga_node");
        let sets = [
            format!("n.name = {}", convert::lit_str(&node.name)),
            format!("n.group_id = {}", convert::lit_str(&node.group_id)),
            format!("n.summary = {}", convert::lit_str(&node.summary)),
            format!("n.created_at = {}", convert::lit_datetime(node.created_at)),
            format!(
                "n.first_episode_uuid = {}",
                convert::opt_lit(node.first_episode_uuid.as_deref().map(convert::lit_str))
            ),
            format!(
                "n.last_episode_uuid = {}",
                convert::opt_lit(node.last_episode_uuid.as_deref().map(convert::lit_str))
            ),
            format!(
                "n.last_summarized_at = {}",
                convert::opt_lit(node.last_summarized_at.map(convert::lit_datetime))
            ),
            format!(
                "n.last_summarized_episode_valid_at = {}",
                convert::opt_lit(
                    node.last_summarized_episode_valid_at
                        .map(convert::lit_datetime)
                )
            ),
        ];
        let cypher = format!(
            "MERGE (n:Saga {{uuid: {}}}) SET {}",
            convert::lit_str(&node.uuid),
            sets.join(", ")
        );
        self.run(&cypher, &HashMap::new(), "save_saga_node").await
    }

    async fn save_has_episode_edge(&self, edge: &HasEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "falkor save_has_episode_edge");
        let cypher = format!(
            "MATCH (s:Saga {{uuid: {}}}), (e:Episodic {{uuid: {}}}) \
             MERGE (s)-[r:HAS_EPISODE {{uuid: {}}}]->(e) \
             SET r.group_id = {}, r.created_at = {}",
            convert::lit_str(&edge.source_node_uuid),
            convert::lit_str(&edge.target_node_uuid),
            convert::lit_str(&edge.uuid),
            convert::lit_str(&edge.group_id),
            convert::lit_datetime(edge.created_at),
        );
        self.run(&cypher, &HashMap::new(), "save_has_episode_edge")
            .await
    }

    async fn save_next_episode_edge(&self, edge: &NextEpisodeEdge) -> Result<(), DriverError> {
        debug!(uuid = edge.uuid, "falkor save_next_episode_edge");
        let cypher = format!(
            "MATCH (a:Episodic {{uuid: {}}}), (b:Episodic {{uuid: {}}}) \
             MERGE (a)-[r:NEXT_EPISODE {{uuid: {}}}]->(b) \
             SET r.group_id = {}, r.created_at = {}",
            convert::lit_str(&edge.source_node_uuid),
            convert::lit_str(&edge.target_node_uuid),
            convert::lit_str(&edge.uuid),
            convert::lit_str(&edge.group_id),
            convert::lit_datetime(edge.created_at),
        );
        self.run(&cypher, &HashMap::new(), "save_next_episode_edge")
            .await
    }

    async fn get_saga_by_name(
        &self,
        name: &str,
        group_id: &str,
    ) -> Result<Option<SagaNode>, DriverError> {
        debug!(name, group_id, "falkor get_saga_by_name");
        let cypher = format!(
            "MATCH (s:Saga {{name: {}, group_id: {}}}) RETURN s",
            convert::lit_str(name),
            convert::lit_str(group_id)
        );
        self.fetch_one_node(
            &cypher,
            &HashMap::new(),
            "get_saga_by_name",
            convert::saga_node_from_props,
        )
        .await
    }

    async fn get_saga_by_uuid(&self, uuid: &str) -> Result<Option<SagaNode>, DriverError> {
        debug!(uuid, "falkor get_saga_by_uuid");
        let cypher = format!(
            "MATCH (s:Saga {{uuid: {}}}) RETURN s",
            convert::lit_str(uuid)
        );
        self.fetch_one_node(
            &cypher,
            &HashMap::new(),
            "get_saga_by_uuid",
            convert::saga_node_from_props,
        )
        .await
    }

    async fn saga_previous_episode_uuid(
        &self,
        saga_uuid: &str,
        current_episode_uuid: &str,
    ) -> Result<Option<String>, DriverError> {
        debug!(saga_uuid, "falkor saga_previous_episode_uuid");
        // Latest prior HAS_EPISODE episode by valid_at (epoch int) DESC, then
        // created_at DESC, excluding the current episode (parity with Neo4j).
        let cypher = format!(
            "MATCH (s:Saga {{uuid: {saga}}})-[:HAS_EPISODE]->(e:Episodic) \
             WHERE e.uuid <> {current} \
             RETURN e.uuid ORDER BY e.valid_at DESC, e.created_at DESC LIMIT 1",
            saga = convert::lit_str(saga_uuid),
            current = convert::lit_str(current_episode_uuid),
        );
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "saga_previous_episode_uuid")
            .await?;
        match rows.first().and_then(|r| r.first()) {
            Some(value) => Ok(Some(convert::string_from_value(value)?)),
            None => Ok(None),
        }
    }

    async fn saga_episode_contents(
        &self,
        saga_uuid: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError> {
        debug!(saga_uuid, limit, "falkor saga_episode_contents");
        // `since` filters `created_at > since` and returns chronological ASC; the
        // no-watermark path takes the latest `limit` (DESC) then reverses to
        // chronological order (parity with Neo4j). All temporal fields are epoch
        // millis so comparisons/orderings are integer comparisons.
        let (cypher, reverse) = match since {
            Some(s) => (
                format!(
                    "MATCH (s:Saga {{uuid: {saga}}})-[:HAS_EPISODE]->(e:Episodic) \
                     WHERE e.created_at > {since} \
                     RETURN e.content, e.valid_at \
                     ORDER BY e.valid_at ASC, e.created_at ASC LIMIT {lim}",
                    saga = convert::lit_str(saga_uuid),
                    since = convert::lit_datetime(s),
                    lim = convert::lit_int(limit as i64),
                ),
                false,
            ),
            None => (
                format!(
                    "MATCH (s:Saga {{uuid: {saga}}})-[:HAS_EPISODE]->(e:Episodic) \
                     RETURN e.content, e.valid_at \
                     ORDER BY e.valid_at DESC, e.created_at DESC LIMIT {lim}",
                    saga = convert::lit_str(saga_uuid),
                    lim = convert::lit_int(limit as i64),
                ),
                true,
            ),
        };
        let rows = self
            .fetch_rows(&cypher, &HashMap::new(), "saga_episode_contents")
            .await?;
        let mut out: Vec<(String, DateTime<Utc>)> = Vec::with_capacity(rows.len());
        for row in &rows {
            let content = convert::string_from_value(row.first().ok_or_else(|| {
                DriverError::Decode("saga episode-content row missing content".into())
            })?)?;
            let valid_at =
                convert::millis_to_datetime(convert::i64_from_value(row.get(1).ok_or_else(
                    || DriverError::Decode("saga episode-content row missing valid_at".into()),
                )?)?)?;
            out.push((content, valid_at));
        }
        if reverse {
            out.reverse();
        }
        Ok(out)
    }
}

// =====================================================================
// SchemaOps
// =====================================================================

#[async_trait]
impl SchemaOps for FalkorDriver {
    async fn build_indices_and_constraints(
        &self,
        delete_existing: bool,
    ) -> Result<(), DriverError> {
        debug!(delete_existing, "falkor build_indices_and_constraints");
        // FalkorDB has no IF NOT EXISTS for these forms; `delete_existing` is a
        // best-effort drop ignored on absence (full drop semantics land in
        // Task 3 alongside cascade deletes). For now, idempotent create covers
        // re-build correctness.
        let _ = delete_existing;

        for (label, prop) in schema::RANGE_INDICES {
            self.run_idempotent(&schema::range_index_ddl(label, prop), "range_index")
                .await?;
        }
        for (label, fields) in schema::NODE_FULLTEXT_INDICES {
            self.run_idempotent(&schema::node_fulltext_ddl(label, fields), "node_fulltext")
                .await?;
        }
        // Relationship fulltext (edge `fact`) — DDL form, not createNodeIndex.
        let (rel, prop) = schema::RELATIONSHIP_FULLTEXT;
        self.run_idempotent(
            &schema::relationship_fulltext_ddl(rel, prop),
            "relationship_fulltext",
        )
        .await?;

        // Vector indices. Entity / Community are node vectors; RELATES_TO is a
        // relationship vector.
        for (label, prop) in schema::VECTOR_INDICES {
            let ddl = if *label == "RELATES_TO" {
                schema::relationship_vector_index_ddl(label, prop, self.embedding_dim)
            } else {
                schema::node_vector_index_ddl(label, prop, self.embedding_dim)
            };
            self.run_idempotent(&ddl, "vector_index").await?;
        }
        Ok(())
    }
}

impl GraphDriver for FalkorDriver {
    fn provider(&self) -> &'static str {
        "falkordb"
    }
}
