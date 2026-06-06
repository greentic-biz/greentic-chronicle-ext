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
//! Task 1 (this commit) implements scaffold + schema + node/edge/episode/
//! community/saga **persistence** and `retrieve_episodes`. All search, bulk and
//! maintenance methods return a loud `DriverError::Query("phase-6 task-2/3
//! pending: ...")` rather than a silent empty success, so the conformance suite
//! fails honestly until Tasks 2/3 land.

mod convert;
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
    EpisodicEdgeOps, GraphDriver, GroupClusterProjection, SagaOps, SchemaOps, SearchOps,
};
use chronicle_core::search::filters::SearchFilters;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

/// Marker for not-yet-implemented (Task 2/3) operations. Loud, never a silent
/// empty success.
fn pending(method: &str) -> DriverError {
    DriverError::Query(format!("phase-6 task-2/3 pending: {method}"))
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
        _episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        Err(pending("get_mentioned_nodes"))
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
        _source_uuid: &str,
        _target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        Err(pending("get_edges_between_nodes"))
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
// BulkSaveOps — Task 3 (transactional save_all). Inherit default sequential
// for now; FalkorDB has no multi-statement atomic batch, so the eventual
// implementation will be documented best-effort. The default impl already does
// the four sequential saves correctly.
// =====================================================================

#[async_trait]
impl BulkSaveOps for FalkorDriver {}

// =====================================================================
// SearchOps — all Task 2.
// =====================================================================

#[async_trait]
impl SearchOps for FalkorDriver {
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
