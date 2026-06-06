use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use chronicle_core::driver::{
    CommunityOps, DriverError, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps,
    GraphDriver, GroupClusterProjection, Neighbor, NodeNeighbors, SagaOps, SchemaOps, SearchOps,
};
use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

// ---------------------------------------------------------------------------
// Inner state — all fields Default
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Inner {
    entity_nodes: HashMap<String, EntityNode>,
    entity_edges: HashMap<String, EntityEdge>,
    episodic_nodes: HashMap<String, EpisodicNode>,
    episodic_edges: Vec<EpisodicEdge>,
    community_nodes: HashMap<String, CommunityNode>,
    community_edges: Vec<CommunityEdge>,
    saga_nodes: HashMap<String, SagaNode>,
    has_episode_edges: Vec<HasEpisodeEdge>,
    next_episode_edges: Vec<NextEpisodeEdge>,
}

// ---------------------------------------------------------------------------
// FakeDriver
// ---------------------------------------------------------------------------

/// In-memory driver suitable for unit and integration tests.
/// All state is guarded by a single `std::sync::Mutex`; lock-poisoning
/// surfaces as `DriverError::Query("poisoned lock")`.
#[derive(Default)]
pub struct FakeDriver {
    inner: Mutex<Inner>,
}

impl FakeDriver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored [`EpisodicEdge`] entries.
    /// If the lock is poisoned, returns `0`.
    pub fn episodic_edge_count(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.episodic_edges.len())
            .unwrap_or(0)
    }

    /// Snapshot of all stored NEXT_EPISODE edges (test inspection).
    /// Returns an empty vec if the lock is poisoned.
    pub fn next_episode_edges(&self) -> Vec<NextEpisodeEdge> {
        self.inner
            .lock()
            .map(|g| g.next_episode_edges.clone())
            .unwrap_or_default()
    }

    /// Snapshot of all stored HAS_EPISODE edges (test inspection).
    /// Returns an empty vec if the lock is poisoned.
    pub fn has_episode_edges(&self) -> Vec<HasEpisodeEdge> {
        self.inner
            .lock()
            .map(|g| g.has_episode_edges.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Helper: cosine similarity (zero-norm → 0.0)
// ---------------------------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

macro_rules! lock {
    ($self:expr) => {
        $self
            .inner
            .lock()
            .map_err(|_| DriverError::Query("poisoned lock".into()))
    };
}

// ---------------------------------------------------------------------------
// SearchFilters honoring (in-memory replica of upstream WHERE builders)
//
// Ports `graphiti_core/search/search_filters.py` @ 34f56e65:
//   - edge_search_filter_query_constructor → `edge_matches_filters`
//   - node_search_filter_query_constructor → `node_matches_filters`
//   - the OUTER-OR / INNER-AND DateFilter group logic → `date_groups_match`
// ---------------------------------------------------------------------------

/// Evaluates a single [`DateFilter`] against an optional timestamp.
///
/// Mirrors upstream `date_filter_query_constructor` semantics for all eight
/// [`ComparisonOperator`] variants. Value operators (`Eq`/`Neq`/`Gt`/`Lt`/
/// `Gte`/`Lte`) require both `value` and the filter's `date` to be `Some`;
/// a missing value never satisfies a value comparison (Cypher `NULL <op> x`
/// yields NULL → not matched). `IsNull` / `IsNotNull` ignore the filter date.
fn date_condition_match(value: Option<DateTime<Utc>>, filter: &DateFilter) -> bool {
    match filter.comparison_operator {
        ComparisonOperator::IsNull => value.is_none(),
        ComparisonOperator::IsNotNull => value.is_some(),
        op => {
            let (Some(v), Some(d)) = (value, filter.date) else {
                return false;
            };
            match op {
                ComparisonOperator::Eq => v == d,
                ComparisonOperator::Neq => v != d,
                ComparisonOperator::Gt => v > d,
                ComparisonOperator::Lt => v < d,
                ComparisonOperator::Gte => v >= d,
                ComparisonOperator::Lte => v <= d,
                // IsNull / IsNotNull handled above; unreachable here.
                ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => unreachable!(),
            }
        }
    }
}

/// Evaluates a date-filter group set against a value.
///
/// `groups` is `Option<Vec<Vec<DateFilter>>>` where the outer vec is OR groups
/// and the inner vec is AND conditions (upstream `((c1 AND c2) OR (c3))`).
/// `None` → no constraint (always matches). An empty outer vec also matches
/// (upstream emits `()` which is vacuously true). An empty AND group matches
/// (no conditions to fail).
fn date_groups_match(value: Option<DateTime<Utc>>, groups: &Option<Vec<Vec<DateFilter>>>) -> bool {
    let Some(or_groups) = groups else {
        return true;
    };
    if or_groups.is_empty() {
        return true;
    }
    or_groups
        .iter()
        .any(|and_group| and_group.iter().all(|f| date_condition_match(value, f)))
}

/// True iff `node_labels` intersects `filter_labels` (upstream Neo4j `n:L1|L2`,
/// i.e. the node carries at least one of the requested labels). An empty
/// `filter_labels` matches everything (upstream emits no fragment).
fn labels_intersect(node_labels: &[String], filter_labels: &[String]) -> bool {
    if filter_labels.is_empty() {
        return true;
    }
    filter_labels.iter().any(|l| node_labels.contains(l))
}

/// Replicates upstream `edge_search_filter_query_constructor` (Neo4j variant).
///
/// All fragments are joined by AND:
///   - `edge_types`   → `e.name IN $edge_types`
///   - `edge_uuids`   → `e.uuid IN $edge_uuids`
///   - `node_labels`  → `n:L1|L2 AND m:L1|L2` (BOTH endpoints each carry ≥1 label)
///   - date groups    → OR-of-ANDs on `valid_at`/`invalid_at`/`created_at`/`expired_at`
///
/// `created_at` is non-optional on [`EntityEdge`]; the others are `Option`.
/// `node_lookup` resolves the edge's endpoint labels from the entity-node map;
/// an endpoint absent from the map is treated as having no labels (so a
/// `node_labels` filter excludes edges whose endpoints we cannot resolve —
/// faithful to a Cypher MATCH that would not bind a missing node).
fn edge_matches_filters(
    edge: &EntityEdge,
    filters: &SearchFilters,
    node_lookup: &HashMap<String, EntityNode>,
) -> bool {
    if let Some(edge_types) = &filters.edge_types
        && !edge_types.contains(&edge.name)
    {
        return false;
    }
    if let Some(edge_uuids) = &filters.edge_uuids
        && !edge_uuids.contains(&edge.uuid)
    {
        return false;
    }
    if let Some(node_labels) = &filters.node_labels {
        let src_labels = node_lookup
            .get(&edge.source_node_uuid)
            .map(|n| n.labels.as_slice())
            .unwrap_or(&[]);
        let tgt_labels = node_lookup
            .get(&edge.target_node_uuid)
            .map(|n| n.labels.as_slice())
            .unwrap_or(&[]);
        if !labels_intersect(src_labels, node_labels) || !labels_intersect(tgt_labels, node_labels)
        {
            return false;
        }
    }
    date_groups_match(edge.valid_at, &filters.valid_at)
        && date_groups_match(edge.invalid_at, &filters.invalid_at)
        && date_groups_match(Some(edge.created_at), &filters.created_at)
        && date_groups_match(edge.expired_at, &filters.expired_at)
}

/// Replicates upstream `node_search_filter_query_constructor` (Neo4j variant):
/// only `node_labels` applies (`n:L1|L2`). All other [`SearchFilters`] fields
/// are edge-scoped upstream and ignored for nodes.
fn node_matches_filters(node: &EntityNode, filters: &SearchFilters) -> bool {
    match &filters.node_labels {
        Some(labels) => labels_intersect(&node.labels, labels),
        None => true,
    }
}

// ---------------------------------------------------------------------------
// EntityNodeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EntityNodeOps for FakeDriver {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for n in nodes {
            g.entity_nodes.insert(n.uuid.clone(), n.clone());
        }
        Ok(())
    }

    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_nodes.get(uuid).cloned())
    }

    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        // Preserve input order; skip missing entries.
        Ok(uuids
            .iter()
            .filter_map(|id| g.entity_nodes.get(id).cloned())
            .collect())
    }

    /// Entity nodes whose `group_id` is in `group_ids` (empty → none). Mirrors
    /// upstream `EntityNode.get_by_group_ids`.
    async fn get_entity_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        Ok(g.entity_nodes
            .values()
            .filter(|n| group_ids.contains(&n.group_id))
            .cloned()
            .collect())
    }

    /// Entity nodes that are MENTIONS targets of any of `episode_uuids` (upstream
    /// `get_mentioned_nodes`). DISTINCT by node uuid; empty input → none.
    async fn get_mentioned_nodes(
        &self,
        episode_uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }
        let eps: std::collections::HashSet<&String> = episode_uuids.iter().collect();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<EntityNode> = Vec::new();
        for me in &g.episodic_edges {
            if eps.contains(&me.source_node_uuid)
                && let Some(node) = g.entity_nodes.get(&me.target_node_uuid)
                && seen.insert(node.uuid.clone())
            {
                out.push(node.clone());
            }
        }
        Ok(out)
    }

    /// Remove Entity nodes by uuid; also drops the RELATES_TO / MENTIONS /
    /// HAS_MEMBER edges that touch them (DETACH-DELETE semantics).
    async fn delete_entity_nodes_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        let set: std::collections::HashSet<&String> = uuids.iter().collect();
        g.entity_nodes.retain(|id, _| !set.contains(id));
        g.entity_edges.retain(|_, e| {
            !set.contains(&e.source_node_uuid) && !set.contains(&e.target_node_uuid)
        });
        g.episodic_edges
            .retain(|e| !set.contains(&e.target_node_uuid));
        g.community_edges
            .retain(|e| !set.contains(&e.source_node_uuid) && !set.contains(&e.target_node_uuid));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EntityEdgeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EntityEdgeOps for FakeDriver {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for e in edges {
            g.entity_edges.insert(e.uuid.clone(), e.clone());
        }
        Ok(())
    }

    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_edges.get(uuid).cloned())
    }

    /// Single-direction lookup: edges where `source_node_uuid == source` AND
    /// `target_node_uuid == target`. Mirrors upstream Neo4j directed MATCH.
    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_edges
            .values()
            .filter(|e| e.source_node_uuid == source_uuid && e.target_node_uuid == target_uuid)
            .cloned()
            .collect())
    }

    /// Entity edges by uuid (upstream `EntityEdge.get_by_uuids`). Preserves input
    /// order; skips missing uuids.
    async fn get_entity_edges_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        Ok(uuids
            .iter()
            .filter_map(|id| g.entity_edges.get(id).cloned())
            .collect())
    }

    /// Remove RELATES_TO edges by uuid (upstream `Edge.delete_by_uuids`). Empty
    /// input is a no-op.
    async fn delete_entity_edges_by_uuids(&self, uuids: &[String]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        let set: std::collections::HashSet<&String> = uuids.iter().collect();
        g.entity_edges.retain(|id, _| !set.contains(id));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EpisodeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EpisodeOps for FakeDriver {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        g.episodic_nodes
            .insert(episode.uuid.clone(), episode.clone());
        Ok(())
    }

    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        Ok(g.episodic_nodes.get(uuid).cloned())
    }

    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        Ok(uuids
            .iter()
            .filter_map(|id| g.episodic_nodes.get(id).cloned())
            .collect())
    }

    /// Mirrors upstream `retrieve_episodes`:
    /// 1. Filter: `valid_at <= reference_time`.
    /// 2. Filter: `group_id` in `group_ids` (only when `group_ids` is non-empty).
    /// 3. Filter: `source == source` (only when `source` is `Some`).
    /// 4. Sort by `valid_at` DESC (most-recent first).
    /// 5. Take `last_n`.
    /// 6. Reverse to chronological order (oldest first) before returning.
    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        let mut eligible: Vec<&EpisodicNode> = g
            .episodic_nodes
            .values()
            .filter(|ep| {
                ep.valid_at <= reference_time
                    && (group_ids.is_empty() || group_ids.contains(&ep.group_id))
                    && source.is_none_or(|s| ep.source == s)
            })
            .collect();

        // Sort DESC by valid_at (most-recent first).
        eligible.sort_by_key(|b| Reverse(b.valid_at));

        // Take last_n most-recent, then reverse to chronological.
        let mut result: Vec<EpisodicNode> = eligible.into_iter().take(last_n).cloned().collect();
        result.reverse();
        Ok(result)
    }

    /// Remove one episode by uuid; also drops MENTIONS / HAS_EPISODE /
    /// NEXT_EPISODE edges that touch it (DETACH-DELETE semantics). Missing uuid is
    /// a no-op.
    async fn delete_episode(&self, uuid: &str) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        g.episodic_nodes.remove(uuid);
        g.episodic_edges.retain(|e| e.source_node_uuid != uuid);
        g.has_episode_edges.retain(|e| e.target_node_uuid != uuid);
        g.next_episode_edges
            .retain(|e| e.source_node_uuid != uuid && e.target_node_uuid != uuid);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EpisodicEdgeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EpisodicEdgeOps for FakeDriver {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for e in edges {
            g.episodic_edges.push(e.clone());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SearchOps
// ---------------------------------------------------------------------------

#[async_trait]
impl SearchOps for FakeDriver {
    /// Approximation of BM25 fulltext: case-insensitive substring match of
    /// `query` against `fact` + `name` fields. Group-filtered when
    /// `group_ids` is non-empty, then `filters` applied via
    /// [`edge_matches_filters`]. Results truncated to `limit`.
    /// NOTE: Not a real BM25 ranking — only tests that need deterministic
    /// recall (not ranking precision) should use this.
    /// Result order within the limit is undefined (HashMap iteration); only assert recall, not ranking.
    async fn edge_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        let results: Vec<EntityEdge> = g
            .entity_edges
            .values()
            .filter(|e| {
                let text = format!("{} {}", e.fact, e.name).to_lowercase();
                text.contains(&q)
                    && (group_ids.is_empty() || group_ids.contains(&e.group_id))
                    && edge_matches_filters(e, filters, &g.entity_nodes)
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Brute-force cosine similarity against stored `fact_embedding`.
    /// Records without embeddings are skipped.
    /// Results filtered by `score > min_score` + `filters`, sorted DESC by
    /// score, truncated to `limit`.
    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        let mut scored: Vec<(f32, EntityEdge)> = g
            .entity_edges
            .values()
            .filter_map(|e| {
                let emb = e.fact_embedding.as_deref()?;
                let score = cosine(search_vector, emb);
                if score > min_score
                    && (group_ids.is_empty() || group_ids.contains(&e.group_id))
                    && edge_matches_filters(e, filters, &g.entity_nodes)
                {
                    Some((score, e.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.uuid.cmp(&b.1.uuid))
        });
        Ok(scored.into_iter().take(limit).map(|(_, e)| e).collect())
    }

    /// Approximation of BM25 fulltext: case-insensitive substring match of
    /// `query` against `name` + `summary` fields. Group-filtered when
    /// `group_ids` is non-empty, then `filters` applied via
    /// [`node_matches_filters`] (node_labels only). Results truncated to `limit`.
    /// Result order within the limit is undefined (HashMap iteration); only assert recall, not ranking.
    async fn node_fulltext_search(
        &self,
        query: &str,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        let results: Vec<EntityNode> = g
            .entity_nodes
            .values()
            .filter(|n| {
                let text = format!("{} {}", n.name, n.summary).to_lowercase();
                text.contains(&q)
                    && (group_ids.is_empty() || group_ids.contains(&n.group_id))
                    && node_matches_filters(n, filters)
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Brute-force cosine similarity against stored `name_embedding`.
    /// Records without embeddings are skipped.
    /// Results filtered by `score > min_score` + `filters` (node_labels), sorted
    /// DESC by score, truncated to `limit`.
    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        let mut scored: Vec<(f32, EntityNode)> = g
            .entity_nodes
            .values()
            .filter_map(|n| {
                let emb = n.name_embedding.as_deref()?;
                let score = cosine(search_vector, emb);
                if score > min_score
                    && (group_ids.is_empty() || group_ids.contains(&n.group_id))
                    && node_matches_filters(n, filters)
                {
                    Some((score, n.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.uuid.cmp(&b.1.uuid))
        });
        Ok(scored.into_iter().take(limit).map(|(_, n)| n).collect())
    }

    /// In-memory BFS over directed `RELATES_TO` (entity→entity) and `MENTIONS`
    /// (episode→entity) adjacency — plan R9 `node_bfs_search`.
    ///
    /// Walk semantics:
    ///   - Origins are label-free uuids (Entity or Episodic).
    ///   - Each hop follows an outgoing edge: RELATES_TO `source→target`,
    ///     MENTIONS `episode→entity`.
    ///   - Entity nodes reached at depth `1..=max_depth` are collected.
    ///   - Upstream `WHERE n.group_id = origin.group_id`: a reached entity is
    ///     kept only if its `group_id` equals the **origin's** group_id (origin
    ///     looked up across entity nodes, episodes, then derived from edges).
    ///   - When `group_ids` is non-empty, both the reached node's group_id and
    ///     the origin's group_id must be in the list.
    ///   - `filters` applied via [`node_matches_filters`] (node_labels).
    /// Returns empty when `origins` is empty or `max_depth < 1`.
    async fn node_bfs_search(
        &self,
        origins: &[String],
        filters: &SearchFilters,
        max_depth: usize,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        if origins.is_empty() || max_depth < 1 {
            return Ok(Vec::new());
        }
        let g = lock!(self)?;

        let mut collected: Vec<EntityNode> = Vec::new();
        let mut seen_results: std::collections::HashSet<String> = std::collections::HashSet::new();

        for origin in origins {
            let Some(origin_group) = origin_group_id(&g, origin) else {
                continue;
            };
            if !group_ids.is_empty() && !group_ids.contains(&origin_group) {
                continue;
            }
            // Directed BFS from this origin up to max_depth hops.
            let mut frontier: Vec<String> = vec![origin.clone()];
            let mut visited: std::collections::HashSet<String> =
                std::collections::HashSet::from([origin.clone()]);
            for _hop in 1..=max_depth {
                let mut next: Vec<String> = Vec::new();
                for current in &frontier {
                    for tgt in outgoing_targets(&g, current) {
                        if visited.insert(tgt.clone()) {
                            next.push(tgt.clone());
                        }
                        // A reached Entity node is a candidate result.
                        if let Some(node) = g.entity_nodes.get(&tgt) {
                            if node.group_id != origin_group {
                                continue;
                            }
                            if !group_ids.is_empty() && !group_ids.contains(&node.group_id) {
                                continue;
                            }
                            if !node_matches_filters(node, filters) {
                                continue;
                            }
                            if seen_results.insert(node.uuid.clone()) {
                                collected.push(node.clone());
                            }
                        }
                    }
                }
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
        }

        collected.truncate(limit);
        Ok(collected)
    }

    /// In-memory BFS returning the `RELATES_TO` edges traversed along paths —
    /// plan R9 `edge_bfs_search`.
    ///
    /// MENTIONS hops extend reach but only RELATES_TO edges are returned (mirrors
    /// upstream: `relationships(path)` includes MENTIONS rels, but the re-MATCH
    /// `(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]` only resolves RELATES_TO
    /// ones). Edges are DISTINCT by uuid, `filters` applied, truncated to `limit`.
    /// Returns empty when `origins` is empty or `max_depth < 1`.
    async fn edge_bfs_search(
        &self,
        origins: &[String],
        max_depth: usize,
        filters: &SearchFilters,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        if origins.is_empty() || max_depth < 1 {
            return Ok(Vec::new());
        }
        let g = lock!(self)?;

        let mut collected: Vec<EntityEdge> = Vec::new();
        let mut seen_edges: std::collections::HashSet<String> = std::collections::HashSet::new();

        for origin in origins {
            // Group gate mirrors node_bfs (origin must be within group_ids).
            if !group_ids.is_empty() {
                match origin_group_id(&g, origin) {
                    Some(og) if group_ids.contains(&og) => {}
                    _ => continue,
                }
            }
            let mut frontier: Vec<String> = vec![origin.clone()];
            let mut visited: std::collections::HashSet<String> =
                std::collections::HashSet::from([origin.clone()]);
            for _hop in 1..=max_depth {
                let mut next: Vec<String> = Vec::new();
                for current in &frontier {
                    // RELATES_TO edges traversed from `current` are returnable.
                    for e in g.entity_edges.values() {
                        if e.source_node_uuid != *current {
                            continue;
                        }
                        if !seen_edges.contains(&e.uuid)
                            && edge_matches_filters(e, filters, &g.entity_nodes)
                        {
                            seen_edges.insert(e.uuid.clone());
                            collected.push(e.clone());
                        }
                    }
                    // Extend reach via both RELATES_TO and MENTIONS targets.
                    for tgt in outgoing_targets(&g, current) {
                        if visited.insert(tgt.clone()) {
                            next.push(tgt);
                        }
                    }
                }
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
        }

        collected.truncate(limit);
        Ok(collected)
    }

    /// Approximation of episode BM25 fulltext (plan R5): case-insensitive
    /// substring match of `query` against `content` + `name` +
    /// `source_description`. Group-filtered when `group_ids` is non-empty,
    /// truncated to `limit`.
    /// NOTE: Not a real BM25 ranking — assert recall, not ranking precision.
    async fn episode_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        let results: Vec<EpisodicNode> = g
            .episodic_nodes
            .values()
            .filter(|ep| {
                let text =
                    format!("{} {} {}", ep.content, ep.name, ep.source_description).to_lowercase();
                text.contains(&q) && (group_ids.is_empty() || group_ids.contains(&ep.group_id))
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Returns `uuid → name_embedding` for the requested Entity nodes that have
    /// a stored embedding (plan R4). UUIDs without an embedding are omitted.
    async fn get_embeddings_for_nodes(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        let g = lock!(self)?;
        let mut map = HashMap::new();
        for id in uuids {
            if let Some(node) = g.entity_nodes.get(id)
                && let Some(emb) = &node.name_embedding
            {
                map.insert(id.clone(), emb.clone());
            }
        }
        Ok(map)
    }

    /// Returns `uuid → fact_embedding` for the requested Entity edges that have
    /// a stored embedding (plan R3). UUIDs without an embedding are omitted.
    async fn get_embeddings_for_edges(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        let g = lock!(self)?;
        let mut map = HashMap::new();
        for id in uuids {
            if let Some(edge) = g.entity_edges.get(id)
                && let Some(emb) = &edge.fact_embedding
            {
                map.insert(id.clone(), emb.clone());
            }
        }
        Ok(map)
    }

    /// 1-hop **undirected** `RELATES_TO` adjacency to `center_uuid` (plan R7):
    /// the subset of `node_uuids` that share a RELATES_TO edge with the center
    /// in either direction.
    async fn nodes_connected_to_center(
        &self,
        node_uuids: &[String],
        center_uuid: &str,
    ) -> Result<Vec<String>, DriverError> {
        let g = lock!(self)?;
        let candidates: std::collections::HashSet<&String> = node_uuids.iter().collect();
        let mut connected: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for e in g.entity_edges.values() {
            let other = if e.source_node_uuid == center_uuid {
                Some(&e.target_node_uuid)
            } else if e.target_node_uuid == center_uuid {
                Some(&e.source_node_uuid)
            } else {
                None
            };
            if let Some(other) = other
                && candidates.contains(other)
                && seen.insert(other.clone())
            {
                connected.push(other.clone());
            }
        }
        Ok(connected)
    }

    /// `MENTIONS` in-degree per Entity node uuid (plan R8): counts stored
    /// [`EpisodicEdge`] entries (episode→entity MENTIONS) whose target is each
    /// requested uuid. UUIDs with zero mentions are omitted from the map.
    async fn episode_mention_counts(
        &self,
        node_uuids: &[String],
    ) -> Result<HashMap<String, u64>, DriverError> {
        let g = lock!(self)?;
        let requested: std::collections::HashSet<&String> = node_uuids.iter().collect();
        let mut counts: HashMap<String, u64> = HashMap::new();
        for me in &g.episodic_edges {
            if requested.contains(&me.target_node_uuid) {
                *counts.entry(me.target_node_uuid.clone()).or_insert(0) += 1;
            }
        }
        Ok(counts)
    }
}

// ---------------------------------------------------------------------------
// BFS adjacency helpers (operate on a locked Inner)
// ---------------------------------------------------------------------------

/// Resolves the group_id of a BFS origin uuid. Checks entity nodes, then
/// episodes, then derives from an edge/episodic-edge with that source. Returns
/// `None` when the origin uuid cannot be resolved to any group.
fn origin_group_id(inner: &Inner, origin: &str) -> Option<String> {
    if let Some(n) = inner.entity_nodes.get(origin) {
        return Some(n.group_id.clone());
    }
    if let Some(ep) = inner.episodic_nodes.get(origin) {
        return Some(ep.group_id.clone());
    }
    if let Some(e) = inner
        .entity_edges
        .values()
        .find(|e| e.source_node_uuid == origin)
    {
        return Some(e.group_id.clone());
    }
    inner
        .episodic_edges
        .iter()
        .find(|e| e.source_node_uuid == origin)
        .map(|e| e.group_id.clone())
}

/// Outgoing directed neighbours of `current`: RELATES_TO targets where
/// `source == current`, plus MENTIONS targets where the episodic edge's
/// `source == current` (episode→entity).
fn outgoing_targets(inner: &Inner, current: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for e in inner.entity_edges.values() {
        if e.source_node_uuid == current {
            out.push(e.target_node_uuid.clone());
        }
    }
    for me in &inner.episodic_edges {
        if me.source_node_uuid == current {
            out.push(me.target_node_uuid.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CommunityOps
// ---------------------------------------------------------------------------

#[async_trait]
impl CommunityOps for FakeDriver {
    async fn save_community_nodes(&self, nodes: &[CommunityNode]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for n in nodes {
            g.community_nodes.insert(n.uuid.clone(), n.clone());
        }
        Ok(())
    }

    /// Upsert HAS_MEMBER edges. De-dups by edge uuid (mirrors the MERGE-by-uuid
    /// upstream save), replacing an existing edge with the same uuid.
    async fn save_community_edges(&self, edges: &[CommunityEdge]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for e in edges {
            if let Some(slot) = g.community_edges.iter_mut().find(|x| x.uuid == e.uuid) {
                *slot = e.clone();
            } else {
                g.community_edges.push(e.clone());
            }
        }
        Ok(())
    }

    async fn get_community_nodes_by_group_ids(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        let g = lock!(self)?;
        if group_ids.is_empty() {
            return Ok(Vec::new());
        }
        Ok(g.community_nodes
            .values()
            .filter(|c| group_ids.contains(&c.group_id))
            .cloned()
            .collect())
    }

    async fn get_community_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<CommunityNode>, DriverError> {
        let g = lock!(self)?;
        Ok(uuids
            .iter()
            .filter_map(|id| g.community_nodes.get(id).cloned())
            .collect())
    }

    /// Brute-force fulltext: case-insensitive substring of `query` against the
    /// community `name`. Group-filtered when `group_ids` is non-empty; truncated
    /// to `limit`. Not a real BM25 ranking — assert recall, not ranking precision.
    async fn community_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        Ok(g.community_nodes
            .values()
            .filter(|c| {
                c.name.to_lowercase().contains(&q)
                    && (group_ids.is_empty() || group_ids.contains(&c.group_id))
            })
            .take(limit)
            .cloned()
            .collect())
    }

    /// Brute-force cosine similarity against stored `name_embedding`. Communities
    /// without an embedding are skipped. Filtered by `score > min_score` + group;
    /// sorted DESC by score (uuid tie-break); truncated to `limit`.
    async fn community_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        let g = lock!(self)?;
        let mut scored: Vec<(f32, CommunityNode)> = g
            .community_nodes
            .values()
            .filter_map(|c| {
                let emb = c.name_embedding.as_deref()?;
                let score = cosine(search_vector, emb);
                if score > min_score && (group_ids.is_empty() || group_ids.contains(&c.group_id)) {
                    Some((score, c.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.uuid.cmp(&b.1.uuid))
        });
        Ok(scored.into_iter().take(limit).map(|(_, c)| c).collect())
    }

    async fn get_embeddings_for_communities(
        &self,
        uuids: &[String],
    ) -> Result<HashMap<String, Vec<f32>>, DriverError> {
        let g = lock!(self)?;
        let mut map = HashMap::new();
        for id in uuids {
            if let Some(c) = g.community_nodes.get(id)
                && let Some(emb) = &c.name_embedding
            {
                map.insert(id.clone(), emb.clone());
            }
        }
        Ok(map)
    }

    /// Drop ALL community nodes + their HAS_MEMBER edges (upstream
    /// `remove_communities`: `MATCH (c:Community) DETACH DELETE c`).
    async fn remove_communities(&self) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        g.community_nodes.clear();
        g.community_edges.clear();
        Ok(())
    }

    /// Per-group RELATES_TO adjacency projection (upstream `get_community_clusters`
    /// / R2). For each group (or all distinct entity group_ids when `group_ids` is
    /// empty), each node's neighbour list is the UNDIRECTED RELATES_TO adjacency
    /// (`-[e:RELATES_TO]-`) restricted to the SAME group, with the edge count per
    /// neighbour uuid.
    async fn get_community_clusters(
        &self,
        group_ids: &[String],
    ) -> Result<Vec<GroupClusterProjection>, DriverError> {
        let g = lock!(self)?;

        // Resolve the group set: explicit, or all distinct entity group_ids.
        let groups: Vec<String> = if group_ids.is_empty() {
            let mut set: Vec<String> = g
                .entity_nodes
                .values()
                .map(|n| n.group_id.clone())
                .collect();
            set.sort();
            set.dedup();
            set
        } else {
            group_ids.to_vec()
        };

        let mut out: Vec<GroupClusterProjection> = Vec::new();
        for group_id in groups {
            // Nodes in this group, sorted by uuid for deterministic ordering.
            let mut nodes: Vec<&EntityNode> = g
                .entity_nodes
                .values()
                .filter(|n| n.group_id == group_id)
                .collect();
            nodes.sort_by(|a, b| a.uuid.cmp(&b.uuid));

            let mut node_neighbors: Vec<NodeNeighbors> = Vec::new();
            for node in nodes {
                // Count RELATES_TO edges to each same-group neighbour (undirected).
                let mut counts: HashMap<String, u64> = HashMap::new();
                for e in g.entity_edges.values() {
                    let other = if e.source_node_uuid == node.uuid {
                        Some(&e.target_node_uuid)
                    } else if e.target_node_uuid == node.uuid {
                        Some(&e.source_node_uuid)
                    } else {
                        None
                    };
                    if let Some(other) = other
                        && let Some(m) = g.entity_nodes.get(other)
                        && m.group_id == group_id
                    {
                        *counts.entry(other.clone()).or_insert(0) += 1;
                    }
                }
                let mut neighbors: Vec<Neighbor> = counts
                    .into_iter()
                    .map(|(node_uuid, edge_count)| Neighbor {
                        node_uuid,
                        edge_count,
                    })
                    .collect();
                neighbors.sort_by(|a, b| a.node_uuid.cmp(&b.node_uuid));
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

    /// Already-member lookup (R4 step a): the first community with a HAS_MEMBER
    /// edge targeting `entity_uuid`.
    async fn community_of_member(
        &self,
        entity_uuid: &str,
    ) -> Result<Option<CommunityNode>, DriverError> {
        let g = lock!(self)?;
        for e in &g.community_edges {
            if e.target_node_uuid == entity_uuid
                && let Some(c) = g.community_nodes.get(&e.source_node_uuid)
            {
                return Ok(Some(c.clone()));
            }
        }
        Ok(None)
    }

    /// Neighbour-vote lookup (R4 step b): ONE community row per RELATES_TO
    /// neighbour of `entity_uuid` that is a HAS_MEMBER of some community. NOT
    /// deduplicated — duplicates carry the plurality the caller mode-counts.
    async fn neighbor_communities(
        &self,
        entity_uuid: &str,
    ) -> Result<Vec<CommunityNode>, DriverError> {
        let g = lock!(self)?;

        // RELATES_TO neighbours of the entity (undirected).
        let mut neighbors: Vec<String> = Vec::new();
        for e in g.entity_edges.values() {
            if e.source_node_uuid == entity_uuid {
                neighbors.push(e.target_node_uuid.clone());
            } else if e.target_node_uuid == entity_uuid {
                neighbors.push(e.source_node_uuid.clone());
            }
        }

        // For each (neighbour, community-membership) pair emit a community row.
        let mut out: Vec<CommunityNode> = Vec::new();
        for neighbor in &neighbors {
            for ce in &g.community_edges {
                if &ce.target_node_uuid == neighbor
                    && let Some(c) = g.community_nodes.get(&ce.source_node_uuid)
                {
                    out.push(c.clone());
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SagaOps
// ---------------------------------------------------------------------------

#[async_trait]
impl SagaOps for FakeDriver {
    async fn save_saga_node(&self, node: &SagaNode) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        g.saga_nodes.insert(node.uuid.clone(), node.clone());
        Ok(())
    }

    async fn save_has_episode_edge(&self, edge: &HasEpisodeEdge) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        if let Some(slot) = g.has_episode_edges.iter_mut().find(|x| x.uuid == edge.uuid) {
            *slot = edge.clone();
        } else {
            g.has_episode_edges.push(edge.clone());
        }
        Ok(())
    }

    async fn save_next_episode_edge(&self, edge: &NextEpisodeEdge) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        if let Some(slot) = g
            .next_episode_edges
            .iter_mut()
            .find(|x| x.uuid == edge.uuid)
        {
            *slot = edge.clone();
        } else {
            g.next_episode_edges.push(edge.clone());
        }
        Ok(())
    }

    /// Get-or-create lookup (R9): the first saga matching `(name, group_id)`.
    async fn get_saga_by_name(
        &self,
        name: &str,
        group_id: &str,
    ) -> Result<Option<SagaNode>, DriverError> {
        let g = lock!(self)?;
        Ok(g.saga_nodes
            .values()
            .find(|s| s.name == name && s.group_id == group_id)
            .cloned())
    }

    /// Most-recent prior episode in a saga (R9): of the saga's HAS_EPISODE
    /// episodes (excluding `current_episode_uuid`), the one with the greatest
    /// `valid_at` (then `created_at`) — mirrors `ORDER BY valid_at DESC,
    /// created_at DESC LIMIT 1`.
    async fn saga_previous_episode_uuid(
        &self,
        saga_uuid: &str,
        current_episode_uuid: &str,
    ) -> Result<Option<String>, DriverError> {
        let g = lock!(self)?;
        let mut candidates: Vec<&EpisodicNode> = g
            .has_episode_edges
            .iter()
            .filter(|e| {
                e.source_node_uuid == saga_uuid && e.target_node_uuid != current_episode_uuid
            })
            .filter_map(|e| g.episodic_nodes.get(&e.target_node_uuid))
            .collect();
        // DESC by (valid_at, created_at); pick the first.
        candidates.sort_by(|a, b| {
            b.valid_at
                .cmp(&a.valid_at)
                .then_with(|| b.created_at.cmp(&a.created_at))
        });
        Ok(candidates.first().map(|e| e.uuid.clone()))
    }

    /// `(content, valid_at)` per saga episode for summarization (R9). When `since`
    /// is `Some`, filters `created_at > since`; returns chronological order
    /// (`valid_at ASC, created_at ASC`); truncated to `limit`.
    async fn saga_episode_contents(
        &self,
        saga_uuid: &str,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>, DriverError> {
        let g = lock!(self)?;
        let mut eps: Vec<&EpisodicNode> = g
            .has_episode_edges
            .iter()
            .filter(|e| e.source_node_uuid == saga_uuid)
            .filter_map(|e| g.episodic_nodes.get(&e.target_node_uuid))
            .filter(|ep| since.is_none_or(|s| ep.created_at > s))
            .collect();
        eps.sort_by(|a, b| {
            a.valid_at
                .cmp(&b.valid_at)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        Ok(eps
            .into_iter()
            .take(limit)
            .map(|ep| (ep.content.clone(), ep.valid_at))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// SchemaOps
// ---------------------------------------------------------------------------

#[async_trait]
impl SchemaOps for FakeDriver {
    async fn build_indices_and_constraints(
        &self,
        _delete_existing: bool,
    ) -> Result<(), DriverError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphDriver (supertrait)
// ---------------------------------------------------------------------------

impl GraphDriver for FakeDriver {
    fn provider(&self) -> &'static str {
        "fake"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_embedder::MockEmbedder;
    use chronicle_core::embedder::EmbedderClient;
    use chrono::{TimeZone, Utc};

    fn make_node(uuid: &str, name: &str, group_id: &str) -> EntityNode {
        let mut n = EntityNode::new(name.into(), group_id.into(), Utc::now());
        n.uuid = uuid.into();
        n
    }

    fn make_edge(
        uuid: &str,
        src: &str,
        tgt: &str,
        name: &str,
        fact: &str,
        group_id: &str,
    ) -> EntityEdge {
        let mut e = EntityEdge::new(
            src.into(),
            tgt.into(),
            name.into(),
            fact.into(),
            group_id.into(),
        );
        e.uuid = uuid.into();
        e
    }

    fn make_episode(
        uuid: &str,
        group_id: &str,
        source: EpisodeType,
        valid_at: DateTime<Utc>,
    ) -> EpisodicNode {
        let mut ep = EpisodicNode::new(
            "ep".into(),
            group_id.into(),
            source,
            "desc".into(),
            "content".into(),
            Utc::now(),
            valid_at,
        );
        ep.uuid = uuid.into();
        ep
    }

    // ------------------------------------------------------------------
    // EntityNode roundtrip
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_entity_node_roundtrip() {
        let driver = FakeDriver::new();
        let node = make_node("n1", "Alice", "g1");
        driver
            .save_entity_nodes(std::slice::from_ref(&node))
            .await
            .unwrap();
        let got = driver.get_entity_node("n1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "n1");
        assert_eq!(got.name, "Alice");
        assert_eq!(got.group_id, "g1");
    }

    #[tokio::test]
    async fn get_entity_nodes_by_uuids_preserves_order_skips_missing() {
        let driver = FakeDriver::new();
        let a = make_node("a", "A", "g1");
        let b = make_node("b", "B", "g1");
        driver.save_entity_nodes(&[a, b]).await.unwrap();
        let got = driver
            .get_entity_nodes_by_uuids(&["b".to_string(), "a".to_string(), "missing".to_string()])
            .await
            .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].uuid, "b");
        assert_eq!(got[1].uuid, "a");
    }

    // ------------------------------------------------------------------
    // EntityEdge roundtrip + directionality
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_entity_edge_roundtrip() {
        let driver = FakeDriver::new();
        let edge = make_edge("e1", "u1", "u2", "WORKS_AT", "Alice works at Acme", "g1");
        driver
            .save_entity_edges(std::slice::from_ref(&edge))
            .await
            .unwrap();
        let got = driver.get_entity_edge("e1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "e1");
        assert_eq!(got.fact, "Alice works at Acme");
    }

    #[tokio::test]
    async fn get_edges_between_nodes_is_directional() {
        let driver = FakeDriver::new();
        let edge = make_edge("e1", "a", "b", "REL", "a to b", "g1");
        driver.save_entity_edges(&[edge]).await.unwrap();

        // Forward direction: found
        let fwd = driver.get_edges_between_nodes("a", "b").await.unwrap();
        assert_eq!(fwd.len(), 1, "forward direction should find the edge");

        // Reverse direction: NOT found
        let rev = driver.get_edges_between_nodes("b", "a").await.unwrap();
        assert!(rev.is_empty(), "reverse direction must return nothing");
    }

    // ------------------------------------------------------------------
    // EpisodicNode roundtrip
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_episode_roundtrip() {
        let driver = FakeDriver::new();
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        let got = driver.get_episode("ep1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "ep1");
        assert_eq!(got.group_id, "g1");
    }

    // ------------------------------------------------------------------
    // retrieve_episodes: ordering, cutoff, last_n
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn retrieve_episodes_chronological_with_cutoff_and_last_n() {
        let driver = FakeDriver::new();
        let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2026, 1, 3, 0, 0, 0).unwrap();
        let future = Utc.with_ymd_and_hms(2026, 1, 10, 0, 0, 0).unwrap();

        // Insert three in order and one in the future
        driver
            .save_episode(&make_episode("ep-t3", "g1", EpisodeType::Message, t3))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode("ep-t1", "g1", EpisodeType::Message, t1))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode("ep-t2", "g1", EpisodeType::Message, t2))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode(
                "ep-future",
                "g1",
                EpisodeType::Message,
                future,
            ))
            .await
            .unwrap();

        // Reference time = t3 (future episode excluded), last_n=2
        let result = driver
            .retrieve_episodes(t3, 2, &["g1".to_string()], None)
            .await
            .unwrap();

        assert_eq!(result.len(), 2, "should return exactly 2 episodes");
        // Chronological order: t2 then t3 (the 2 most-recent of the 3 eligible)
        assert_eq!(result[0].uuid, "ep-t2");
        assert_eq!(result[1].uuid, "ep-t3");
    }

    // ------------------------------------------------------------------
    // Similarity search: alpha ranks first
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn edge_similarity_search_ranks_matching_string_first() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut edge_alpha = make_edge("e-alpha", "u1", "u2", "REL", "alpha", "g1");
        edge_alpha.fact_embedding = Some(emb.create("alpha").await.unwrap());

        let mut edge_beta = make_edge("e-beta", "u3", "u4", "REL", "beta", "g1");
        edge_beta.fact_embedding = Some(emb.create("beta").await.unwrap());

        driver
            .save_entity_edges(&[edge_alpha, edge_beta])
            .await
            .unwrap();

        let query_vec = emb.create("alpha").await.unwrap();
        let results = driver
            .edge_similarity_search(&query_vec, &SearchFilters::default(), &[], 10, 0.0)
            .await
            .unwrap();

        assert!(!results.is_empty(), "should return at least one result");
        assert_eq!(results[0].fact, "alpha", "alpha edge should rank first");

        // Verify score ~ 1.0 for exact match
        let score = cosine(&query_vec, &emb.create("alpha").await.unwrap());
        assert!((score - 1.0).abs() < 1e-5, "cosine score should be ~1.0");
    }

    // ------------------------------------------------------------------
    // Fulltext: substring + group filter
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn edge_fulltext_search_substring_and_group_filter() {
        let driver = FakeDriver::new();
        let e1 = make_edge("e1", "u1", "u2", "REL", "Alice works at Acme", "grp-a");
        let e2 = make_edge("e2", "u3", "u4", "REL", "Bob works at Beta Inc", "grp-b");
        driver.save_entity_edges(&[e1, e2]).await.unwrap();

        // Substring match without group filter — both match "works"
        let all = driver
            .edge_fulltext_search("works", &SearchFilters::default(), &[], 10)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // Group filter restricts to grp-a only
        let filtered = driver
            .edge_fulltext_search(
                "works",
                &SearchFilters::default(),
                &["grp-a".to_string()],
                10,
            )
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].uuid, "e1");
    }

    #[tokio::test]
    async fn node_fulltext_search_substring_match() {
        let driver = FakeDriver::new();
        let mut n1 = make_node("n1", "Alice Smith", "g1");
        n1.summary = "engineer at Acme".into();
        let n2 = make_node("n2", "Bob Jones", "g1");
        driver.save_entity_nodes(&[n1, n2]).await.unwrap();

        let results = driver
            .node_fulltext_search("alice", &SearchFilters::default(), &[], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uuid, "n1");
    }

    // ------------------------------------------------------------------
    // EpisodicEdge save
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_episodic_edges_accumulates() {
        let driver = FakeDriver::new();
        let e1 = EpisodicEdge::new("ep1".into(), "n1".into(), "g1".into(), Utc::now());
        let e2 = EpisodicEdge::new("ep2".into(), "n2".into(), "g1".into(), Utc::now());
        driver.save_episodic_edges(&[e1]).await.unwrap();
        driver.save_episodic_edges(&[e2]).await.unwrap();
        assert_eq!(driver.episodic_edge_count(), 2);
    }

    // ------------------------------------------------------------------
    // Node similarity search: ranking + skip-no-embedding
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn node_similarity_search_ranks_matching_string_first() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut node_alpha = make_node("n-alpha", "alpha", "g1");
        node_alpha.name_embedding = Some(emb.create("alpha").await.unwrap());

        let mut node_beta = make_node("n-beta", "beta", "g1");
        node_beta.name_embedding = Some(emb.create("beta").await.unwrap());

        // node without embedding — must be skipped, not panic
        let node_no_emb = make_node("n-no-emb", "gamma", "g1");

        driver
            .save_entity_nodes(&[node_alpha, node_beta, node_no_emb])
            .await
            .unwrap();

        let query_vec = emb.create("alpha").await.unwrap();
        let results = driver
            .node_similarity_search(&query_vec, &SearchFilters::default(), &[], 10, 0.0)
            .await
            .unwrap();

        // node without embedding must not appear in results
        assert!(
            results.iter().all(|n| n.uuid != "n-no-emb"),
            "node without embedding must be skipped"
        );
        assert!(!results.is_empty(), "should return at least one result");
        assert_eq!(results[0].name, "alpha", "alpha node should rank first");

        // Verify score ~ 1.0 for exact match
        let score = cosine(&query_vec, &emb.create("alpha").await.unwrap());
        assert!((score - 1.0).abs() < 1e-5, "cosine score should be ~1.0");
    }

    // ------------------------------------------------------------------
    // retrieve_episodes: empty group_ids returns all groups
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn retrieve_episodes_empty_group_ids_returns_all_groups() {
        let driver = FakeDriver::new();
        let now = Utc::now();
        let ep_g1 = make_episode("ep-g1", "g1", EpisodeType::Message, now);
        let ep_g2 = make_episode("ep-g2", "g2", EpisodeType::Message, now);
        driver.save_episode(&ep_g1).await.unwrap();
        driver.save_episode(&ep_g2).await.unwrap();

        // Empty group_ids slice — must return episodes from ALL groups
        let results = driver.retrieve_episodes(now, 100, &[], None).await.unwrap();

        let uuids: Vec<&str> = results.iter().map(|ep| ep.uuid.as_str()).collect();
        assert!(
            uuids.contains(&"ep-g1"),
            "ep-g1 (group g1) should be returned"
        );
        assert!(
            uuids.contains(&"ep-g2"),
            "ep-g2 (group g2) should be returned"
        );
    }

    // ------------------------------------------------------------------
    // Schema ops no-op
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn schema_build_is_noop() {
        let driver = FakeDriver::new();
        driver.build_indices_and_constraints(true).await.unwrap();
    }

    // ------------------------------------------------------------------
    // Provider name
    // ------------------------------------------------------------------
    #[test]
    fn provider_name_is_fake() {
        let driver = FakeDriver::new();
        assert_eq!(driver.provider(), "fake");
    }

    // ==================================================================
    // Phase-2: SearchFilters semantics
    // ==================================================================

    fn date_filter(op: ComparisonOperator, date: Option<DateTime<Utc>>) -> DateFilter {
        DateFilter {
            date,
            comparison_operator: op,
        }
    }

    // ── edge_types: e.name IN $edge_types ──────────────────────────────
    #[tokio::test]
    async fn edge_filter_edge_types_restricts_by_name() {
        let driver = FakeDriver::new();
        let works = make_edge("e1", "a", "b", "WORKS_AT", "x works", "g1");
        let lives = make_edge("e2", "a", "c", "LIVES_IN", "x lives", "g1");
        driver.save_entity_edges(&[works, lives]).await.unwrap();

        let filters = SearchFilters {
            edge_types: Some(vec!["WORKS_AT".to_string()]),
            ..Default::default()
        };
        let hits = driver
            .edge_fulltext_search("x", &filters, &[], 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].uuid, "e1");
    }

    // ── edge_uuids whitelist ───────────────────────────────────────────
    #[tokio::test]
    async fn edge_filter_edge_uuids_whitelist() {
        let driver = FakeDriver::new();
        let e1 = make_edge("e1", "a", "b", "REL", "match me", "g1");
        let e2 = make_edge("e2", "a", "c", "REL", "match me too", "g1");
        let e3 = make_edge("e3", "a", "d", "REL", "match me three", "g1");
        driver.save_entity_edges(&[e1, e2, e3]).await.unwrap();

        let filters = SearchFilters {
            edge_uuids: Some(vec!["e1".to_string(), "e3".to_string()]),
            ..Default::default()
        };
        let hits = driver
            .edge_fulltext_search("match", &filters, &[], 10)
            .await
            .unwrap();
        let mut uuids: Vec<String> = hits.iter().map(|e| e.uuid.clone()).collect();
        uuids.sort();
        assert_eq!(uuids, vec!["e1".to_string(), "e3".to_string()]);
    }

    // ── date OR-of-ANDs window incl. IsNull ────────────────────────────
    #[tokio::test]
    async fn edge_filter_date_or_of_ands_window_and_is_null() {
        let driver = FakeDriver::new();
        let t0 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2026, 1, 5, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 1, 10, 0, 0, 0).unwrap();

        // inside [t0, t2): valid_at = t1
        let mut inside = make_edge("inside", "a", "b", "REL", "f", "g1");
        inside.valid_at = Some(t1);
        // outside window: valid_at far future
        let mut outside = make_edge("outside", "a", "c", "REL", "f", "g1");
        outside.valid_at = Some(Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap());
        // null valid_at
        let null_edge = make_edge("null", "a", "d", "REL", "f", "g1");

        driver
            .save_entity_edges(&[inside, outside, null_edge])
            .await
            .unwrap();

        // ((valid_at >= t0 AND valid_at < t2) OR valid_at IS NULL)
        let filters = SearchFilters {
            valid_at: Some(vec![
                vec![
                    date_filter(ComparisonOperator::Gte, Some(t0)),
                    date_filter(ComparisonOperator::Lt, Some(t2)),
                ],
                vec![date_filter(ComparisonOperator::IsNull, None)],
            ]),
            ..Default::default()
        };
        let hits = driver
            .edge_fulltext_search("f", &filters, &[], 10)
            .await
            .unwrap();
        let mut uuids: Vec<String> = hits.iter().map(|e| e.uuid.clone()).collect();
        uuids.sort();
        assert_eq!(uuids, vec!["inside".to_string(), "null".to_string()]);
    }

    // ── node_labels: BOTH endpoints must carry a requested label ───────
    #[tokio::test]
    async fn edge_filter_node_labels_both_endpoints() {
        let driver = FakeDriver::new();
        let mut person = make_node("p", "Alice", "g1");
        person.labels = vec!["Entity".into(), "Person".into()];
        let mut company = make_node("c", "Acme", "g1");
        company.labels = vec!["Entity".into(), "Company".into()];
        let mut other = make_node("o", "Other", "g1");
        other.labels = vec!["Entity".into()];
        driver
            .save_entity_nodes(&[person, company, other])
            .await
            .unwrap();

        // both endpoints labelled (Person/Company both share "Entity")
        let e_both = make_edge("e-both", "p", "c", "REL", "fact", "g1");
        // target endpoint lacks "Person"|"Company"
        let e_one = make_edge("e-one", "p", "o", "REL", "fact", "g1");
        driver.save_entity_edges(&[e_both, e_one]).await.unwrap();

        // require both endpoints have at least one of {Person, Company}
        let filters = SearchFilters {
            node_labels: Some(vec!["Person".to_string(), "Company".to_string()]),
            ..Default::default()
        };
        let hits = driver
            .edge_fulltext_search("fact", &filters, &[], 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "only the edge with both endpoints labelled");
        assert_eq!(hits[0].uuid, "e-both");
    }

    // ── node search honors node_labels ─────────────────────────────────
    #[tokio::test]
    async fn node_filter_node_labels() {
        let driver = FakeDriver::new();
        let mut person = make_node("p", "Alice", "g1");
        person.labels = vec!["Entity".into(), "Person".into()];
        let plain = make_node("q", "Alice Two", "g1");
        driver.save_entity_nodes(&[person, plain]).await.unwrap();

        let filters = SearchFilters {
            node_labels: Some(vec!["Person".to_string()]),
            ..Default::default()
        };
        let hits = driver
            .node_fulltext_search("alice", &filters, &[], 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].uuid, "p");
    }

    // ==================================================================
    // Phase-2: BFS
    // ==================================================================

    /// Build a chain a -RELATES_TO-> b -RELATES_TO-> c (all group g1).
    async fn seed_chain(driver: &FakeDriver) {
        let a = make_node("a", "A", "g1");
        let b = make_node("b", "B", "g1");
        let c = make_node("c", "C", "g1");
        driver.save_entity_nodes(&[a, b, c]).await.unwrap();
        let ab = make_edge("ab", "a", "b", "REL", "a-b", "g1");
        let bc = make_edge("bc", "b", "c", "REL", "b-c", "g1");
        driver.save_entity_edges(&[ab, bc]).await.unwrap();
    }

    #[tokio::test]
    async fn node_bfs_depth_1_vs_depth_2() {
        let driver = FakeDriver::new();
        seed_chain(&driver).await;

        let depth1 = driver
            .node_bfs_search(&["a".to_string()], &SearchFilters::default(), 1, &[], 100)
            .await
            .unwrap();
        let d1: Vec<String> = depth1.iter().map(|n| n.uuid.clone()).collect();
        assert_eq!(d1, vec!["b".to_string()], "depth 1 reaches only b");

        let depth2 = driver
            .node_bfs_search(&["a".to_string()], &SearchFilters::default(), 2, &[], 100)
            .await
            .unwrap();
        let mut d2: Vec<String> = depth2.iter().map(|n| n.uuid.clone()).collect();
        d2.sort();
        assert_eq!(
            d2,
            vec!["b".to_string(), "c".to_string()],
            "depth 2 reaches b and c"
        );
    }

    #[tokio::test]
    async fn node_bfs_empty_origins_or_zero_depth_returns_empty() {
        let driver = FakeDriver::new();
        seed_chain(&driver).await;

        let no_origin = driver
            .node_bfs_search(&[], &SearchFilters::default(), 3, &[], 100)
            .await
            .unwrap();
        assert!(no_origin.is_empty());

        let zero_depth = driver
            .node_bfs_search(&["a".to_string()], &SearchFilters::default(), 0, &[], 100)
            .await
            .unwrap();
        assert!(zero_depth.is_empty());
    }

    #[tokio::test]
    async fn node_bfs_via_mentions_origin_reaches_entities() {
        let driver = FakeDriver::new();
        // entity b, c
        let b = make_node("b", "B", "g1");
        let c = make_node("c", "C", "g1");
        driver.save_entity_nodes(&[b, c]).await.unwrap();
        // episode ep1 MENTIONS b ; b RELATES_TO c
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        let mention = EpisodicEdge::new("ep1".into(), "b".into(), "g1".into(), Utc::now());
        driver.save_episodic_edges(&[mention]).await.unwrap();
        let bc = make_edge("bc", "b", "c", "REL", "b-c", "g1");
        driver.save_entity_edges(&[bc]).await.unwrap();

        // origin = episode uuid; depth 2 reaches b (hop 1 via MENTIONS) and c (hop 2)
        let hits = driver
            .node_bfs_search(&["ep1".to_string()], &SearchFilters::default(), 2, &[], 100)
            .await
            .unwrap();
        let mut uuids: Vec<String> = hits.iter().map(|n| n.uuid.clone()).collect();
        uuids.sort();
        assert_eq!(uuids, vec!["b".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn edge_bfs_returns_only_relates_to_edges() {
        let driver = FakeDriver::new();
        // entity b, c ; episode ep1 MENTIONS b ; b RELATES_TO c
        let b = make_node("b", "B", "g1");
        let c = make_node("c", "C", "g1");
        driver.save_entity_nodes(&[b, c]).await.unwrap();
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        let mention = EpisodicEdge::new("ep1".into(), "b".into(), "g1".into(), Utc::now());
        driver.save_episodic_edges(&[mention]).await.unwrap();
        let bc = make_edge("bc", "b", "c", "REL", "b-c", "g1");
        driver.save_entity_edges(&[bc]).await.unwrap();

        let edges = driver
            .edge_bfs_search(&["ep1".to_string()], 2, &SearchFilters::default(), &[], 100)
            .await
            .unwrap();
        // Only the RELATES_TO edge bc is returned; the MENTIONS hop is not an edge.
        let uuids: Vec<String> = edges.iter().map(|e| e.uuid.clone()).collect();
        assert_eq!(uuids, vec!["bc".to_string()]);
    }

    #[tokio::test]
    async fn node_bfs_respects_group_id_of_origin() {
        let driver = FakeDriver::new();
        // origin a in g1 ; b in g2 (different group) reachable but filtered out
        let a = make_node("a", "A", "g1");
        let b = make_node("b", "B", "g2");
        driver.save_entity_nodes(&[a, b]).await.unwrap();
        let ab = make_edge("ab", "a", "b", "REL", "a-b", "g1");
        driver.save_entity_edges(&[ab]).await.unwrap();

        let hits = driver
            .node_bfs_search(&["a".to_string()], &SearchFilters::default(), 2, &[], 100)
            .await
            .unwrap();
        assert!(
            hits.is_empty(),
            "b is in a different group than origin a → excluded"
        );
    }

    // ==================================================================
    // Phase-2: reranker-support primitives
    // ==================================================================

    #[tokio::test]
    async fn nodes_connected_to_center_is_undirected() {
        let driver = FakeDriver::new();
        // center <- x (incoming) ; center -> y (outgoing) ; z unconnected
        let in_edge = make_edge("e-in", "x", "center", "REL", "x-center", "g1");
        let out_edge = make_edge("e-out", "center", "y", "REL", "center-y", "g1");
        driver
            .save_entity_edges(&[in_edge, out_edge])
            .await
            .unwrap();

        let connected = driver
            .nodes_connected_to_center(
                &["x".to_string(), "y".to_string(), "z".to_string()],
                "center",
            )
            .await
            .unwrap();
        let mut got = connected.clone();
        got.sort();
        assert_eq!(
            got,
            vec!["x".to_string(), "y".to_string()],
            "both incoming and outgoing neighbours count (undirected)"
        );
    }

    #[tokio::test]
    async fn episode_mention_counts_per_target() {
        let driver = FakeDriver::new();
        // n1 mentioned twice, n2 once, n3 zero
        let m1 = EpisodicEdge::new("ep1".into(), "n1".into(), "g1".into(), Utc::now());
        let m2 = EpisodicEdge::new("ep2".into(), "n1".into(), "g1".into(), Utc::now());
        let m3 = EpisodicEdge::new("ep1".into(), "n2".into(), "g1".into(), Utc::now());
        driver.save_episodic_edges(&[m1, m2, m3]).await.unwrap();

        let counts = driver
            .episode_mention_counts(&["n1".to_string(), "n2".to_string(), "n3".to_string()])
            .await
            .unwrap();
        assert_eq!(counts.get("n1"), Some(&2));
        assert_eq!(counts.get("n2"), Some(&1));
        assert_eq!(counts.get("n3"), None, "unmentioned uuid omitted from map");
    }

    #[tokio::test]
    async fn embeddings_loaders_return_only_present() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut node_with = make_node("nw", "with", "g1");
        node_with.name_embedding = Some(emb.create("with").await.unwrap());
        let node_without = make_node("nx", "without", "g1");
        driver
            .save_entity_nodes(&[node_with, node_without])
            .await
            .unwrap();

        let mut edge_with = make_edge("ew", "a", "b", "REL", "with", "g1");
        edge_with.fact_embedding = Some(emb.create("with").await.unwrap());
        let edge_without = make_edge("ex", "a", "c", "REL", "without", "g1");
        driver
            .save_entity_edges(&[edge_with, edge_without])
            .await
            .unwrap();

        let node_map = driver
            .get_embeddings_for_nodes(&["nw".to_string(), "nx".to_string()])
            .await
            .unwrap();
        assert!(node_map.contains_key("nw"));
        assert!(
            !node_map.contains_key("nx"),
            "node without embedding omitted"
        );

        let edge_map = driver
            .get_embeddings_for_edges(&["ew".to_string(), "ex".to_string()])
            .await
            .unwrap();
        assert!(edge_map.contains_key("ew"));
        assert!(
            !edge_map.contains_key("ex"),
            "edge without embedding omitted"
        );
    }

    #[tokio::test]
    async fn episode_fulltext_search_substring_and_group() {
        let driver = FakeDriver::new();
        let mut ep1 = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        ep1.content = "Alice joined Acme".into();
        let mut ep2 = make_episode("ep2", "g2", EpisodeType::Message, Utc::now());
        ep2.content = "Alice left town".into();
        driver.save_episode(&ep1).await.unwrap();
        driver.save_episode(&ep2).await.unwrap();

        let all = driver
            .episode_fulltext_search("alice", &[], 10)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        let g1_only = driver
            .episode_fulltext_search("alice", &["g1".to_string()], 10)
            .await
            .unwrap();
        assert_eq!(g1_only.len(), 1);
        assert_eq!(g1_only[0].uuid, "ep1");
    }

    // ==================================================================
    // Phase-4: Community ops
    // ==================================================================

    fn make_community(uuid: &str, name: &str, group_id: &str) -> CommunityNode {
        let mut c = CommunityNode::new(name.into(), group_id.into(), Utc::now());
        c.uuid = uuid.into();
        c
    }

    fn make_saga(uuid: &str, name: &str, group_id: &str) -> SagaNode {
        let mut s = SagaNode::new(name.into(), group_id.into(), Utc::now());
        s.uuid = uuid.into();
        s
    }

    #[tokio::test]
    async fn community_save_and_get_by_group_and_uuid() {
        let driver = FakeDriver::new();
        let c1 = make_community("c1", "Tech", "g1");
        let c2 = make_community("c2", "Finance", "g2");
        driver.save_community_nodes(&[c1, c2]).await.unwrap();

        let g1 = driver
            .get_community_nodes_by_group_ids(&["g1".to_string()])
            .await
            .unwrap();
        assert_eq!(g1.len(), 1);
        assert_eq!(g1[0].uuid, "c1");

        let by_uuid = driver
            .get_community_nodes_by_uuids(&["c2".to_string(), "missing".to_string()])
            .await
            .unwrap();
        assert_eq!(by_uuid.len(), 1);
        assert_eq!(by_uuid[0].uuid, "c2");

        // Empty group_ids → empty (upstream semantics).
        let none = driver.get_community_nodes_by_group_ids(&[]).await.unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn community_fulltext_substring_and_group_filter() {
        let driver = FakeDriver::new();
        driver
            .save_community_nodes(&[
                make_community("c1", "Tech Companies", "g1"),
                make_community("c2", "Finance Sector", "g2"),
            ])
            .await
            .unwrap();

        let all = driver
            .community_fulltext_search("tech", &[], 10)
            .await
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].uuid, "c1");

        let scoped = driver
            .community_fulltext_search("sector", &["g1".to_string()], 10)
            .await
            .unwrap();
        assert!(scoped.is_empty(), "c2 is in g2, excluded by g1 filter");
    }

    #[tokio::test]
    async fn community_similarity_ranks_and_skips_no_embedding() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut alpha = make_community("c-alpha", "alpha", "g1");
        alpha.name_embedding = Some(emb.create("alpha").await.unwrap());
        let mut beta = make_community("c-beta", "beta", "g1");
        beta.name_embedding = Some(emb.create("beta").await.unwrap());
        let no_emb = make_community("c-none", "gamma", "g1");
        driver
            .save_community_nodes(&[alpha, beta, no_emb])
            .await
            .unwrap();

        let qv = emb.create("alpha").await.unwrap();
        let results = driver
            .community_similarity_search(&qv, &[], 10, 0.0)
            .await
            .unwrap();
        assert!(results.iter().all(|c| c.uuid != "c-none"));
        assert_eq!(results[0].uuid, "c-alpha");
    }

    #[tokio::test]
    async fn community_embeddings_loader_omits_missing() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);
        let mut with = make_community("cw", "with", "g1");
        with.name_embedding = Some(emb.create("with").await.unwrap());
        let without = make_community("cx", "without", "g1");
        driver.save_community_nodes(&[with, without]).await.unwrap();

        let map = driver
            .get_embeddings_for_communities(&["cw".to_string(), "cx".to_string()])
            .await
            .unwrap();
        assert!(map.contains_key("cw"));
        assert!(!map.contains_key("cx"));
    }

    #[tokio::test]
    async fn remove_communities_clears_nodes_and_edges() {
        let driver = FakeDriver::new();
        driver
            .save_community_nodes(&[make_community("c1", "Tech", "g1")])
            .await
            .unwrap();
        let edge = CommunityEdge::new("c1".into(), "n1".into(), "g1".into(), Utc::now());
        driver.save_community_edges(&[edge]).await.unwrap();

        driver.remove_communities().await.unwrap();
        let nodes = driver
            .get_community_nodes_by_group_ids(&["g1".to_string()])
            .await
            .unwrap();
        assert!(nodes.is_empty());
        // membership lookup now finds nothing
        let mem = driver.community_of_member("n1").await.unwrap();
        assert!(mem.is_none());
    }

    #[tokio::test]
    async fn community_clusters_projection_shape() {
        let driver = FakeDriver::new();
        // g1: a-b (2 edges) and a-c (1 edge); g2: standalone d.
        let a = make_node("a", "A", "g1");
        let b = make_node("b", "B", "g1");
        let c = make_node("c", "C", "g1");
        let d = make_node("d", "D", "g2");
        driver.save_entity_nodes(&[a, b, c, d]).await.unwrap();
        driver
            .save_entity_edges(&[
                make_edge("ab1", "a", "b", "REL", "f", "g1"),
                make_edge("ab2", "a", "b", "REL", "f", "g1"),
                make_edge("ac", "a", "c", "REL", "f", "g1"),
            ])
            .await
            .unwrap();

        let clusters = driver
            .get_community_clusters(&["g1".to_string()])
            .await
            .unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].group_id, "g1");
        // node "a" has neighbours b (count 2) and c (count 1)
        let a_proj = clusters[0]
            .nodes
            .iter()
            .find(|n| n.node_uuid == "a")
            .expect("node a present");
        let b_n = a_proj
            .neighbors
            .iter()
            .find(|n| n.node_uuid == "b")
            .unwrap();
        let c_n = a_proj
            .neighbors
            .iter()
            .find(|n| n.node_uuid == "c")
            .unwrap();
        assert_eq!(b_n.edge_count, 2, "a-b has 2 RELATES_TO edges");
        assert_eq!(c_n.edge_count, 1, "a-c has 1 RELATES_TO edge");

        // Empty group_ids → all distinct entity group_ids projected.
        let all = driver.get_community_clusters(&[]).await.unwrap();
        let gids: Vec<&str> = all.iter().map(|p| p.group_id.as_str()).collect();
        assert!(gids.contains(&"g1"));
        assert!(gids.contains(&"g2"));
    }

    #[tokio::test]
    async fn membership_already_member_vs_neighbor_vote() {
        let driver = FakeDriver::new();
        // Entities a, b, x; community c1 has member a; a RELATES_TO x.
        driver
            .save_entity_nodes(&[
                make_node("a", "A", "g1"),
                make_node("b", "B", "g1"),
                make_node("x", "X", "g1"),
            ])
            .await
            .unwrap();
        driver
            .save_community_nodes(&[make_community("c1", "C1", "g1")])
            .await
            .unwrap();
        driver
            .save_community_edges(&[CommunityEdge::new(
                "c1".into(),
                "a".into(),
                "g1".into(),
                Utc::now(),
            )])
            .await
            .unwrap();
        driver
            .save_entity_edges(&[make_edge("ax", "a", "x", "REL", "f", "g1")])
            .await
            .unwrap();

        // a is already a member of c1
        let mem = driver.community_of_member("a").await.unwrap().unwrap();
        assert_eq!(mem.uuid, "c1");

        // x is NOT a member; its neighbour a belongs to c1 → neighbour-vote row
        assert!(driver.community_of_member("x").await.unwrap().is_none());
        let votes = driver.neighbor_communities("x").await.unwrap();
        assert_eq!(votes.len(), 1, "one row per neighbour-community membership");
        assert_eq!(votes[0].uuid, "c1");
    }

    #[tokio::test]
    async fn neighbor_communities_emits_row_per_neighbor_not_deduped() {
        let driver = FakeDriver::new();
        // x's neighbours a and b are BOTH members of c1 → two rows for c1.
        driver
            .save_entity_nodes(&[
                make_node("a", "A", "g1"),
                make_node("b", "B", "g1"),
                make_node("x", "X", "g1"),
            ])
            .await
            .unwrap();
        driver
            .save_community_nodes(&[make_community("c1", "C1", "g1")])
            .await
            .unwrap();
        driver
            .save_community_edges(&[
                CommunityEdge::new("c1".into(), "a".into(), "g1".into(), Utc::now()),
                CommunityEdge::new("c1".into(), "b".into(), "g1".into(), Utc::now()),
            ])
            .await
            .unwrap();
        driver
            .save_entity_edges(&[
                make_edge("xa", "x", "a", "REL", "f", "g1"),
                make_edge("xb", "x", "b", "REL", "f", "g1"),
            ])
            .await
            .unwrap();

        let votes = driver.neighbor_communities("x").await.unwrap();
        assert_eq!(votes.len(), 2, "NOT deduped: one row per neighbour vote");
        assert!(votes.iter().all(|c| c.uuid == "c1"));
    }

    // ==================================================================
    // Phase-4: Saga ops
    // ==================================================================

    #[tokio::test]
    async fn saga_get_by_name_scopes_group() {
        let driver = FakeDriver::new();
        driver
            .save_saga_node(&make_saga("s1", "onboarding", "g1"))
            .await
            .unwrap();
        driver
            .save_saga_node(&make_saga("s2", "onboarding", "g2"))
            .await
            .unwrap();

        let hit = driver
            .get_saga_by_name("onboarding", "g1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hit.uuid, "s1");
        assert!(
            driver
                .get_saga_by_name("missing", "g1")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn saga_previous_episode_picks_latest_by_valid_at() {
        let driver = FakeDriver::new();
        let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2026, 1, 3, 0, 0, 0).unwrap();
        for (uuid, t) in [("e1", t1), ("e2", t2), ("e3", t3)] {
            driver
                .save_episode(&make_episode(uuid, "g1", EpisodeType::Message, t))
                .await
                .unwrap();
        }
        driver
            .save_saga_node(&make_saga("s1", "saga", "g1"))
            .await
            .unwrap();
        for uuid in ["e1", "e2", "e3"] {
            driver
                .save_has_episode_edge(&HasEpisodeEdge::new(
                    "s1".into(),
                    uuid.into(),
                    "g1".into(),
                    Utc::now(),
                ))
                .await
                .unwrap();
        }

        // current = e3 → previous should be e2 (latest valid_at among the rest)
        let prev = driver.saga_previous_episode_uuid("s1", "e3").await.unwrap();
        assert_eq!(prev.as_deref(), Some("e2"));
    }

    #[tokio::test]
    async fn saga_episode_contents_since_filters_and_orders() {
        let driver = FakeDriver::new();
        let v1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let v2 = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        // created_at watermark
        let c_old = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let c_new = Utc.with_ymd_and_hms(2026, 6, 1, 0, 0, 0).unwrap();

        let mut e_old = make_episode("e-old", "g1", EpisodeType::Message, v1);
        e_old.created_at = c_old;
        e_old.content = "old content".into();
        let mut e_new = make_episode("e-new", "g1", EpisodeType::Message, v2);
        e_new.created_at = c_new;
        e_new.content = "new content".into();
        driver.save_episode(&e_old).await.unwrap();
        driver.save_episode(&e_new).await.unwrap();

        driver
            .save_saga_node(&make_saga("s1", "saga", "g1"))
            .await
            .unwrap();
        for uuid in ["e-old", "e-new"] {
            driver
                .save_has_episode_edge(&HasEpisodeEdge::new(
                    "s1".into(),
                    uuid.into(),
                    "g1".into(),
                    Utc::now(),
                ))
                .await
                .unwrap();
        }

        // since just after c_old → only e-new (created_at > since)
        let since = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap();
        let filtered = driver
            .saga_episode_contents("s1", Some(since), 200)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0, "new content");

        // no watermark → both, chronological by valid_at
        let all = driver.saga_episode_contents("s1", None, 200).await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "old content");
        assert_eq!(all[1].0, "new content");
    }

    // ==================================================================
    // Phase-4: maintenance (get_mentioned_nodes + deletes)
    // ==================================================================

    #[tokio::test]
    async fn get_mentioned_nodes_from_mentions_targets() {
        let driver = FakeDriver::new();
        driver
            .save_entity_nodes(&[make_node("n1", "N1", "g1"), make_node("n2", "N2", "g1")])
            .await
            .unwrap();
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        driver
            .save_episodic_edges(&[
                EpisodicEdge::new("ep1".into(), "n1".into(), "g1".into(), Utc::now()),
                EpisodicEdge::new("ep1".into(), "n2".into(), "g1".into(), Utc::now()),
            ])
            .await
            .unwrap();

        let mentioned = driver
            .get_mentioned_nodes(&["ep1".to_string()])
            .await
            .unwrap();
        let mut uuids: Vec<String> = mentioned.iter().map(|n| n.uuid.clone()).collect();
        uuids.sort();
        assert_eq!(uuids, vec!["n1".to_string(), "n2".to_string()]);

        // empty input → empty
        assert!(driver.get_mentioned_nodes(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_entity_edges_and_nodes_and_episode() {
        let driver = FakeDriver::new();
        driver
            .save_entity_nodes(&[make_node("n1", "N1", "g1")])
            .await
            .unwrap();
        driver
            .save_entity_edges(&[make_edge("e1", "n1", "n2", "REL", "f", "g1")])
            .await
            .unwrap();
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        driver
            .save_episodic_edges(&[EpisodicEdge::new(
                "ep1".into(),
                "n1".into(),
                "g1".into(),
                Utc::now(),
            )])
            .await
            .unwrap();

        driver
            .delete_entity_edges_by_uuids(&["e1".to_string()])
            .await
            .unwrap();
        assert!(driver.get_entity_edge("e1").await.unwrap().is_none());

        driver
            .delete_entity_nodes_by_uuids(&["n1".to_string()])
            .await
            .unwrap();
        assert!(driver.get_entity_node("n1").await.unwrap().is_none());
        // its MENTIONS edge is gone too
        assert_eq!(driver.episodic_edge_count(), 0);

        driver.delete_episode("ep1").await.unwrap();
        assert!(driver.get_episode("ep1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_entity_edges_by_uuids_preserves_order() {
        let driver = FakeDriver::new();
        driver
            .save_entity_edges(&[
                make_edge("e1", "a", "b", "REL", "f1", "g1"),
                make_edge("e2", "a", "c", "REL", "f2", "g1"),
            ])
            .await
            .unwrap();
        let got = driver
            .get_entity_edges_by_uuids(&["e2".to_string(), "e1".to_string(), "miss".to_string()])
            .await
            .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].uuid, "e2");
        assert_eq!(got[1].uuid, "e1");
    }
}
