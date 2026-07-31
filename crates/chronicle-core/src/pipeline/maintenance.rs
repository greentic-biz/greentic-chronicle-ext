// Ported from graphiti_core/graphiti.py @ 34f56e65 (v0.29.1):
//   - add_triplet (~1645-1763)
//   - remove_episode (~1765-1793)
//   - get_nodes_and_edges_by_episode (~1632-1643)
//
// Maintenance operations on an existing graph. `add_triplet` inserts a single
// (source)-[edge]->(target) fact with full node/edge resolution + invalidation
// but WITHOUT episode/episodic-edge/community side effects. `remove_episode`
// performs the upstream cascade (primary-source edges + single-mention nodes +
// the episode). `get_nodes_and_edges_by_episode` is a trivial fan-in.

use std::collections::HashSet;

use crate::errors::ChronicleError;
use crate::helpers::utc_now;
use crate::pipeline::bulk::add_nodes_and_edges_bulk;
use crate::pipeline::clients::Clients;
use crate::pipeline::edge_ops::resolve_extracted_edge;
use crate::pipeline::node_ops::resolve_extracted_nodes;
use crate::search::edge_search::edge_search_simple;
use crate::search::filters::SearchFilters;
use crate::search::recipes::edge_hybrid_search_rrf;
use crate::search::results::SearchResults;
use crate::types::{EntityEdge, EntityNode, EpisodeType, EpisodicNode};

/// Result of [`add_triplet`].
///
/// Port of upstream `AddTripletResults` (graphiti.py). `edges` is the resolved
/// edge followed by any invalidated edges; `nodes` is the resolved
/// `[source, target]` pair.
#[derive(Debug, Clone)]
pub struct AddTripletResults {
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
}

/// Insert a single `(source)-[edge]->(target)` triplet with full resolution +
/// invalidation, but no episode/episodic-edge/community side effects.
///
/// Port of upstream `Graphiti.add_triplet` (graphiti.py ~1645-1763):
/// 1. Embed node names / edge fact when missing.
/// 2. Resolve each node: reuse the persisted node if its UUID already exists,
///    else run single-node `resolve_extracted_nodes` and take the first result.
/// 3. Merge caller-provided attributes/summary/labels into the resolved nodes.
/// 4. Point the edge at the resolved node UUIDs.
/// 5. UUID-conflict regen: if an edge with `edge.uuid` already exists but with
///    different endpoints, mint a fresh UUID so we create a new edge instead of
///    overwriting the unrelated one.
/// 6. `valid_edges` = edges already between the two nodes; `related_edges` =
///    hybrid fact search filtered to those `valid_edges` (the D-3 pattern); the
///    invalidation pool (`existing_edges`) = unfiltered hybrid fact search.
/// 7. `resolve_extracted_edge` against a synthetic minimal episode (full dedup /
///    invalidation), then persist nodes + (resolved + invalidated) edges via the
///    bulk save path.
pub async fn add_triplet(
    clients: &Clients,
    source_node: EntityNode,
    edge: EntityEdge,
    target_node: EntityNode,
) -> Result<AddTripletResults, ChronicleError> {
    let mut source_node = source_node;
    let mut target_node = target_node;
    let mut edge = edge;

    // 1. Embed missing name / fact embeddings (upstream generate_*_embedding).
    if source_node.name_embedding.is_none() {
        source_node.name_embedding = Some(
            clients
                .embedder
                .create(&source_node.name.replace('\n', " "))
                .await?,
        );
    }
    if target_node.name_embedding.is_none() {
        target_node.name_embedding = Some(
            clients
                .embedder
                .create(&target_node.name.replace('\n', " "))
                .await?,
        );
    }
    if edge.fact_embedding.is_none() {
        edge.fact_embedding = Some(
            clients
                .embedder
                .create(&edge.fact.replace('\n', " "))
                .await?,
        );
    }

    // Synthetic episode for the single-node resolution + edge resolution
    // (upstream passes `EpisodicNode(name='', ..., valid_at=edge.valid_at or
    // utc_now())`). resolve_extracted_nodes also uses it as the primary episode.
    let synthetic_episode = EpisodicNode::new(
        String::new(),
        edge.group_id.clone(),
        EpisodeType::Text,
        String::new(),
        String::new(),
        utc_now(),
        edge.valid_at.unwrap_or_else(utc_now),
    );

    // 2. Resolve source / target nodes.
    let mut resolved_source = match clients.driver.get_entity_node(&source_node.uuid).await? {
        Some(existing) => existing,
        None => {
            let outcome = resolve_extracted_nodes(
                clients,
                vec![source_node.clone()],
                &synthetic_episode,
                &[],
            )
            .await?;
            outcome.nodes.into_iter().next().ok_or_else(|| {
                ChronicleError::InvalidInput("source node resolution empty".into())
            })?
        }
    };
    let mut resolved_target = match clients.driver.get_entity_node(&target_node.uuid).await? {
        Some(existing) => existing,
        None => {
            let outcome = resolve_extracted_nodes(
                clients,
                vec![target_node.clone()],
                &synthetic_episode,
                &[],
            )
            .await?;
            outcome.nodes.into_iter().next().ok_or_else(|| {
                ChronicleError::InvalidInput("target node resolution empty".into())
            })?
        }
    };

    // 3. Merge caller-provided attributes / summary / labels into the resolved
    //    nodes (upstream: dict update for attributes, replace summary when the
    //    caller passed a non-empty one, set-union for labels).
    for (key, value) in &source_node.attributes {
        resolved_source
            .attributes
            .insert(key.clone(), value.clone());
    }
    for (key, value) in &target_node.attributes {
        resolved_target
            .attributes
            .insert(key.clone(), value.clone());
    }
    if !source_node.summary.is_empty() {
        resolved_source.summary = source_node.summary.clone();
    }
    if !target_node.summary.is_empty() {
        resolved_target.summary = target_node.summary.clone();
    }
    if !source_node.labels.is_empty() {
        resolved_source.labels = union_labels(&resolved_source.labels, &source_node.labels);
    }
    if !target_node.labels.is_empty() {
        resolved_target.labels = union_labels(&resolved_target.labels, &target_node.labels);
    }

    // 4. Point the edge at the resolved nodes.
    edge.source_node_uuid = resolved_source.uuid.clone();
    edge.target_node_uuid = resolved_target.uuid.clone();

    // 5. UUID-conflict regen: an existing edge with this UUID but different
    //    endpoints would be overwritten by save — mint a new UUID instead.
    if let Some(existing_edge) = clients.driver.get_entity_edge(&edge.uuid).await?
        && (existing_edge.source_node_uuid != edge.source_node_uuid
            || existing_edge.target_node_uuid != edge.target_node_uuid)
    {
        let old_uuid = edge.uuid.clone();
        edge.uuid = uuid::Uuid::new_v4().to_string();
        tracing::info!(
            old_uuid = %old_uuid,
            new_uuid = %edge.uuid,
            "edge UUID already exists with different source/target; generated new UUID"
        );
    }

    // 6. Candidate pools for resolution (D-3 pattern).
    let valid_edges = clients
        .driver
        .get_edges_between_nodes(&edge.source_node_uuid, &edge.target_node_uuid)
        .await?;

    let related_filter = SearchFilters {
        edge_uuids: Some(valid_edges.iter().map(|e| e.uuid.clone()).collect()),
        ..SearchFilters::default()
    };
    let group_ids = [edge.group_id.clone()];
    let related_edges = edge_search_simple(
        clients.driver.as_ref(),
        clients.embedder.as_ref(),
        None,
        &edge.fact,
        &group_ids,
        &edge_hybrid_search_rrf(),
        &related_filter,
    )
    .await?;
    let existing_edges = edge_search_simple(
        clients.driver.as_ref(),
        clients.embedder.as_ref(),
        None,
        &edge.fact,
        &group_ids,
        &edge_hybrid_search_rrf(),
        &SearchFilters::default(),
    )
    .await?;

    // 7. Resolve the edge (dedup + invalidation) against the pools.
    let (resolved_edge, invalidated_edges) = resolve_extracted_edge(
        clients,
        edge,
        related_edges,
        existing_edges,
        &synthetic_episode,
    )
    .await?;

    let mut nodes = vec![resolved_source, resolved_target];
    let mut edges = vec![resolved_edge];
    edges.extend(invalidated_edges);

    // Persist via the bulk save path (no episodes / episodic edges).
    add_nodes_and_edges_bulk(clients, &[], &[], &mut nodes, &mut edges).await?;

    Ok(AddTripletResults { nodes, edges })
}

/// Set-union of two label lists, sorted for determinism.
///
/// Upstream uses `list(set(a) | set(b))` (order non-deterministic); we sort to
/// keep persisted/returned labels stable across runs.
fn union_labels(base: &[String], extra: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = base.iter().cloned().collect();
    set.extend(extra.iter().cloned());
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
}

/// Remove an episode and the graph state it solely contributed.
///
/// Port of upstream `Graphiti.remove_episode` (graphiti.py ~1765-1793, plan R5):
/// 1. Load the episode (UUID-not-found surfaces [`ChronicleError::EpisodeNotFound`]).
/// 2. Fetch its `entity_edges`; delete only those whose `episodes[0]` equals this
///    episode's UUID (the episode that originally created the edge).
/// 3. Fetch its mentioned nodes; delete only those mentioned by exactly one
///    episode (this one).
/// 4. Delete the episode (detach).
///
/// No saga / community cascade (upstream).
pub async fn remove_episode(clients: &Clients, episode_uuid: &str) -> Result<(), ChronicleError> {
    let episode = clients
        .driver
        .get_episode(episode_uuid)
        .await?
        .ok_or_else(|| ChronicleError::EpisodeNotFound {
            uuid: episode_uuid.to_string(),
        })?;

    // Edges mentioned by the episode; keep only those it originally created.
    let edges = clients
        .driver
        .get_entity_edges_by_uuids(&episode.entity_edges)
        .await?;
    let edges_to_delete: Vec<String> = edges
        .iter()
        .filter(|e| e.episodes.first().map(String::as_str) == Some(episode.uuid.as_str()))
        .map(|e| e.uuid.clone())
        .collect();

    // Nodes mentioned by the episode; keep only single-mention ones.
    let nodes = clients
        .driver
        .get_mentioned_nodes(std::slice::from_ref(&episode.uuid))
        .await?;
    let node_uuids: Vec<String> = nodes.iter().map(|n| n.uuid.clone()).collect();
    let mention_counts = clients.driver.episode_mention_counts(&node_uuids).await?;
    let nodes_to_delete: Vec<String> = nodes
        .iter()
        .filter(|n| mention_counts.get(&n.uuid).copied() == Some(1))
        .map(|n| n.uuid.clone())
        .collect();

    clients
        .driver
        .delete_entity_edges_by_uuids(&edges_to_delete)
        .await?;
    clients
        .driver
        .delete_entity_nodes_by_uuids(&nodes_to_delete)
        .await?;
    clients.driver.delete_episode(&episode.uuid).await?;

    Ok(())
}

/// Collect the nodes and entity edges attributed to a single episode.
///
/// Port of upstream `Graphiti.get_nodes_and_edges_by_episode` (graphiti.py
/// ~1632-1643, plan R10): load the episode, fetch its mentioned nodes and its
/// `entity_edges`, and return them in a [`SearchResults`] with all other scopes /
/// score vectors empty.
pub async fn get_nodes_and_edges_by_episode(
    clients: &Clients,
    episode_uuid: &str,
) -> Result<SearchResults, ChronicleError> {
    let episode = clients
        .driver
        .get_episode(episode_uuid)
        .await?
        .ok_or_else(|| ChronicleError::EpisodeNotFound {
            uuid: episode_uuid.to_string(),
        })?;

    let edges = clients
        .driver
        .get_entity_edges_by_uuids(&episode.entity_edges)
        .await?;
    let nodes = clients
        .driver
        .get_mentioned_nodes(std::slice::from_ref(&episode.uuid))
        .await?;

    Ok(SearchResults {
        edges,
        nodes,
        ..SearchResults::default()
    })
}
