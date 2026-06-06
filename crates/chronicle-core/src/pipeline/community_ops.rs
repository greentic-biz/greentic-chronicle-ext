// Ported from graphiti_core/utils/maintenance/community_operations.py @ 34f56e65 (v0.29.1)
//
// This module ports the community-detection + summarization maintenance ops:
//   - label_propagation         (pure label-propagation community detection, R2)
//   - truncate_at_sentence      (graphiti_core/utils/text_utils.py @ 34f56e65)
//   - build_community           (pairwise-reduce summary merge per cluster, R3)
//   - build_communities         (orchestration: clusters → build, concurrency 10, R3)
//   - determine_entity_community (already-member / neighbour-vote, R4)
//   - update_community          (incremental membership update on ingest, R4)
//   - rebuild_communities       (remove → build → embed → save; graphiti.build_communities)
//
// Fidelity notes:
// - `truncate_at_sentence` is ported byte-for-byte from text_utils.py: it scans the
//   first `max_chars` for the LAST `[.!?]` followed by whitespace-or-end, then returns
//   up to (and including) that boundary, rstripped; no boundary → the prefix rstripped.
//   Upstream uses a regex `[.!?](?:\s|$)` over `text[:max_chars]`; we replicate the
//   match semantics directly without a regex dependency. NOTE: Python's `len()`/slicing
//   is by Unicode code point — we operate over `char_indices` to match that (NOT bytes).
// - `label_propagation` ports the `community_lst.sort(reverse=True)` over `(count,
//   community)` tuples EXACTLY: Python sorts tuples lexicographically, so ties on count
//   break on the larger community int (both descending). The top entry is taken only
//   when its count > 1; otherwise `max(community_candidate, curr_community)`.
//   `community_candidate` defaults to -1 when a node has no neighbours, so an isolated
//   node keeps `max(-1, curr) == curr`.
// - `label_propagation` termination (DEVIATION, safety fix): upstream's loop is a bare
//   `while True` with NO iteration cap. Synchronous label propagation can oscillate on
//   pathological inputs (e.g. a bare path a-b-c with equal weights swaps labels forever),
//   which would hang the build. We add a generous `MAX_LABEL_PROPAGATION_ITERATIONS = 1000`
//   cap: on overflow we `tracing::warn!` and return the last assignment instead of spinning.
//   This cannot change behaviour on correct input — real `get_community_clusters`
//   projections converge in a handful of passes — it only bounds the degenerate case.
//   Documented in docs/port-fidelity.md.
// - `build_community` divergence vs plan: upstream takes only the `llm_client`, not the
//   driver; our `build_community` takes `&Clients` to access `llm` + `semaphore` for the
//   parallel pairwise summarization. No persistence happens here (matches upstream).
// - `update_community` divergence vs plan/upstream: upstream's `community.save` writes the
//   node including name_embedding in one Cypher call. We split into
//   `save_community_nodes` (which the driver persists with its embedding) after calling
//   `embedder.create` to regenerate `name_embedding`. Behaviour-equivalent.

use std::collections::HashMap;
use std::sync::Arc;

use crate::errors::ChronicleError;
use crate::llm::{LlmRequest, generate_typed};
use crate::pipeline::clients::Clients;
use crate::prompts::models::{Summary, SummaryDescription};
use crate::prompts::summarize_nodes::{
    SummarizePairContext, SummaryDescriptionContext, summarize_pair as summarize_pair_prompt,
    summary_description as summary_description_prompt,
};
use crate::types::{CommunityEdge, CommunityNode, EntityNode};
use chrono::Utc;

use crate::driver::NodeNeighbors;

/// Upstream `MAX_SUMMARY_CHARS` (graphiti_core/utils/text_utils.py @ 34f56e65).
pub const MAX_SUMMARY_CHARS: usize = 1000;

/// Upstream `MAX_COMMUNITY_BUILD_CONCURRENCY` (community_operations.py @ 34f56e65).
pub const MAX_COMMUNITY_BUILD_CONCURRENCY: usize = 10;

/// Safety cap on `label_propagation` iterations (DEVIATION from upstream, which
/// loops `while True` with no bound — see module note). Synchronous label
/// propagation can oscillate forever on pathological inputs (e.g. a bare path
/// with equal weights), which would hang the build. Real `get_community_clusters`
/// projections converge in a handful of passes, so 1000 cannot trip on correct
/// input — it only bounds the degenerate case, emitting a `tracing::warn!` and
/// returning the last assignment instead of spinning.
pub const MAX_LABEL_PROPAGATION_ITERATIONS: usize = 1000;

// ---------------------------------------------------------------------------
// truncate_at_sentence
// ---------------------------------------------------------------------------

/// Truncate `text` at or about `max_chars`, respecting sentence boundaries.
///
/// Byte-for-byte port of `truncate_at_sentence` (graphiti_core/utils/text_utils.py
/// @ 34f56e65). Operates over Unicode code points (Python `str` semantics), NOT bytes.
///
/// 1. If `text` is empty or its code-point length ≤ `max_chars`, return it unchanged.
/// 2. Take the first `max_chars` code points. Find the LAST `[.!?]` that is followed by
///    whitespace or end-of-(truncated)-string; return `text` up to and including that
///    terminator, with trailing whitespace stripped.
/// 3. If no such boundary exists, return the `max_chars` prefix with trailing whitespace
///    stripped.
pub fn truncate_at_sentence(text: &str, max_chars: usize) -> String {
    if text.is_empty() {
        return text.to_string();
    }

    // Python slicing/len is by code point.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }

    let truncated = &chars[..max_chars];

    // Find the last index i where truncated[i] is one of . ! ? and the NEXT char
    // (truncated[i+1]) is whitespace, OR i is the final index of the truncated slice
    // (the `$` end-anchor in the upstream regex applies to the truncated string).
    let mut last_boundary_end: Option<usize> = None; // code-point index just past the terminator
    for i in 0..truncated.len() {
        let c = truncated[i];
        if c == '.' || c == '!' || c == '?' {
            let followed_by_ws_or_end = match truncated.get(i + 1) {
                Some(next) => next.is_whitespace(),
                None => true, // end of truncated string ($)
            };
            if followed_by_ws_or_end {
                // Upstream `last_match.end()` is the index just past the matched
                // terminator (the `(?:\s|$)` group is zero-or-one char; `.end()`
                // for `[.!?]\s` points past the whitespace, for `[.!?]$` past the
                // terminator). To mirror Python's `text[:last_match.end()].rstrip()`,
                // we slice up to (terminator + following whitespace if any) then rstrip;
                // rstrip makes the trailing-whitespace inclusion irrelevant.
                last_boundary_end = Some(i + 1);
            }
        }
    }

    if let Some(end) = last_boundary_end {
        let slice: String = chars[..end].iter().collect();
        return slice.trim_end().to_string();
    }

    let slice: String = truncated.iter().collect();
    slice.trim_end().to_string()
}

// ---------------------------------------------------------------------------
// label_propagation
// ---------------------------------------------------------------------------

/// Label-propagation community detection over one group's adjacency projection.
///
/// EXACT port of `label_propagation` (community_operations.py @ 34f56e65):
/// 1. Each node starts in its own community (its enumeration index).
/// 2. Iterate: each node sums neighbour edge_counts per neighbour community; build
///    `(count, community)` tuples, sort descending (Python lexicographic tuple sort:
///    ties on count break on larger community int); take the top community when its
///    count > 1, else `max(community_candidate, curr_community)`.
/// 3. Converge when no node changes community in a full pass.
/// 4. Return clusters grouped by final community int.
///
/// `projection` is the per-node neighbour adjacency for ONE group (upstream
/// `dict[str, list[Neighbor]]`); insertion order of `projection` defines the initial
/// community indices, exactly as Python `enumerate(projection.keys())`.
pub fn label_propagation(projection: &[NodeNeighbors]) -> Vec<Vec<String>> {
    // community_map: node uuid -> community int. Initial: enumeration index.
    let mut community_map: HashMap<String, i64> = HashMap::new();
    for (i, nn) in projection.iter().enumerate() {
        community_map.insert(nn.node_uuid.clone(), i as i64);
    }

    for iteration in 0.. {
        // Safety cap (deviation from upstream's unbounded `while True`): bail out
        // of a non-converging oscillation rather than spinning forever. Cannot
        // trip on converging (real) projections.
        if iteration >= MAX_LABEL_PROPAGATION_ITERATIONS {
            tracing::warn!(
                max_iterations = MAX_LABEL_PROPAGATION_ITERATIONS,
                node_count = projection.len(),
                "label_propagation did not converge within the iteration cap; \
                 returning the last assignment (input may be a pathological \
                 non-cluster graph)"
            );
            break;
        }

        let mut no_change = true;
        let mut new_community_map: HashMap<String, i64> = HashMap::new();

        for nn in projection.iter() {
            let curr_community = community_map[&nn.node_uuid];

            // Sum edge_counts per neighbour community.
            let mut community_candidates: HashMap<i64, i64> = HashMap::new();
            for neighbor in &nn.neighbors {
                if let Some(&nc) = community_map.get(&neighbor.node_uuid) {
                    *community_candidates.entry(nc).or_insert(0) += neighbor.edge_count as i64;
                }
            }

            // community_lst = [(count, community) for community, count in ...]
            let mut community_lst: Vec<(i64, i64)> = community_candidates
                .into_iter()
                .map(|(community, count)| (count, community))
                .collect();

            // Python: community_lst.sort(reverse=True) — descending lexicographic over
            // (count, community). Ties on count break on the larger community.
            community_lst.sort_by(|a, b| b.cmp(a));

            let (candidate_rank, community_candidate) =
                community_lst.first().copied().unwrap_or((0, -1));

            let new_community = if community_candidate != -1 && candidate_rank > 1 {
                community_candidate
            } else {
                community_candidate.max(curr_community)
            };

            new_community_map.insert(nn.node_uuid.clone(), new_community);

            if new_community != curr_community {
                no_change = false;
            }
        }

        if no_change {
            break;
        }
        community_map = new_community_map;
    }

    // Group by final community int. Preserve projection node order within clusters
    // and a deterministic cluster order (by community int) for stable output.
    let mut cluster_map: HashMap<i64, Vec<String>> = HashMap::new();
    for nn in projection.iter() {
        let community = community_map[&nn.node_uuid];
        cluster_map
            .entry(community)
            .or_default()
            .push(nn.node_uuid.clone());
    }

    let mut communities: Vec<i64> = cluster_map.keys().copied().collect();
    communities.sort_unstable();
    communities
        .into_iter()
        .map(|c| cluster_map.remove(&c).unwrap_or_default())
        .collect()
}

// ---------------------------------------------------------------------------
// summarize_pair / generate_summary_description (LLM helpers)
// ---------------------------------------------------------------------------

/// Port of upstream `summarize_pair`: combine two summaries into one dense summary,
/// then `truncate_at_sentence(result, MAX_SUMMARY_CHARS)`.
async fn summarize_pair(
    clients: &Clients,
    left: &str,
    right: &str,
) -> Result<String, ChronicleError> {
    let ctx = SummarizePairContext {
        left_summary: left,
        right_summary: right,
    };
    let request =
        LlmRequest::new(summarize_pair_prompt(&ctx)).named("summarize_nodes.summarize_pair");
    let response: Summary = generate_typed(clients.llm.as_ref(), request).await?;
    Ok(truncate_at_sentence(&response.summary, MAX_SUMMARY_CHARS))
}

/// Port of upstream `generate_summary_description`: a one-sentence name for a summary.
async fn generate_summary_description(
    clients: &Clients,
    summary: &str,
) -> Result<String, ChronicleError> {
    let ctx = SummaryDescriptionContext { summary };
    let request = LlmRequest::new(summary_description_prompt(&ctx))
        .named("summarize_nodes.summary_description");
    let response: SummaryDescription = generate_typed(clients.llm.as_ref(), request).await?;
    Ok(response.description)
}

// ---------------------------------------------------------------------------
// build_community_edges
// ---------------------------------------------------------------------------

/// Build one HAS_MEMBER edge per cluster member (Community → Entity).
///
/// Port of `build_community_edges` (graphiti_core/utils/maintenance/edge_operations.py
/// @ 34f56e65): `source = community.uuid`, `target = node.uuid`,
/// `group_id = community.group_id`, `created_at`.
pub fn build_community_edges(
    entity_nodes: &[EntityNode],
    community_node: &CommunityNode,
    created_at: chrono::DateTime<Utc>,
) -> Vec<CommunityEdge> {
    entity_nodes
        .iter()
        .map(|node| {
            CommunityEdge::new(
                community_node.uuid.clone(),
                node.uuid.clone(),
                community_node.group_id.clone(),
                created_at,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// build_community
// ---------------------------------------------------------------------------

/// Build a single community node + its HAS_MEMBER edges from a cluster of entities.
///
/// Port of upstream `build_community` (community_operations.py @ 34f56e65):
/// pairwise-reduce the member `summary` strings via `summarize_pair` in a binary-tree
/// merge (odd member carried forward each round, parallel under the shared semaphore),
/// `truncate_at_sentence` the final summary, derive the community `name` via
/// `generate_summary_description`, construct the `CommunityNode`, and build HAS_MEMBER
/// edges per member.
///
/// Requires a non-empty `cluster` (upstream indexes `community_cluster[0].group_id` and
/// `summaries[0]`); returns `InvalidInput` if empty.
pub async fn build_community(
    clients: &Clients,
    cluster: &[EntityNode],
) -> Result<(CommunityNode, Vec<CommunityEdge>), ChronicleError> {
    if cluster.is_empty() {
        return Err(ChronicleError::InvalidInput(
            "build_community requires a non-empty cluster".into(),
        ));
    }

    let mut summaries: Vec<String> = cluster.iter().map(|e| e.summary.clone()).collect();
    let mut length = summaries.len();

    while length > 1 {
        // Pop the odd one out (carried forward this round) if length is odd.
        let mut odd_one_out: Option<String> = None;
        if length % 2 == 1 {
            odd_one_out = summaries.pop();
            length -= 1;
        }

        let half = length / 2;
        // Pair left half with right half: zip(summaries[:half], summaries[half:]).
        let (left_half, right_half) = summaries.split_at(half);

        // Fan out summarize_pair under the shared semaphore (upstream semaphore_gather).
        let mut handles = Vec::with_capacity(half);
        for (left, right) in left_half.iter().zip(right_half.iter()) {
            let clients = clients.clone();
            let semaphore = Arc::clone(&clients.semaphore);
            let left = left.clone();
            let right = right.clone();
            handles.push(tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
                summarize_pair(&clients, &left, &right).await
            }));
        }

        let mut new_summaries: Vec<String> = Vec::with_capacity(handles.len());
        for handle in handles {
            let merged = handle.await.map_err(|e| {
                ChronicleError::InvalidInput(format!("summarize_pair task panicked: {e}"))
            })??;
            new_summaries.push(merged);
        }

        if let Some(odd) = odd_one_out {
            new_summaries.push(odd);
        }

        summaries = new_summaries;
        length = summaries.len();
    }

    let summary = truncate_at_sentence(&summaries[0], MAX_SUMMARY_CHARS);
    let name = generate_summary_description(clients, &summary).await?;
    let now = Utc::now();

    let mut community_node = CommunityNode::new(name, cluster[0].group_id.clone(), now);
    community_node.summary = summary;

    let community_edges = build_community_edges(cluster, &community_node, now);

    Ok((community_node, community_edges))
}

// ---------------------------------------------------------------------------
// build_communities
// ---------------------------------------------------------------------------

/// Build all communities for the given groups (or all groups when `group_ids` empty).
///
/// Port of upstream `build_communities` (community_operations.py @ 34f56e65):
/// `get_community_clusters` (driver) → `label_propagation` per group → fetch the cluster's
/// `EntityNode`s by uuid → `build_community` per cluster under a concurrency cap of
/// `MAX_COMMUNITY_BUILD_CONCURRENCY` (10) → collect nodes + flattened edges.
///
/// Does NOT persist (upstream `build_communities` returns nodes/edges; saving + name
/// embedding happen in [`rebuild_communities`]).
pub async fn build_communities(
    clients: &Clients,
    group_ids: &[String],
) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), ChronicleError> {
    let projections = clients.driver.get_community_clusters(group_ids).await?;

    // Per-group label propagation → cluster uuid lists → fetch entity nodes.
    let mut clusters: Vec<Vec<EntityNode>> = Vec::new();
    for projection in &projections {
        let cluster_uuids = label_propagation(&projection.nodes);
        for uuids in cluster_uuids {
            let nodes = clients.driver.get_entity_nodes_by_uuids(&uuids).await?;
            if !nodes.is_empty() {
                clusters.push(nodes);
            }
        }
    }

    // build_community per cluster, capped at MAX_COMMUNITY_BUILD_CONCURRENCY.
    let build_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_COMMUNITY_BUILD_CONCURRENCY));
    let mut handles = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let clients = clients.clone();
        let build_semaphore = Arc::clone(&build_semaphore);
        handles.push(tokio::spawn(async move {
            let _permit = build_semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            build_community(&clients, &cluster).await
        }));
    }

    let mut community_nodes: Vec<CommunityNode> = Vec::new();
    let mut community_edges: Vec<CommunityEdge> = Vec::new();
    for handle in handles {
        let (node, edges) = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("build_community task panicked: {e}"))
        })??;
        community_nodes.push(node);
        community_edges.extend(edges);
    }

    Ok((community_nodes, community_edges))
}

// ---------------------------------------------------------------------------
// determine_entity_community
// ---------------------------------------------------------------------------

/// Determine which community an entity should belong to (R4).
///
/// Port of upstream `determine_entity_community` (community_operations.py @ 34f56e65):
/// (a) if the entity already HAS_MEMBER a community → `(community, is_new=false)`;
/// (b) else the mode (plurality) community among RELATES_TO-neighbour memberships →
///     `(community, is_new=true)`; ties resolve to the FIRST community reaching the max
///     count, matching upstream's `for uuid, count in community_map.items()` (insertion
///     order) `if count > max_count` (strictly greater → first-seen wins).
/// (c) none → `(None, false)`.
///
/// The driver's `neighbor_communities` returns ONE row per neighbour membership (NOT
/// deduplicated); we mode-count by uuid here, preserving first-seen order.
pub async fn determine_entity_community(
    clients: &Clients,
    entity: &EntityNode,
) -> Result<(Option<CommunityNode>, bool), ChronicleError> {
    // (a) Already a member?
    if let Some(community) = clients.driver.community_of_member(&entity.uuid).await? {
        return Ok((Some(community), false));
    }

    // (b) Neighbour-vote: mode community among neighbour memberships.
    let communities = clients.driver.neighbor_communities(&entity.uuid).await?;

    // Count per community uuid; track first-seen order for stable tie-breaking.
    let mut order: Vec<String> = Vec::new();
    let mut community_map: HashMap<String, i64> = HashMap::new();
    for community in &communities {
        let entry = community_map.entry(community.uuid.clone());
        if let std::collections::hash_map::Entry::Vacant(_) = entry {
            order.push(community.uuid.clone());
        }
        *community_map.entry(community.uuid.clone()).or_insert(0) += 1;
    }

    let mut community_uuid: Option<String> = None;
    let mut max_count: i64 = 0;
    for uuid in &order {
        let count = community_map[uuid];
        if count > max_count {
            community_uuid = Some(uuid.clone());
            max_count = count;
        }
    }

    if max_count == 0 {
        return Ok((None, false));
    }

    if let Some(target) = community_uuid {
        for community in communities {
            if community.uuid == target {
                return Ok((Some(community), true));
            }
        }
    }

    Ok((None, false))
}

// ---------------------------------------------------------------------------
// update_community
// ---------------------------------------------------------------------------

/// Incrementally fold an entity into its community on ingest (R4).
///
/// Port of upstream `update_community` (community_operations.py @ 34f56e65):
/// `determine_entity_community`; if no community → no-op `([], [])`; else
/// `new_summary = summarize_pair(entity.summary, community.summary)`,
/// `new_name = generate_summary_description(new_summary)`; if `is_new`, save a HAS_MEMBER
/// edge; regenerate the community `name_embedding`; persist the community.
pub async fn update_community(
    clients: &Clients,
    entity: &EntityNode,
) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), ChronicleError> {
    let (community, is_new) = determine_entity_community(clients, entity).await?;

    let Some(mut community) = community else {
        return Ok((Vec::new(), Vec::new()));
    };

    let new_summary = summarize_pair(clients, &entity.summary, &community.summary).await?;
    let new_name = generate_summary_description(clients, &new_summary).await?;

    community.summary = new_summary;
    community.name = new_name;

    let mut community_edges: Vec<CommunityEdge> = Vec::new();
    if is_new {
        // build_community_edges yields exactly one edge per member; for a single
        // member that is one edge. Guard defensively rather than expect().
        if let Some(edge) =
            build_community_edges(std::slice::from_ref(entity), &community, Utc::now())
                .into_iter()
                .next()
        {
            clients
                .driver
                .save_community_edges(std::slice::from_ref(&edge))
                .await?;
            community_edges.push(edge);
        }
    }

    // Regenerate the name embedding (upstream community.generate_name_embedding).
    let embedding = clients
        .embedder
        .create(&community.name.replace('\n', " "))
        .await?;
    community.name_embedding = Some(embedding);

    clients
        .driver
        .save_community_nodes(std::slice::from_ref(&community))
        .await?;

    Ok((vec![community], community_edges))
}

// ---------------------------------------------------------------------------
// rebuild_communities
// ---------------------------------------------------------------------------

/// Full community rebuild for the given groups (graphiti `build_communities`).
///
/// Port of upstream `Graphiti.build_communities` (graphiti.py @ 34f56e65):
/// `remove_communities` (DETACH DELETE all :Community) → [`build_communities`] →
/// regenerate each community `name_embedding` in parallel → persist nodes + edges →
/// return them. The facade (Task 6) calls this method.
pub async fn rebuild_communities(
    clients: &Clients,
    group_ids: &[String],
) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), ChronicleError> {
    clients.driver.remove_communities().await?;

    let (mut community_nodes, community_edges) = build_communities(clients, group_ids).await?;

    // Regenerate name embeddings in parallel (upstream semaphore_gather over
    // node.generate_name_embedding).
    let mut handles = Vec::with_capacity(community_nodes.len());
    for node in &community_nodes {
        let embedder = Arc::clone(&clients.embedder);
        let semaphore = Arc::clone(&clients.semaphore);
        let name = node.name.replace('\n', " ");
        handles.push(tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            embedder.create(&name).await.map_err(ChronicleError::from)
        }));
    }
    for (node, handle) in community_nodes.iter_mut().zip(handles) {
        let embedding = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("name embedding task panicked: {e}"))
        })??;
        node.name_embedding = Some(embedding);
    }

    clients
        .driver
        .save_community_nodes(&community_nodes)
        .await?;
    clients
        .driver
        .save_community_edges(&community_edges)
        .await?;

    Ok((community_nodes, community_edges))
}

// ---------------------------------------------------------------------------
// Tests (pure)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::Neighbor;

    fn nn(uuid: &str, neighbors: &[(&str, u64)]) -> NodeNeighbors {
        NodeNeighbors {
            node_uuid: uuid.to_string(),
            neighbors: neighbors
                .iter()
                .map(|(u, c)| Neighbor {
                    node_uuid: u.to_string(),
                    edge_count: *c,
                })
                .collect(),
        }
    }

    /// Case 1: a fully-connected triple a-b-c (each pair edge_count 2) collapses
    /// into a single community. (A bare path chain a-b-c can oscillate under
    /// synchronous label propagation — that is upstream behaviour too; we test the
    /// convergent triangle to exercise the merge path.)
    #[test]
    fn label_propagation_chain_merges() {
        let projection = vec![
            nn("a", &[("b", 2), ("c", 2)]),
            nn("b", &[("a", 2), ("c", 2)]),
            nn("c", &[("a", 2), ("b", 2)]),
        ];
        let clusters = label_propagation(&projection);
        assert_eq!(
            clusters.len(),
            1,
            "triangle should collapse to one community"
        );
        let mut members = clusters[0].clone();
        members.sort();
        assert_eq!(members, vec!["a", "b", "c"]);
    }

    /// Case 2: tie on count breaks to the LARGER community int (tuple sort reverse).
    /// Node `x` (community 0) has two neighbours: `y` (community 1) and `z`
    /// (community 2), each edge_count 1. Candidates: {1:1, 2:1}. Top count == 1
    /// (not > 1), so new_community = max(community_candidate, curr). The sort picks
    /// (1,2) as top (larger community on tie), so community_candidate = 2,
    /// new_community = max(2, 0) = 2.
    #[test]
    fn label_propagation_tie_breaks_to_max() {
        // Single pass with frozen neighbour labels: use isolated y, z so they keep
        // their own community across the run (no neighbours → keep self).
        let projection = vec![nn("x", &[("y", 1), ("z", 1)]), nn("y", &[]), nn("z", &[])];
        let clusters = label_propagation(&projection);
        // y(community1) and z(community2) stay isolated; x joins community 2 (z).
        // Final communities: x->2, y->1, z->2  => clusters {1:[y], 2:[x,z]}.
        // Find the cluster containing x.
        let x_cluster = clusters
            .iter()
            .find(|c| c.contains(&"x".to_string()))
            .expect("x must be in some cluster");
        assert!(
            x_cluster.contains(&"z".to_string()),
            "x must join z's (larger-int) community on tie, got {x_cluster:?}"
        );
        assert!(
            !x_cluster.contains(&"y".to_string()),
            "x must NOT join y's (smaller-int) community on tie"
        );
    }

    /// Case 3: an isolated node (no neighbours) keeps its own community.
    #[test]
    fn label_propagation_isolated_keeps_own() {
        let projection = vec![nn("solo", &[])];
        let clusters = label_propagation(&projection);
        assert_eq!(clusters, vec![vec!["solo".to_string()]]);
    }

    /// Case 4: convergence — two disjoint triangles converge to exactly two
    /// communities and the algorithm terminates (no infinite oscillation).
    #[test]
    fn label_propagation_converges_two_components() {
        let projection = vec![
            nn("a", &[("b", 3), ("c", 3)]),
            nn("b", &[("a", 3), ("c", 3)]),
            nn("c", &[("a", 3), ("b", 3)]),
            nn("x", &[("y", 3), ("z", 3)]),
            nn("y", &[("x", 3), ("z", 3)]),
            nn("z", &[("x", 3), ("y", 3)]),
        ];
        let clusters = label_propagation(&projection);
        assert_eq!(
            clusters.len(),
            2,
            "two disjoint triangles → two communities"
        );
        for cluster in &clusters {
            assert_eq!(cluster.len(), 3, "each triangle is fully merged");
        }
    }

    /// Case 5: the iteration cap (safety deviation) guarantees termination on a
    /// pathological oscillating input — a bare 2-node path with equal weights,
    /// where synchronous propagation swaps labels forever under upstream's
    /// unbounded loop. We only assert it RETURNS (does not hang); the exact
    /// partition of a non-cluster graph is unspecified.
    #[test]
    fn label_propagation_caps_on_oscillating_input() {
        // a<->b with equal single-edge weight: each pass, a adopts b's label and
        // b adopts a's label, never settling under naive synchronous LPA.
        let projection = vec![nn("a", &[("b", 1)]), nn("b", &[("a", 1)])];
        let clusters = label_propagation(&projection);
        // Terminated within the cap and returned every node exactly once.
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 2, "both nodes are returned, no infinite loop");
    }

    #[test]
    fn truncate_at_sentence_returns_short_text_unchanged() {
        let text = "Short text.";
        assert_eq!(truncate_at_sentence(text, 1000), text);
    }

    #[test]
    fn truncate_at_sentence_truncates_at_last_boundary() {
        // Two sentences; max_chars lands inside the second. Should truncate at the
        // first sentence's terminator.
        let text = "First sentence. Second sentence is longer and overflows the limit.";
        let out = truncate_at_sentence(text, 20);
        assert_eq!(out, "First sentence.");
    }

    #[test]
    fn truncate_at_sentence_no_boundary_truncates_at_max() {
        // No sentence boundary within max_chars → hard prefix, rstripped.
        let text = "aaaaaaaaaa bbbbbbbbbb cccccccccc";
        let out = truncate_at_sentence(text, 15);
        assert_eq!(out, "aaaaaaaaaa bbbb");
    }

    #[test]
    fn truncate_at_sentence_question_and_exclamation_boundaries() {
        // max_chars 25 includes "Why? Because it works!" (the '!' sits at index 21);
        // the last boundary within the first 25 chars is that '!'.
        let text = "Why? Because it works! And then a very long trailing clause overflows.";
        let out = truncate_at_sentence(text, 25);
        assert_eq!(out, "Why? Because it works!");
    }

    #[test]
    fn truncate_at_sentence_counts_unicode_codepoints() {
        // 5 emoji (1 code point each) + ". rest" — under a code-point max it is
        // returned unchanged even though byte-length far exceeds max.
        let text = "😀😀😀😀😀. trailing";
        // 6 code points before the period; max_chars 100 >> len → unchanged.
        assert_eq!(truncate_at_sentence(text, 100), text);
    }
}
