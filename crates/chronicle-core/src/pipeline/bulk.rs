// Ported from graphiti_core/utils/bulk_utils.py (RawEpisode, _build_directed_uuid_map,
//   UnionFind / compress_uuid_map, retrieve_previous_episodes_bulk,
//   extract_nodes_and_edges_bulk (_separate variant), dedupe_nodes_bulk,
//   dedupe_edges_bulk, add_nodes_and_edges_bulk, resolve_edge_pointers, CHUNK_SIZE)
//   and graphiti.py::add_episode_bulk (~1230-1488), _extract_and_dedupe_nodes_bulk
//   (~783-813), _resolve_nodes_and_edges_bulk (~815-924), AddBulkEpisodeResults
//   (~123-129), _get_or_create_saga (~346-392), _saga_get_previous_episode_uuid
//   (~394-420) @ 34f56e65 (v0.29.1).
//
// Multi-episode bulk ingestion with cross-episode entity/edge dedup. Communities
// are never updated in the bulk path (upstream — communities/community_edges are
// always empty in the result). Combined extraction and Kuzu/IoC save branches are
// out of scope: we port the `_separate` extraction variant and persist via the
// existing per-collection driver save ops (the single-transaction save is Phase-4
// Task 7 — see `add_nodes_and_edges_bulk` atomicity note).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::ChronicleError;
use crate::helpers::{EPISODE_WINDOW_LEN, normalize_string_exact, utc_now};
use crate::pipeline::clients::Clients;
use crate::pipeline::dedup_helpers::{DedupResolutionState, build_candidate_indexes};
use crate::pipeline::edge_ops::{
    extract_edges, hydrate_node_summaries, resolve_extracted_edge, resolve_extracted_edges,
};
use crate::pipeline::node_ops::{extract_nodes, resolve_extracted_nodes, resolve_with_similarity};
use crate::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};

/// Upstream `CHUNK_SIZE` (bulk_utils.py:66). Defined upstream as a batch-size
/// hint; it is **caller-chunked, not applied internally** — `add_episode_bulk`
/// processes whatever slice it is handed in one pass. Callers ingesting very
/// large batches should split into chunks of at most this size themselves.
pub const CHUNK_SIZE: usize = 10;

/// Word-overlap / cosine threshold for the bulk edge candidate gate
/// (bulk_utils.py:498 `min_score = 0.6`).
const EDGE_DEDUP_MIN_SCORE: f32 = 0.6;

/// Raw caller-supplied episode prior to graph materialization.
///
/// Port of upstream `RawEpisode` (bulk_utils.py:101-107). `uuid` is `Some` only
/// when the caller is re-ingesting an already-persisted episode (looked up via
/// the driver); otherwise a fresh `EpisodicNode` is minted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEpisode {
    pub name: String,
    #[serde(default)]
    pub uuid: Option<String>,
    pub content: String,
    pub source_description: String,
    pub source: EpisodeType,
    pub reference_time: DateTime<Utc>,
}

/// Result of a bulk ingest.
///
/// Port of upstream `AddBulkEpisodeResults` (graphiti.py:123-129). `communities`
/// and `community_edges` are present for shape parity but are **always empty** in
/// the bulk path: upstream never updates communities during bulk ingest (the
/// `add_episode_bulk` return always passes `communities=[], community_edges=[]`).
#[derive(Debug, Clone)]
pub struct AddBulkEpisodeResults {
    pub episodes: Vec<EpisodicNode>,
    pub episodic_edges: Vec<EpisodicEdge>,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
    /// Always empty in bulk (upstream parity).
    pub communities: Vec<CommunityNode>,
    /// Always empty in bulk (upstream parity).
    pub community_edges: Vec<CommunityEdge>,
}

// ---------------------------------------------------------------------------
// Union-find helpers
// ---------------------------------------------------------------------------

/// Collapse alias → canonical chains while **preserving direction**.
///
/// Line-for-line port of upstream `_build_directed_uuid_map` (bulk_utils.py:69-98).
/// The incoming `pairs` are directed `(source, target)` mappings discovered during
/// node dedupe; we use union-find with iterative path compression so every source
/// UUID resolves to its ultimate canonical target, even when an alias is
/// lexicographically smaller than its canonical UUID. Direction is preserved
/// because we attach `find(source)` under `find(target)`
/// (`parent[find(source)] = find(target)`), NOT the smaller-uuid root.
pub fn build_directed_uuid_map(pairs: &[(String, String)]) -> HashMap<String, String> {
    let mut parent: HashMap<String, String> = HashMap::new();

    // Iterative path-compression find. Mirrors upstream's nested `find`: first
    // walk to the root, then re-walk compressing every node on the path to root.
    fn find(parent: &mut HashMap<String, String>, uuid: &str) -> String {
        parent
            .entry(uuid.to_string())
            .or_insert_with(|| uuid.to_string());

        // Walk to root.
        let mut root = uuid.to_string();
        while parent[&root] != root {
            root = parent[&root].clone();
        }

        // Compress: point every node on the path directly at root.
        let mut cur = uuid.to_string();
        while parent[&cur] != root {
            let next = parent[&cur].clone();
            parent.insert(cur, root.clone());
            cur = next;
        }

        root
    }

    for (source_uuid, target_uuid) in pairs {
        parent
            .entry(source_uuid.clone())
            .or_insert_with(|| source_uuid.clone());
        parent
            .entry(target_uuid.clone())
            .or_insert_with(|| target_uuid.clone());
        let rs = find(&mut parent, source_uuid);
        let rt = find(&mut parent, target_uuid);
        // Direction-preserving union: source's root points at target's root.
        parent.insert(rs, rt);
    }

    let keys: Vec<String> = parent.keys().cloned().collect();
    keys.into_iter()
        .map(|uuid| {
            let root = find(&mut parent, &uuid);
            (uuid, root)
        })
        .collect()
}

/// Undirected union-find collapsing each UUID to the **lexicographically smallest**
/// UUID in its duplicate set.
///
/// Port of upstream `UnionFind` + `compress_uuid_map` (bulk_utils.py:584-621).
/// `union(a, b)` attaches the lexicographically larger root under the smaller, so
/// the canonical root of every set is its smallest member. Unlike
/// [`build_directed_uuid_map`], direction is irrelevant here — duplicate edge
/// pairs are symmetric, so the smallest-uuid choice is purely for determinism.
pub fn compress_uuid_map(pairs: &[(String, String)]) -> HashMap<String, String> {
    let mut parent: HashMap<String, String> = HashMap::new();
    for (a, b) in pairs {
        parent.entry(a.clone()).or_insert_with(|| a.clone());
        parent.entry(b.clone()).or_insert_with(|| b.clone());
    }

    // Recursive-style find with full path compression (upstream `find`).
    fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
        let p = parent[x].clone();
        if p != x {
            let root = find(parent, &p);
            parent.insert(x.to_string(), root.clone());
            root
        } else {
            p
        }
    }

    for (a, b) in pairs {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            continue;
        }
        // Attach the lexicographically larger root under the smaller.
        if ra < rb {
            parent.insert(rb, ra);
        } else {
            parent.insert(ra, rb);
        }
    }

    let keys: Vec<String> = parent.keys().cloned().collect();
    keys.into_iter()
        .map(|uuid| {
            let root = find(&mut parent, &uuid);
            (uuid, root)
        })
        .collect()
}

/// Remap each edge's source/target node UUID through `uuid_map`
/// (bulk_utils.py:627-634). UUIDs absent from the map are left unchanged.
pub fn resolve_edge_pointers(edges: &mut [EntityEdge], uuid_map: &HashMap<String, String>) {
    for edge in edges {
        if let Some(canonical) = uuid_map.get(&edge.source_node_uuid) {
            edge.source_node_uuid = canonical.clone();
        }
        if let Some(canonical) = uuid_map.get(&edge.target_node_uuid) {
            edge.target_node_uuid = canonical.clone();
        }
    }
}

/// Remap each episodic (MENTIONS) edge's target node UUID through `uuid_map`.
/// Episodic edges point episode → entity, so only the target is a node UUID.
fn resolve_episodic_edge_pointers(edges: &mut [EpisodicEdge], uuid_map: &HashMap<String, String>) {
    for edge in edges {
        if let Some(canonical) = uuid_map.get(&edge.target_node_uuid) {
            edge.target_node_uuid = canonical.clone();
        }
    }
}

// ---------------------------------------------------------------------------
// retrieve_previous_episodes_bulk
// ---------------------------------------------------------------------------

/// Fetch the `EPISODE_WINDOW_LEN` previous episodes for each episode, in parallel.
///
/// Port of upstream `retrieve_previous_episodes_bulk` (bulk_utils.py:110-125).
/// Returns `(episode, previous_episodes)` tuples in the SAME order as `episodes`.
pub async fn retrieve_previous_episodes_bulk(
    clients: &Clients,
    episodes: &[EpisodicNode],
) -> Result<Vec<(EpisodicNode, Vec<EpisodicNode>)>, ChronicleError> {
    let mut handles = Vec::with_capacity(episodes.len());
    for episode in episodes {
        let driver = Arc::clone(&clients.driver);
        let semaphore = Arc::clone(&clients.semaphore);
        let valid_at = episode.valid_at;
        let group_id = episode.group_id.clone();
        handles.push(tokio::spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| ChronicleError::InvalidInput(format!("semaphore closed: {e}")))?;
            driver
                .retrieve_episodes(
                    valid_at,
                    EPISODE_WINDOW_LEN,
                    std::slice::from_ref(&group_id),
                    None,
                )
                .await
                .map_err(ChronicleError::from)
        }));
    }

    let mut episode_tuples = Vec::with_capacity(episodes.len());
    for (episode, handle) in episodes.iter().zip(handles) {
        let previous = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("retrieve_previous_episodes task panicked: {e}"))
        })??;
        episode_tuples.push((episode.clone(), previous));
    }

    Ok(episode_tuples)
}

// ---------------------------------------------------------------------------
// extract_nodes_and_edges_bulk
// ---------------------------------------------------------------------------

/// Extract nodes then edges for every episode in parallel (the `_separate`
/// upstream variant — combined single-call extraction is out of scope).
///
/// Port of upstream `_extract_nodes_and_edges_bulk_separate` (bulk_utils.py:330-371):
/// fan out `extract_nodes` per episode, then `extract_edges` per episode over that
/// episode's extracted nodes. Returns `(nodes_per_episode, edges_per_episode)` in
/// episode order.
pub async fn extract_nodes_and_edges_bulk(
    clients: &Clients,
    episode_tuples: &[(EpisodicNode, Vec<EpisodicNode>)],
    entity_types: Option<&Value>,
    custom_extraction_instructions: Option<&str>,
) -> Result<(Vec<Vec<EntityNode>>, Vec<Vec<EntityEdge>>), ChronicleError> {
    // Pass 1: extract nodes per episode.
    let mut node_handles = Vec::with_capacity(episode_tuples.len());
    for (episode, previous_episodes) in episode_tuples {
        let clients = clients.clone();
        let episode = episode.clone();
        let previous_episodes = previous_episodes.clone();
        let entity_types = entity_types.cloned();
        let custom = custom_extraction_instructions.map(str::to_string);
        node_handles.push(tokio::spawn(async move {
            extract_nodes(
                &clients,
                &episode,
                &previous_episodes,
                entity_types.as_ref(),
                custom.as_deref(),
            )
            .await
        }));
    }

    let mut extracted_nodes_bulk: Vec<Vec<EntityNode>> = Vec::with_capacity(node_handles.len());
    for handle in node_handles {
        let nodes = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("extract_nodes task panicked: {e}"))
        })??;
        extracted_nodes_bulk.push(nodes);
    }

    // Pass 2: extract edges per episode over that episode's extracted nodes.
    let mut edge_handles = Vec::with_capacity(episode_tuples.len());
    for (i, (episode, previous_episodes)) in episode_tuples.iter().enumerate() {
        let clients = clients.clone();
        let episode = episode.clone();
        let previous_episodes = previous_episodes.clone();
        let nodes = extracted_nodes_bulk[i].clone();
        let group_id = episode.group_id.clone();
        let custom = custom_extraction_instructions.map(str::to_string);
        edge_handles.push(tokio::spawn(async move {
            extract_edges(
                &clients,
                &episode,
                &nodes,
                &previous_episodes,
                &group_id,
                custom.as_deref(),
            )
            .await
        }));
    }

    let mut extracted_edges_bulk: Vec<Vec<EntityEdge>> = Vec::with_capacity(edge_handles.len());
    for handle in edge_handles {
        let edges = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("extract_edges task panicked: {e}"))
        })??;
        extracted_edges_bulk.push(edges);
    }

    Ok((extracted_nodes_bulk, extracted_edges_bulk))
}

// ---------------------------------------------------------------------------
// dedupe_nodes_bulk
// ---------------------------------------------------------------------------

/// Two-pass cross-episode node dedup.
///
/// Port of upstream `dedupe_nodes_bulk` (bulk_utils.py:374-486).
///
/// 1. **Pass 1 (vs graph):** run `resolve_extracted_nodes` for every episode in
///    parallel, reconciling each batch item against the live graph exactly like
///    the per-episode flow.
/// 2. **Pass 2 (intra-batch):** over the union of resolved nodes, accumulate a
///    canonical pool; for each node try an exact normalized-name match, else
///    build the MinHash/LSH candidate indexes over the current pool and run
///    `resolve_with_similarity`. Duplicate pairs found here, unioned with every
///    per-episode `uuid_map`, feed `build_directed_uuid_map`.
///
/// Returns `(nodes_by_episode_uuid, compressed_directed_map)`.
pub async fn dedupe_nodes_bulk(
    clients: &Clients,
    extracted_nodes: Vec<Vec<EntityNode>>,
    episode_tuples: &[(EpisodicNode, Vec<EpisodicNode>)],
    _entity_types: Option<&Value>,
) -> Result<(HashMap<String, Vec<EntityNode>>, HashMap<String, String>), ChronicleError> {
    // --- Pass 1: resolve each episode's nodes against the graph (parallel) ----
    let mut handles = Vec::with_capacity(extracted_nodes.len());
    for (i, nodes) in extracted_nodes.into_iter().enumerate() {
        let clients = clients.clone();
        let episode = episode_tuples[i].0.clone();
        let previous = episode_tuples[i].1.clone();
        handles.push(tokio::spawn(async move {
            resolve_extracted_nodes(&clients, nodes, &episode, &previous).await
        }));
    }

    let mut episode_resolutions: Vec<(String, Vec<EntityNode>)> = Vec::with_capacity(handles.len());
    let mut per_episode_uuid_maps: Vec<HashMap<String, String>> = Vec::with_capacity(handles.len());
    let mut duplicate_pairs: Vec<(String, String)> = Vec::new();

    for (i, handle) in handles.into_iter().enumerate() {
        let outcome = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("resolve_extracted_nodes task panicked: {e}"))
        })??;
        episode_resolutions.push((episode_tuples[i].0.uuid.clone(), outcome.nodes));
        per_episode_uuid_maps.push(outcome.uuid_map);
        for (source, target) in outcome.duplicates {
            duplicate_pairs.push((source.uuid, target.uuid));
        }
    }

    // --- Pass 2: intra-batch dedup over the union of resolved nodes -----------
    // O(n^2) like upstream: rebuild the MinHash index over the accumulated pool
    // per node (typical batches are <= CHUNK_SIZE, so this stays cheap).
    let mut canonical_nodes: HashMap<String, EntityNode> = HashMap::new();
    for (_, resolved_nodes) in &episode_resolutions {
        for node in resolved_nodes {
            if canonical_nodes.is_empty() {
                canonical_nodes.insert(node.uuid.clone(), node.clone());
                continue;
            }

            let existing_candidates: Vec<EntityNode> = canonical_nodes.values().cloned().collect();

            // Exact normalized-name match first.
            let normalized = normalize_string_exact(&node.name);
            let exact_match = existing_candidates
                .iter()
                .find(|candidate| normalize_string_exact(&candidate.name) == normalized);
            if let Some(exact_match) = exact_match {
                if exact_match.uuid != node.uuid {
                    duplicate_pairs.push((node.uuid.clone(), exact_match.uuid.clone()));
                }
                continue;
            }

            // MinHash/LSH similarity against the canonical pool.
            let indexes = build_candidate_indexes(existing_candidates);
            let mut state = DedupResolutionState {
                resolved_nodes: vec![None],
                uuid_map: HashMap::new(),
                unresolved_indices: Vec::new(),
                duplicate_pairs: Vec::new(),
            };
            resolve_with_similarity(std::slice::from_ref(node), &indexes, &mut state);

            match state.resolved_nodes[0].take() {
                None => {
                    canonical_nodes.insert(node.uuid.clone(), node.clone());
                }
                Some(resolved) => {
                    let canonical_uuid = resolved.uuid.clone();
                    canonical_nodes
                        .entry(canonical_uuid.clone())
                        .or_insert(resolved);
                    if canonical_uuid != node.uuid {
                        duplicate_pairs.push((node.uuid.clone(), canonical_uuid));
                    }
                }
            }
        }
    }

    // --- Build the compressed directed map ------------------------------------
    let mut union_pairs: Vec<(String, String)> = Vec::new();
    for uuid_map in &per_episode_uuid_maps {
        for (k, v) in uuid_map {
            union_pairs.push((k.clone(), v.clone()));
        }
    }
    union_pairs.extend(duplicate_pairs);

    let compressed_map = build_directed_uuid_map(&union_pairs);

    // --- Project resolved nodes per episode through the canonical map ---------
    let mut nodes_by_episode: HashMap<String, Vec<EntityNode>> = HashMap::new();
    for (episode_uuid, resolved_nodes) in &episode_resolutions {
        let mut deduped_nodes: Vec<EntityNode> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for node in resolved_nodes {
            let canonical_uuid = compressed_map
                .get(&node.uuid)
                .cloned()
                .unwrap_or_else(|| node.uuid.clone());
            if !seen.insert(canonical_uuid.clone()) {
                continue;
            }
            let canonical_node = canonical_nodes
                .get(&canonical_uuid)
                .cloned()
                .unwrap_or_else(|| {
                    // Upstream logs an error and falls back to the pre-map node.
                    tracing::error!(
                        canonical_uuid = %canonical_uuid,
                        fallback_uuid = %node.uuid,
                        "canonical node missing during batch dedupe; falling back"
                    );
                    node.clone()
                });
            deduped_nodes.push(canonical_node);
        }
        nodes_by_episode.insert(episode_uuid.clone(), deduped_nodes);
    }

    Ok((nodes_by_episode, compressed_map))
}

// ---------------------------------------------------------------------------
// dedupe_edges_bulk
// ---------------------------------------------------------------------------

/// Cross-episode edge dedup.
///
/// Port of upstream `dedupe_edges_bulk` (bulk_utils.py:489-581). Embeds every
/// extracted edge's fact, then for each edge collects same-`(source, target)`
/// candidates that either share a word with its fact (approximate BM25) OR have
/// cosine fact-embedding similarity `>= 0.6`. Each `(episode, edge, candidates)`
/// is resolved via `resolve_extracted_edge` in parallel; the resulting duplicate
/// pairs feed `compress_uuid_map` (undirected, smallest-uuid root) and the per-
/// episode edge lists are projected through it.
pub async fn dedupe_edges_bulk(
    clients: &Clients,
    mut extracted_edges: Vec<Vec<EntityEdge>>,
    episode_tuples: &[(EpisodicNode, Vec<EpisodicNode>)],
) -> Result<HashMap<String, Vec<EntityEdge>>, ChronicleError> {
    // Embed all facts that are missing an embedding (upstream
    // create_entity_edge_embeddings per episode list).
    let mut missing: Vec<(usize, usize)> = Vec::new();
    let mut facts: Vec<String> = Vec::new();
    for (i, edges) in extracted_edges.iter().enumerate() {
        for (j, edge) in edges.iter().enumerate() {
            if edge.fact_embedding.is_none() {
                missing.push((i, j));
                facts.push(edge.fact.clone());
            }
        }
    }
    if !facts.is_empty() {
        let embeddings = clients.embedder.create_batch(&facts).await?;
        for ((i, j), emb) in missing.into_iter().zip(embeddings) {
            extracted_edges[i][j].fact_embedding = Some(emb);
        }
    }

    // Flat pool of all edges (used as both candidate source and uuid lookup).
    let all_edges: Vec<EntityEdge> = extracted_edges.iter().flatten().cloned().collect();

    // Build (episode, edge, candidates) dedupe tuples.
    let mut dedupe_tuples: Vec<(EpisodicNode, EntityEdge, Vec<EntityEdge>)> = Vec::new();
    for (i, edges_i) in extracted_edges.iter().enumerate() {
        for edge in edges_i {
            let mut candidates: Vec<EntityEdge> = Vec::new();
            for existing_edge in &all_edges {
                if edge.uuid == existing_edge.uuid {
                    continue;
                }
                // Same endpoints required.
                if edge.source_node_uuid != existing_edge.source_node_uuid
                    || edge.target_node_uuid != existing_edge.target_node_uuid
                {
                    continue;
                }

                // Approximate BM25 via word overlap (faster, wider net than BM25).
                let edge_words: HashSet<String> = edge
                    .fact
                    .to_lowercase()
                    .split_whitespace()
                    .map(String::from)
                    .collect();
                let existing_words: HashSet<String> = existing_edge
                    .fact
                    .to_lowercase()
                    .split_whitespace()
                    .map(String::from)
                    .collect();
                let has_overlap = !edge_words.is_disjoint(&existing_words);
                if has_overlap {
                    candidates.push(existing_edge.clone());
                    continue;
                }

                // Semantic similarity fallback (cosine on L2-normalized facts).
                let similarity = cosine_l2(
                    edge.fact_embedding.as_deref().unwrap_or(&[]),
                    existing_edge.fact_embedding.as_deref().unwrap_or(&[]),
                );
                if similarity >= EDGE_DEDUP_MIN_SCORE {
                    candidates.push(existing_edge.clone());
                }
            }
            dedupe_tuples.push((episode_tuples[i].0.clone(), edge.clone(), candidates));
        }
    }

    // Resolve each edge against its candidates (parallel). Candidates serve as
    // both the related (duplicate) and existing (invalidation) pool upstream.
    let mut handles = Vec::with_capacity(dedupe_tuples.len());
    for (episode, edge, candidates) in &dedupe_tuples {
        let clients = clients.clone();
        let episode = episode.clone();
        let edge = edge.clone();
        let candidates = candidates.clone();
        handles.push(tokio::spawn(async move {
            resolve_extracted_edge(&clients, edge, candidates.clone(), candidates, &episode).await
        }));
    }

    let mut duplicate_pairs: Vec<(String, String)> = Vec::new();
    for (idx, handle) in handles.into_iter().enumerate() {
        let (_resolved, duplicates) = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("resolve_extracted_edge task panicked: {e}"))
        })??;
        let edge = &dedupe_tuples[idx].1;
        for duplicate in duplicates {
            duplicate_pairs.push((edge.uuid.clone(), duplicate.uuid.clone()));
        }
    }

    // Compress to smallest-uuid canonical and project per episode.
    let compressed_map = compress_uuid_map(&duplicate_pairs);

    let edge_uuid_map: HashMap<String, EntityEdge> =
        all_edges.into_iter().map(|e| (e.uuid.clone(), e)).collect();

    let mut edges_by_episode: HashMap<String, Vec<EntityEdge>> = HashMap::new();
    for (i, edges) in extracted_edges.iter().enumerate() {
        let episode_uuid = episode_tuples[i].0.uuid.clone();
        let mapped: Vec<EntityEdge> = edges
            .iter()
            .map(|edge| {
                let canonical_uuid = compressed_map
                    .get(&edge.uuid)
                    .cloned()
                    .unwrap_or_else(|| edge.uuid.clone());
                edge_uuid_map
                    .get(&canonical_uuid)
                    .cloned()
                    .unwrap_or_else(|| edge.clone())
            })
            .collect();
        edges_by_episode.insert(episode_uuid, mapped);
    }

    Ok(edges_by_episode)
}

/// Cosine similarity over L2-normalized vectors (upstream `np.dot(normalize_l2(a),
/// normalize_l2(b))`). Returns 0.0 for empty/zero/mismatched inputs.
fn cosine_l2(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| (x / na) * (y / nb)).sum()
}

// ---------------------------------------------------------------------------
// add_nodes_and_edges_bulk
// ---------------------------------------------------------------------------

/// Persist all bulk artifacts: embed missing node names + edge facts, then save
/// episodes, entity nodes, entity edges and episodic edges.
///
/// Port of upstream `add_nodes_and_edges_bulk` (bulk_utils.py:128-260).
///
/// **Atomicity (closed Phase 4):** upstream wraps the four saves in a single
/// write transaction. We embed missing node names + edge facts first, then call
/// the transactional [`crate::driver::BulkSaveOps::save_all`], so a mid-batch
/// failure rolls back cleanly on a transactional backend (Neo4j). The in-memory
/// `FakeDriver` inherits the default sequential save (nothing can partially
/// fail).
pub async fn add_nodes_and_edges_bulk(
    clients: &Clients,
    episodes: &[EpisodicNode],
    episodic_edges: &[EpisodicEdge],
    nodes: &mut [EntityNode],
    edges: &mut [EntityEdge],
) -> Result<(), ChronicleError> {
    embed_missing_node_names(clients, nodes).await?;
    embed_missing_edge_facts(clients, edges).await?;

    clients
        .driver
        .save_all(episodes, episodic_edges, nodes, edges)
        .await?;

    Ok(())
}

async fn embed_missing_node_names(
    clients: &Clients,
    nodes: &mut [EntityNode],
) -> Result<(), ChronicleError> {
    let missing: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name_embedding.is_none())
        .map(|(i, _)| i)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let names: Vec<String> = missing.iter().map(|&i| nodes[i].name.clone()).collect();
    let embeddings = clients.embedder.create_batch(&names).await?;
    for (&i, emb) in missing.iter().zip(embeddings) {
        nodes[i].name_embedding = Some(emb);
    }
    Ok(())
}

async fn embed_missing_edge_facts(
    clients: &Clients,
    edges: &mut [EntityEdge],
) -> Result<(), ChronicleError> {
    let missing: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.fact_embedding.is_none())
        .map(|(i, _)| i)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let facts: Vec<String> = missing.iter().map(|&i| edges[i].fact.clone()).collect();
    let embeddings = clients.embedder.create_batch(&facts).await?;
    for (&i, emb) in missing.iter().zip(embeddings) {
        edges[i].fact_embedding = Some(emb);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// add_episode_bulk
// ---------------------------------------------------------------------------

/// Ingest multiple episodes in one batch with cross-episode entity/edge dedup.
///
/// Port of upstream `Graphiti.add_episode_bulk` (graphiti.py:1230-1488) plus the
/// `_extract_and_dedupe_nodes_bulk` / `_resolve_nodes_and_edges_bulk` helpers.
///
/// Pipeline:
/// 1. Materialize episodes (look up by uuid when `RawEpisode.uuid` is `Some`,
///    else mint a fresh `EpisodicNode`) and save them.
/// 2. `retrieve_previous_episodes_bulk` for context.
/// 3. `extract_nodes_and_edges_bulk` (parallel per-episode extract).
/// 4. `dedupe_nodes_bulk` → `(nodes_by_episode, uuid_map)`.
/// 5. Build MENTIONS episodic edges per episode's deduped nodes.
/// 6. Remap extracted edge pointers through `uuid_map`, then `dedupe_edges_bulk`.
/// 7. Resolve nodes (hydrate summaries) + edges per episode against the graph;
///    remap episodic-edge pointers through the final node map.
/// 8. `add_nodes_and_edges_bulk` (single save).
/// 9. Optional saga association: sort episodes by `valid_at`, chain
///    `NEXT_EPISODE` + `HAS_EPISODE`, update saga first/last episode pointers.
///
/// Communities are never updated (upstream — the result's `communities` /
/// `community_edges` are always empty).
pub async fn add_episode_bulk(
    clients: &Clients,
    raw_episodes: Vec<RawEpisode>,
    group_id: &str,
    saga: Option<&str>,
    entity_types: Option<&Value>,
    custom_extraction_instructions: Option<&str>,
) -> Result<AddBulkEpisodeResults, ChronicleError> {
    let now = utc_now();

    // 1. Materialize + save episodes.
    let mut episodes: Vec<EpisodicNode> = Vec::with_capacity(raw_episodes.len());
    for raw in &raw_episodes {
        let episode = match &raw.uuid {
            Some(uuid) => clients
                .driver
                .get_episode(uuid)
                .await?
                .ok_or_else(|| ChronicleError::EpisodeNotFound { uuid: uuid.clone() })?,
            None => EpisodicNode::new(
                raw.name.clone(),
                group_id.to_string(),
                raw.source,
                raw.source_description.clone(),
                raw.content.clone(),
                now,
                raw.reference_time,
            ),
        };
        episodes.push(episode);
    }
    for episode in &episodes {
        clients.driver.save_episode(episode).await?;
    }

    // 2. Previous-episode context.
    let episode_context = retrieve_previous_episodes_bulk(clients, &episodes).await?;

    // 3. Extract nodes + edges in bulk.
    let (extracted_nodes_bulk, extracted_edges_bulk) = extract_nodes_and_edges_bulk(
        clients,
        &episode_context,
        entity_types,
        custom_extraction_instructions,
    )
    .await?;

    // 4. Dedupe nodes across the batch.
    let (nodes_by_episode, uuid_map) = dedupe_nodes_bulk(
        clients,
        extracted_nodes_bulk,
        &episode_context,
        entity_types,
    )
    .await?;

    // 5. Build MENTIONS episodic edges per episode.
    let mut episodic_edges: Vec<EpisodicEdge> = Vec::new();
    for (episode_uuid, nodes) in &nodes_by_episode {
        for node in nodes {
            episodic_edges.push(EpisodicEdge::new(
                episode_uuid.clone(),
                node.uuid.clone(),
                group_id.to_string(),
                now,
            ));
        }
    }

    // 6. Remap extracted edge pointers through the node uuid_map, then dedupe.
    let mut remapped_edges_bulk = extracted_edges_bulk;
    for edges in &mut remapped_edges_bulk {
        resolve_edge_pointers(edges, &uuid_map);
    }
    let edges_by_episode =
        dedupe_edges_bulk(clients, remapped_edges_bulk, &episode_context).await?;

    // 7. Final reconcile against the graph: resolve nodes (+ summaries) and edges
    //    per episode, accumulating a final node uuid_map.
    let (final_hydrated_nodes, resolved_edges, final_uuid_map) = resolve_nodes_and_edges_bulk(
        clients,
        &nodes_by_episode,
        &edges_by_episode,
        &episode_context,
    )
    .await?;

    // Remap episodic-edge pointers through the final node map.
    resolve_episodic_edge_pointers(&mut episodic_edges, &final_uuid_map);

    // 8. Persist everything.
    let mut final_nodes = final_hydrated_nodes;
    let mut final_edges = resolved_edges;
    add_nodes_and_edges_bulk(
        clients,
        &episodes,
        &episodic_edges,
        &mut final_nodes,
        &mut final_edges,
    )
    .await?;

    // 9. Saga association.
    if let Some(saga_name) = saga {
        associate_saga(clients, saga_name, group_id, &episodes, now).await?;
    }

    Ok(AddBulkEpisodeResults {
        episodes,
        episodic_edges,
        nodes: final_nodes,
        edges: final_edges,
        communities: Vec::new(),
        community_edges: Vec::new(),
    })
}

/// Resolve deduped nodes (with summary hydration) and edges per episode against
/// the live graph, returning unique hydrated nodes, resolved+invalidated edges,
/// and the node uuid_map produced by the final resolution pass.
///
/// Port of upstream `_resolve_nodes_and_edges_bulk` (graphiti.py:815-924).
async fn resolve_nodes_and_edges_bulk(
    clients: &Clients,
    nodes_by_episode: &HashMap<String, Vec<EntityNode>>,
    edges_by_episode: &HashMap<String, Vec<EntityEdge>>,
    episode_context: &[(EpisodicNode, Vec<EpisodicNode>)],
) -> Result<(Vec<EntityNode>, Vec<EntityEdge>, HashMap<String, String>), ChronicleError> {
    // Unique node list per episode (dedup by uuid across the whole batch, in
    // episode order — upstream nodes_by_episode_unique).
    let mut nodes_by_episode_unique: HashMap<String, Vec<EntityNode>> = HashMap::new();
    let mut seen_uuids: HashSet<String> = HashSet::new();
    for (episode, _) in episode_context {
        let mut unique: Vec<EntityNode> = Vec::new();
        if let Some(nodes) = nodes_by_episode.get(&episode.uuid) {
            for node in nodes {
                if seen_uuids.insert(node.uuid.clone()) {
                    unique.push(node.clone());
                }
            }
        }
        nodes_by_episode_unique.insert(episode.uuid.clone(), unique);
    }

    // Resolve nodes per episode against the graph (parallel), accumulating the
    // resolution uuid_map.
    let mut node_handles = Vec::with_capacity(episode_context.len());
    for (episode, previous) in episode_context {
        let clients = clients.clone();
        let episode = episode.clone();
        let previous = previous.clone();
        let unique = nodes_by_episode_unique
            .get(&episode.uuid)
            .cloned()
            .unwrap_or_default();
        node_handles.push(tokio::spawn(async move {
            let outcome = resolve_extracted_nodes(&clients, unique, &episode, &previous).await?;
            // Hydrate summaries from the current episode (upstream
            // extract_attributes_from_nodes; we reuse hydrate_node_summaries).
            let hydrated =
                hydrate_node_summaries(&clients, outcome.nodes, &episode, &previous).await?;
            Ok::<(Vec<EntityNode>, HashMap<String, String>), ChronicleError>((
                hydrated,
                outcome.uuid_map,
            ))
        }));
    }

    let mut final_hydrated_nodes: Vec<EntityNode> = Vec::new();
    let mut uuid_map: HashMap<String, String> = HashMap::new();
    for handle in node_handles {
        let (hydrated, map) = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("resolve nodes bulk task panicked: {e}"))
        })??;
        final_hydrated_nodes.extend(hydrated);
        uuid_map.extend(map);
    }

    // Resolve edges per episode with the updated node pointers (parallel, dedup
    // edge uuids across the batch).
    let mut edges_uuid_set: HashSet<String> = HashSet::new();
    let mut edge_handles = Vec::with_capacity(episode_context.len());
    for (episode, _) in episode_context {
        let mut edges: Vec<EntityEdge> = edges_by_episode
            .get(&episode.uuid)
            .cloned()
            .unwrap_or_default();
        resolve_edge_pointers(&mut edges, &uuid_map);
        let unique: Vec<EntityEdge> = edges
            .into_iter()
            .filter(|e| edges_uuid_set.insert(e.uuid.clone()))
            .collect();

        let clients = clients.clone();
        let episode = episode.clone();
        let nodes = final_hydrated_nodes.clone();
        edge_handles.push(tokio::spawn(async move {
            resolve_extracted_edges(&clients, unique, &episode, &nodes).await
        }));
    }

    let mut resolved_edges: Vec<EntityEdge> = Vec::new();
    let mut invalidated_edges: Vec<EntityEdge> = Vec::new();
    for handle in edge_handles {
        let outcome = handle.await.map_err(|e| {
            ChronicleError::InvalidInput(format!("resolve edges bulk task panicked: {e}"))
        })??;
        resolved_edges.extend(outcome.resolved_edges);
        invalidated_edges.extend(outcome.invalidated_edges);
    }

    resolved_edges.extend(invalidated_edges);
    Ok((final_hydrated_nodes, resolved_edges, uuid_map))
}

/// Associate all episodes with a saga: get-or-create by name, chain `NEXT_EPISODE`
/// edges in `valid_at` order, save a `HAS_EPISODE` edge per episode, and update
/// the saga's first/last episode pointers.
///
/// Port of upstream `add_episode_bulk` saga block (graphiti.py:1410-1459) +
/// `_get_or_create_saga` (346-392) + `_saga_get_previous_episode_uuid` (394-420).
async fn associate_saga(
    clients: &Clients,
    saga_name: &str,
    group_id: &str,
    episodes: &[EpisodicNode],
    now: DateTime<Utc>,
) -> Result<(), ChronicleError> {
    // Get-or-create the saga, anchoring a fresh saga's created_at to the earliest
    // episode reference time in the batch.
    let saga_created_at = episodes.iter().map(|e| e.valid_at).min().unwrap_or(now);
    let mut saga_node = match clients.driver.get_saga_by_name(saga_name, group_id).await? {
        Some(existing) => existing,
        None => {
            let saga = SagaNode::new(saga_name.to_string(), group_id.to_string(), saga_created_at);
            clients.driver.save_saga_node(&saga).await?;
            saga
        }
    };

    // Chain in valid_at order (stable for equal valid_at — preserves input order).
    let mut sorted_episodes: Vec<&EpisodicNode> = episodes.iter().collect();
    sorted_episodes.sort_by_key(|a| a.valid_at);

    // Most-recent episode already in the saga (for chaining onto an existing run).
    let mut previous_episode_uuid = clients
        .driver
        .saga_previous_episode_uuid(&saga_node.uuid, "")
        .await?;

    for episode in &sorted_episodes {
        if let Some(prev_uuid) = &previous_episode_uuid {
            let next_edge = NextEpisodeEdge::new(
                prev_uuid.clone(),
                episode.uuid.clone(),
                group_id.to_string(),
                now,
            );
            clients.driver.save_next_episode_edge(&next_edge).await?;
        }

        let has_edge = HasEpisodeEdge::new(
            saga_node.uuid.clone(),
            episode.uuid.clone(),
            group_id.to_string(),
            now,
        );
        clients.driver.save_has_episode_edge(&has_edge).await?;

        previous_episode_uuid = Some(episode.uuid.clone());
    }

    if let Some(first) = sorted_episodes.first()
        && saga_node.first_episode_uuid.is_none()
    {
        saga_node.first_episode_uuid = Some(first.uuid.clone());
    }
    if let Some(last) = sorted_episodes.last() {
        saga_node.last_episode_uuid = Some(last.uuid.clone());
    }
    if !sorted_episodes.is_empty() {
        clients.driver.save_saga_node(&saga_node).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    #[test]
    fn directed_uuid_map_preserves_direction_and_compresses_paths() {
        // a -> b, b -> c  =>  a -> c, b -> c (canonical = chain target, not smallest).
        let pairs = vec![pair("a", "b"), pair("b", "c")];
        let map = build_directed_uuid_map(&pairs);
        assert_eq!(map.get("a").unwrap(), "c");
        assert_eq!(map.get("b").unwrap(), "c");
        assert_eq!(map.get("c").unwrap(), "c");
    }

    #[test]
    fn directed_uuid_map_respects_target_even_if_lexicographically_smaller() {
        // z -> a: direction must point z at a (target), not collapse by uuid order.
        let pairs = vec![pair("z", "a")];
        let map = build_directed_uuid_map(&pairs);
        assert_eq!(map.get("z").unwrap(), "a");
        assert_eq!(map.get("a").unwrap(), "a");
    }

    #[test]
    fn directed_uuid_map_alias_smaller_than_canonical() {
        // alias "a" maps to canonical "m"; "m" maps to canonical "z". Even though
        // "a" < "m" < "z", everything resolves to the ultimate directed target "z".
        let pairs = vec![pair("a", "m"), pair("m", "z")];
        let map = build_directed_uuid_map(&pairs);
        assert_eq!(map.get("a").unwrap(), "z");
        assert_eq!(map.get("m").unwrap(), "z");
        assert_eq!(map.get("z").unwrap(), "z");
    }

    #[test]
    fn compress_uuid_map_picks_smallest_root_undirected() {
        // 3 <-> 2 and 2 <-> 1  =>  all map to "1" (lexicographically smallest).
        let pairs = vec![pair("3", "2"), pair("2", "1")];
        let map = compress_uuid_map(&pairs);
        assert_eq!(map.get("1").unwrap(), "1");
        assert_eq!(map.get("2").unwrap(), "1");
        assert_eq!(map.get("3").unwrap(), "1");
    }

    #[test]
    fn compress_uuid_map_is_direction_insensitive() {
        // Reversing the pair order yields the same smallest-root result.
        let map = compress_uuid_map(&[pair("a", "z"), pair("z", "m")]);
        assert_eq!(map.get("a").unwrap(), "a");
        assert_eq!(map.get("m").unwrap(), "a");
        assert_eq!(map.get("z").unwrap(), "a");
    }

    #[test]
    fn resolve_edge_pointers_remaps_known_uuids_only() {
        let mut map = HashMap::new();
        map.insert("ex-src".to_string(), "canon-src".to_string());
        let mut edge = EntityEdge::new(
            "ex-src".into(),
            "unknown-tgt".into(),
            "R".into(),
            "f".into(),
            "g".into(),
        );
        resolve_edge_pointers(std::slice::from_mut(&mut edge), &map);
        assert_eq!(edge.source_node_uuid, "canon-src");
        assert_eq!(edge.target_node_uuid, "unknown-tgt");
    }
}
