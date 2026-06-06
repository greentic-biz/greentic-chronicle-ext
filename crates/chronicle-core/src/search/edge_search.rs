// Ported from graphiti_core/search/search.py::edge_search @ 34f56e65.
//
// Phase-2 full rewrite: all five rerankers (RRF / MMR / NodeDistance /
// EpisodeMentions / CrossEncoder), BFS self-seed, SearchFilters threading.
//
// Upstream conventions verified against search.py lines 253-460:
//   - config None → return ([], []).
//   - Each enabled method fetches 2*limit candidates (lines 286/297/309).
//   - BFS self-seed (lines 332-353): when BFS is a configured method AND
//     bfs_origin_node_uuids is None, run an extra edge_bfs_search seeded from
//     the source_node_uuids collected across the initial (non-self-seed) result
//     sets, appended as another result list.
//   - edge_uuid_map = {edge.uuid: edge for result in results for edge in result}
//     (line 355). Python dict preserves FIRST-SEEN insertion order, which the
//     cross_encoder branch relies on via `list(edge_uuid_map.values())[:limit]`.
//     We replicate this with an explicit insertion-ordered Vec + a HashMap for
//     O(1) lookup — NOT HashMap iteration order.
//   - Reranker dispatch (lines 368-445):
//       rrf | episode_mentions → rrf(uuid_lists); episode_mentions then RE-SORTS
//         the reranked edges descending by edge.episodes.len() (line 449-450).
//       mmr → get_embeddings_for_edges → maximal_marginal_relevance.
//       cross_encoder → fact→uuid map over the FIRST `limit` map values; rank;
//         keep score >= reranker_min_score.
//       node_distance → center required (else InvalidInput); rrf-presort; group
//         edge uuids by source_node_uuid; node_distance_rerank over the source
//         uuids; expand back to edge uuids in returned node order.
//   - Slice [:limit] both lists (line 460).

use std::collections::HashMap;

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;
use crate::types::EntityEdge;

use super::config::{EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, SearchConfig};
use super::filters::SearchFilters;
use super::rerank::maximal_marginal_relevance;
use super::rrf::rrf;

/// Build an insertion-ordered uuid→edge map mirroring upstream
/// `{edge.uuid: edge for result in results for edge in result}`.
///
/// Python dicts preserve first-seen insertion order; later duplicates overwrite
/// the value but do NOT change the key's position. We return both an ordered
/// Vec of uuids (first-seen order) and a HashMap for O(1) lookup, so that
/// `ordered_values()[:limit]` semantics are reproducible.
fn ordered_edge_map(
    result_lists: &[Vec<EntityEdge>],
) -> (Vec<String>, HashMap<String, EntityEdge>) {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, EntityEdge> = HashMap::new();
    for list in result_lists {
        for edge in list {
            if !map.contains_key(&edge.uuid) {
                order.push(edge.uuid.clone());
            }
            // Later duplicate overwrites the value (matches Python dict), but the
            // key keeps its first-seen position in `order`.
            map.insert(edge.uuid.clone(), edge.clone());
        }
    }
    (order, map)
}

/// Hybrid edge search with the full Phase-2 reranker dispatch.
///
/// Upstream: `graphiti_core/search/search.py::edge_search`.
///
/// `config` `None` → `(vec![], vec![])`. `query_vector` is the pre-embedded
/// query (caller decides whether embedding is needed; for cosine/MMR it must be
/// the real vector, otherwise a zero/unused vector is acceptable). Returns the
/// reranked edges and their reranker scores, both sliced to `limit`.
#[allow(clippy::too_many_arguments)]
pub async fn edge_search(
    driver: &dyn GraphDriver,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    query_vector: &[f32],
    group_ids: &[String],
    config: Option<&EdgeSearchConfig>,
    filters: &SearchFilters,
    center_node_uuid: Option<&str>,
    bfs_origin_node_uuids: Option<&[String]>,
    limit: usize,
    reranker_min_score: f64,
) -> Result<(Vec<EntityEdge>, Vec<f64>), ChronicleError> {
    let Some(config) = config else {
        return Ok((Vec::new(), Vec::new()));
    };

    let candidate_limit = 2 * limit;

    let want_bm25 = config.search_methods.contains(&EdgeSearchMethod::Bm25);
    let want_cosine = config
        .search_methods
        .contains(&EdgeSearchMethod::CosineSimilarity);
    let want_bfs = config
        .search_methods
        .contains(&EdgeSearchMethod::BreadthFirstSearch);

    // Initial method fan-out (upstream runs these concurrently via
    // semaphore_gather; here we await sequentially to keep the dispatch readable
    // — order of result lists matches the upstream task-append order:
    // bm25, cosine, bfs).
    let mut result_lists: Vec<Vec<EntityEdge>> = Vec::new();

    if want_bm25 {
        result_lists.push(
            driver
                .edge_fulltext_search(query, filters, group_ids, candidate_limit)
                .await?,
        );
    }
    if want_cosine {
        result_lists.push(
            driver
                .edge_similarity_search(
                    query_vector,
                    filters,
                    group_ids,
                    candidate_limit,
                    config.sim_min_score,
                )
                .await?,
        );
    }
    if want_bfs {
        result_lists.push(
            driver
                .edge_bfs_search(
                    bfs_origin_node_uuids.unwrap_or(&[]),
                    config.bfs_max_depth,
                    filters,
                    group_ids,
                    candidate_limit,
                )
                .await?,
        );
    }

    // BFS lazy self-seed (upstream lines 332-353): origins None → seed BFS from
    // the source_node_uuids collected across the initial result sets.
    if want_bfs && bfs_origin_node_uuids.is_none() {
        let source_node_uuids: Vec<String> = result_lists
            .iter()
            .flatten()
            .map(|e| e.source_node_uuid.clone())
            .collect();
        result_lists.push(
            driver
                .edge_bfs_search(
                    &source_node_uuids,
                    config.bfs_max_depth,
                    filters,
                    group_ids,
                    candidate_limit,
                )
                .await?,
        );
    }

    let (uuid_order, edge_uuid_map) = ordered_edge_map(&result_lists);
    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|e| e.uuid.clone()).collect())
        .collect();

    // Reranker dispatch.
    let (reranked_uuids, edge_scores): (Vec<String>, Vec<f64>) = match config.reranker {
        EdgeReranker::Rrf | EdgeReranker::EpisodeMentions => {
            rrf(&uuid_lists, 1, reranker_min_score)
        }
        EdgeReranker::Mmr => {
            // Load fresh embeddings for ALL candidate edges (insertion order).
            let emb_map = driver.get_embeddings_for_edges(&uuid_order).await?;
            // Preserve insertion order for the candidate list passed to MMR.
            let candidates: Vec<(String, Vec<f32>)> = uuid_order
                .iter()
                .filter_map(|u| emb_map.get(u).map(|v| (u.clone(), v.clone())))
                .collect();
            maximal_marginal_relevance(
                query_vector,
                &candidates,
                config.mmr_lambda,
                reranker_min_score,
            )
        }
        EdgeReranker::CrossEncoder => {
            let cross_encoder = cross_encoder.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "edge cross_encoder reranker requested but no CrossEncoderClient was provided"
                        .to_string(),
                )
            })?;
            // fact → uuid over the FIRST `limit` map values (insertion order).
            // Build preserving first-seen FACT order (Python dict comprehension
            // over the truncated values; duplicate facts collapse to first-seen).
            let mut fact_order: Vec<String> = Vec::new();
            let mut fact_to_uuid: HashMap<String, String> = HashMap::new();
            for u in uuid_order.iter().take(limit) {
                if let Some(edge) = edge_uuid_map.get(u) {
                    if !fact_to_uuid.contains_key(&edge.fact) {
                        fact_order.push(edge.fact.clone());
                    }
                    fact_to_uuid.insert(edge.fact.clone(), edge.uuid.clone());
                }
            }
            let ranked = cross_encoder.rank(query, &fact_order).await?;
            let mut uuids = Vec::new();
            let mut scores = Vec::new();
            for (fact, score) in ranked {
                if score >= reranker_min_score
                    && let Some(u) = fact_to_uuid.get(&fact)
                {
                    uuids.push(u.clone());
                    scores.push(score);
                }
            }
            (uuids, scores)
        }
        EdgeReranker::NodeDistance => {
            let center = center_node_uuid.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "No center node provided for Node Distance reranker".to_string(),
                )
            })?;
            // rrf-presort the candidates (upstream seeds node_distance via rrf).
            let (sorted_uuids, _) = rrf(&uuid_lists, 1, reranker_min_score);

            // Group edge uuids by source_node_uuid, preserving first-seen source
            // order (defaultdict(list) over the rrf-sorted edges).
            let mut source_order: Vec<String> = Vec::new();
            let mut source_to_edges: HashMap<String, Vec<String>> = HashMap::new();
            for u in &sorted_uuids {
                if let Some(edge) = edge_uuid_map.get(u) {
                    if !source_to_edges.contains_key(&edge.source_node_uuid) {
                        source_order.push(edge.source_node_uuid.clone());
                    }
                    source_to_edges
                        .entry(edge.source_node_uuid.clone())
                        .or_default()
                        .push(edge.uuid.clone());
                }
            }

            // node_distance_rerank does NO internal presort — the dispatch layer
            // already rrf-presorted `source_order` (carried-forward Task 3 note).
            let (ranked_nodes, node_scores) = super::rerank::node_distance_rerank(
                driver,
                &source_order,
                center,
                reranker_min_score,
            )
            .await?;

            // Expand node order back to edge uuids.
            let mut uuids = Vec::new();
            for node_uuid in &ranked_nodes {
                if let Some(edges) = source_to_edges.get(node_uuid) {
                    uuids.extend(edges.iter().cloned());
                }
            }
            // UPSTREAM QUIRK (search.py line 460): the returned `edge_scores` are
            // the node-distance scores (one per SOURCE node), NOT aligned 1:1 with
            // the expanded edge list. Both `reranked_edges` and `edge_scores` are
            // sliced to `limit` independently. We return `node_scores` verbatim to
            // preserve this behaviour.
            (uuids, node_scores)
        }
    };

    let mut reranked_edges: Vec<EntityEdge> = reranked_uuids
        .iter()
        .filter_map(|u| edge_uuid_map.get(u).cloned())
        .collect();

    // episode_mentions: re-sort the reranked edges descending by episode count.
    let mut edge_scores = edge_scores;
    if config.reranker == EdgeReranker::EpisodeMentions {
        // Stable sort descending by episodes.len() (upstream list.sort is stable).
        reranked_edges.sort_by_key(|e| std::cmp::Reverse(e.episodes.len()));
        // The score list is no longer aligned to the resorted edges; upstream
        // also leaves edge_scores as the pre-sort rrf scores and slices both to
        // limit. We keep the rrf scores as-is (sliced below).
    }

    reranked_edges.truncate(limit);
    edge_scores.truncate(limit);

    Ok((reranked_edges, edge_scores))
}

/// Embed-and-search convenience for the edge scope only — used by the pipeline
/// (invalidation-candidate search) and the public facade `search()`.
///
/// Applies the R1 embed decision for the edge scope: the query is embedded iff
/// the edge config uses CosineSimilarity OR the MMR reranker; otherwise a
/// zero vector (unused) is passed. Returns edges only (drops reranker scores).
#[allow(clippy::too_many_arguments)]
pub async fn edge_search_simple(
    driver: &dyn GraphDriver,
    embedder: &dyn EmbedderClient,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    group_ids: &[String],
    config: &SearchConfig,
    filters: &SearchFilters,
) -> Result<Vec<EntityEdge>, ChronicleError> {
    let Some(edge_config) = &config.edge_config else {
        return Ok(Vec::new());
    };

    let needs_embedding = edge_config
        .search_methods
        .contains(&EdgeSearchMethod::CosineSimilarity)
        || edge_config.reranker == EdgeReranker::Mmr;

    let query_vector = if needs_embedding {
        embedder.create(&query.replace('\n', " ")).await?
    } else {
        vec![0.0_f32; embedder.embedding_dim()]
    };

    let (edges, _scores) = edge_search(
        driver,
        cross_encoder,
        query,
        &query_vector,
        group_ids,
        Some(edge_config),
        filters,
        None,
        None,
        config.limit,
        config.reranker_min_score,
    )
    .await?;

    Ok(edges)
}
