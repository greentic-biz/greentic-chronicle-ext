// Ported from graphiti_core/search/search.py::node_search @ 34f56e65.
//
// Phase-2: mirrors edge_search structure for the node scope. Differences vs
// edge_search (verified against search.py lines 463-660):
//   - methods: bm25 (node_fulltext), cosine (node_similarity), bfs (node_bfs).
//   - BFS self-seed collects node.uuid (not source_node_uuid) across initial
//     result sets (line 540-559).
//   - cross_encoder ranks node.name over ALL uuid-map values — NO [:limit]
//     pre-truncate, UNLIKE edges (line 599).
//   - episode_mentions calls episode_mentions_rerank(driver, uuid_LISTS,
//     min_score) — a DB-count-based reranker (line 617-625), NOT an in-memory
//     post-sort (that post-sort is edge-only).
//   - node_distance: rrf-presort → seeded uuids → node_distance_rerank
//     (line 626-648).
//   - Slice [:limit] both lists (line 660).

use std::collections::HashMap;

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::errors::ChronicleError;
use crate::types::EntityNode;

use super::config::{NodeReranker, NodeSearchConfig, NodeSearchMethod};
use super::filters::SearchFilters;
use super::rerank::{episode_mentions_rerank, maximal_marginal_relevance, node_distance_rerank};
use super::rrf::rrf;

/// Insertion-ordered uuid→node map mirroring upstream
/// `{node.uuid: node for result in results for node in result}`.
fn ordered_node_map(
    result_lists: &[Vec<EntityNode>],
) -> (Vec<String>, HashMap<String, EntityNode>) {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, EntityNode> = HashMap::new();
    for list in result_lists {
        for node in list {
            if !map.contains_key(&node.uuid) {
                order.push(node.uuid.clone());
            }
            map.insert(node.uuid.clone(), node.clone());
        }
    }
    (order, map)
}

/// Hybrid node search with the full Phase-2 reranker dispatch.
///
/// Upstream: `graphiti_core/search/search.py::node_search`. `config` `None` →
/// `(vec![], vec![])`.
#[allow(clippy::too_many_arguments)]
pub async fn node_search(
    driver: &dyn GraphDriver,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    query_vector: &[f32],
    group_ids: &[String],
    config: Option<&NodeSearchConfig>,
    filters: &SearchFilters,
    center_node_uuid: Option<&str>,
    bfs_origin_node_uuids: Option<&[String]>,
    limit: usize,
    reranker_min_score: f64,
) -> Result<(Vec<EntityNode>, Vec<f64>), ChronicleError> {
    let Some(config) = config else {
        return Ok((Vec::new(), Vec::new()));
    };

    let candidate_limit = 2 * limit;

    let want_bm25 = config.search_methods.contains(&NodeSearchMethod::Bm25);
    let want_cosine = config
        .search_methods
        .contains(&NodeSearchMethod::CosineSimilarity);
    let want_bfs = config
        .search_methods
        .contains(&NodeSearchMethod::BreadthFirstSearch);

    let mut result_lists: Vec<Vec<EntityNode>> = Vec::new();

    if want_bm25 {
        result_lists.push(
            driver
                .node_fulltext_search(query, filters, group_ids, candidate_limit)
                .await?,
        );
    }
    if want_cosine {
        result_lists.push(
            driver
                .node_similarity_search(
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
                .node_bfs_search(
                    bfs_origin_node_uuids.unwrap_or(&[]),
                    filters,
                    config.bfs_max_depth,
                    group_ids,
                    candidate_limit,
                )
                .await?,
        );
    }

    // BFS lazy self-seed (upstream lines 540-559): origins None → seed BFS from
    // the node uuids collected across the initial result sets.
    if want_bfs && bfs_origin_node_uuids.is_none() {
        let origin_node_uuids: Vec<String> = result_lists
            .iter()
            .flatten()
            .map(|n| n.uuid.clone())
            .collect();
        result_lists.push(
            driver
                .node_bfs_search(
                    &origin_node_uuids,
                    filters,
                    config.bfs_max_depth,
                    group_ids,
                    candidate_limit,
                )
                .await?,
        );
    }

    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|n| n.uuid.clone()).collect())
        .collect();
    let (uuid_order, node_uuid_map) = ordered_node_map(&result_lists);

    let (reranked_uuids, node_scores): (Vec<String>, Vec<f64>) = match config.reranker {
        NodeReranker::Rrf => rrf(&uuid_lists, 1, reranker_min_score),
        NodeReranker::Mmr => {
            let emb_map = driver.get_embeddings_for_nodes(&uuid_order).await?;
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
        NodeReranker::CrossEncoder => {
            let cross_encoder = cross_encoder.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "node cross_encoder reranker requested but no CrossEncoderClient was provided"
                        .to_string(),
                )
            })?;
            // name → uuid over ALL map values (NO [:limit] pre-truncate — edges
            // differ here). First-seen NAME order; duplicate names collapse.
            let mut name_order: Vec<String> = Vec::new();
            let mut name_to_uuid: HashMap<String, String> = HashMap::new();
            for u in &uuid_order {
                if let Some(node) = node_uuid_map.get(u) {
                    if !name_to_uuid.contains_key(&node.name) {
                        name_order.push(node.name.clone());
                    }
                    name_to_uuid.insert(node.name.clone(), node.uuid.clone());
                }
            }
            let ranked = cross_encoder.rank(query, &name_order).await?;
            let mut uuids = Vec::new();
            let mut scores = Vec::new();
            for (name, score) in ranked {
                if score >= reranker_min_score
                    && let Some(u) = name_to_uuid.get(&name)
                {
                    uuids.push(u.clone());
                    scores.push(score);
                }
            }
            (uuids, scores)
        }
        NodeReranker::EpisodeMentions => {
            // DB-count-based reranker over the per-method uuid lists (NOT an
            // in-memory post-sort — that variant is edge-only).
            episode_mentions_rerank(driver, &uuid_lists, reranker_min_score).await?
        }
        NodeReranker::NodeDistance => {
            let center = center_node_uuid.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "No center node provided for Node Distance reranker".to_string(),
                )
            })?;
            // rrf-presort → seeded uuids; node_distance_rerank does NO internal
            // presort (carried-forward Task 3 note — dispatch presorts).
            let (seeded_uuids, _) = rrf(&uuid_lists, 1, reranker_min_score);
            node_distance_rerank(driver, &seeded_uuids, center, reranker_min_score).await?
        }
    };

    let mut reranked_nodes: Vec<EntityNode> = reranked_uuids
        .iter()
        .filter_map(|u| node_uuid_map.get(u).cloned())
        .collect();
    let mut node_scores = node_scores;

    reranked_nodes.truncate(limit);
    node_scores.truncate(limit);

    Ok((reranked_nodes, node_scores))
}
