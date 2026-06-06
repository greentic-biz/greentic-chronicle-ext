// Ported from graphiti_core/search/search.py::community_search @ 34f56e65.
//
// Phase-4: community scope. Differences vs the other scopes (verified against
// search.py lines 763-870):
//   - ALWAYS runs BOTH community_fulltext_search AND community_similarity_search,
//     regardless of the config's `search_methods` list. This is an UPSTREAM QUIRK:
//     unlike edge/node scopes (which build their task list from `search_methods`),
//     community_search hardcodes the two-method `semaphore_gather` and never
//     inspects `config.search_methods` when deciding what to run. The
//     `search_methods` list still matters for the top-level embed decision
//     (cosine_similarity → embed the query) but NOT for method selection here.
//     Replicated bug-for-bug.
//   - Both methods run at candidate_limit = 2 * limit, in parallel.
//   - Rerankers (community-only set):
//       rrf          → rrf(uuid_lists, 1, reranker_min_score).
//       mmr          → get_embeddings_for_communities(map values) →
//                      maximal_marginal_relevance over (uuid, embedding) pairs.
//       cross_encoder→ rank community.name over ALL map values (NO [:limit]
//                      pre-truncate), keep score >= reranker_min_score.
//     There is NO node_distance / episode_mentions for communities, and NO BFS
//     method (CommunitySearchMethod has only cosine + bm25).
//   - cross_encoder builds a `name → uuid` map over the search results
//     (first-seen NAME order; duplicate names collapse), mirroring upstream
//     `{node.name: node.uuid for ...}`.
//   - Slice [:limit] both lists (line 870).

use std::collections::HashMap;

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::errors::ChronicleError;
use crate::types::CommunityNode;

use super::config::{CommunityReranker, CommunitySearchConfig};
use super::rerank::maximal_marginal_relevance;
use super::rrf::rrf;

/// Insertion-ordered uuid→community map mirroring upstream
/// `{community.uuid: community for result in results for community in result}`.
/// Returns the first-seen uuid order alongside the map (used to drive MMR
/// candidate ordering, like the edge/node scopes).
fn ordered_community_map(
    result_lists: &[Vec<CommunityNode>],
) -> (Vec<String>, HashMap<String, CommunityNode>) {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, CommunityNode> = HashMap::new();
    for list in result_lists {
        for community in list {
            if !map.contains_key(&community.uuid) {
                order.push(community.uuid.clone());
            }
            map.insert(community.uuid.clone(), community.clone());
        }
    }
    (order, map)
}

/// Hybrid community search with the Phase-4 reranker dispatch.
///
/// Upstream: `graphiti_core/search/search.py::community_search`. `config` `None`
/// → `(vec![], vec![])`.
#[allow(clippy::too_many_arguments)]
pub async fn community_search(
    driver: &dyn GraphDriver,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    query_vector: &[f32],
    group_ids: &[String],
    config: Option<&CommunitySearchConfig>,
    limit: usize,
    reranker_min_score: f64,
) -> Result<(Vec<CommunityNode>, Vec<f64>), ChronicleError> {
    let Some(config) = config else {
        return Ok((Vec::new(), Vec::new()));
    };

    let candidate_limit = 2 * limit;

    // UPSTREAM QUIRK: community_search ALWAYS runs BOTH fulltext + similarity,
    // regardless of `config.search_methods`. Run them in parallel.
    let (fulltext_res, similarity_res) = tokio::join!(
        driver.community_fulltext_search(query, group_ids, candidate_limit),
        driver.community_similarity_search(
            query_vector,
            group_ids,
            candidate_limit,
            config.sim_min_score,
        ),
    );
    let fulltext = fulltext_res?;
    let similarity = similarity_res?;
    let result_lists = vec![fulltext, similarity];

    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|c| c.uuid.clone()).collect())
        .collect();
    let (uuid_order, community_uuid_map) = ordered_community_map(&result_lists);

    let (reranked_uuids, community_scores): (Vec<String>, Vec<f64>) = match config.reranker {
        CommunityReranker::Rrf => rrf(&uuid_lists, 1, reranker_min_score),
        CommunityReranker::Mmr => {
            let emb_map = driver.get_embeddings_for_communities(&uuid_order).await?;
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
        CommunityReranker::CrossEncoder => {
            let cross_encoder = cross_encoder.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "community cross_encoder reranker requested but no CrossEncoderClient was provided"
                        .to_string(),
                )
            })?;
            // name → uuid over ALL map values (NO [:limit] pre-truncate).
            // First-seen NAME order; duplicate names collapse (mirrors upstream
            // `{node.name: node.uuid for ...}`).
            let mut name_order: Vec<String> = Vec::new();
            let mut name_to_uuid: HashMap<String, String> = HashMap::new();
            for u in &uuid_order {
                if let Some(community) = community_uuid_map.get(u) {
                    if !name_to_uuid.contains_key(&community.name) {
                        name_order.push(community.name.clone());
                    }
                    name_to_uuid.insert(community.name.clone(), community.uuid.clone());
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
    };

    let mut reranked_communities: Vec<CommunityNode> = reranked_uuids
        .iter()
        .filter_map(|u| community_uuid_map.get(u).cloned())
        .collect();
    let mut community_scores = community_scores;

    reranked_communities.truncate(limit);
    community_scores.truncate(limit);

    Ok((reranked_communities, community_scores))
}
