// Ported from graphiti_core/search/search.py::episode_search @ 34f56e65.
//
// Phase-2: episode scope. Only method = bm25 (episode_fulltext_search over the
// `episode_content` index). Rerankers (verified search.py lines 663-760):
//   - rrf → rrf(uuid_lists) (line 717-720).
//   - cross_encoder → rrf-presort, take FIRST `limit` rrf results, map
//     episode.content → uuid, rank, keep score >= reranker_min_score
//     (line 721-749).
// No similarity / MMR / node_distance / episode_mentions for episodes.
// `query_vector` is unused (upstream names it `_query_vector`).

use std::collections::HashMap;

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::errors::ChronicleError;
use crate::types::EpisodicNode;

use super::config::{EpisodeReranker, EpisodeSearchConfig};
use super::filters::SearchFilters;
use super::rrf::rrf;

/// Insertion-ordered uuid→episode map mirroring upstream
/// `{episode.uuid: episode for result in results for episode in result}`.
fn ordered_episode_map(result_lists: &[Vec<EpisodicNode>]) -> HashMap<String, EpisodicNode> {
    let mut map: HashMap<String, EpisodicNode> = HashMap::new();
    for list in result_lists {
        for ep in list {
            map.entry(ep.uuid.clone()).or_insert_with(|| ep.clone());
        }
    }
    map
}

/// Episode fulltext (BM25) search with RRF / CrossEncoder reranking.
///
/// Upstream: `graphiti_core/search/search.py::episode_search`. `config` `None`
/// → `(vec![], vec![])`. `filters` is currently unused by the episode fulltext
/// driver method (upstream threads SearchFilters into the episode query but the
/// chronicle `episode_fulltext_search` does not yet take it — see SearchOps).
#[allow(clippy::too_many_arguments)]
pub async fn episode_search(
    driver: &dyn GraphDriver,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    _query_vector: &[f32],
    group_ids: &[String],
    config: Option<&EpisodeSearchConfig>,
    _filters: &SearchFilters,
    limit: usize,
    reranker_min_score: f64,
) -> Result<(Vec<EpisodicNode>, Vec<f64>), ChronicleError> {
    let Some(config) = config else {
        return Ok((Vec::new(), Vec::new()));
    };

    let candidate_limit = 2 * limit;

    // Episodes only ever run a single bm25 method (upstream hardcodes the task
    // list to episode_fulltext_search).
    let bm25 = driver
        .episode_fulltext_search(query, group_ids, candidate_limit)
        .await?;
    let result_lists = vec![bm25];

    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|e| e.uuid.clone()).collect())
        .collect();
    let episode_uuid_map = ordered_episode_map(&result_lists);

    let (reranked_uuids, episode_scores): (Vec<String>, Vec<f64>) = match config.reranker {
        EpisodeReranker::Rrf => rrf(&uuid_lists, 1, reranker_min_score),
        EpisodeReranker::CrossEncoder => {
            let cross_encoder = cross_encoder.ok_or_else(|| {
                ChronicleError::InvalidInput(
                    "episode cross_encoder reranker requested but no CrossEncoderClient was provided"
                        .to_string(),
                )
            })?;
            // rrf-presort → take FIRST `limit` → map content → uuid.
            let (rrf_uuids, _) = rrf(&uuid_lists, 1, reranker_min_score);
            let mut content_order: Vec<String> = Vec::new();
            let mut content_to_uuid: HashMap<String, String> = HashMap::new();
            for u in rrf_uuids.iter().take(limit) {
                if let Some(ep) = episode_uuid_map.get(u) {
                    if !content_to_uuid.contains_key(&ep.content) {
                        content_order.push(ep.content.clone());
                    }
                    content_to_uuid.insert(ep.content.clone(), ep.uuid.clone());
                }
            }
            let ranked = cross_encoder.rank(query, &content_order).await?;
            let mut uuids = Vec::new();
            let mut scores = Vec::new();
            for (content, score) in ranked {
                if score >= reranker_min_score
                    && let Some(u) = content_to_uuid.get(&content)
                {
                    uuids.push(u.clone());
                    scores.push(score);
                }
            }
            (uuids, scores)
        }
    };

    let mut reranked_episodes: Vec<EpisodicNode> = reranked_uuids
        .iter()
        .filter_map(|u| episode_uuid_map.get(u).cloned())
        .collect();
    let mut episode_scores = episode_scores;

    reranked_episodes.truncate(limit);
    episode_scores.truncate(limit);

    Ok((reranked_episodes, episode_scores))
}
