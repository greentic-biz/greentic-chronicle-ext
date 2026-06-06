// Ported from graphiti_core/search/search.py::edge_search @ 34f56e65.
//
// Upstream conventions verified:
//   - Candidate limit: 2 * config.limit (confirmed search.py lines 286, 297, 309).
//   - Parallel execution: upstream uses `semaphore_gather(*search_tasks)` for all methods.
//     Phase-1 implementation uses `tokio::join!` for the two-method BM25+cosine case,
//     providing parity for the common path without requiring a semaphore abstraction.
//     Three-or-more-method combinations (with BFS in Phase 2) will need a
//     `futures::future::join_all` or semaphore_gather equivalent.
//   - edge_config == None: return empty list (upstream returns [], []).
//   - RRF reranker_min_score: passed as min_score to rrf (upstream reranker_min_score=0 default).

use std::collections::HashMap;

use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;
use crate::types::EntityEdge;

use super::config::{EdgeReranker, EdgeSearchMethod, SearchConfig};
use super::rrf::rrf;

/// Hybrid edge search with configurable methods and RRF reranking.
///
/// Upstream: `graphiti_core/search/search.py::edge_search`.
///
/// # Parallel execution
/// When both BM25 and CosineSimilarity are enabled, the two searches run
/// concurrently via `tokio::join!`. This matches upstream `semaphore_gather`
/// semantics for the two-method case. BFS (Phase 2) will extend this to
/// `join_all` / a semaphore approach.
///
/// # Candidate limit
/// Each method fetches `2 * config.limit` candidates before RRF fusion,
/// mirroring the upstream 2× over-fetch convention.
pub async fn edge_search(
    driver: &dyn GraphDriver,
    embedder: &dyn EmbedderClient,
    query: &str,
    group_ids: &[String],
    config: &SearchConfig,
) -> Result<Vec<EntityEdge>, ChronicleError> {
    let Some(edge_config) = &config.edge_config else {
        return Ok(Vec::new());
    };

    let candidate_limit = 2 * config.limit;

    let want_bm25 = edge_config.search_methods.contains(&EdgeSearchMethod::Bm25);
    let want_cosine = edge_config
        .search_methods
        .contains(&EdgeSearchMethod::CosineSimilarity);

    // Parallel fetch when both methods are enabled (mirrors upstream semaphore_gather).
    let (bm25_result, cosine_result) = match (want_bm25, want_cosine) {
        (true, true) => {
            let vector = embedder.create(query).await?;
            let (bm25, cosine) = tokio::join!(
                driver.edge_fulltext_search(query, group_ids, candidate_limit),
                driver.edge_similarity_search(
                    &vector,
                    group_ids,
                    candidate_limit,
                    edge_config.sim_min_score,
                ),
            );
            (Some(bm25?), Some(cosine?))
        }
        (true, false) => {
            let result = driver
                .edge_fulltext_search(query, group_ids, candidate_limit)
                .await?;
            (Some(result), None)
        }
        (false, true) => {
            let vector = embedder.create(query).await?;
            let result = driver
                .edge_similarity_search(
                    &vector,
                    group_ids,
                    candidate_limit,
                    edge_config.sim_min_score,
                )
                .await?;
            (None, Some(result))
        }
        (false, false) => (None, None),
    };

    let mut result_lists: Vec<Vec<EntityEdge>> = Vec::new();
    if let Some(r) = bm25_result {
        result_lists.push(r);
    }
    if let Some(r) = cosine_result {
        result_lists.push(r);
    }

    // Build UUID→edge map (last write wins on duplicate UUIDs across result sets,
    // which is acceptable since fields are identical for the same edge).
    let edge_uuid_map: HashMap<String, EntityEdge> = result_lists
        .iter()
        .flatten()
        .map(|e| (e.uuid.clone(), e.clone()))
        .collect();

    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|e| e.uuid.clone()).collect())
        .collect();

    // Rerank. RRF is the only reranker wired in Phase 1.
    // Mmr, NodeDistance, EpisodeMentions, CrossEncoder are implemented in Phase 2
    // (Task 3: rerankers; Task 5: scope search wiring). Until then, invoking a
    // Phase-2 reranker falls back to RRF so that the config types can be used
    // in recipes without compile errors. Phase-5 callers that pass these configs
    // will get RRF results until the full reranker dispatch lands.
    let rank_const = match edge_config.reranker {
        EdgeReranker::Rrf
        | EdgeReranker::Mmr
        | EdgeReranker::NodeDistance
        | EdgeReranker::EpisodeMentions
        | EdgeReranker::CrossEncoder => 1usize,
    };
    let (ranked, _scores) = rrf(&uuid_lists, rank_const, config.reranker_min_score);

    Ok(ranked
        .into_iter()
        .filter_map(|u| edge_uuid_map.get(&u).cloned())
        .take(config.limit)
        .collect())
}

// Integration tests for edge_search live in chronicle-testkit/src/edge_search_tests.rs
// to avoid a dev-dependency cycle (chronicle-core → chronicle-testkit → chronicle-core).
// See: crates/chronicle-testkit/src/edge_search_tests.rs
