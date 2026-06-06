// Ported from graphiti_core/search/search.py::search @ 34f56e65 (top-level).
//
// Phase-2: multi-scope entrypoint. Verified against search.py lines 98-250:
//   - Empty (trimmed) query → default SearchResults immediately (line 117-118).
//   - Embed the query ONCE iff any scope uses cosine_similarity OR the mmr
//     reranker; embed text = query.replace('\n', " "); else a zero vector that
//     the scopes will not consult (line 120-152).
//   - group_ids `[""]` / empty → treated as None / no filter (line 155). Here
//     "no filter" maps to an empty slice (the driver fulltext/similarity methods
//     treat an empty group_ids as "all groups").
//   - 3 scope searches run in parallel via tokio::join! (upstream
//     semaphore_gather; community scope deferred to Phase 4).
//   - center_node_uuid / bfs_origin_node_uuids forwarded ONLY to edge + node
//     scopes (the episode scope takes neither).

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;

use super::config::{EdgeReranker, NodeReranker, SearchConfig};
use super::edge_search::edge_search;
use super::episode_search::episode_search;
use super::filters::SearchFilters;
use super::node_search::node_search;
use super::results::SearchResults;

/// Returns true iff any configured scope needs the query embedding (cosine
/// search method or the MMR reranker), mirroring upstream's embed decision.
fn needs_query_embedding(config: &SearchConfig) -> bool {
    let edge_needs = config.edge_config.as_ref().is_some_and(|c| {
        c.search_methods
            .contains(&super::config::EdgeSearchMethod::CosineSimilarity)
            || c.reranker == EdgeReranker::Mmr
    });
    let node_needs = config.node_config.as_ref().is_some_and(|c| {
        c.search_methods
            .contains(&super::config::NodeSearchMethod::CosineSimilarity)
            || c.reranker == NodeReranker::Mmr
    });
    // Episodes never use cosine/MMR (bm25 + rrf/cross_encoder only).
    edge_needs || node_needs
}

/// Top-level multi-scope search.
///
/// Upstream: `graphiti_core/search/search.py::search`. Runs the edge, node, and
/// episode scopes (community deferred to Phase 4) and assembles a
/// [`SearchResults`]. An empty (trimmed) query short-circuits to the default
/// (all-empty) results.
#[allow(clippy::too_many_arguments)]
pub async fn search(
    driver: &dyn GraphDriver,
    embedder: &dyn EmbedderClient,
    cross_encoder: Option<&dyn CrossEncoderClient>,
    query: &str,
    group_ids: &[String],
    config: &SearchConfig,
    filters: &SearchFilters,
    center_node_uuid: Option<&str>,
    bfs_origin_node_uuids: Option<&[String]>,
) -> Result<SearchResults, ChronicleError> {
    if query.trim().is_empty() {
        return Ok(SearchResults::default());
    }

    // Embed once iff any scope needs it; otherwise an unused zero vector.
    let query_vector: Vec<f32> = if needs_query_embedding(config) {
        embedder.create(&query.replace('\n', " ")).await?
    } else {
        vec![0.0_f32; embedder.embedding_dim()]
    };

    // group_ids [""] / empty → no filter (empty slice).
    let normalized: Vec<String> = if group_ids.is_empty() || group_ids == [String::new()] {
        Vec::new()
    } else {
        group_ids.to_vec()
    };

    // Run the three scopes in parallel. Each scope short-circuits when its
    // sub-config is None.
    let (edge_res, node_res, episode_res) = tokio::join!(
        edge_search(
            driver,
            cross_encoder,
            query,
            &query_vector,
            &normalized,
            config.edge_config.as_ref(),
            filters,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        node_search(
            driver,
            cross_encoder,
            query,
            &query_vector,
            &normalized,
            config.node_config.as_ref(),
            filters,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        episode_search(
            driver,
            cross_encoder,
            query,
            &query_vector,
            &normalized,
            config.episode_config.as_ref(),
            filters,
            config.limit,
            config.reranker_min_score,
        ),
    );

    let (edges, edge_reranker_scores) = edge_res?;
    let (nodes, node_reranker_scores) = node_res?;
    let (episodes, episode_reranker_scores) = episode_res?;

    Ok(SearchResults {
        edges,
        edge_reranker_scores,
        nodes,
        node_reranker_scores,
        episodes,
        episode_reranker_scores,
    })
}
