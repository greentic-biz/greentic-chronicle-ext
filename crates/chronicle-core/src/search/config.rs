// Ported shape from graphiti_core/search/search_config.py @ 34f56e65 (Phase-1 edge subset).
// Constants sourced from graphiti_core/search/search_utils.py lines 65-67.
//
// Phase-1 scope: EdgeSearchConfig (BM25 + CosineSimilarity + RRF) only.
// NodeSearchConfig, EpisodeSearchConfig, CommunitySearchConfig — Phase 2+.
// Rerankers beyond RRF (Mmr, NodeDistance, EpisodeMentions, CrossEncoder) — Phase 2.

/// Upstream DEFAULT_SEARCH_LIMIT (search_config.py).
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Upstream DEFAULT_MIN_SCORE = 0.6 (search_utils.py:65).
/// Used as the per-method similarity threshold on embedding search.
pub const DEFAULT_MIN_SCORE: f32 = 0.6;

/// Upstream DEFAULT_MMR_LAMBDA = 0.5 (search_utils.py:66).
/// Relevance/diversity trade-off weight for MMR reranking (Phase 2).
pub const DEFAULT_MMR_LAMBDA: f64 = 0.5;

/// Upstream MAX_SEARCH_DEPTH = 3 (search_utils.py:67).
/// Maximum BFS traversal depth (Phase 2).
pub const MAX_SEARCH_DEPTH: usize = 3;

/// Upstream EdgeSearchMethod enum — Phase-1 subset (BFS is Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSearchMethod {
    CosineSimilarity,
    Bm25,
    /// BFS traversal — implementation deferred to Phase 2.
    BreadthFirstSearch,
}

/// Upstream EdgeReranker enum — Phase-1 subset (RRF only).
/// Mmr, NodeDistance, EpisodeMentions, CrossEncoder are Phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeReranker {
    Rrf,
    // Mmr, NodeDistance, EpisodeMentions, CrossEncoder — Phase 2
}

/// Phase-1 edge search configuration.
/// Upstream equivalent: EdgeSearchConfig (search_config.py).
#[derive(Debug, Clone)]
pub struct EdgeSearchConfig {
    /// Which search methods to invoke (all enabled by default: BM25 + CosineSimilarity).
    pub search_methods: Vec<EdgeSearchMethod>,
    /// Reranker applied to fuse results from multiple search methods.
    pub reranker: EdgeReranker,
    /// Minimum cosine similarity score for embedding search results.
    /// Upstream: `sim_min_score` field on EdgeSearchConfig.
    pub sim_min_score: f32,
    /// MMR lambda (Phase 2 only; carried here to avoid a breaking config change).
    pub mmr_lambda: f64,
    /// Maximum BFS depth (Phase 2 only).
    pub bfs_max_depth: usize,
}

impl Default for EdgeSearchConfig {
    fn default() -> Self {
        Self {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Rrf,
            sim_min_score: DEFAULT_MIN_SCORE,
            mmr_lambda: DEFAULT_MMR_LAMBDA,
            bfs_max_depth: MAX_SEARCH_DEPTH,
        }
    }
}

/// Top-level search configuration passed to [`super::edge_search::edge_search`].
/// Upstream equivalent: combination of SearchConfig + per-entity-type configs (search_config.py).
///
/// # Why no `Default` derive
/// A derived `Default` would set `limit = 0`, producing an empty result set.
/// Use [`edge_hybrid_search_rrf`] as the canonical default recipe instead.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Edge search sub-config. `None` causes `edge_search` to return an empty result.
    pub edge_config: Option<EdgeSearchConfig>,
    /// Maximum number of results to return after reranking.
    pub limit: usize,
    /// Minimum RRF score to include in final output (passed as `min_score` to `rrf`).
    pub reranker_min_score: f64,
}

/// Upstream EDGE_HYBRID_SEARCH_RRF recipe (search_config_recipes.py).
/// BM25 + CosineSimilarity, RRF reranker, default limit, no post-rerank score filter.
pub fn edge_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig::default()),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_edge_search_config_has_both_methods() {
        let cfg = EdgeSearchConfig::default();
        assert!(cfg.search_methods.contains(&EdgeSearchMethod::Bm25));
        assert!(
            cfg.search_methods
                .contains(&EdgeSearchMethod::CosineSimilarity)
        );
        assert_eq!(cfg.reranker, EdgeReranker::Rrf);
        assert!((cfg.sim_min_score - DEFAULT_MIN_SCORE).abs() < f32::EPSILON);
        assert!((cfg.mmr_lambda - DEFAULT_MMR_LAMBDA).abs() < f64::EPSILON);
        assert_eq!(cfg.bfs_max_depth, MAX_SEARCH_DEPTH);
    }

    #[test]
    fn edge_hybrid_search_rrf_recipe_is_sane() {
        let cfg = edge_hybrid_search_rrf();
        assert!(cfg.edge_config.is_some());
        assert_eq!(cfg.limit, DEFAULT_SEARCH_LIMIT);
        assert_eq!(cfg.reranker_min_score, 0.0);
    }

    #[test]
    fn constants_match_upstream() {
        assert_eq!(DEFAULT_SEARCH_LIMIT, 10);
        assert!((DEFAULT_MIN_SCORE - 0.6_f32).abs() < f32::EPSILON);
        assert!((DEFAULT_MMR_LAMBDA - 0.5_f64).abs() < f64::EPSILON);
        assert_eq!(MAX_SEARCH_DEPTH, 3);
    }
}
