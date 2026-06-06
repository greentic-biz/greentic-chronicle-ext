// Ported from graphiti_core/search/search_config.py @ 34f56e65.
// Phase-1 subset expanded to full Phase-2 surface in this commit.
// Constants sourced from graphiti_core/search/search_utils.py lines 65-67.
//
// Phase-1 scope was: EdgeSearchConfig (BM25 + CosineSimilarity + RRF) only.
// Phase-2 additions: NodeSearchConfig, EpisodeSearchConfig, full reranker enums,
//   NodeSearchMethod, EpisodeSearchMethod, SearchConfig gains node_config/episode_config.
//
// community scope deferred to Phase 4 (upstream community_config; needs CommunityNode storage + CommunitySearchMethod enum) — ledger

/// Upstream DEFAULT_SEARCH_LIMIT (search_config.py).
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Upstream DEFAULT_MIN_SCORE = 0.6 (search_utils.py:65).
/// Used as the per-method similarity threshold on embedding search.
pub const DEFAULT_MIN_SCORE: f32 = 0.6;

/// Upstream DEFAULT_MMR_LAMBDA = 0.5 (search_utils.py:66).
/// Relevance/diversity trade-off weight for MMR reranking.
pub const DEFAULT_MMR_LAMBDA: f64 = 0.5;

/// Upstream MAX_SEARCH_DEPTH = 3 (search_utils.py:67).
/// Maximum BFS traversal depth.
pub const MAX_SEARCH_DEPTH: usize = 3;

// ── Search Methods ───────────────────────────────────────────────────────────

/// Upstream `EdgeSearchMethod` enum (search_config.py).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSearchMethod {
    CosineSimilarity,
    Bm25,
    /// BFS (breadth-first) traversal from seed nodes.
    BreadthFirstSearch,
}

/// Upstream `NodeSearchMethod` enum (search_config.py).
/// Phase-2 addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSearchMethod {
    CosineSimilarity,
    Bm25,
    /// BFS traversal from origin nodes.
    BreadthFirstSearch,
}

/// Upstream `EpisodeSearchMethod` enum (search_config.py).
/// Phase-2 addition. Only BM25 (fulltext) is supported upstream for episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeSearchMethod {
    Bm25,
}

// ── Rerankers ────────────────────────────────────────────────────────────────

/// Upstream `EdgeReranker` enum (search_config.py). Phase-2 full set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeReranker {
    /// Reciprocal rank fusion.
    Rrf,
    /// Maximal marginal relevance — diversity-aware fusion.
    Mmr,
    /// 1-hop graph distance from a center node.
    NodeDistance,
    /// Sort by episode mention count (note: ascending sort — see R8 in plan).
    EpisodeMentions,
    /// LLM-based cross-encoder relevance scoring.
    CrossEncoder,
}

/// Upstream `NodeReranker` enum (search_config.py). Phase-2 addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeReranker {
    Rrf,
    Mmr,
    NodeDistance,
    EpisodeMentions,
    CrossEncoder,
}

/// Upstream `EpisodeReranker` enum (search_config.py). Phase-2 addition.
/// Episodes support only RRF and CrossEncoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeReranker {
    Rrf,
    CrossEncoder,
}

// ── Per-scope search configs ─────────────────────────────────────────────────

/// Upstream `EdgeSearchConfig` model (search_config.py).
#[derive(Debug, Clone)]
pub struct EdgeSearchConfig {
    /// Which search methods to invoke.
    pub search_methods: Vec<EdgeSearchMethod>,
    /// Reranker applied to fuse results from multiple search methods.
    pub reranker: EdgeReranker,
    /// Minimum cosine similarity score for embedding search results.
    /// Upstream: `sim_min_score` field.
    pub sim_min_score: f32,
    /// MMR lambda — relevance/diversity weight (used when reranker = Mmr).
    /// Upstream default: 0.5.
    pub mmr_lambda: f64,
    /// Maximum BFS depth.
    /// Upstream default: 3.
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

/// Upstream `NodeSearchConfig` model (search_config.py). Phase-2 addition.
#[derive(Debug, Clone)]
pub struct NodeSearchConfig {
    pub search_methods: Vec<NodeSearchMethod>,
    pub reranker: NodeReranker,
    pub sim_min_score: f32,
    pub mmr_lambda: f64,
    pub bfs_max_depth: usize,
}

impl Default for NodeSearchConfig {
    fn default() -> Self {
        Self {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::Rrf,
            sim_min_score: DEFAULT_MIN_SCORE,
            mmr_lambda: DEFAULT_MMR_LAMBDA,
            bfs_max_depth: MAX_SEARCH_DEPTH,
        }
    }
}

/// Upstream `EpisodeSearchConfig` model (search_config.py). Phase-2 addition.
#[derive(Debug, Clone)]
pub struct EpisodeSearchConfig {
    pub search_methods: Vec<EpisodeSearchMethod>,
    pub reranker: EpisodeReranker,
    pub sim_min_score: f32,
    pub mmr_lambda: f64,
    pub bfs_max_depth: usize,
}

impl Default for EpisodeSearchConfig {
    fn default() -> Self {
        Self {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::Rrf,
            sim_min_score: DEFAULT_MIN_SCORE,
            mmr_lambda: DEFAULT_MMR_LAMBDA,
            bfs_max_depth: MAX_SEARCH_DEPTH,
        }
    }
}

// ── Top-level SearchConfig ───────────────────────────────────────────────────

/// Top-level search configuration passed to the search functions.
///
/// Upstream equivalent: `SearchConfig` in `search_config.py`.
///
/// # Why no `Default` derive
/// A derived `Default` would set `limit = 0`, producing an empty result set.
/// Use the recipe functions in `search/recipes.rs` as starting points.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Edge search sub-config. `None` skips edge search.
    pub edge_config: Option<EdgeSearchConfig>,
    /// Node search sub-config. `None` skips node search.
    /// Phase-2 addition.
    pub node_config: Option<NodeSearchConfig>,
    /// Episode search sub-config. `None` skips episode search.
    /// Phase-2 addition.
    pub episode_config: Option<EpisodeSearchConfig>,
    // community scope deferred to Phase 4 (upstream community_config; needs CommunityNode storage + CommunitySearchMethod enum) — ledger
    /// Maximum number of results to return after reranking.
    pub limit: usize,
    /// Minimum reranker score to include in final output.
    /// Upstream: `reranker_min_score` (default 0).
    pub reranker_min_score: f64,
}

// ── Backward-compatible recipe kept in config.rs (re-exported from recipes.rs) ─

/// Upstream EDGE_HYBRID_SEARCH_RRF recipe (search_config_recipes.py).
/// BM25 + CosineSimilarity, RRF reranker, default limit, no post-rerank score filter.
///
/// Maintained here for Phase-1 backward compatibility.
/// The canonical definition is in `search/recipes.rs`; this function delegates to it
/// and the public path remains `search::edge_hybrid_search_rrf` via `mod.rs`.
pub fn edge_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig::default()),
        node_config: None,
        episode_config: None,
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
    fn default_node_search_config_has_both_methods() {
        let cfg = NodeSearchConfig::default();
        assert!(cfg.search_methods.contains(&NodeSearchMethod::Bm25));
        assert!(
            cfg.search_methods
                .contains(&NodeSearchMethod::CosineSimilarity)
        );
        assert_eq!(cfg.reranker, NodeReranker::Rrf);
    }

    #[test]
    fn default_episode_search_config_has_bm25_only() {
        let cfg = EpisodeSearchConfig::default();
        assert_eq!(cfg.search_methods.len(), 1);
        assert!(cfg.search_methods.contains(&EpisodeSearchMethod::Bm25));
        assert_eq!(cfg.reranker, EpisodeReranker::Rrf);
    }

    #[test]
    fn edge_hybrid_search_rrf_recipe_is_sane() {
        let cfg = edge_hybrid_search_rrf();
        assert!(cfg.edge_config.is_some());
        assert!(cfg.node_config.is_none());
        assert!(cfg.episode_config.is_none());
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

    #[test]
    fn search_config_fields_accessible() {
        let cfg = edge_hybrid_search_rrf();
        // Verify Phase-2 fields exist and are None by default for backward-compat recipe.
        let _ = cfg.node_config;
        let _ = cfg.episode_config;
    }
}
