// Ported from graphiti_core/search/search_config_recipes.py @ 34f56e65.
//
// Each recipe is a constructor function returning a `SearchConfig` value that
// matches the upstream constant exactly (methods, reranker, limit, mmr_lambda).
//
// Naming convention: snake_case mirrors the upstream UPPER_SNAKE_CASE constant
// names (e.g. `EDGE_HYBRID_SEARCH_RRF` → `edge_hybrid_search_rrf`).
//
// Community recipes (COMMUNITY_HYBRID_SEARCH_*) are NOT ported — community scope is
// deferred to Phase 4 (upstream community_config; needs CommunityNode storage +
// CommunitySearchMethod enum) — ledger.
//
// Note on COMBINED_HYBRID_SEARCH_MMR: upstream sets mmr_lambda=1 (integer) for
// edge, node, and community scopes. Verified: Python `1` and Python `1.0` are
// identical for a `float` field — upstream intent is pure-relevance MMR (no
// diversity penalty). Episode scope uses RRF (no MMR option for episodes).

use super::config::{
    DEFAULT_SEARCH_LIMIT, EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, EpisodeReranker,
    EpisodeSearchConfig, EpisodeSearchMethod, NodeReranker, NodeSearchConfig, NodeSearchMethod,
    SearchConfig,
};

// ── Combined multi-scope recipes ─────────────────────────────────────────────

/// Upstream `COMBINED_HYBRID_SEARCH_RRF`.
/// BM25 + CosineSimilarity with RRF reranking across edges, nodes, and episodes.
pub fn combined_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Rrf,
            ..EdgeSearchConfig::default()
        }),
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::Rrf,
            ..NodeSearchConfig::default()
        }),
        episode_config: Some(EpisodeSearchConfig {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::Rrf,
            ..EpisodeSearchConfig::default()
        }),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `COMBINED_HYBRID_SEARCH_MMR`.
/// BM25 + CosineSimilarity with MMR reranking (mmr_lambda=1.0) for edges and
/// nodes; episodes use RRF.
///
/// Note: mmr_lambda=1.0 means pure-relevance order (no diversity penalty).
/// Upstream Python uses integer `1` which coerces to `1.0` for a float field.
pub fn combined_hybrid_search_mmr() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Mmr,
            mmr_lambda: 1.0,
            ..EdgeSearchConfig::default()
        }),
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::Mmr,
            mmr_lambda: 1.0,
            ..NodeSearchConfig::default()
        }),
        episode_config: Some(EpisodeSearchConfig {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::Rrf,
            ..EpisodeSearchConfig::default()
        }),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`.
/// BM25 + CosineSimilarity + BFS with CrossEncoder reranking for edges and nodes;
/// episodes use BM25 + CrossEncoder.
pub fn combined_hybrid_search_cross_encoder() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![
                EdgeSearchMethod::Bm25,
                EdgeSearchMethod::CosineSimilarity,
                EdgeSearchMethod::BreadthFirstSearch,
            ],
            reranker: EdgeReranker::CrossEncoder,
            ..EdgeSearchConfig::default()
        }),
        node_config: Some(NodeSearchConfig {
            search_methods: vec![
                NodeSearchMethod::Bm25,
                NodeSearchMethod::CosineSimilarity,
                NodeSearchMethod::BreadthFirstSearch,
            ],
            reranker: NodeReranker::CrossEncoder,
            ..NodeSearchConfig::default()
        }),
        episode_config: Some(EpisodeSearchConfig {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::CrossEncoder,
            ..EpisodeSearchConfig::default()
        }),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

// ── Edge-only recipes ─────────────────────────────────────────────────────────

/// Upstream `EDGE_HYBRID_SEARCH_RRF`.
/// BM25 + CosineSimilarity, RRF reranker.
/// Also re-exported from `search::edge_hybrid_search_rrf` for backward compatibility.
pub fn edge_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Rrf,
            ..EdgeSearchConfig::default()
        }),
        node_config: None,
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `EDGE_HYBRID_SEARCH_MMR`.
/// BM25 + CosineSimilarity, MMR reranker (default mmr_lambda=0.5).
pub fn edge_hybrid_search_mmr() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Mmr,
            ..EdgeSearchConfig::default()
        }),
        node_config: None,
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `EDGE_HYBRID_SEARCH_NODE_DISTANCE`.
/// BM25 + CosineSimilarity, NodeDistance reranker.
pub fn edge_hybrid_search_node_distance() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::NodeDistance,
            ..EdgeSearchConfig::default()
        }),
        node_config: None,
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS`.
/// BM25 + CosineSimilarity, EpisodeMentions reranker.
pub fn edge_hybrid_search_episode_mentions() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::EpisodeMentions,
            ..EdgeSearchConfig::default()
        }),
        node_config: None,
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `EDGE_HYBRID_SEARCH_CROSS_ENCODER`.
/// BM25 + CosineSimilarity + BFS, CrossEncoder reranker.
/// Explicit limit=10 (matches upstream constant which sets `limit=10` explicitly).
pub fn edge_hybrid_search_cross_encoder() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![
                EdgeSearchMethod::Bm25,
                EdgeSearchMethod::CosineSimilarity,
                EdgeSearchMethod::BreadthFirstSearch,
            ],
            reranker: EdgeReranker::CrossEncoder,
            ..EdgeSearchConfig::default()
        }),
        node_config: None,
        episode_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    }
}

// ── Node-only recipes ─────────────────────────────────────────────────────────

/// Upstream `NODE_HYBRID_SEARCH_RRF`.
/// BM25 + CosineSimilarity, RRF reranker.
pub fn node_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::Rrf,
            ..NodeSearchConfig::default()
        }),
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `NODE_HYBRID_SEARCH_MMR`.
/// BM25 + CosineSimilarity, MMR reranker (default mmr_lambda=0.5).
pub fn node_hybrid_search_mmr() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::Mmr,
            ..NodeSearchConfig::default()
        }),
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `NODE_HYBRID_SEARCH_NODE_DISTANCE`.
/// BM25 + CosineSimilarity, NodeDistance reranker.
pub fn node_hybrid_search_node_distance() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::NodeDistance,
            ..NodeSearchConfig::default()
        }),
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `NODE_HYBRID_SEARCH_EPISODE_MENTIONS`.
/// BM25 + CosineSimilarity, EpisodeMentions reranker.
pub fn node_hybrid_search_episode_mentions() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::CosineSimilarity],
            reranker: NodeReranker::EpisodeMentions,
            ..NodeSearchConfig::default()
        }),
        episode_config: None,

        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `NODE_HYBRID_SEARCH_CROSS_ENCODER`.
/// BM25 + CosineSimilarity + BFS, CrossEncoder reranker.
/// Explicit limit=10 (matches upstream constant).
pub fn node_hybrid_search_cross_encoder() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: Some(NodeSearchConfig {
            search_methods: vec![
                NodeSearchMethod::Bm25,
                NodeSearchMethod::CosineSimilarity,
                NodeSearchMethod::BreadthFirstSearch,
            ],
            reranker: NodeReranker::CrossEncoder,
            ..NodeSearchConfig::default()
        }),
        episode_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::config::{
        DEFAULT_MMR_LAMBDA, DEFAULT_SEARCH_LIMIT, EdgeReranker, EdgeSearchMethod, EpisodeReranker,
        EpisodeSearchMethod, NodeReranker, NodeSearchMethod,
    };

    // ── Spot-check 1: combined_cross_encoder has BFS in edge+node + episode bm25 ──

    #[test]
    fn combined_cross_encoder_bfs_in_edge_and_node_methods() {
        let cfg = combined_hybrid_search_cross_encoder();

        let ec = cfg.edge_config.expect("edge_config must be Some");
        assert!(
            ec.search_methods
                .contains(&EdgeSearchMethod::BreadthFirstSearch),
            "edge_config must contain BFS"
        );
        assert!(ec.search_methods.contains(&EdgeSearchMethod::Bm25));
        assert!(
            ec.search_methods
                .contains(&EdgeSearchMethod::CosineSimilarity)
        );
        assert_eq!(ec.reranker, EdgeReranker::CrossEncoder);

        let nc = cfg.node_config.expect("node_config must be Some");
        assert!(
            nc.search_methods
                .contains(&NodeSearchMethod::BreadthFirstSearch),
            "node_config must contain BFS"
        );
        assert_eq!(nc.reranker, NodeReranker::CrossEncoder);

        let epc = cfg.episode_config.expect("episode_config must be Some");
        assert_eq!(epc.search_methods, vec![EpisodeSearchMethod::Bm25]);
        assert_eq!(epc.reranker, EpisodeReranker::CrossEncoder);
    }

    // ── Spot-check 2: edge_cross_encoder limit is exactly 10 ─────────────────

    #[test]
    fn edge_cross_encoder_has_explicit_limit_10() {
        let cfg = edge_hybrid_search_cross_encoder();
        assert_eq!(cfg.limit, 10);
        let ec = cfg.edge_config.expect("edge_config must be Some");
        assert!(
            ec.search_methods
                .contains(&EdgeSearchMethod::BreadthFirstSearch)
        );
        assert_eq!(ec.reranker, EdgeReranker::CrossEncoder);
    }

    // ── Spot-check 3: combined_mmr has mmr_lambda=1.0 on edge+node ───────────

    #[test]
    fn combined_mmr_lambda_is_1_0_on_edge_and_node() {
        let cfg = combined_hybrid_search_mmr();

        let ec = cfg.edge_config.expect("edge_config must be Some");
        assert_eq!(ec.reranker, EdgeReranker::Mmr);
        assert!(
            (ec.mmr_lambda - 1.0_f64).abs() < f64::EPSILON,
            "edge mmr_lambda must be 1.0, got {}",
            ec.mmr_lambda
        );

        let nc = cfg.node_config.expect("node_config must be Some");
        assert_eq!(nc.reranker, NodeReranker::Mmr);
        assert!(
            (nc.mmr_lambda - 1.0_f64).abs() < f64::EPSILON,
            "node mmr_lambda must be 1.0, got {}",
            nc.mmr_lambda
        );

        // Episodes use RRF, not MMR.
        let epc = cfg.episode_config.expect("episode_config must be Some");
        assert_eq!(epc.reranker, EpisodeReranker::Rrf);
    }

    // ── Spot-check 4: node_distance recipe reranker is NodeDistance ───────────

    #[test]
    fn node_distance_recipe_has_node_distance_reranker() {
        // Edge variant.
        let edge_cfg = edge_hybrid_search_node_distance();
        let ec = edge_cfg.edge_config.expect("edge_config must be Some");
        assert_eq!(ec.reranker, EdgeReranker::NodeDistance);
        assert!(ec.search_methods.contains(&EdgeSearchMethod::Bm25));
        assert!(
            ec.search_methods
                .contains(&EdgeSearchMethod::CosineSimilarity)
        );

        // Node variant.
        let node_cfg = node_hybrid_search_node_distance();
        let nc = node_cfg.node_config.expect("node_config must be Some");
        assert_eq!(nc.reranker, NodeReranker::NodeDistance);
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn edge_hybrid_search_rrf_backward_compat() {
        let cfg = edge_hybrid_search_rrf();
        let ec = cfg.edge_config.expect("edge_config must be Some");
        assert_eq!(ec.reranker, EdgeReranker::Rrf);
        assert!(
            !ec.search_methods
                .contains(&EdgeSearchMethod::BreadthFirstSearch)
        );
        assert_eq!(cfg.limit, DEFAULT_SEARCH_LIMIT);
        assert!(cfg.node_config.is_none());
        assert!(cfg.episode_config.is_none());
    }

    #[test]
    fn node_cross_encoder_has_explicit_limit_10() {
        let cfg = node_hybrid_search_cross_encoder();
        assert_eq!(cfg.limit, 10);
        let nc = cfg.node_config.expect("node_config must be Some");
        assert!(
            nc.search_methods
                .contains(&NodeSearchMethod::BreadthFirstSearch)
        );
        assert_eq!(nc.reranker, NodeReranker::CrossEncoder);
    }

    #[test]
    fn combined_rrf_all_scopes_set() {
        let cfg = combined_hybrid_search_rrf();
        assert!(cfg.edge_config.is_some());
        assert!(cfg.node_config.is_some());
        assert!(cfg.episode_config.is_some());
    }

    #[test]
    fn edge_episode_mentions_reranker() {
        let cfg = edge_hybrid_search_episode_mentions();
        let ec = cfg.edge_config.expect("edge_config");
        assert_eq!(ec.reranker, EdgeReranker::EpisodeMentions);
    }

    #[test]
    fn node_episode_mentions_reranker() {
        let cfg = node_hybrid_search_episode_mentions();
        let nc = cfg.node_config.expect("node_config");
        assert_eq!(nc.reranker, NodeReranker::EpisodeMentions);
    }

    #[test]
    fn default_mmr_lambda_is_0_5_for_non_mmr_combined_recipe() {
        // Edge MMR recipe without explicit override should use default lambda 0.5.
        let cfg = edge_hybrid_search_mmr();
        let ec = cfg.edge_config.expect("edge_config");
        assert_eq!(ec.reranker, EdgeReranker::Mmr);
        assert!(
            (ec.mmr_lambda - DEFAULT_MMR_LAMBDA).abs() < f64::EPSILON,
            "default mmr_lambda should be 0.5"
        );
    }
}
