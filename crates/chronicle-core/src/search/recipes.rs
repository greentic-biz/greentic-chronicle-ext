// Ported from graphiti_core/search/search_config_recipes.py @ 34f56e65.
//
// Each recipe is a constructor function returning a `SearchConfig` value that
// matches the upstream constant exactly (methods, reranker, limit, mmr_lambda).
//
// Naming convention: snake_case mirrors the upstream UPPER_SNAKE_CASE constant
// names (e.g. `EDGE_HYBRID_SEARCH_RRF` → `edge_hybrid_search_rrf`).
//
// Phase-4: community recipes (COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER}) are
// now ported, and community_config is added to the three COMBINED_* recipes.
//
// Note on COMBINED_HYBRID_SEARCH_MMR: upstream sets mmr_lambda=1 (integer) for
// edge, node, AND community scopes (verified search_config_recipes.py lines 56-78:
// the community_config block carries `mmr_lambda=1` too). Python `1` and `1.0` are
// identical for a `float` field — upstream intent is pure-relevance MMR (no
// diversity penalty). Episode scope uses RRF (no MMR option for episodes).
//
// Note on COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER: upstream sets `limit=3`
// explicitly (verified line 217-224). The RRF/MMR community recipes use the
// default limit. The COMBINED_* recipes' community_config uses the default
// per-config limit; the top-level SearchConfig limit is unchanged.

use super::config::{
    CommunityReranker, CommunitySearchConfig, CommunitySearchMethod, DEFAULT_SEARCH_LIMIT,
    EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, EpisodeReranker, EpisodeSearchConfig,
    EpisodeSearchMethod, NodeReranker, NodeSearchConfig, NodeSearchMethod, SearchConfig,
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
        community_config: Some(CommunitySearchConfig {
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::Rrf,
            ..CommunitySearchConfig::default()
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
        community_config: Some(CommunitySearchConfig {
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::Mmr,
            // Upstream community_config sets mmr_lambda=1 (verified) — same as
            // edge/node here. Pure-relevance MMR (no diversity penalty).
            mmr_lambda: 1.0,
            ..CommunitySearchConfig::default()
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
        community_config: Some(CommunitySearchConfig {
            // Communities have NO BFS method — upstream uses bm25 + cosine only
            // (verified search_config_recipes.py lines 104-107).
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::CrossEncoder,
            ..CommunitySearchConfig::default()
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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

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
        community_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    }
}

// ── Community-only recipes ──────────────────────────────────────────────────

/// Upstream `COMMUNITY_HYBRID_SEARCH_RRF`.
/// BM25 + CosineSimilarity, RRF reranker, default limit.
pub fn community_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: None,
        episode_config: None,
        community_config: Some(CommunitySearchConfig {
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::Rrf,
            ..CommunitySearchConfig::default()
        }),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `COMMUNITY_HYBRID_SEARCH_MMR`.
/// BM25 + CosineSimilarity, MMR reranker (default mmr_lambda=0.5 — the COMMUNITY_*
/// recipe does NOT override lambda; only the COMBINED_* recipe sets it to 1.0).
pub fn community_hybrid_search_mmr() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: None,
        episode_config: None,
        community_config: Some(CommunitySearchConfig {
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::Mmr,
            ..CommunitySearchConfig::default()
        }),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}

/// Upstream `COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER`.
/// BM25 + CosineSimilarity, CrossEncoder reranker. Explicit `limit=3` (matches
/// the upstream constant which sets `limit=3` — verified).
pub fn community_hybrid_search_cross_encoder() -> SearchConfig {
    SearchConfig {
        edge_config: None,
        node_config: None,
        episode_config: None,
        community_config: Some(CommunitySearchConfig {
            search_methods: vec![
                CommunitySearchMethod::Bm25,
                CommunitySearchMethod::CosineSimilarity,
            ],
            reranker: CommunityReranker::CrossEncoder,
            ..CommunitySearchConfig::default()
        }),
        limit: 3,
        reranker_min_score: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::config::{
        CommunityReranker, CommunitySearchMethod, DEFAULT_MMR_LAMBDA, DEFAULT_SEARCH_LIMIT,
        EdgeReranker, EdgeSearchMethod, EpisodeReranker, EpisodeSearchMethod, NodeReranker,
        NodeSearchMethod,
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

        // Community scope ALSO sets mmr_lambda=1.0 in COMBINED_MMR (verified
        // upstream search_config_recipes.py).
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::Mmr);
        assert!(
            (cc.mmr_lambda - 1.0_f64).abs() < f64::EPSILON,
            "community mmr_lambda must be 1.0 in COMBINED_MMR, got {}",
            cc.mmr_lambda
        );
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
        // Phase-4: combined recipes now include the community scope.
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::Rrf);
        assert!(cc.search_methods.contains(&CommunitySearchMethod::Bm25));
        assert!(
            cc.search_methods
                .contains(&CommunitySearchMethod::CosineSimilarity)
        );
    }

    #[test]
    fn combined_cross_encoder_community_has_no_bfs() {
        let cfg = combined_hybrid_search_cross_encoder();
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::CrossEncoder);
        // Communities have no BFS variant — methods are exactly bm25 + cosine.
        assert_eq!(cc.search_methods.len(), 2);
        assert!(cc.search_methods.contains(&CommunitySearchMethod::Bm25));
        assert!(
            cc.search_methods
                .contains(&CommunitySearchMethod::CosineSimilarity)
        );
    }

    #[test]
    fn community_rrf_recipe_shape() {
        let cfg = community_hybrid_search_rrf();
        assert!(cfg.edge_config.is_none());
        assert!(cfg.node_config.is_none());
        assert!(cfg.episode_config.is_none());
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::Rrf);
        assert_eq!(cfg.limit, DEFAULT_SEARCH_LIMIT);
    }

    #[test]
    fn community_mmr_recipe_uses_default_lambda() {
        // The COMMUNITY_* MMR recipe does NOT override lambda (default 0.5);
        // only the COMBINED_* recipe sets community lambda=1.0.
        let cfg = community_hybrid_search_mmr();
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::Mmr);
        assert!(
            (cc.mmr_lambda - DEFAULT_MMR_LAMBDA).abs() < f64::EPSILON,
            "COMMUNITY_MMR uses default lambda 0.5"
        );
    }

    #[test]
    fn community_cross_encoder_recipe_has_explicit_limit_3() {
        let cfg = community_hybrid_search_cross_encoder();
        assert_eq!(cfg.limit, 3, "COMMUNITY_CROSS_ENCODER sets limit=3");
        let cc = cfg.community_config.expect("community_config must be Some");
        assert_eq!(cc.reranker, CommunityReranker::CrossEncoder);
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
