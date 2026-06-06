// Search module — Phase-2 full surface.
// Upstream origin: graphiti_core/search/ @ 34f56e65.
//
// Phase-1 scope: RRF fusion, edge hybrid search (BM25 + CosineSimilarity).
// Phase-2 additions: SearchFilters, NodeSearchConfig, EpisodeSearchConfig,
//   full reranker enums (Mmr/NodeDistance/EpisodeMentions/CrossEncoder),
//   BFS traversal in method enums, complete recipe set.
//
// Community scope deferred to Phase 4 (ledger note in config.rs / recipes.rs).

pub mod config;
pub mod edge_search;
pub mod episode_search;
pub mod filters;
pub mod node_search;
pub mod recipes;
pub mod rerank;
pub mod results;
pub mod rrf;
#[allow(clippy::module_inception)]
pub mod search;

// ── config re-exports ────────────────────────────────────────────────────────
pub use config::{
    DEFAULT_MIN_SCORE,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SEARCH_LIMIT,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    MAX_SEARCH_DEPTH,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    // backward-compat: edge_hybrid_search_rrf also lives in recipes; both paths work
    edge_hybrid_search_rrf,
};

// ── filters re-exports ────────────────────────────────────────────────────────
pub use filters::{ComparisonOperator, DateFilter, PropertyFilter, SearchFilters};

// ── recipes re-exports ────────────────────────────────────────────────────────
pub use recipes::{
    combined_hybrid_search_cross_encoder, combined_hybrid_search_mmr, combined_hybrid_search_rrf,
    edge_hybrid_search_cross_encoder, edge_hybrid_search_episode_mentions, edge_hybrid_search_mmr,
    edge_hybrid_search_node_distance, node_hybrid_search_cross_encoder,
    node_hybrid_search_episode_mentions, node_hybrid_search_mmr, node_hybrid_search_node_distance,
    node_hybrid_search_rrf,
};

// ── search function re-exports ────────────────────────────────────────────────
pub use edge_search::{edge_search, edge_search_simple};
pub use episode_search::episode_search;
pub use node_search::node_search;
pub use rerank::{episode_mentions_rerank, maximal_marginal_relevance, node_distance_rerank};
pub use results::SearchResults;
pub use rrf::rrf;
pub use search::search;
