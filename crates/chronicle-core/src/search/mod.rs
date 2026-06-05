// Search module — Phase-1 edge subset.
// Upstream origin: graphiti_core/search/ @ 34f56e65.
//
// Phase-1 scope: RRF fusion, edge hybrid search (BM25 + CosineSimilarity).
// Phase-2: NodeSearchConfig, EpisodeSearchConfig, CommunitySearchConfig,
//          BFS traversal, MMR / NodeDistance / EpisodeMentions / CrossEncoder rerankers.

pub mod config;
pub mod edge_search;
pub mod rrf;

pub use config::{
    DEFAULT_MIN_SCORE, DEFAULT_MMR_LAMBDA, DEFAULT_SEARCH_LIMIT, EdgeReranker, EdgeSearchConfig,
    EdgeSearchMethod, MAX_SEARCH_DEPTH, SearchConfig, edge_hybrid_search_rrf,
};
pub use edge_search::edge_search;
pub use rrf::rrf;
