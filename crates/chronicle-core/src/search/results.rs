// Ported from graphiti_core/search/search_config.py::SearchResults @ 34f56e65.
//
// DIVERGENCE (ledger note): upstream `SearchResults` carries `communities` +
// `community_reranker_scores`. The community scope is DEFERRED to Phase 4 (see
// search/config.rs + recipes.rs notes); the community fields are OMITTED from
// this struct in Phase 2 and will be added when the community scope lands. All
// other fields mirror upstream 1:1.

use crate::types::{EntityEdge, EntityNode, EpisodicNode};

/// Aggregated multi-scope search results.
///
/// Upstream: `SearchResults` model. Each result vector is paired with its
/// reranker score vector (same length / order as produced by the per-scope
/// search). Community fields are deferred to Phase 4.
#[derive(Debug, Default, Clone)]
pub struct SearchResults {
    /// Reranked entity edges (edge scope).
    pub edges: Vec<EntityEdge>,
    /// Reranker scores aligned to `edges` (see per-scope quirks for node_distance).
    pub edge_reranker_scores: Vec<f64>,
    /// Reranked entity nodes (node scope).
    pub nodes: Vec<EntityNode>,
    /// Reranker scores aligned to `nodes`.
    pub node_reranker_scores: Vec<f64>,
    /// Reranked episodes (episode scope).
    pub episodes: Vec<EpisodicNode>,
    /// Reranker scores aligned to `episodes`.
    pub episode_reranker_scores: Vec<f64>,
    // community fields deferred to Phase 4 (upstream community scope) — ledger
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_empty() {
        let r = SearchResults::default();
        assert!(r.edges.is_empty());
        assert!(r.edge_reranker_scores.is_empty());
        assert!(r.nodes.is_empty());
        assert!(r.node_reranker_scores.is_empty());
        assert!(r.episodes.is_empty());
        assert!(r.episode_reranker_scores.is_empty());
    }
}
