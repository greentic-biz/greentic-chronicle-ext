// Ported from graphiti_core/search/search_config.py::SearchResults @ 34f56e65.
//
// Phase-4: the community scope landed; `communities` + `community_reranker_scores`
// are now present (default-empty), so this struct mirrors upstream 1:1. The
// Phase-2 "communities omitted" ledger note is hereby removed.

use crate::types::{CommunityNode, EntityEdge, EntityNode, EpisodicNode};

/// Aggregated multi-scope search results.
///
/// Upstream: `SearchResults` model. Each result vector is paired with its
/// reranker score vector (same length / order as produced by the per-scope
/// search).
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
    /// Reranked communities (community scope). Phase-4 addition.
    pub communities: Vec<CommunityNode>,
    /// Reranker scores aligned to `communities`. Phase-4 addition.
    pub community_reranker_scores: Vec<f64>,
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
        assert!(r.communities.is_empty());
        assert!(r.community_reranker_scores.is_empty());
    }
}
