use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use chronicle_core::driver::{
    DriverError, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SchemaOps,
    SearchOps,
};
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};

// ---------------------------------------------------------------------------
// Inner state — all fields Default
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Inner {
    entity_nodes: HashMap<String, EntityNode>,
    entity_edges: HashMap<String, EntityEdge>,
    episodic_nodes: HashMap<String, EpisodicNode>,
    episodic_edges: Vec<EpisodicEdge>,
}

// ---------------------------------------------------------------------------
// FakeDriver
// ---------------------------------------------------------------------------

/// In-memory driver suitable for unit and integration tests.
/// All state is guarded by a single `std::sync::Mutex`; lock-poisoning
/// surfaces as `DriverError::Query("poisoned lock")`.
#[derive(Default)]
pub struct FakeDriver {
    inner: Mutex<Inner>,
}

impl FakeDriver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored [`EpisodicEdge`] entries.
    /// If the lock is poisoned, returns `0`.
    pub fn episodic_edge_count(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.episodic_edges.len())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Helper: cosine similarity (zero-norm → 0.0)
// ---------------------------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

macro_rules! lock {
    ($self:expr) => {
        $self
            .inner
            .lock()
            .map_err(|_| DriverError::Query("poisoned lock".into()))
    };
}

// ---------------------------------------------------------------------------
// EntityNodeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EntityNodeOps for FakeDriver {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for n in nodes {
            g.entity_nodes.insert(n.uuid.clone(), n.clone());
        }
        Ok(())
    }

    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_nodes.get(uuid).cloned())
    }

    async fn get_entity_nodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        // Preserve input order; skip missing entries.
        Ok(uuids
            .iter()
            .filter_map(|id| g.entity_nodes.get(id).cloned())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// EntityEdgeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EntityEdgeOps for FakeDriver {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for e in edges {
            g.entity_edges.insert(e.uuid.clone(), e.clone());
        }
        Ok(())
    }

    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_edges.get(uuid).cloned())
    }

    /// Single-direction lookup: edges where `source_node_uuid == source` AND
    /// `target_node_uuid == target`. Mirrors upstream Neo4j directed MATCH.
    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        Ok(g.entity_edges
            .values()
            .filter(|e| e.source_node_uuid == source_uuid && e.target_node_uuid == target_uuid)
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// EpisodeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EpisodeOps for FakeDriver {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        g.episodic_nodes
            .insert(episode.uuid.clone(), episode.clone());
        Ok(())
    }

    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        Ok(g.episodic_nodes.get(uuid).cloned())
    }

    async fn get_episodes_by_uuids(
        &self,
        uuids: &[String],
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        Ok(uuids
            .iter()
            .filter_map(|id| g.episodic_nodes.get(id).cloned())
            .collect())
    }

    /// Mirrors upstream `retrieve_episodes`:
    /// 1. Filter: `valid_at <= reference_time`.
    /// 2. Filter: `group_id` in `group_ids` (only when `group_ids` is non-empty).
    /// 3. Filter: `source == source` (only when `source` is `Some`).
    /// 4. Sort by `valid_at` DESC (most-recent first).
    /// 5. Take `last_n`.
    /// 6. Reverse to chronological order (oldest first) before returning.
    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError> {
        let g = lock!(self)?;
        let mut eligible: Vec<&EpisodicNode> = g
            .episodic_nodes
            .values()
            .filter(|ep| {
                ep.valid_at <= reference_time
                    && (group_ids.is_empty() || group_ids.contains(&ep.group_id))
                    && source.is_none_or(|s| ep.source == s)
            })
            .collect();

        // Sort DESC by valid_at (most-recent first).
        eligible.sort_by_key(|b| Reverse(b.valid_at));

        // Take last_n most-recent, then reverse to chronological.
        let mut result: Vec<EpisodicNode> = eligible.into_iter().take(last_n).cloned().collect();
        result.reverse();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// EpisodicEdgeOps
// ---------------------------------------------------------------------------

#[async_trait]
impl EpisodicEdgeOps for FakeDriver {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError> {
        let mut g = lock!(self)?;
        for e in edges {
            g.episodic_edges.push(e.clone());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SearchOps
// ---------------------------------------------------------------------------

#[async_trait]
impl SearchOps for FakeDriver {
    /// Approximation of BM25 fulltext: case-insensitive substring match of
    /// `query` against `fact` + `name` fields. Group-filtered when
    /// `group_ids` is non-empty. Results truncated to `limit`.
    /// NOTE: Not a real BM25 ranking — only tests that need deterministic
    /// recall (not ranking precision) should use this.
    /// Result order within the limit is undefined (HashMap iteration); only assert recall, not ranking.
    async fn edge_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        let results: Vec<EntityEdge> = g
            .entity_edges
            .values()
            .filter(|e| {
                let text = format!("{} {}", e.fact, e.name).to_lowercase();
                text.contains(&q) && (group_ids.is_empty() || group_ids.contains(&e.group_id))
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Brute-force cosine similarity against stored `fact_embedding`.
    /// Records without embeddings are skipped.
    /// Results filtered by `score > min_score`, sorted DESC by score, truncated to `limit`.
    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError> {
        let g = lock!(self)?;
        let mut scored: Vec<(f32, EntityEdge)> = g
            .entity_edges
            .values()
            .filter_map(|e| {
                let emb = e.fact_embedding.as_deref()?;
                let score = cosine(search_vector, emb);
                if score > min_score && (group_ids.is_empty() || group_ids.contains(&e.group_id)) {
                    Some((score, e.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.uuid.cmp(&b.1.uuid))
        });
        Ok(scored.into_iter().take(limit).map(|(_, e)| e).collect())
    }

    /// Approximation of BM25 fulltext: case-insensitive substring match of
    /// `query` against `name` + `summary` fields. Group-filtered when
    /// `group_ids` is non-empty. Results truncated to `limit`.
    /// Result order within the limit is undefined (HashMap iteration); only assert recall, not ranking.
    async fn node_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        let q = query.to_lowercase();
        let results: Vec<EntityNode> = g
            .entity_nodes
            .values()
            .filter(|n| {
                let text = format!("{} {}", n.name, n.summary).to_lowercase();
                text.contains(&q) && (group_ids.is_empty() || group_ids.contains(&n.group_id))
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Brute-force cosine similarity against stored `name_embedding`.
    /// Records without embeddings are skipped.
    /// Results filtered by `score > min_score`, sorted DESC by score, truncated to `limit`.
    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError> {
        let g = lock!(self)?;
        let mut scored: Vec<(f32, EntityNode)> = g
            .entity_nodes
            .values()
            .filter_map(|n| {
                let emb = n.name_embedding.as_deref()?;
                let score = cosine(search_vector, emb);
                if score > min_score && (group_ids.is_empty() || group_ids.contains(&n.group_id)) {
                    Some((score, n.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.uuid.cmp(&b.1.uuid))
        });
        Ok(scored.into_iter().take(limit).map(|(_, n)| n).collect())
    }
}

// ---------------------------------------------------------------------------
// SchemaOps
// ---------------------------------------------------------------------------

#[async_trait]
impl SchemaOps for FakeDriver {
    async fn build_indices_and_constraints(
        &self,
        _delete_existing: bool,
    ) -> Result<(), DriverError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphDriver (supertrait)
// ---------------------------------------------------------------------------

impl GraphDriver for FakeDriver {
    fn provider(&self) -> &'static str {
        "fake"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_embedder::MockEmbedder;
    use chronicle_core::embedder::EmbedderClient;
    use chrono::{TimeZone, Utc};

    fn make_node(uuid: &str, name: &str, group_id: &str) -> EntityNode {
        let mut n = EntityNode::new(name.into(), group_id.into(), Utc::now());
        n.uuid = uuid.into();
        n
    }

    fn make_edge(
        uuid: &str,
        src: &str,
        tgt: &str,
        name: &str,
        fact: &str,
        group_id: &str,
    ) -> EntityEdge {
        let mut e = EntityEdge::new(
            src.into(),
            tgt.into(),
            name.into(),
            fact.into(),
            group_id.into(),
        );
        e.uuid = uuid.into();
        e
    }

    fn make_episode(
        uuid: &str,
        group_id: &str,
        source: EpisodeType,
        valid_at: DateTime<Utc>,
    ) -> EpisodicNode {
        let mut ep = EpisodicNode::new(
            "ep".into(),
            group_id.into(),
            source,
            "desc".into(),
            "content".into(),
            Utc::now(),
            valid_at,
        );
        ep.uuid = uuid.into();
        ep
    }

    // ------------------------------------------------------------------
    // EntityNode roundtrip
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_entity_node_roundtrip() {
        let driver = FakeDriver::new();
        let node = make_node("n1", "Alice", "g1");
        driver
            .save_entity_nodes(std::slice::from_ref(&node))
            .await
            .unwrap();
        let got = driver.get_entity_node("n1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "n1");
        assert_eq!(got.name, "Alice");
        assert_eq!(got.group_id, "g1");
    }

    #[tokio::test]
    async fn get_entity_nodes_by_uuids_preserves_order_skips_missing() {
        let driver = FakeDriver::new();
        let a = make_node("a", "A", "g1");
        let b = make_node("b", "B", "g1");
        driver.save_entity_nodes(&[a, b]).await.unwrap();
        let got = driver
            .get_entity_nodes_by_uuids(&["b".to_string(), "a".to_string(), "missing".to_string()])
            .await
            .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].uuid, "b");
        assert_eq!(got[1].uuid, "a");
    }

    // ------------------------------------------------------------------
    // EntityEdge roundtrip + directionality
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_entity_edge_roundtrip() {
        let driver = FakeDriver::new();
        let edge = make_edge("e1", "u1", "u2", "WORKS_AT", "Alice works at Acme", "g1");
        driver
            .save_entity_edges(std::slice::from_ref(&edge))
            .await
            .unwrap();
        let got = driver.get_entity_edge("e1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "e1");
        assert_eq!(got.fact, "Alice works at Acme");
    }

    #[tokio::test]
    async fn get_edges_between_nodes_is_directional() {
        let driver = FakeDriver::new();
        let edge = make_edge("e1", "a", "b", "REL", "a to b", "g1");
        driver.save_entity_edges(&[edge]).await.unwrap();

        // Forward direction: found
        let fwd = driver.get_edges_between_nodes("a", "b").await.unwrap();
        assert_eq!(fwd.len(), 1, "forward direction should find the edge");

        // Reverse direction: NOT found
        let rev = driver.get_edges_between_nodes("b", "a").await.unwrap();
        assert!(rev.is_empty(), "reverse direction must return nothing");
    }

    // ------------------------------------------------------------------
    // EpisodicNode roundtrip
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_get_episode_roundtrip() {
        let driver = FakeDriver::new();
        let ep = make_episode("ep1", "g1", EpisodeType::Message, Utc::now());
        driver.save_episode(&ep).await.unwrap();
        let got = driver.get_episode("ep1").await.unwrap().unwrap();
        assert_eq!(got.uuid, "ep1");
        assert_eq!(got.group_id, "g1");
    }

    // ------------------------------------------------------------------
    // retrieve_episodes: ordering, cutoff, last_n
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn retrieve_episodes_chronological_with_cutoff_and_last_n() {
        let driver = FakeDriver::new();
        let t1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2026, 1, 3, 0, 0, 0).unwrap();
        let future = Utc.with_ymd_and_hms(2026, 1, 10, 0, 0, 0).unwrap();

        // Insert three in order and one in the future
        driver
            .save_episode(&make_episode("ep-t3", "g1", EpisodeType::Message, t3))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode("ep-t1", "g1", EpisodeType::Message, t1))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode("ep-t2", "g1", EpisodeType::Message, t2))
            .await
            .unwrap();
        driver
            .save_episode(&make_episode(
                "ep-future",
                "g1",
                EpisodeType::Message,
                future,
            ))
            .await
            .unwrap();

        // Reference time = t3 (future episode excluded), last_n=2
        let result = driver
            .retrieve_episodes(t3, 2, &["g1".to_string()], None)
            .await
            .unwrap();

        assert_eq!(result.len(), 2, "should return exactly 2 episodes");
        // Chronological order: t2 then t3 (the 2 most-recent of the 3 eligible)
        assert_eq!(result[0].uuid, "ep-t2");
        assert_eq!(result[1].uuid, "ep-t3");
    }

    // ------------------------------------------------------------------
    // Similarity search: alpha ranks first
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn edge_similarity_search_ranks_matching_string_first() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut edge_alpha = make_edge("e-alpha", "u1", "u2", "REL", "alpha", "g1");
        edge_alpha.fact_embedding = Some(emb.create("alpha").await.unwrap());

        let mut edge_beta = make_edge("e-beta", "u3", "u4", "REL", "beta", "g1");
        edge_beta.fact_embedding = Some(emb.create("beta").await.unwrap());

        driver
            .save_entity_edges(&[edge_alpha, edge_beta])
            .await
            .unwrap();

        let query_vec = emb.create("alpha").await.unwrap();
        let results = driver
            .edge_similarity_search(&query_vec, &[], 10, 0.0)
            .await
            .unwrap();

        assert!(!results.is_empty(), "should return at least one result");
        assert_eq!(results[0].fact, "alpha", "alpha edge should rank first");

        // Verify score ~ 1.0 for exact match
        let score = cosine(&query_vec, &emb.create("alpha").await.unwrap());
        assert!((score - 1.0).abs() < 1e-5, "cosine score should be ~1.0");
    }

    // ------------------------------------------------------------------
    // Fulltext: substring + group filter
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn edge_fulltext_search_substring_and_group_filter() {
        let driver = FakeDriver::new();
        let e1 = make_edge("e1", "u1", "u2", "REL", "Alice works at Acme", "grp-a");
        let e2 = make_edge("e2", "u3", "u4", "REL", "Bob works at Beta Inc", "grp-b");
        driver.save_entity_edges(&[e1, e2]).await.unwrap();

        // Substring match without group filter — both match "works"
        let all = driver.edge_fulltext_search("works", &[], 10).await.unwrap();
        assert_eq!(all.len(), 2);

        // Group filter restricts to grp-a only
        let filtered = driver
            .edge_fulltext_search("works", &["grp-a".to_string()], 10)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].uuid, "e1");
    }

    #[tokio::test]
    async fn node_fulltext_search_substring_match() {
        let driver = FakeDriver::new();
        let mut n1 = make_node("n1", "Alice Smith", "g1");
        n1.summary = "engineer at Acme".into();
        let n2 = make_node("n2", "Bob Jones", "g1");
        driver.save_entity_nodes(&[n1, n2]).await.unwrap();

        let results = driver.node_fulltext_search("alice", &[], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uuid, "n1");
    }

    // ------------------------------------------------------------------
    // EpisodicEdge save
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn save_episodic_edges_accumulates() {
        let driver = FakeDriver::new();
        let e1 = EpisodicEdge::new("ep1".into(), "n1".into(), "g1".into(), Utc::now());
        let e2 = EpisodicEdge::new("ep2".into(), "n2".into(), "g1".into(), Utc::now());
        driver.save_episodic_edges(&[e1]).await.unwrap();
        driver.save_episodic_edges(&[e2]).await.unwrap();
        assert_eq!(driver.episodic_edge_count(), 2);
    }

    // ------------------------------------------------------------------
    // Node similarity search: ranking + skip-no-embedding
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn node_similarity_search_ranks_matching_string_first() {
        let driver = FakeDriver::new();
        let emb = MockEmbedder::new(8);

        let mut node_alpha = make_node("n-alpha", "alpha", "g1");
        node_alpha.name_embedding = Some(emb.create("alpha").await.unwrap());

        let mut node_beta = make_node("n-beta", "beta", "g1");
        node_beta.name_embedding = Some(emb.create("beta").await.unwrap());

        // node without embedding — must be skipped, not panic
        let node_no_emb = make_node("n-no-emb", "gamma", "g1");

        driver
            .save_entity_nodes(&[node_alpha, node_beta, node_no_emb])
            .await
            .unwrap();

        let query_vec = emb.create("alpha").await.unwrap();
        let results = driver
            .node_similarity_search(&query_vec, &[], 10, 0.0)
            .await
            .unwrap();

        // node without embedding must not appear in results
        assert!(
            results.iter().all(|n| n.uuid != "n-no-emb"),
            "node without embedding must be skipped"
        );
        assert!(!results.is_empty(), "should return at least one result");
        assert_eq!(results[0].name, "alpha", "alpha node should rank first");

        // Verify score ~ 1.0 for exact match
        let score = cosine(&query_vec, &emb.create("alpha").await.unwrap());
        assert!((score - 1.0).abs() < 1e-5, "cosine score should be ~1.0");
    }

    // ------------------------------------------------------------------
    // retrieve_episodes: empty group_ids returns all groups
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn retrieve_episodes_empty_group_ids_returns_all_groups() {
        let driver = FakeDriver::new();
        let now = Utc::now();
        let ep_g1 = make_episode("ep-g1", "g1", EpisodeType::Message, now);
        let ep_g2 = make_episode("ep-g2", "g2", EpisodeType::Message, now);
        driver.save_episode(&ep_g1).await.unwrap();
        driver.save_episode(&ep_g2).await.unwrap();

        // Empty group_ids slice — must return episodes from ALL groups
        let results = driver.retrieve_episodes(now, 100, &[], None).await.unwrap();

        let uuids: Vec<&str> = results.iter().map(|ep| ep.uuid.as_str()).collect();
        assert!(
            uuids.contains(&"ep-g1"),
            "ep-g1 (group g1) should be returned"
        );
        assert!(
            uuids.contains(&"ep-g2"),
            "ep-g2 (group g2) should be returned"
        );
    }

    // ------------------------------------------------------------------
    // Schema ops no-op
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn schema_build_is_noop() {
        let driver = FakeDriver::new();
        driver.build_indices_and_constraints(true).await.unwrap();
    }

    // ------------------------------------------------------------------
    // Provider name
    // ------------------------------------------------------------------
    #[test]
    fn provider_name_is_fake() {
        let driver = FakeDriver::new();
        assert_eq!(driver.provider(), "fake");
    }
}
