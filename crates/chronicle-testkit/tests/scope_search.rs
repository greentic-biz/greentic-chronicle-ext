// Integration tests for the Phase-2 scope searches + top-level search.
//
// Placed here (in chronicle-testkit) rather than in chronicle-core to avoid a
// Cargo dev-dependency cycle: chronicle-core → chronicle-testkit → chronicle-core.
//
// Covers (plan Task 5):
//   - edge per-reranker dispatch: rrf baseline, mmr ordering, episode_mentions
//     post-sort, node_distance happy + missing-center error, cross_encoder
//     incl. min_score filter + missing-encoder error.
//   - node bfs self-seed (no origins → reaches neighbor of a bm25 hit).
//   - episode rrf + cross_encoder.
//   - top-level multi-scope assembly + empty-query guard + embed decision
//     (MockEmbedder call count == 0 when no scope uses cosine/mmr).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use chrono::Utc;

use chronicle_core::driver::{EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps};
use chronicle_core::embedder::{EmbedderClient, EmbedderError};
use chronicle_core::search::config::{
    EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, EpisodeReranker, EpisodeSearchConfig,
    EpisodeSearchMethod, NodeReranker, NodeSearchConfig, NodeSearchMethod, SearchConfig,
};
use chronicle_core::search::edge_search::edge_search;
use chronicle_core::search::episode_search::episode_search;
use chronicle_core::search::filters::SearchFilters;
use chronicle_core::search::node_search::node_search;
use chronicle_core::search::search::search;
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};

use chronicle_testkit::{FakeDriver, MockCrossEncoder, MockEmbedder};

// ── helpers ──────────────────────────────────────────────────────────────────

fn edge(uuid: &str, src: &str, tgt: &str, fact: &str, group: &str) -> EntityEdge {
    let mut e = EntityEdge::new(
        src.into(),
        tgt.into(),
        "REL".into(),
        fact.into(),
        group.into(),
    );
    e.uuid = uuid.into();
    e
}

fn node(uuid: &str, name: &str, group: &str) -> EntityNode {
    let mut n = EntityNode::new(name.into(), group.into(), Utc::now());
    n.uuid = uuid.into();
    n
}

fn episode(uuid: &str, content: &str, group: &str) -> EpisodicNode {
    let mut ep = EpisodicNode::new(
        "ep".into(),
        group.into(),
        EpisodeType::Message,
        "desc".into(),
        content.into(),
        Utc::now(),
        Utc::now(),
    );
    ep.uuid = uuid.into();
    ep
}

/// Counting embedder wrapper: records how many times `create` is called so we
/// can pin the top-level embed decision (0 calls when no cosine/mmr scope).
struct CountingEmbedder {
    inner: MockEmbedder,
    calls: Arc<AtomicUsize>,
}

impl CountingEmbedder {
    fn new(dim: usize) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                inner: MockEmbedder::new(dim),
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait]
impl EmbedderClient for CountingEmbedder {
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.create(input).await
    }
    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.calls.fetch_add(inputs.len(), Ordering::SeqCst);
        self.inner.create_batch(inputs).await
    }
}

fn edge_cfg(methods: Vec<EdgeSearchMethod>, reranker: EdgeReranker) -> EdgeSearchConfig {
    EdgeSearchConfig {
        search_methods: methods,
        reranker,
        sim_min_score: 0.0,
        mmr_lambda: 0.5,
        bfs_max_depth: 3,
    }
}

// ── edge: RRF baseline ───────────────────────────────────────────────────────

#[tokio::test]
async fn edge_rrf_baseline_fuses_bm25_and_cosine() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);
    let q = "alpha";

    let mut e1 = edge("e1", "s1", "t1", "alpha in fact", "g1");
    e1.fact_embedding = Some(emb.create(q).await.unwrap());
    let e2 = edge("e2", "s2", "t2", "alpha appears too", "g1"); // bm25 only
    let mut e3 = edge("e3", "s3", "t3", "unrelated zzz", "g1");
    e3.fact_embedding = Some(emb.create(q).await.unwrap()); // cosine only
    driver.save_entity_edges(&[e1, e2, e3]).await.unwrap();

    let qv = emb.create(q).await.unwrap();
    let cfg = edge_cfg(
        vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
        EdgeReranker::Rrf,
    );
    let (edges, scores) = edge_search(
        &driver,
        None,
        q,
        &qv,
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap();

    assert_eq!(edges.len(), 3);
    assert_eq!(edges[0].uuid, "e1", "e1 in both lists → top RRF");
    assert_eq!(scores.len(), 3);
    assert!(scores[0] >= scores[1]);
}

// ── edge: MMR ordering with seeded embeddings ────────────────────────────────

#[tokio::test]
async fn edge_mmr_orders_by_relevance_lambda_one() {
    // lambda=1.0 → pure relevance; query aligned with e_near.
    let driver = FakeDriver::new();
    // 2-d embeddings (cosine-search uses fact_embedding; MMR reloads it).
    let mut e_near = edge("e_near", "s", "t", "near fact", "g1");
    e_near.fact_embedding = Some(vec![1.0, 0.0]);
    let mut e_mid = edge("e_mid", "s", "t", "mid fact", "g1");
    e_mid.fact_embedding = Some(vec![1.0, 1.0]);
    let mut e_far = edge("e_far", "s", "t", "far fact", "g1");
    e_far.fact_embedding = Some(vec![0.0, 1.0]);
    driver
        .save_entity_edges(&[e_near, e_mid, e_far])
        .await
        .unwrap();

    let qv = vec![1.0_f32, 0.0];
    let cfg = EdgeSearchConfig {
        search_methods: vec![EdgeSearchMethod::CosineSimilarity],
        reranker: EdgeReranker::Mmr,
        sim_min_score: -1.0,
        mmr_lambda: 1.0,
        bfs_max_depth: 3,
    };
    let (edges, _scores) = edge_search(
        &driver,
        None,
        "q",
        &qv,
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        -2.0,
    )
    .await
    .unwrap();

    let order: Vec<&str> = edges.iter().map(|e| e.uuid.as_str()).collect();
    assert_eq!(order, vec!["e_near", "e_mid", "e_far"]);
}

// ── edge: episode_mentions post-sort by episodes.len() desc ──────────────────

#[tokio::test]
async fn edge_episode_mentions_postsort_by_episode_count() {
    let driver = FakeDriver::new();
    let mut e_few = edge("e_few", "s", "t", "fact many", "g1");
    e_few.episodes = vec!["ep1".into()]; // 1 episode
    let mut e_many = edge("e_many", "s", "t", "fact many too", "g1");
    e_many.episodes = vec!["ep1".into(), "ep2".into(), "ep3".into()]; // 3 episodes
    driver.save_entity_edges(&[e_few, e_many]).await.unwrap();

    let cfg = edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::EpisodeMentions);
    let (edges, _scores) = edge_search(
        &driver,
        None,
        "fact",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap();

    assert_eq!(edges.len(), 2);
    assert_eq!(
        edges[0].uuid, "e_many",
        "edge with more episodes sorts first (desc by episodes.len())"
    );
    assert_eq!(edges[1].uuid, "e_few");
}

// ── edge: node_distance happy path ───────────────────────────────────────────

#[tokio::test]
async fn edge_node_distance_orders_adjacent_source_first() {
    let driver = FakeDriver::new();
    // center; near_src is RELATES_TO-adjacent to center; far_src is not.
    // Edges to rank: from near_src and far_src.
    let adj = edge("adj", "near_src", "center", "link", "g1"); // near_src - center adjacency
    let e_near = edge("e_near", "near_src", "x", "fact near", "g1");
    let e_far = edge("e_far", "far_src", "y", "fact far", "g1");
    driver
        .save_entity_edges(&[adj, e_near, e_far])
        .await
        .unwrap();

    let cfg = edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::NodeDistance);
    let (edges, _scores) = edge_search(
        &driver,
        None,
        "fact",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        Some("center"),
        None,
        10,
        0.0,
    )
    .await
    .unwrap();

    // e_near's source (near_src) is adjacent to center → ranks before e_far.
    let near_pos = edges.iter().position(|e| e.uuid == "e_near");
    let far_pos = edges.iter().position(|e| e.uuid == "e_far");
    assert!(near_pos.is_some() && far_pos.is_some());
    assert!(
        near_pos < far_pos,
        "adjacent source's edge ranks before non-adjacent (order {:?})",
        edges.iter().map(|e| &e.uuid).collect::<Vec<_>>()
    );
}

// ── edge: node_distance missing center → InvalidInput ────────────────────────

#[tokio::test]
async fn edge_node_distance_missing_center_errors() {
    let driver = FakeDriver::new();
    let e = edge("e1", "s", "t", "fact", "g1");
    driver.save_entity_edges(&[e]).await.unwrap();

    let cfg = edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::NodeDistance);
    let err = edge_search(
        &driver,
        None,
        "fact",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("center node"), "got: {msg}");
}

// ── edge: cross_encoder incl. min_score filter ───────────────────────────────

#[tokio::test]
async fn edge_cross_encoder_ranks_facts_and_filters_min_score() {
    let driver = FakeDriver::new();
    let e1 = edge("e1", "s", "t", "relevant fact", "g1");
    let e2 = edge("e2", "s", "t", "marginal fact", "g1");
    let e3 = edge("e3", "s", "t", "irrelevant fact", "g1");
    driver.save_entity_edges(&[e1, e2, e3]).await.unwrap();

    let ce = MockCrossEncoder::from_pairs([
        ("relevant fact", 0.9),
        ("marginal fact", 0.5),
        ("irrelevant fact", 0.1),
    ]);
    let cfg = edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::CrossEncoder);
    let (edges, scores) = edge_search(
        &driver,
        Some(&ce),
        "fact",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.4, // min_score filters out e3 (0.1)
    )
    .await
    .unwrap();

    let order: Vec<&str> = edges.iter().map(|e| e.uuid.as_str()).collect();
    assert_eq!(order, vec!["e1", "e2"], "e3 below min_score → dropped");
    assert!(scores[0] >= scores[1]);
}

// ── edge: cross_encoder missing encoder → InvalidInput ───────────────────────

#[tokio::test]
async fn edge_cross_encoder_missing_client_errors() {
    let driver = FakeDriver::new();
    let e = edge("e1", "s", "t", "fact", "g1");
    driver.save_entity_edges(&[e]).await.unwrap();

    let cfg = edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::CrossEncoder);
    let err = edge_search(
        &driver,
        None,
        "fact",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap_err();
    assert!(format!("{err}").contains("cross_encoder"));
}

// ── edge: config None → empty ────────────────────────────────────────────────

#[tokio::test]
async fn edge_search_none_config_returns_empty() {
    let driver = FakeDriver::new();
    let (edges, scores) = edge_search(
        &driver,
        None,
        "q",
        &[],
        &[],
        None,
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap();
    assert!(edges.is_empty() && scores.is_empty());
}

// ── node: bfs self-seed reaches neighbor of bm25 hit ─────────────────────────

#[tokio::test]
async fn node_bfs_self_seed_reaches_neighbor() {
    let driver = FakeDriver::new();
    // bm25 hits "alpha" node; alpha -RELATES_TO-> beta (not a bm25 hit).
    let alpha = node("alpha", "alpha", "g1");
    let beta = node("beta", "beta", "g1");
    driver.save_entity_nodes(&[alpha, beta]).await.unwrap();
    let ab = edge("ab", "alpha", "beta", "a-b", "g1");
    driver.save_entity_edges(&[ab]).await.unwrap();

    let cfg = NodeSearchConfig {
        search_methods: vec![NodeSearchMethod::Bm25, NodeSearchMethod::BreadthFirstSearch],
        reranker: NodeReranker::Rrf,
        sim_min_score: 0.0,
        mmr_lambda: 0.5,
        bfs_max_depth: 2,
    };
    // bfs_origin None → self-seed from bm25 hit (alpha) → reaches beta.
    let (nodes, _scores) = node_search(
        &driver,
        None,
        "alpha",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap();

    let uuids: Vec<&str> = nodes.iter().map(|n| n.uuid.as_str()).collect();
    assert!(uuids.contains(&"alpha"), "bm25 hit present");
    assert!(
        uuids.contains(&"beta"),
        "self-seeded BFS reaches neighbor beta (order {uuids:?})"
    );
}

// ── node: episode_mentions reranker (DB-count based) ─────────────────────────

#[tokio::test]
async fn node_episode_mentions_keeps_unmentioned_last() {
    let driver = FakeDriver::new();
    let n_mentioned = node("nm", "alpha mentioned", "g1");
    let n_unmentioned = node("nu", "alpha unmentioned", "g1");
    driver
        .save_entity_nodes(&[n_mentioned, n_unmentioned])
        .await
        .unwrap();
    // nm mentioned once; nu has zero mentions.
    let m = EpisodicEdge::new("ep1".into(), "nm".into(), "g1".into(), Utc::now());
    driver.save_episodic_edges(&[m]).await.unwrap();

    let cfg = NodeSearchConfig {
        search_methods: vec![NodeSearchMethod::Bm25],
        reranker: NodeReranker::EpisodeMentions,
        sim_min_score: 0.0,
        mmr_lambda: 0.5,
        bfs_max_depth: 3,
    };
    let (nodes, _scores) = node_search(
        &driver,
        None,
        "alpha",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        None,
        None,
        10,
        0.0,
    )
    .await
    .unwrap();

    // UPSTREAM QUIRK: ASC by mention count (fewer first), unmentioned (inf) last.
    assert_eq!(nodes.len(), 2);
    assert_eq!(
        nodes[0].uuid, "nm",
        "mentioned (count 1) sorts before unmentioned (inf)"
    );
    assert_eq!(nodes[1].uuid, "nu");
}

// ── episode: rrf ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn episode_rrf_returns_bm25_hits() {
    let driver = FakeDriver::new();
    let ep1 = episode("ep1", "alpha content", "g1");
    let ep2 = episode("ep2", "alpha also", "g1");
    let ep3 = episode("ep3", "unrelated", "g1");
    driver.save_episode(&ep1).await.unwrap();
    driver.save_episode(&ep2).await.unwrap();
    driver.save_episode(&ep3).await.unwrap();

    let cfg = EpisodeSearchConfig {
        search_methods: vec![EpisodeSearchMethod::Bm25],
        reranker: EpisodeReranker::Rrf,
        sim_min_score: 0.0,
        mmr_lambda: 0.5,
        bfs_max_depth: 3,
    };
    let (episodes, scores) = episode_search(
        &driver,
        None,
        "alpha",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        10,
        0.0,
    )
    .await
    .unwrap();

    let uuids: Vec<&str> = episodes.iter().map(|e| e.uuid.as_str()).collect();
    assert_eq!(episodes.len(), 2, "two alpha hits");
    assert!(uuids.contains(&"ep1") && uuids.contains(&"ep2"));
    assert_eq!(scores.len(), 2);
}

// ── episode: cross_encoder ───────────────────────────────────────────────────

#[tokio::test]
async fn episode_cross_encoder_ranks_content() {
    let driver = FakeDriver::new();
    let ep1 = episode("ep1", "alpha primary", "g1");
    let ep2 = episode("ep2", "alpha secondary", "g1");
    driver.save_episode(&ep1).await.unwrap();
    driver.save_episode(&ep2).await.unwrap();

    let ce = MockCrossEncoder::from_pairs([("alpha primary", 0.9), ("alpha secondary", 0.3)]);
    let cfg = EpisodeSearchConfig {
        search_methods: vec![EpisodeSearchMethod::Bm25],
        reranker: EpisodeReranker::CrossEncoder,
        sim_min_score: 0.0,
        mmr_lambda: 0.5,
        bfs_max_depth: 3,
    };
    let (episodes, _scores) = episode_search(
        &driver,
        Some(&ce),
        "alpha",
        &[],
        &["g1".into()],
        Some(&cfg),
        &SearchFilters::default(),
        10,
        0.0,
    )
    .await
    .unwrap();

    assert_eq!(episodes[0].uuid, "ep1", "higher cross-encoder score first");
}

// ── top-level: multi-scope assembly ──────────────────────────────────────────

#[tokio::test]
async fn top_level_search_assembles_all_three_scopes() {
    let driver = FakeDriver::new();
    let n = node("n1", "alpha node", "g1");
    driver.save_entity_nodes(&[n]).await.unwrap();
    let e = edge("e1", "n1", "n2", "alpha fact", "g1");
    driver.save_entity_edges(&[e]).await.unwrap();
    let ep = episode("ep1", "alpha episode", "g1");
    driver.save_episode(&ep).await.unwrap();

    let cfg = SearchConfig {
        edge_config: Some(edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::Rrf)),
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25],
            reranker: NodeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        episode_config: Some(EpisodeSearchConfig {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        limit: 10,
        reranker_min_score: 0.0,
    };
    let emb = MockEmbedder::new(8);
    let results = search(
        &driver,
        &emb,
        None,
        "alpha",
        &["g1".into()],
        &cfg,
        &SearchFilters::default(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(results.edges.len(), 1);
    assert_eq!(results.nodes.len(), 1);
    assert_eq!(results.episodes.len(), 1);
}

// ── top-level: empty query guard ─────────────────────────────────────────────

#[tokio::test]
async fn top_level_empty_query_returns_default() {
    let driver = FakeDriver::new();
    let e = edge("e1", "s", "t", "alpha", "g1");
    driver.save_entity_edges(&[e]).await.unwrap();

    let cfg = SearchConfig {
        edge_config: Some(edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::Rrf)),
        node_config: None,
        episode_config: None,
        limit: 10,
        reranker_min_score: 0.0,
    };
    let emb = MockEmbedder::new(8);
    let results = search(
        &driver,
        &emb,
        None,
        "   ",
        &["g1".into()],
        &cfg,
        &SearchFilters::default(),
        None,
        None,
    )
    .await
    .unwrap();
    assert!(results.edges.is_empty() && results.nodes.is_empty() && results.episodes.is_empty());
}

// ── top-level: embed decision — no cosine/mmr → 0 embedder calls ─────────────

#[tokio::test]
async fn top_level_no_cosine_or_mmr_skips_embedding() {
    let driver = FakeDriver::new();
    let e = edge("e1", "s", "t", "alpha", "g1");
    driver.save_entity_edges(&[e]).await.unwrap();

    // bm25-only edge scope + bm25 episode scope; no cosine, no mmr anywhere.
    let cfg = SearchConfig {
        edge_config: Some(edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::Rrf)),
        node_config: Some(NodeSearchConfig {
            search_methods: vec![NodeSearchMethod::Bm25],
            reranker: NodeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        episode_config: Some(EpisodeSearchConfig {
            search_methods: vec![EpisodeSearchMethod::Bm25],
            reranker: EpisodeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        limit: 10,
        reranker_min_score: 0.0,
    };
    let (emb, calls) = CountingEmbedder::new(8);
    let _ = search(
        &driver,
        &emb,
        None,
        "alpha",
        &["g1".into()],
        &cfg,
        &SearchFilters::default(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no cosine/mmr scope → query must NOT be embedded"
    );
}

#[tokio::test]
async fn top_level_cosine_scope_embeds_once() {
    let driver = FakeDriver::new();
    let mut e = edge("e1", "s", "t", "alpha", "g1");
    e.fact_embedding = Some(MockEmbedder::new(8).create("alpha").await.unwrap());
    driver.save_entity_edges(&[e]).await.unwrap();

    let cfg = SearchConfig {
        edge_config: Some(edge_cfg(
            vec![EdgeSearchMethod::CosineSimilarity],
            EdgeReranker::Rrf,
        )),
        node_config: None,
        episode_config: None,
        limit: 10,
        reranker_min_score: 0.0,
    };
    let (emb, calls) = CountingEmbedder::new(8);
    let _ = search(
        &driver,
        &emb,
        None,
        "alpha",
        &["g1".into()],
        &cfg,
        &SearchFilters::default(),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cosine scope → query embedded exactly once"
    );
}

// ── top-level: group_ids [""] normalized to no-filter ────────────────────────

#[tokio::test]
async fn top_level_empty_group_id_string_is_no_filter() {
    let driver = FakeDriver::new();
    let e_a = edge("ea", "s", "t", "alpha", "ga");
    let e_b = edge("eb", "s", "t", "alpha", "gb");
    driver.save_entity_edges(&[e_a, e_b]).await.unwrap();

    let cfg = SearchConfig {
        edge_config: Some(edge_cfg(vec![EdgeSearchMethod::Bm25], EdgeReranker::Rrf)),
        node_config: None,
        episode_config: None,
        limit: 10,
        reranker_min_score: 0.0,
    };
    let emb = MockEmbedder::new(8);
    // group_ids = [""] → treated as no filter → both groups returned.
    let results = search(
        &driver,
        &emb,
        None,
        "alpha",
        &[String::new()],
        &cfg,
        &SearchFilters::default(),
        None,
        None,
    )
    .await
    .unwrap();
    let uuids: Vec<&str> = results.edges.iter().map(|e| e.uuid.as_str()).collect();
    assert_eq!(results.edges.len(), 2, "both groups returned (no filter)");
    assert!(uuids.contains(&"ea") && uuids.contains(&"eb"));
}
