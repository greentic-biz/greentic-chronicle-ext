// Integration tests for chronicle_core::search::rerank node_distance_rerank and
// episode_mentions_rerank (the driver-backed rerankers).
//
// Placed here (in chronicle-testkit) rather than in chronicle-core to avoid a Cargo
// dev-dependency cycle: chronicle-core → chronicle-testkit → chronicle-core.
// chronicle-testkit already depends on chronicle-core, so tests here get both.
// (The pure maximal_marginal_relevance unit tests live in chronicle-core itself.)

use chrono::Utc;

use chronicle_core::driver::{EntityEdgeOps, EpisodicEdgeOps};
use chronicle_core::search::rerank::{episode_mentions_rerank, node_distance_rerank};
use chronicle_core::types::{EntityEdge, EpisodicEdge};

use chronicle_testkit::FakeDriver;

fn make_edge(uuid: &str, src: &str, tgt: &str, group_id: &str) -> EntityEdge {
    let mut e = EntityEdge::new(
        src.into(),
        tgt.into(),
        "REL".into(),
        format!("{src}-{tgt}"),
        group_id.into(),
    );
    e.uuid = uuid.into();
    e
}

// ── node_distance_rerank ──────────────────────────────────────────────────────

/// Adjacent node (1/1.0 = 1.0) ranks above an unreachable one (1/inf = 0.0).
/// With min_score = 0, the unreachable node is RETAINED at the end (0.0 >= 0).
#[tokio::test]
async fn node_distance_adjacent_above_unreachable_unreachable_kept_at_zero() {
    let driver = FakeDriver::new();
    // center -[REL]- adj ; "far" is unconnected.
    driver
        .save_entity_edges(&[make_edge("e1", "center", "adj", "g1")])
        .await
        .unwrap();

    let (uuids, scores) = node_distance_rerank(
        &driver,
        &["adj".to_string(), "far".to_string()],
        "center",
        0.0,
    )
    .await
    .unwrap();

    assert_eq!(uuids, vec!["adj".to_string(), "far".to_string()]);
    assert_eq!(scores, vec![1.0, 0.0], "1/1.0=1.0 then 1/inf=0.0");
}

/// With min_score > 0 the unreachable node (1/inf = 0.0) is filtered out.
#[tokio::test]
async fn node_distance_min_score_filters_unreachable() {
    let driver = FakeDriver::new();
    driver
        .save_entity_edges(&[make_edge("e1", "center", "adj", "g1")])
        .await
        .unwrap();

    let (uuids, scores) = node_distance_rerank(
        &driver,
        &["adj".to_string(), "far".to_string()],
        "center",
        0.5,
    )
    .await
    .unwrap();

    assert_eq!(uuids, vec!["adj".to_string()], "far (0.0) < 0.5 → dropped");
    assert_eq!(scores, vec![1.0]);
}

/// Center present in input → prepended with score 1/0.1 = 10.0.
#[tokio::test]
async fn node_distance_center_in_input_prepended_with_ten() {
    let driver = FakeDriver::new();
    driver
        .save_entity_edges(&[make_edge("e1", "center", "adj", "g1")])
        .await
        .unwrap();

    let (uuids, scores) = node_distance_rerank(
        &driver,
        &["adj".to_string(), "center".to_string()],
        "center",
        0.0,
    )
    .await
    .unwrap();

    assert_eq!(uuids, vec!["center".to_string(), "adj".to_string()]);
    assert_eq!(scores, vec![10.0, 1.0], "center 1/0.1=10.0 prepended");
}

/// Center has no adjacency in the graph at all → all candidates unreachable.
#[tokio::test]
async fn node_distance_missing_center_adjacency_all_unreachable() {
    let driver = FakeDriver::new();
    // No edges seeded → nothing is adjacent to center.
    let (uuids, scores) =
        node_distance_rerank(&driver, &["a".to_string(), "b".to_string()], "center", 0.0)
            .await
            .unwrap();

    // Both unreachable → 1/inf = 0.0, both retained (0.0 >= 0.0). Stable order.
    assert_eq!(uuids, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(scores, vec![0.0, 0.0]);
}

// ── episode_mentions_rerank ───────────────────────────────────────────────────

fn mention(episode: &str, target: &str) -> EpisodicEdge {
    EpisodicEdge::new(episode.into(), target.into(), "g1".into(), Utc::now())
}

/// UPSTREAM QUIRK: ASCENDING sort by mention count — the node with count 1 ranks
/// BEFORE the node with count 5.
#[tokio::test]
async fn episode_mentions_ascending_count_order() {
    let driver = FakeDriver::new();
    // low: 1 mention ; high: 5 mentions.
    let mut edges = vec![mention("ep0", "low")];
    for i in 0..5 {
        edges.push(mention(&format!("ep{i}"), "high"));
    }
    driver.save_episodic_edges(&edges).await.unwrap();

    let lists = vec![vec!["high".to_string(), "low".to_string()]];
    let (uuids, scores) = episode_mentions_rerank(&driver, &lists, 0.0).await.unwrap();

    assert_eq!(
        uuids,
        vec!["low".to_string(), "high".to_string()],
        "fewer mentions ranks higher (ASC quirk)"
    );
    assert_eq!(scores, vec![1.0, 5.0], "raw counts, ascending");
}

/// UPSTREAM QUIRK: unmentioned node carries `inf`; with min_score 0, `inf >= 0` is
/// true, so it is RETAINED at the END with score inf.
#[tokio::test]
async fn episode_mentions_unmentioned_retained_at_end_as_inf() {
    let driver = FakeDriver::new();
    driver
        .save_episodic_edges(&[mention("ep0", "mentioned")])
        .await
        .unwrap();

    let lists = vec![vec!["mentioned".to_string(), "ghost".to_string()]];
    let (uuids, scores) = episode_mentions_rerank(&driver, &lists, 0.0).await.unwrap();

    assert_eq!(uuids, vec!["mentioned".to_string(), "ghost".to_string()]);
    assert_eq!(scores[0], 1.0);
    assert!(scores[1].is_infinite(), "unmentioned → inf, kept at end");
}

/// The RRF presort over multiple lists feeds the (ascending) count sort. Here both
/// nodes have the SAME mention count, so the count sort is stable and the RRF order
/// survives as the tiebreak.
#[tokio::test]
async fn episode_mentions_rrf_presort_breaks_count_ties() {
    let driver = FakeDriver::new();
    // both x and y mentioned exactly once → equal count.
    driver
        .save_episodic_edges(&[mention("ep0", "x"), mention("ep1", "y")])
        .await
        .unwrap();

    // RRF over two lists ranks y strictly above x: y is rank-1 in BOTH lists
    // (RRF 1/1 + 1/1 = 2.0), x is rank-2 in both (1/2 + 1/2 = 1.0).
    let lists = vec![
        vec!["y".to_string(), "x".to_string()],
        vec!["y".to_string(), "x".to_string()],
    ];
    let (uuids, scores) = episode_mentions_rerank(&driver, &lists, 0.0).await.unwrap();

    assert_eq!(scores, vec![1.0, 1.0], "equal counts");
    assert_eq!(
        uuids,
        vec!["y".to_string(), "x".to_string()],
        "RRF presort tiebreak: y (higher RRF) leads on equal counts"
    );
}
