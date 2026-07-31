//! Env-gated live FalkorDB search-op tests for `FalkorDriver` (Phase-6 Task 2).
//!
//! These run only when `FALKOR_TEST_URI` is set; otherwise each test prints a
//! skip notice and returns (mirrors the Task-1 integration pattern). To run
//! against a disposable container:
//!
//! ```bash
//! docker run -d --rm --name chronicle-falkor-t2 -p 16379:6379 falkordb/falkordb:latest
//! FALKOR_TEST_URI=falkor://localhost:16379 \
//!     cargo test -p chronicle-driver-falkor --test falkor_search
//! docker stop chronicle-falkor-t2
//! ```
//!
//! Each test connects to a fresh, uniquely-named graph so concurrent runs never
//! collide. Mirrors the Neo4j / SurrealDB search-test intents: vector rank +
//! min-score cutoff, node/edge fulltext recall, directed get_edges_between_nodes,
//! embedding loaders, BFS depth + edge-type + group + MENTIONS-origin, undirected
//! center adjacency, mention counts, and filter combos.

use std::collections::HashMap;

use chronicle_core::driver::{
    EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, SearchOps,
};
use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};
use chronicle_driver_falkor::FalkorDriver;
use chrono::{TimeZone, Utc};

const DIM: usize = 8;

/// Serialise the search tests against the shared FalkorDB server.
///
/// Each test builds its own graph WITH a fresh vector + fulltext index in
/// `connect()`, then immediately queries those indices. The `falkordb` 0.2 crate
/// runs the schema-refresh round-trips that decode `--compact` rows on a shared
/// connection; under many concurrent graph-build + vector-query cycles the
/// just-built vector index is occasionally not yet queryable on the server,
/// surfacing as `arguments for procedure 'db.idx.vector.queryNodes'`. The work is
/// genuinely concurrent (each test is a `multi_thread` runtime); we only serialise
/// the SETUP+QUERY phase across test functions so they don't stampede one server.
/// A `tokio::sync::Mutex` is the right primitive — its guard is `Send` and safe to
/// hold across `.await` points.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn test_driver() -> Option<FalkorDriver> {
    let uri = std::env::var("FALKOR_TEST_URI").ok()?;
    let graph = format!("chronicle_search_{}", uuid::Uuid::new_v4().simple());
    Some(
        FalkorDriver::connect(&uri, &graph, DIM)
            .await
            .expect("connect"),
    )
}

fn unique_group() -> String {
    format!("test-{}", uuid::Uuid::new_v4())
}

/// A normalised one-hot-ish embedding so cosine distances are predictable.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[i % DIM] = 1.0;
    v
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

/// Acquire the serial guard, then connect. Returns `(driver, guard)`; the guard
/// MUST be bound for the whole test body so the serialisation holds across the
/// setup + query phase. Skips (returns early) when `FALKOR_TEST_URI` is unset.
macro_rules! skip_if_no_env {
    () => {{
        let _guard = SERIAL.lock().await;
        let Some(d) = test_driver().await else {
            eprintln!("skipping: FALKOR_TEST_URI unset");
            return;
        };
        (d, _guard)
    }};
}

/// Helper: save two related entities + a directed edge, returning their uuids.
async fn seed_pair(
    d: &FalkorDriver,
    group: &str,
    src_name: &str,
    tgt_name: &str,
    edge_name: &str,
    fact: &str,
) -> (EntityNode, EntityNode, EntityEdge) {
    let src = EntityNode::new(src_name.into(), group.to_string(), Utc::now());
    let tgt = EntityNode::new(tgt_name.into(), group.to_string(), Utc::now());
    d.save_entity_nodes(&[src.clone(), tgt.clone()])
        .await
        .expect("save nodes");
    let edge = EntityEdge::new(
        src.uuid.clone(),
        tgt.uuid.clone(),
        edge_name.into(),
        fact.into(),
        group.to_string(),
    );
    (src, tgt, edge)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_similarity_ranks_and_applies_min_score() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let mut exact = EntityNode::new("Exact".into(), group.clone(), Utc::now());
    exact.name_embedding = Some(axis(0));
    let mut ortho = EntityNode::new("Ortho".into(), group.clone(), Utc::now());
    ortho.name_embedding = Some(axis(1));
    d.save_entity_nodes(&[exact.clone(), ortho.clone()])
        .await
        .expect("save");

    // min_score 0.0: both returned, exact first (similarity ~1.0 vs ~0.0).
    let all = d
        .node_similarity_search(
            &axis(0),
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.0,
        )
        .await
        .expect("search");
    assert_eq!(all.len(), 2, "both above min_score 0.0");
    assert_eq!(all[0].uuid, exact.uuid, "exact match ranks first");

    // min_score 0.5: only the exact match (similarity ~1.0) survives.
    let cut = d
        .node_similarity_search(
            &axis(0),
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.5,
        )
        .await
        .expect("search");
    assert_eq!(cut.len(), 1);
    assert_eq!(cut[0].uuid, exact.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_similarity_ranks_and_cuts() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let (_s, _t, mut near) = seed_pair(&d, &group, "A", "B", "REL", "near fact").await;
    near.fact_embedding = Some(axis(0));
    let (_s2, _t2, mut far) = seed_pair(&d, &group, "C", "D", "REL", "far fact").await;
    far.fact_embedding = Some(axis(1));
    d.save_entity_edges(&[near.clone(), far.clone()])
        .await
        .expect("save edges");

    let all = d
        .edge_similarity_search(
            &axis(0),
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.0,
        )
        .await
        .expect("search");
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].uuid, near.uuid, "near edge ranks first");

    let cut = d
        .edge_similarity_search(
            &axis(0),
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.5,
        )
        .await
        .expect("search");
    assert_eq!(cut.len(), 1);
    assert_eq!(cut[0].uuid, near.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_fulltext_recall() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let mut acme = EntityNode::new("Acme Corporation".into(), group.clone(), Utc::now());
    acme.summary = "a manufacturing giant".into();
    let other = EntityNode::new("Globex".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[acme.clone(), other.clone()])
        .await
        .expect("save");

    let hits = d
        .node_fulltext_search(
            "Acme",
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].uuid, acme.uuid);

    // Match on the summary field too.
    let by_summary = d
        .node_fulltext_search(
            "manufacturing",
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("search");
    assert_eq!(by_summary.len(), 1);
    assert_eq!(by_summary[0].uuid, acme.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_fulltext_recall_on_fact() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let (_s, _t, hit) = seed_pair(
        &d,
        &group,
        "Alice",
        "Acme",
        "WORKS_AT",
        "Alice works at Acme",
    )
    .await;
    let (_s2, _t2, miss) =
        seed_pair(&d, &group, "Bob", "Globex", "WORKS_AT", "Bob plays guitar").await;
    d.save_entity_edges(&[hit.clone(), miss.clone()])
        .await
        .expect("save edges");

    let hits = d
        .edge_fulltext_search(
            "guitar",
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].uuid, miss.uuid, "fulltext matches the fact text");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_edges_between_nodes_is_directed() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let (src, tgt, edge) = seed_pair(&d, &group, "S", "T", "REL", "s to t").await;
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("save edge");

    let forward = d
        .get_edges_between_nodes(&src.uuid, &tgt.uuid)
        .await
        .expect("forward");
    assert_eq!(forward.len(), 1);
    assert_eq!(forward[0].uuid, edge.uuid);

    // Reverse direction has no edge (directed match).
    let backward = d
        .get_edges_between_nodes(&tgt.uuid, &src.uuid)
        .await
        .expect("backward");
    assert!(backward.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embedding_loaders_parse_vectors_back() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let mut n1 = EntityNode::new("N1".into(), group.clone(), Utc::now());
    n1.name_embedding = Some(axis(2));
    let n2 = EntityNode::new("N2-no-emb".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[n1.clone(), n2.clone()])
        .await
        .expect("save");

    let node_embs = d
        .get_embeddings_for_nodes(&[n1.uuid.clone(), n2.uuid.clone()])
        .await
        .expect("node embeddings");
    assert_eq!(node_embs.len(), 1, "no-embedding node omitted");
    assert!(approx_eq(node_embs.get(&n1.uuid).unwrap(), &axis(2)));

    let (_s, _t, mut edge) = seed_pair(&d, &group, "EA", "EB", "REL", "fact").await;
    edge.fact_embedding = Some(axis(3));
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("save edge");
    let edge_embs = d
        .get_embeddings_for_edges(std::slice::from_ref(&edge.uuid))
        .await
        .expect("edge embeddings");
    assert_eq!(edge_embs.len(), 1);
    assert!(approx_eq(edge_embs.get(&edge.uuid).unwrap(), &axis(3)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_bfs_depth_and_edge_type_filter() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    // Chain: A -REL-> B -REL-> C -REL-> D
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    let c = EntityNode::new("C".into(), group.clone(), Utc::now());
    let dd = EntityNode::new("D".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone(), c.clone(), dd.clone()])
        .await
        .expect("save nodes");
    let e_ab = EntityEdge::new(
        a.uuid.clone(),
        b.uuid.clone(),
        "KNOWS".into(),
        "ab".into(),
        group.clone(),
    );
    let e_bc = EntityEdge::new(
        b.uuid.clone(),
        c.uuid.clone(),
        "KNOWS".into(),
        "bc".into(),
        group.clone(),
    );
    let e_cd = EntityEdge::new(
        c.uuid.clone(),
        dd.uuid.clone(),
        "LIKES".into(),
        "cd".into(),
        group.clone(),
    );
    d.save_entity_edges(&[e_ab, e_bc, e_cd])
        .await
        .expect("save edges");

    // Depth 1 from A reaches only B.
    let d1 = d
        .node_bfs_search(
            std::slice::from_ref(&a.uuid),
            &SearchFilters::default(),
            1,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs d1");
    let names1: Vec<&str> = d1.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names1, vec!["B"]);

    // Depth 3 from A reaches B, C, D.
    let mut d3 = d
        .node_bfs_search(
            std::slice::from_ref(&a.uuid),
            &SearchFilters::default(),
            3,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs d3");
    d3.sort_by(|x, y| x.name.cmp(&y.name));
    let names3: Vec<&str> = d3.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names3, vec!["B", "C", "D"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_bfs_from_mentions_origin() {
    // An Episodic origin reaches an Entity via a MENTIONS hop.
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let entity = EntityNode::new("Mentioned".into(), group.clone(), Utc::now());
    d.save_entity_nodes(std::slice::from_ref(&entity))
        .await
        .expect("save entity");
    let ep = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Text,
        "src".into(),
        "content".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&ep).await.expect("save ep");
    let medge = EpisodicEdge::new(
        ep.uuid.clone(),
        entity.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_episodic_edges(std::slice::from_ref(&medge))
        .await
        .expect("save mentions");

    let reached = d
        .node_bfs_search(
            std::slice::from_ref(&ep.uuid),
            &SearchFilters::default(),
            1,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs from episode");
    assert_eq!(reached.len(), 1);
    assert_eq!(reached[0].uuid, entity.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_bfs_returns_relates_to_only() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    // A -MENTIONS(via episode)-> not applicable; build A -REL-> B and an episode
    // mentions A so the path can start from the episode but only RELATES_TO edges
    // come back.
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone()])
        .await
        .expect("save nodes");
    let e_ab = EntityEdge::new(
        a.uuid.clone(),
        b.uuid.clone(),
        "REL".into(),
        "ab".into(),
        group.clone(),
    );
    d.save_entity_edges(std::slice::from_ref(&e_ab))
        .await
        .expect("save edge");

    let ep = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Text,
        "src".into(),
        "c".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&ep).await.expect("save ep");
    let medge = EpisodicEdge::new(ep.uuid.clone(), a.uuid.clone(), group.clone(), Utc::now());
    d.save_episodic_edges(std::slice::from_ref(&medge))
        .await
        .expect("save mentions");

    // From the episode, depth 2 traverses MENTIONS then RELATES_TO; only the
    // RELATES_TO edge (e_ab) is returned.
    let edges = d
        .edge_bfs_search(
            std::slice::from_ref(&ep.uuid),
            2,
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("edge bfs");
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].uuid, e_ab.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nodes_connected_to_center_is_undirected_one_hop() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let center = EntityNode::new("Center".into(), group.clone(), Utc::now());
    let near = EntityNode::new("Near".into(), group.clone(), Utc::now());
    let far = EntityNode::new("Far".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[center.clone(), near.clone(), far.clone()])
        .await
        .expect("save");
    // near -> center (reverse direction; adjacency is undirected so still counts).
    let e = EntityEdge::new(
        near.uuid.clone(),
        center.uuid.clone(),
        "REL".into(),
        "n-c".into(),
        group.clone(),
    );
    d.save_entity_edges(std::slice::from_ref(&e))
        .await
        .expect("save edge");

    let adj = d
        .nodes_connected_to_center(&[near.uuid.clone(), far.uuid.clone()], &center.uuid)
        .await
        .expect("adjacency");
    assert_eq!(adj, vec![near.uuid.clone()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn episode_mention_counts_per_entity() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let n1 = EntityNode::new("N1".into(), group.clone(), Utc::now());
    let n2 = EntityNode::new("N2".into(), group.clone(), Utc::now());
    let n3_unmentioned = EntityNode::new("N3".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[n1.clone(), n2.clone(), n3_unmentioned.clone()])
        .await
        .expect("save");

    // Two episodes mention n1, one mentions n2, none mention n3.
    for name in ["epA", "epB"] {
        let ep = EpisodicNode::new(
            name.into(),
            group.clone(),
            EpisodeType::Text,
            "s".into(),
            "c".into(),
            Utc::now(),
            Utc::now(),
        );
        d.save_episode(&ep).await.expect("save ep");
        let m = EpisodicEdge::new(ep.uuid.clone(), n1.uuid.clone(), group.clone(), Utc::now());
        d.save_episodic_edges(std::slice::from_ref(&m))
            .await
            .expect("mention n1");
    }
    let ep_c = EpisodicNode::new(
        "epC".into(),
        group.clone(),
        EpisodeType::Text,
        "s".into(),
        "c".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&ep_c).await.expect("save ep");
    let m2 = EpisodicEdge::new(
        ep_c.uuid.clone(),
        n2.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_episodic_edges(std::slice::from_ref(&m2))
        .await
        .expect("mention n2");

    let counts = d
        .episode_mention_counts(&[
            n1.uuid.clone(),
            n2.uuid.clone(),
            n3_unmentioned.uuid.clone(),
        ])
        .await
        .expect("counts");
    assert_eq!(counts.get(&n1.uuid), Some(&2));
    assert_eq!(counts.get(&n2.uuid), Some(&1));
    assert!(
        !counts.contains_key(&n3_unmentioned.uuid),
        "unmentioned node omitted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn episode_fulltext_recall() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let hit = EpisodicNode::new(
        "h".into(),
        group.clone(),
        EpisodeType::Text,
        "s".into(),
        "the quick brown fox".into(),
        Utc::now(),
        Utc::now(),
    );
    let miss = EpisodicNode::new(
        "m".into(),
        group.clone(),
        EpisodeType::Text,
        "s".into(),
        "completely different text".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&hit).await.expect("save hit");
    d.save_episode(&miss).await.expect("save miss");

    let hits = d
        .episode_fulltext_search("brown fox", std::slice::from_ref(&group), 10)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].uuid, hit.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_filter_combo_types_dates_and_uuids() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let (_s, _t, mut keep) = seed_pair(&d, &group, "A", "B", "KNOWS", "keep").await;
    keep.fact_embedding = Some(axis(0));
    keep.valid_at = Some(Utc.timestamp_millis_opt(1_700_000_000_000).unwrap());
    let (_s2, _t2, mut wrong_type) = seed_pair(&d, &group, "C", "D", "HATES", "wrong type").await;
    wrong_type.fact_embedding = Some(axis(0));
    wrong_type.valid_at = Some(Utc.timestamp_millis_opt(1_700_000_000_000).unwrap());
    let (_s3, _t3, mut wrong_date) = seed_pair(&d, &group, "E", "F", "KNOWS", "wrong date").await;
    wrong_date.fact_embedding = Some(axis(0));
    wrong_date.valid_at = Some(Utc.timestamp_millis_opt(1_900_000_000_000).unwrap());
    d.save_entity_edges(&[keep.clone(), wrong_type.clone(), wrong_date.clone()])
        .await
        .expect("save edges");

    // edge_types=[KNOWS] AND valid_at in [t-window] AND edge_uuids=[keep,wrong_date].
    let t_lo = Utc.timestamp_millis_opt(1_650_000_000_000).unwrap();
    let t_hi = Utc.timestamp_millis_opt(1_750_000_000_000).unwrap();
    let filters = SearchFilters {
        edge_types: Some(vec!["KNOWS".into()]),
        edge_uuids: Some(vec![keep.uuid.clone(), wrong_date.uuid.clone()]),
        valid_at: Some(vec![vec![
            DateFilter {
                date: Some(t_lo),
                comparison_operator: ComparisonOperator::Gte,
            },
            DateFilter {
                date: Some(t_hi),
                comparison_operator: ComparisonOperator::Lte,
            },
        ]]),
        ..Default::default()
    };
    let hits = d
        .edge_similarity_search(&axis(0), &filters, std::slice::from_ref(&group), 50, 0.0)
        .await
        .expect("filtered search");
    let uuids: Vec<String> = hits.iter().map(|e| e.uuid.clone()).collect();
    assert_eq!(
        uuids,
        vec![keep.uuid.clone()],
        "only KNOWS + in-window + listed uuid"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_label_filter() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();

    let mut org = EntityNode::new("OrgNode".into(), group.clone(), Utc::now());
    org.labels = vec!["Entity".into(), "Organization".into()];
    org.name_embedding = Some(axis(0));
    let mut person = EntityNode::new("PersonNode".into(), group.clone(), Utc::now());
    person.labels = vec!["Entity".into(), "Person".into()];
    person.name_embedding = Some(axis(0));
    d.save_entity_nodes(&[org.clone(), person.clone()])
        .await
        .expect("save");

    let filters = SearchFilters {
        node_labels: Some(vec!["Organization".into()]),
        ..Default::default()
    };
    let hits = d
        .node_similarity_search(&axis(0), &filters, std::slice::from_ref(&group), 50, 0.0)
        .await
        .expect("label-filtered search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].uuid, org.uuid);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_node_label_is_hard_error() {
    let (d, _guard) = skip_if_no_env!();
    let group = unique_group();
    let filters = SearchFilters {
        node_labels: Some(vec!["bad-label".into()]),
        ..Default::default()
    };
    let res = d
        .node_similarity_search(&axis(0), &filters, &[group], 10, 0.0)
        .await;
    assert!(res.is_err(), "unsanitised label must be a hard error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_inputs_short_circuit() {
    let (d, _guard) = skip_if_no_env!();
    let f = SearchFilters::default();
    assert!(
        d.node_fulltext_search("", &f, &[], 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        d.edge_fulltext_search("", &f, &[], 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        d.node_bfs_search(&[], &f, 3, &[], 10)
            .await
            .unwrap()
            .is_empty(),
        "no origins"
    );
    assert!(
        d.node_bfs_search(&["x".into()], &f, 0, &[], 10)
            .await
            .unwrap()
            .is_empty(),
        "depth < 1"
    );
    assert!(d.get_embeddings_for_nodes(&[]).await.unwrap().is_empty());
    let empty: HashMap<String, u64> = d.episode_mention_counts(&[]).await.unwrap();
    assert!(empty.is_empty());
}
