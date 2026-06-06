//! Env-gated live FalkorDB integration tests for `FalkorDriver`.
//!
//! These run only when `FALKOR_TEST_URI` is set; otherwise each test prints a
//! skip notice and returns (mirrors the Neo4j integration pattern). To run
//! against a disposable container:
//!
//! ```bash
//! docker run -d --rm --name chronicle-falkor-test -p 6379:6379 \
//!     falkordb/falkordb:latest
//! FALKOR_TEST_URI=falkor://localhost:6379 \
//!     cargo test -p chronicle-driver-falkor --test falkor_integration
//! docker stop chronicle-falkor-test
//! ```

use chronicle_core::driver::{
    CommunityOps, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SagaOps,
    SchemaOps,
};
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};
use chronicle_driver_falkor::FalkorDriver;
use chrono::{Duration, TimeZone, Utc};
use serde_json::json;

const DIM: usize = 8;

/// Connect to a per-test graph (unique name) so concurrent runs don't collide.
async fn test_driver() -> Option<FalkorDriver> {
    let uri = std::env::var("FALKOR_TEST_URI").ok()?;
    let graph = format!("chronicle_test_{}", uuid::Uuid::new_v4().simple());
    Some(
        FalkorDriver::connect(&uri, &graph, DIM)
            .await
            .expect("connect"),
    )
}

fn unique_group() -> String {
    format!("test-{}", uuid::Uuid::new_v4())
}

fn sample_embedding(seed: f32) -> Vec<f32> {
    (0..DIM).map(|i| seed + i as f32 * 0.01).collect()
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

macro_rules! skip_if_no_env {
    () => {{
        let Some(d) = test_driver().await else {
            eprintln!("skipping: FALKOR_TEST_URI unset");
            return;
        };
        d
    }};
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_is_falkordb() {
    let d = skip_if_no_env!();
    assert_eq!(d.provider(), "falkordb");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_indices_is_idempotent() {
    let d = skip_if_no_env!();
    d.build_indices_and_constraints(false)
        .await
        .expect("first build");
    d.build_indices_and_constraints(false)
        .await
        .expect("second build (idempotent)");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn entity_node_roundtrip_with_embedding_and_attrs() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let mut node = EntityNode::new("Acme Corp".into(), group.clone(), Utc::now());
    node.summary = "A big company".into();
    node.labels = vec!["Entity".into(), "Organization".into()];
    node.name_embedding = Some(sample_embedding(0.1));
    node.attributes.insert("founded".into(), json!(1990));
    node.attributes.insert("ticker".into(), json!("ACME"));
    node.attributes
        .insert("nested".into(), json!({"hq": "NYC", "public": true}));

    d.save_entity_nodes(std::slice::from_ref(&node))
        .await
        .expect("save");
    let got = d
        .get_entity_node(&node.uuid)
        .await
        .expect("get")
        .expect("present");

    assert_eq!(got.uuid, node.uuid);
    assert_eq!(got.name, "Acme Corp");
    assert_eq!(got.group_id, group);
    assert_eq!(got.summary, "A big company");
    assert!(got.labels.contains(&"Organization".to_string()));
    assert!(approx_eq(
        got.name_embedding.as_ref().unwrap(),
        node.name_embedding.as_ref().unwrap()
    ));
    // Attributes (including the nested object) survive via JSON-string storage.
    assert_eq!(got.attributes.get("founded"), Some(&json!(1990)));
    assert_eq!(got.attributes.get("ticker"), Some(&json!("ACME")));
    assert_eq!(
        got.attributes.get("nested"),
        Some(&json!({"hq": "NYC", "public": true}))
    );

    d.delete_entity_nodes_by_uuids(&[node.uuid.clone()])
        .await
        .expect("delete");
    assert!(d.get_entity_node(&node.uuid).await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn entity_node_handles_quote_injection_safely() {
    let d = skip_if_no_env!();
    let group = unique_group();
    // A name with quotes / backslashes must round-trip, proving literal escaping.
    let nasty = "O'Brien \\ \"X\" '); DROP";
    let mut node = EntityNode::new(nasty.into(), group, Utc::now());
    node.summary = "line1\nline2".into();
    d.save_entity_nodes(std::slice::from_ref(&node))
        .await
        .expect("save");
    let got = d.get_entity_node(&node.uuid).await.unwrap().unwrap();
    assert_eq!(got.name, nasty);
    assert_eq!(got.summary, "line1\nline2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_entity_nodes_by_uuids_preserves_order() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    let c = EntityNode::new("C".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .expect("save");

    let order = vec![c.uuid.clone(), a.uuid.clone(), b.uuid.clone()];
    let got = d.get_entity_nodes_by_uuids(&order).await.expect("get");
    let got_uuids: Vec<String> = got.iter().map(|n| n.uuid.clone()).collect();
    assert_eq!(got_uuids, order);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn entity_edge_roundtrip_bitemporal_and_embedding() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let src = EntityNode::new("Alice".into(), group.clone(), Utc::now());
    let tgt = EntityNode::new("Acme".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[src.clone(), tgt.clone()])
        .await
        .expect("save nodes");

    let mut edge = EntityEdge::new(
        src.uuid.clone(),
        tgt.uuid.clone(),
        "WORKS_AT".into(),
        "Alice works at Acme".into(),
        group.clone(),
    );
    edge.valid_at = Some(Utc.timestamp_millis_opt(1_700_000_000_000).unwrap());
    edge.expired_at = Some(Utc.timestamp_millis_opt(1_800_000_000_000).unwrap());
    edge.episodes = vec!["ep-1".into(), "ep-2".into()];
    edge.fact_embedding = Some(sample_embedding(0.3));
    edge.attributes.insert("confidence".into(), json!(0.9));

    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("save edge");
    let got = d
        .get_entity_edge(&edge.uuid)
        .await
        .expect("get")
        .expect("present");

    assert_eq!(got.fact, "Alice works at Acme");
    assert_eq!(got.source_node_uuid, src.uuid);
    assert_eq!(got.target_node_uuid, tgt.uuid);
    assert_eq!(got.valid_at, edge.valid_at);
    assert_eq!(got.expired_at, edge.expired_at);
    assert!(got.invalid_at.is_none());
    assert_eq!(got.episodes, vec!["ep-1".to_string(), "ep-2".into()]);
    assert!(approx_eq(
        got.fact_embedding.as_ref().unwrap(),
        edge.fact_embedding.as_ref().unwrap()
    ));
    assert_eq!(got.attributes.get("confidence"), Some(&json!(0.9)));

    let by_uuids = d
        .get_entity_edges_by_uuids(&[edge.uuid.clone()])
        .await
        .expect("by uuids");
    assert_eq!(by_uuids.len(), 1);

    d.delete_entity_edges_by_uuids(&[edge.uuid.clone()])
        .await
        .expect("delete edge");
    assert!(d.get_entity_edge(&edge.uuid).await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn episode_roundtrip_and_episodic_edge() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let entity = EntityNode::new("Bob".into(), group.clone(), Utc::now());
    d.save_entity_nodes(std::slice::from_ref(&entity))
        .await
        .expect("save entity");

    let mut ep = EpisodicNode::new(
        "msg-1".into(),
        group.clone(),
        EpisodeType::Message,
        "chat".into(),
        "Bob said hi".into(),
        Utc::now(),
        Utc.timestamp_millis_opt(1_700_000_500_000).unwrap(),
    );
    ep.entity_edges = vec!["edge-a".into()];
    d.save_episode(&ep).await.expect("save episode");

    let got = d.get_episode(&ep.uuid).await.unwrap().unwrap();
    assert_eq!(got.content, "Bob said hi");
    assert_eq!(got.source, EpisodeType::Message);
    assert_eq!(got.valid_at, ep.valid_at);
    assert_eq!(got.entity_edges, vec!["edge-a".to_string()]);

    let medge = EpisodicEdge::new(ep.uuid.clone(), entity.uuid.clone(), group, Utc::now());
    d.save_episodic_edges(std::slice::from_ref(&medge))
        .await
        .expect("save mentions");

    d.delete_episode(&ep.uuid).await.expect("delete episode");
    assert!(d.get_episode(&ep.uuid).await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn datetime_stored_as_int_and_filters_correctly() {
    // The KEY delta guard: temporal fields stored as epoch-millis ints,
    // round-trip to the same chrono value, AND an int-comparison query filters.
    let d = skip_if_no_env!();
    let group = unique_group();

    let t1 = Utc.timestamp_millis_opt(1_700_000_000_000).unwrap();
    let t2 = Utc.timestamp_millis_opt(1_750_000_000_000).unwrap();
    let t3 = Utc.timestamp_millis_opt(1_800_000_000_000).unwrap();

    for (n, t) in [("e1", t1), ("e2", t2), ("e3", t3)] {
        let ep = EpisodicNode::new(
            n.into(),
            group.clone(),
            EpisodeType::Text,
            "src".into(),
            format!("content {n}"),
            Utc::now(),
            t,
        );
        d.save_episode(&ep).await.expect("save");
    }

    // retrieve_episodes uses `valid_at <= reference` as an INT comparison; cutoff
    // at t2 must return exactly {e1, e2}, ordered chronologically (e1 then e2).
    let got = d
        .retrieve_episodes(t2, 10, std::slice::from_ref(&group), None)
        .await
        .expect("retrieve");
    let names: Vec<String> = got.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, vec!["e1".to_string(), "e2".into()]);

    // valid_at round-trips to the exact chrono value (proving int storage is
    // lossless for millisecond precision).
    assert_eq!(got[0].valid_at, t1);
    assert_eq!(got[1].valid_at, t2);

    // last_n cap + DESC-then-reverse ordering: cap at 1 keeps the LATEST <= cutoff.
    let last1 = d
        .retrieve_episodes(t2, 1, std::slice::from_ref(&group), None)
        .await
        .expect("retrieve last 1");
    assert_eq!(last1.len(), 1);
    assert_eq!(last1[0].name, "e2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieve_episodes_source_filter() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let now = Utc::now();
    let base = now - Duration::hours(1);

    let text_ep = EpisodicNode::new(
        "t".into(),
        group.clone(),
        EpisodeType::Text,
        "src".into(),
        "text".into(),
        now,
        base,
    );
    let msg_ep = EpisodicNode::new(
        "m".into(),
        group.clone(),
        EpisodeType::Message,
        "src".into(),
        "msg".into(),
        now,
        base,
    );
    d.save_episode(&text_ep).await.unwrap();
    d.save_episode(&msg_ep).await.unwrap();

    let only_msg = d
        .retrieve_episodes(
            now,
            10,
            std::slice::from_ref(&group),
            Some(EpisodeType::Message),
        )
        .await
        .expect("retrieve");
    assert_eq!(only_msg.len(), 1);
    assert_eq!(only_msg[0].source, EpisodeType::Message);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn community_roundtrip() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let mut comm = CommunityNode::new("Tech".into(), group.clone(), Utc::now());
    comm.summary = "Tech companies".into();
    comm.name_embedding = Some(sample_embedding(0.5));
    d.save_community_nodes(std::slice::from_ref(&comm))
        .await
        .expect("save community");

    let member = EntityNode::new("Member".into(), group.clone(), Utc::now());
    d.save_entity_nodes(std::slice::from_ref(&member))
        .await
        .expect("save member");
    let edge = CommunityEdge::new(
        comm.uuid.clone(),
        member.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_community_edges(std::slice::from_ref(&edge))
        .await
        .expect("save has_member");

    let by_group = d
        .get_community_nodes_by_group_ids(std::slice::from_ref(&group))
        .await
        .expect("by group");
    assert_eq!(by_group.len(), 1);
    assert_eq!(by_group[0].name, "Tech");
    assert!(approx_eq(
        by_group[0].name_embedding.as_ref().unwrap(),
        comm.name_embedding.as_ref().unwrap()
    ));

    let by_uuid = d
        .get_community_nodes_by_uuids(&[comm.uuid.clone()])
        .await
        .expect("by uuid");
    assert_eq!(by_uuid.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saga_roundtrip_and_threading_edges() {
    let d = skip_if_no_env!();
    let group = unique_group();
    let mut saga = SagaNode::new("onboarding".into(), group.clone(), Utc::now());
    saga.summary = "Customer onboarding".into();
    saga.first_episode_uuid = Some("ep-first".into());
    saga.last_summarized_at = Some(Utc.timestamp_millis_opt(1_700_000_000_000).unwrap());
    d.save_saga_node(&saga).await.expect("save saga");

    let by_name = d
        .get_saga_by_name("onboarding", &group)
        .await
        .expect("by name")
        .expect("present");
    assert_eq!(by_name.uuid, saga.uuid);
    assert_eq!(by_name.summary, "Customer onboarding");
    assert_eq!(by_name.first_episode_uuid, Some("ep-first".to_string()));
    assert_eq!(by_name.last_summarized_at, saga.last_summarized_at);
    assert!(by_name.last_episode_uuid.is_none());

    let by_uuid = d
        .get_saga_by_uuid(&saga.uuid)
        .await
        .expect("by uuid")
        .expect("present");
    assert_eq!(by_uuid.name, "onboarding");

    // HAS_EPISODE + NEXT_EPISODE edges save against real episodes.
    let ep1 = EpisodicNode::new(
        "e1".into(),
        group.clone(),
        EpisodeType::Text,
        "s".into(),
        "c1".into(),
        Utc::now(),
        Utc::now(),
    );
    let ep2 = EpisodicNode::new(
        "e2".into(),
        group.clone(),
        EpisodeType::Text,
        "s".into(),
        "c2".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&ep1).await.unwrap();
    d.save_episode(&ep2).await.unwrap();
    let has = HasEpisodeEdge::new(
        saga.uuid.clone(),
        ep1.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_has_episode_edge(&has).await.expect("has_episode");
    let next = NextEpisodeEdge::new(ep1.uuid.clone(), ep2.uuid.clone(), group, Utc::now());
    d.save_next_episode_edge(&next).await.expect("next_episode");
}
