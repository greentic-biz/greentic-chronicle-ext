//! Env-gated live Neo4j integration tests for `Neo4jDriver`.
//!
//! These run only when `NEO4J_TEST_URI` is set; otherwise each test prints a skip
//! notice and returns. To run against a disposable container:
//!
//! ```bash
//! docker run -d --rm --name chronicle-neo4j-test -p 7687:7687 \
//!     -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//! NEO4J_TEST_URI=bolt://localhost:7687 \
//!     cargo test -p chronicle-driver-neo4j --test neo4j_integration
//! docker stop chronicle-neo4j-test
//! ```

use chronicle_core::driver::{
    EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SchemaOps, SearchOps,
};
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};
use chronicle_driver_neo4j::Neo4jDriver;
use chrono::{Duration, Utc};
use serde_json::json;

async fn test_driver() -> Option<Neo4jDriver> {
    let uri = std::env::var("NEO4J_TEST_URI").ok()?;
    let user = std::env::var("NEO4J_TEST_USER").unwrap_or_else(|_| "neo4j".into());
    let pass = std::env::var("NEO4J_TEST_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    Some(
        Neo4jDriver::connect(&uri, &user, &pass, "neo4j")
            .await
            .expect("connect"),
    )
}

/// Unique group_id per test invocation so concurrent/repeat runs do not collide.
fn unique_group() -> String {
    format!("test-{}", uuid::Uuid::new_v4())
}

/// Detach-delete everything in a group (cleanup at test end).
async fn cleanup(driver: &Neo4jDriver, group_id: &str) {
    // Reuse the public save path indirectly is not possible for delete; issue a
    // raw cleanup through a tiny throwaway edge/node save is unnecessary. We rely
    // on a direct cypher run via a fresh connection-less helper: simplest is to
    // build a node with a sentinel and then DETACH DELETE via the driver's
    // search-independent path. Since the driver exposes no delete op (Phase 1),
    // we open a short-lived raw neo4rs graph for cleanup only.
    let uri = std::env::var("NEO4J_TEST_URI").unwrap();
    let user = std::env::var("NEO4J_TEST_USER").unwrap_or_else(|_| "neo4j".into());
    let pass = std::env::var("NEO4J_TEST_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let graph = neo4rs::Graph::new(&uri, &user, &pass).await.unwrap();
    let _ = driver; // keep signature symmetric
    graph
        .run(neo4rs::query("MATCH (n {group_id: $g}) DETACH DELETE n").param("g", group_id))
        .await
        .expect("cleanup");
}

#[tokio::test]
async fn build_indices_is_idempotent() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    d.build_indices_and_constraints(false)
        .await
        .expect("first build");
    d.build_indices_and_constraints(false)
        .await
        .expect("second build (idempotent)");
    // delete_existing path also exercised.
    d.build_indices_and_constraints(true)
        .await
        .expect("build with delete_existing");
}

#[tokio::test]
async fn provider_is_neo4j() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    assert_eq!(d.provider(), "neo4j");
}

#[tokio::test]
async fn entity_node_roundtrip() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let mut node = EntityNode::new("Alice".into(), group.clone(), Utc::now());
    node.summary = "a person".into();
    node.labels = vec!["Entity".into(), "Person".into()];
    node.name_embedding = Some(vec![0.1, 0.2, 0.3, 0.4]);
    node.attributes.insert("age".into(), json!(30));
    node.attributes.insert("city".into(), json!("Wonderland"));

    d.save_entity_nodes(std::slice::from_ref(&node))
        .await
        .expect("save");

    let got = d
        .get_entity_node(&node.uuid)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.uuid, node.uuid);
    assert_eq!(got.name, "Alice");
    assert_eq!(got.summary, "a person");
    assert_eq!(got.group_id, group);
    assert!(got.labels.contains(&"Person".to_string()));
    assert!(got.labels.contains(&"Entity".to_string()));
    assert_eq!(got.attributes.get("age"), Some(&json!(30)));
    assert_eq!(got.attributes.get("city"), Some(&json!("Wonderland")));
    let emb = got.name_embedding.expect("embedding");
    assert_eq!(emb.len(), 4);
    assert!((emb[0] - 0.1).abs() < 1e-5);

    // by-uuids path
    let many = d
        .get_entity_nodes_by_uuids(&[node.uuid.clone()])
        .await
        .expect("by uuids");
    assert_eq!(many.len(), 1);

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn entity_edge_roundtrip_with_temporal_and_episodes() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
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
    let now = Utc::now();
    edge.valid_at = Some(now);
    edge.invalid_at = Some(now + Duration::days(1));
    edge.expired_at = Some(now + Duration::days(2));
    edge.episodes = vec!["ep-1".into(), "ep-2".into()];
    edge.fact_embedding = Some(vec![0.5, 0.6, 0.7]);
    edge.attributes.insert("confidence".into(), json!(0.9));

    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("save edge");

    let got = d
        .get_entity_edge(&edge.uuid)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.name, "WORKS_AT");
    assert_eq!(got.fact, "Alice works at Acme");
    assert_eq!(got.source_node_uuid, src.uuid);
    assert_eq!(got.target_node_uuid, tgt.uuid);
    assert_eq!(got.episodes, vec!["ep-1".to_string(), "ep-2".to_string()]);
    assert!(got.valid_at.is_some());
    assert!(got.invalid_at.is_some());
    assert!(got.expired_at.is_some());
    // millisecond-level fidelity on temporal fields
    assert_eq!(
        got.valid_at.unwrap().timestamp_millis(),
        now.timestamp_millis()
    );
    assert_eq!(got.attributes.get("confidence"), Some(&json!(0.9)));
    assert!(got.fact_embedding.is_some());

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn edges_between_nodes_is_directional() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone()])
        .await
        .expect("nodes");

    let edge = EntityEdge::new(
        a.uuid.clone(),
        b.uuid.clone(),
        "KNOWS".into(),
        "A knows B".into(),
        group.clone(),
    );
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("edge");

    // a -> b returns the edge
    let forward = d
        .get_edges_between_nodes(&a.uuid, &b.uuid)
        .await
        .expect("forward");
    assert_eq!(forward.len(), 1);
    assert_eq!(forward[0].uuid, edge.uuid);

    // b -> a must NOT return it (directed match)
    let backward = d
        .get_edges_between_nodes(&b.uuid, &a.uuid)
        .await
        .expect("backward");
    assert!(backward.is_empty());

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn episode_roundtrip_and_retrieve_ordering() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let base = Utc::now() - Duration::hours(10);

    // Three episodes with ascending valid_at.
    let mut e1 = EpisodicNode::new(
        "ep1".into(),
        group.clone(),
        EpisodeType::Message,
        "src".into(),
        "first".into(),
        base,
        base,
    );
    e1.entity_edges = vec!["x".into()];
    let e2 = EpisodicNode::new(
        "ep2".into(),
        group.clone(),
        EpisodeType::Text,
        "src".into(),
        "second".into(),
        base + Duration::hours(1),
        base + Duration::hours(1),
    );
    let e3 = EpisodicNode::new(
        "ep3".into(),
        group.clone(),
        EpisodeType::Json,
        "src".into(),
        "third".into(),
        base + Duration::hours(2),
        base + Duration::hours(2),
    );

    for e in [&e1, &e2, &e3] {
        d.save_episode(e).await.expect("save episode");
    }

    // roundtrip
    let got = d
        .get_episode(&e1.uuid)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.name, "ep1");
    assert_eq!(got.source, EpisodeType::Message);
    assert_eq!(got.content, "first");
    assert_eq!(got.entity_edges, vec!["x".to_string()]);

    // retrieve last 2 as-of now -> chronological order [e2, e3]
    let recent = d
        .retrieve_episodes(Utc::now(), 2, std::slice::from_ref(&group), None)
        .await
        .expect("retrieve");
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].name, "ep2");
    assert_eq!(recent[1].name, "ep3");

    // source filter
    let only_text = d
        .retrieve_episodes(
            Utc::now(),
            10,
            std::slice::from_ref(&group),
            Some(EpisodeType::Text),
        )
        .await
        .expect("retrieve text");
    assert_eq!(only_text.len(), 1);
    assert_eq!(only_text[0].name, "ep2");

    // reference_time excludes future episodes
    let as_of_early = d
        .retrieve_episodes(
            base + Duration::minutes(30),
            10,
            std::slice::from_ref(&group),
            None,
        )
        .await
        .expect("retrieve early");
    assert_eq!(as_of_early.len(), 1);
    assert_eq!(as_of_early[0].name, "ep1");

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn episodic_edge_save_and_mentions() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let entity = EntityNode::new("Bob".into(), group.clone(), Utc::now());
    d.save_entity_nodes(std::slice::from_ref(&entity))
        .await
        .expect("node");
    let episode = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Message,
        "src".into(),
        "Bob said hi".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&episode).await.expect("episode");

    let me = EpisodicEdge::new(
        episode.uuid.clone(),
        entity.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_episodic_edges(&[me]).await.expect("mentions edge");

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn fulltext_search_finds_edge_by_fact_word() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    // Ensure fulltext indices exist and are online.
    d.build_indices_and_constraints(false)
        .await
        .expect("indices");

    let group = unique_group();
    let a = EntityNode::new("Alice".into(), group.clone(), Utc::now());
    let b = EntityNode::new("Acme".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone()])
        .await
        .expect("nodes");
    let mut edge = EntityEdge::new(
        a.uuid.clone(),
        b.uuid.clone(),
        "WORKS_AT".into(),
        "Alice is employed at Acme Corporation".into(),
        group.clone(),
    );
    edge.fact_embedding = Some(vec![0.1, 0.2, 0.3]);
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("edge");

    // Node fulltext: search "Alice"
    let nodes = d
        .node_fulltext_search("Alice", std::slice::from_ref(&group), 10)
        .await
        .expect("node fts");
    assert!(nodes.iter().any(|n| n.uuid == a.uuid));

    // Edge fulltext: search by a distinctive fact word
    let edges = d
        .edge_fulltext_search("employed", std::slice::from_ref(&group), 10)
        .await
        .expect("edge fts");
    assert!(edges.iter().any(|e| e.uuid == edge.uuid));

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn similarity_search_with_real_vector() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let a = EntityNode::new("Vec".into(), group.clone(), Utc::now());
    let mut node = a.clone();
    node.name_embedding = Some(vec![1.0, 0.0, 0.0]);
    d.save_entity_nodes(std::slice::from_ref(&node))
        .await
        .expect("node");

    let b = EntityNode::new("Vsrc".into(), group.clone(), Utc::now());
    let tgt = b.clone();
    d.save_entity_nodes(std::slice::from_ref(&tgt))
        .await
        .expect("tgt");
    let mut edge = EntityEdge::new(
        node.uuid.clone(),
        tgt.uuid.clone(),
        "REL".into(),
        "fact".into(),
        group.clone(),
    );
    edge.fact_embedding = Some(vec![0.0, 1.0, 0.0]);
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("edge");

    // Query vector nearly identical to node embedding -> high cosine.
    let node_hits = d
        .node_similarity_search(&[1.0, 0.0, 0.0], std::slice::from_ref(&group), 10, 0.5)
        .await
        .expect("node sim");
    assert!(node_hits.iter().any(|n| n.uuid == node.uuid));

    let edge_hits = d
        .edge_similarity_search(&[0.0, 1.0, 0.0], std::slice::from_ref(&group), 10, 0.5)
        .await
        .expect("edge sim");
    assert!(edge_hits.iter().any(|e| e.uuid == edge.uuid));

    cleanup(&d, &group).await;
}
