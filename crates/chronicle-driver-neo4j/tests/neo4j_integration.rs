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
    CommunityOps, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SagaOps,
    SchemaOps, SearchOps,
};
use chronicle_core::search::filters::SearchFilters;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    HasEpisodeEdge, NextEpisodeEdge, SagaNode,
};
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
        .node_fulltext_search(
            "Alice",
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("node fts");
    assert!(nodes.iter().any(|n| n.uuid == a.uuid));

    // Edge fulltext: search by a distinctive fact word
    let edges = d
        .edge_fulltext_search(
            "employed",
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
        )
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
        .node_similarity_search(
            &[1.0, 0.0, 0.0],
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.5,
        )
        .await
        .expect("node sim");
    assert!(node_hits.iter().any(|n| n.uuid == node.uuid));

    let edge_hits = d
        .edge_similarity_search(
            &[0.0, 1.0, 0.0],
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            10,
            0.5,
        )
        .await
        .expect("edge sim");
    assert!(edge_hits.iter().any(|e| e.uuid == edge.uuid));

    cleanup(&d, &group).await;
}

// =====================================================================
// Phase-2 search-primitive integration tests (Task 6)
// =====================================================================

/// Helper: build + save a RELATES_TO edge `src -> tgt` with a given name/fact.
async fn save_rel(
    d: &Neo4jDriver,
    src: &EntityNode,
    tgt: &EntityNode,
    name: &str,
    fact: &str,
    group: &str,
) -> EntityEdge {
    let edge = EntityEdge::new(
        src.uuid.clone(),
        tgt.uuid.clone(),
        name.into(),
        fact.into(),
        group.to_string(),
    );
    d.save_entity_edges(std::slice::from_ref(&edge))
        .await
        .expect("save rel");
    edge
}

/// BFS over a directed chain a -> b -> c: depth 1 reaches {b}, depth 2 reaches
/// {b, c}. Origin is `a`.
#[tokio::test]
async fn node_bfs_depth_1_vs_2_chain() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    let c = EntityNode::new("C".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .expect("nodes");
    save_rel(&d, &a, &b, "AB", "a to b", &group).await;
    save_rel(&d, &b, &c, "BC", "b to c", &group).await;

    let depth1 = d
        .node_bfs_search(
            std::slice::from_ref(&a.uuid),
            &SearchFilters::default(),
            1,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs d1");
    let d1: std::collections::HashSet<_> = depth1.iter().map(|n| n.uuid.clone()).collect();
    assert!(d1.contains(&b.uuid), "depth 1 reaches b");
    assert!(!d1.contains(&c.uuid), "depth 1 must NOT reach c");

    let depth2 = d
        .node_bfs_search(
            std::slice::from_ref(&a.uuid),
            &SearchFilters::default(),
            2,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs d2");
    let d2: std::collections::HashSet<_> = depth2.iter().map(|n| n.uuid.clone()).collect();
    assert!(
        d2.contains(&b.uuid) && d2.contains(&c.uuid),
        "depth 2 reaches b and c"
    );

    cleanup(&d, &group).await;
}

/// BFS seeded from an Episodic origin via a MENTIONS hop: episode -MENTIONS-> a
/// -RELATES_TO-> b. Origin = episode uuid; depth 2 reaches {a, b}.
#[tokio::test]
async fn node_bfs_via_mentions_episode_origin() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let a = EntityNode::new("Ent".into(), group.clone(), Utc::now());
    let b = EntityNode::new("Ent2".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone()])
        .await
        .expect("nodes");
    save_rel(&d, &a, &b, "AB", "a to b", &group).await;

    let episode = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Message,
        "src".into(),
        "mentions Ent".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&episode).await.expect("episode");
    let me = EpisodicEdge::new(
        episode.uuid.clone(),
        a.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_episodic_edges(&[me]).await.expect("mentions");

    let hits = d
        .node_bfs_search(
            std::slice::from_ref(&episode.uuid),
            &SearchFilters::default(),
            2,
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("bfs mentions");
    let set: std::collections::HashSet<_> = hits.iter().map(|n| n.uuid.clone()).collect();
    assert!(set.contains(&a.uuid), "MENTIONS hop reaches a");
    assert!(
        set.contains(&b.uuid),
        "a -RELATES_TO-> b reached at depth 2"
    );

    cleanup(&d, &group).await;
}

/// Edge BFS returns DISTINCT edges reachable from the origin.
#[tokio::test]
async fn edge_bfs_distinct_edges() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    let c = EntityNode::new("C".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone(), c.clone()])
        .await
        .expect("nodes");
    let e_ab = save_rel(&d, &a, &b, "AB", "a to b", &group).await;
    let e_bc = save_rel(&d, &b, &c, "BC", "b to c", &group).await;

    let edges = d
        .edge_bfs_search(
            std::slice::from_ref(&a.uuid),
            2,
            &SearchFilters::default(),
            std::slice::from_ref(&group),
            50,
        )
        .await
        .expect("edge bfs");
    let set: std::collections::HashSet<_> = edges.iter().map(|e| e.uuid.clone()).collect();
    assert!(set.contains(&e_ab.uuid), "AB edge present");
    assert!(set.contains(&e_bc.uuid), "BC edge present");
    // Both relationship uuids are reachable from origin `a`.
    assert_eq!(set.len(), 2, "exactly the two chain edges are reachable");
    // NOTE: upstream's UNDIRECTED second MATCH `(n)-[e {uuid}]-(m)` matches each
    // relationship in BOTH orientations, so `RETURN DISTINCT` (which distincts on
    // the full row, including the swapped n.uuid/m.uuid) yields two rows per edge.
    // This mirrors upstream `edge_bfs_search` exactly; downstream rerankers dedup
    // by edge uuid. We therefore assert on the distinct-uuid SET, not row count.

    cleanup(&d, &group).await;
}

/// Episode fulltext search over the `episode_content` index.
#[tokio::test]
async fn episode_fulltext_finds_by_content_word() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    d.build_indices_and_constraints(false)
        .await
        .expect("indices");
    let group = unique_group();
    let episode = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Message,
        "src".into(),
        "the quick brown vorpalfox jumped".into(),
        Utc::now(),
        Utc::now(),
    );
    d.save_episode(&episode).await.expect("episode");

    let hits = d
        .episode_fulltext_search("vorpalfox", std::slice::from_ref(&group), 10)
        .await
        .expect("episode fts");
    assert!(hits.iter().any(|e| e.uuid == episode.uuid));

    cleanup(&d, &group).await;
}

/// Embeddings loaders: only nodes/edges WITH an embedding are returned.
#[tokio::test]
async fn embeddings_loaders_omit_missing() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let mut with_emb = EntityNode::new("WithEmb".into(), group.clone(), Utc::now());
    with_emb.name_embedding = Some(vec![0.1, 0.2, 0.3]);
    let without = EntityNode::new("NoEmb".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[with_emb.clone(), without.clone()])
        .await
        .expect("nodes");

    let node_embs = d
        .get_embeddings_for_nodes(&[with_emb.uuid.clone(), without.uuid.clone()])
        .await
        .expect("node embeddings");
    assert!(
        node_embs.contains_key(&with_emb.uuid),
        "embedded node present"
    );
    assert!(
        !node_embs.contains_key(&without.uuid),
        "node without embedding omitted"
    );
    assert_eq!(node_embs.get(&with_emb.uuid).unwrap().len(), 3);

    // Edge with embedding vs without.
    let mut e_with = EntityEdge::new(
        with_emb.uuid.clone(),
        without.uuid.clone(),
        "REL".into(),
        "fact".into(),
        group.clone(),
    );
    e_with.fact_embedding = Some(vec![0.4, 0.5]);
    d.save_entity_edges(std::slice::from_ref(&e_with))
        .await
        .expect("edge with emb");
    let e_without = save_rel(&d, &without, &with_emb, "REL2", "fact2", &group).await;

    let edge_embs = d
        .get_embeddings_for_edges(&[e_with.uuid.clone(), e_without.uuid.clone()])
        .await
        .expect("edge embeddings");
    assert!(
        edge_embs.contains_key(&e_with.uuid),
        "embedded edge present"
    );
    assert!(
        !edge_embs.contains_key(&e_without.uuid),
        "edge without embedding omitted"
    );

    cleanup(&d, &group).await;
}

/// Center adjacency is UNDIRECTED: a node connected to center via an edge in
/// EITHER direction is reported.
#[tokio::test]
async fn nodes_connected_to_center_undirected() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let center = EntityNode::new("Center".into(), group.clone(), Utc::now());
    let out_neighbor = EntityNode::new("Out".into(), group.clone(), Utc::now());
    let in_neighbor = EntityNode::new("In".into(), group.clone(), Utc::now());
    let unrelated = EntityNode::new("Far".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[
        center.clone(),
        out_neighbor.clone(),
        in_neighbor.clone(),
        unrelated.clone(),
    ])
    .await
    .expect("nodes");
    // center -> out_neighbor (outgoing), in_neighbor -> center (incoming)
    save_rel(&d, &center, &out_neighbor, "OUT", "center to out", &group).await;
    save_rel(&d, &in_neighbor, &center, "IN", "in to center", &group).await;

    let adjacent = d
        .nodes_connected_to_center(
            &[
                out_neighbor.uuid.clone(),
                in_neighbor.uuid.clone(),
                unrelated.uuid.clone(),
            ],
            &center.uuid,
        )
        .await
        .expect("adjacency");
    let set: std::collections::HashSet<_> = adjacent.into_iter().collect();
    assert!(
        set.contains(&out_neighbor.uuid),
        "outgoing neighbor adjacent"
    );
    assert!(
        set.contains(&in_neighbor.uuid),
        "incoming neighbor adjacent (undirected)"
    );
    assert!(
        !set.contains(&unrelated.uuid),
        "unrelated node not adjacent"
    );

    cleanup(&d, &group).await;
}

/// MENTIONS in-degree counts per node uuid.
#[tokio::test]
async fn episode_mention_counts_basic() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    let group = unique_group();
    let mentioned = EntityNode::new("Mentioned".into(), group.clone(), Utc::now());
    let never = EntityNode::new("Never".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[mentioned.clone(), never.clone()])
        .await
        .expect("nodes");
    // Two episodes both mention `mentioned`.
    for i in 0..2 {
        let ep = EpisodicNode::new(
            format!("ep{i}"),
            group.clone(),
            EpisodeType::Message,
            "src".into(),
            "mentions it".into(),
            Utc::now(),
            Utc::now(),
        );
        d.save_episode(&ep).await.expect("episode");
        let me = EpisodicEdge::new(
            ep.uuid.clone(),
            mentioned.uuid.clone(),
            group.clone(),
            Utc::now(),
        );
        d.save_episodic_edges(&[me]).await.expect("mentions");
    }

    let counts = d
        .episode_mention_counts(&[mentioned.uuid.clone(), never.uuid.clone()])
        .await
        .expect("counts");
    assert_eq!(counts.get(&mentioned.uuid).copied(), Some(2));
    // `never` has no MENTIONS → no row → absent from the map (caller maps to inf).
    assert!(!counts.contains_key(&never.uuid), "unmentioned node absent");

    cleanup(&d, &group).await;
}

/// Filtered edge fulltext: edge_types whitelist + edge_uuids whitelist +
/// valid_at OR-of-ANDs window (two groups) + node_labels on both endpoints.
#[tokio::test]
async fn filtered_edge_search_combo() {
    let Some(d) = test_driver().await else {
        eprintln!("skipping: NEO4J_TEST_URI unset");
        return;
    };
    d.build_indices_and_constraints(false)
        .await
        .expect("indices");
    let group = unique_group();

    let mut alice = EntityNode::new("Alice".into(), group.clone(), Utc::now());
    alice.labels = vec!["Entity".into(), "Person".into()];
    let mut acme = EntityNode::new("Acme".into(), group.clone(), Utc::now());
    acme.labels = vec!["Entity".into(), "Company".into()];
    d.save_entity_nodes(&[alice.clone(), acme.clone()])
        .await
        .expect("nodes");

    let now = Utc::now();
    // Target edge: name WORKS_AT, valid_at = now, both endpoints labelled.
    let mut wanted = EntityEdge::new(
        alice.uuid.clone(),
        acme.uuid.clone(),
        "WORKS_AT".into(),
        "Alice is employed at distinctiveword Acme".into(),
        group.clone(),
    );
    wanted.valid_at = Some(now);
    d.save_entity_edges(std::slice::from_ref(&wanted))
        .await
        .expect("wanted");

    // Decoy edge: name KNOWS (excluded by edge_types), valid_at far in past.
    let mut decoy = EntityEdge::new(
        alice.uuid.clone(),
        acme.uuid.clone(),
        "KNOWS".into(),
        "Alice knows distinctiveword Acme".into(),
        group.clone(),
    );
    decoy.valid_at = Some(now - Duration::days(365));
    d.save_entity_edges(std::slice::from_ref(&decoy))
        .await
        .expect("decoy");

    use chronicle_core::search::filters::{ComparisonOperator, DateFilter};
    let filters = SearchFilters {
        edge_types: Some(vec!["WORKS_AT".into()]),
        edge_uuids: Some(vec![wanted.uuid.clone(), decoy.uuid.clone()]),
        node_labels: Some(vec!["Person".into(), "Company".into()]),
        // OR-of-ANDs: ((valid_at >= now-1h AND valid_at <= now+1h) OR (valid_at >= now+10y))
        valid_at: Some(vec![
            vec![
                DateFilter {
                    date: Some(now - Duration::hours(1)),
                    comparison_operator: ComparisonOperator::Gte,
                },
                DateFilter {
                    date: Some(now + Duration::hours(1)),
                    comparison_operator: ComparisonOperator::Lte,
                },
            ],
            vec![DateFilter {
                date: Some(now + Duration::days(3650)),
                comparison_operator: ComparisonOperator::Gte,
            }],
        ]),
        ..Default::default()
    };

    let hits = d
        .edge_fulltext_search(
            "distinctiveword",
            &filters,
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("filtered edge fts");
    let set: std::collections::HashSet<_> = hits.iter().map(|e| e.uuid.clone()).collect();
    assert!(
        set.contains(&wanted.uuid),
        "WORKS_AT edge in the valid_at window matches"
    );
    assert!(
        !set.contains(&decoy.uuid),
        "KNOWS decoy excluded by edge_types + valid_at window"
    );

    // node_labels enforcement: a third edge between two UNLABELLED nodes must be
    // excluded even though it matches edge_types + window.
    let plain_src = EntityNode::new("Plain1".into(), group.clone(), Utc::now());
    let plain_tgt = EntityNode::new("Plain2".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[plain_src.clone(), plain_tgt.clone()])
        .await
        .expect("plain nodes");
    let mut plain_edge = EntityEdge::new(
        plain_src.uuid.clone(),
        plain_tgt.uuid.clone(),
        "WORKS_AT".into(),
        "plain distinctiveword link".into(),
        group.clone(),
    );
    plain_edge.valid_at = Some(now);
    d.save_entity_edges(std::slice::from_ref(&plain_edge))
        .await
        .expect("plain edge");

    let filters2 = SearchFilters {
        edge_types: Some(vec!["WORKS_AT".into()]),
        node_labels: Some(vec!["Person".into(), "Company".into()]),
        ..Default::default()
    };
    let hits2 = d
        .edge_fulltext_search(
            "distinctiveword",
            &filters2,
            std::slice::from_ref(&group),
            10,
        )
        .await
        .expect("label-filtered fts");
    let set2: std::collections::HashSet<_> = hits2.iter().map(|e| e.uuid.clone()).collect();
    assert!(set2.contains(&wanted.uuid), "labelled endpoints pass");
    assert!(
        !set2.contains(&plain_edge.uuid),
        "unlabelled endpoints excluded by node_labels"
    );

    cleanup(&d, &group).await;
}

// =====================================================================
// Phase-4: community + saga + maintenance ops (env-gated; skip w/o DB)
// =====================================================================

macro_rules! skip_without_db {
    () => {{
        match test_driver().await {
            Some(d) => d,
            None => {
                eprintln!("skipping: NEO4J_TEST_URI unset");
                return;
            }
        }
    }};
}

#[tokio::test]
async fn community_node_and_edge_roundtrip() {
    let d = skip_without_db!();
    d.build_indices_and_constraints(false).await.expect("build");
    let group = unique_group();

    // An entity to be a HAS_MEMBER target.
    let mut entity = EntityNode::new("Member".into(), group.clone(), Utc::now());
    entity.name_embedding = Some(vec![0.1; 8]);
    d.save_entity_nodes(std::slice::from_ref(&entity))
        .await
        .expect("save entity");

    let mut community = CommunityNode::new("Tech Cluster".into(), group.clone(), Utc::now());
    community.summary = "tech firms".into();
    community.name_embedding = Some(vec![0.2; 8]);
    d.save_community_nodes(std::slice::from_ref(&community))
        .await
        .expect("save community");

    let edge = CommunityEdge::new(
        community.uuid.clone(),
        entity.uuid.clone(),
        group.clone(),
        Utc::now(),
    );
    d.save_community_edges(std::slice::from_ref(&edge))
        .await
        .expect("save community edge");

    let by_group = d
        .get_community_nodes_by_group_ids(std::slice::from_ref(&group))
        .await
        .expect("by group");
    assert_eq!(by_group.len(), 1);
    assert_eq!(by_group[0].uuid, community.uuid);
    assert_eq!(by_group[0].summary, "tech firms");

    let by_uuid = d
        .get_community_nodes_by_uuids(&[community.uuid.clone()])
        .await
        .expect("by uuid");
    assert_eq!(by_uuid.len(), 1);

    // membership already-member
    let mem = d
        .community_of_member(&entity.uuid)
        .await
        .expect("member lookup");
    assert_eq!(mem.map(|c| c.uuid), Some(community.uuid.clone()));

    // embeddings loader
    let emb = d
        .get_embeddings_for_communities(&[community.uuid.clone()])
        .await
        .expect("embeddings");
    assert!(emb.contains_key(&community.uuid));

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn community_fulltext_and_similarity_search_live() {
    let d = skip_without_db!();
    d.build_indices_and_constraints(false).await.expect("build");
    let group = unique_group();

    let mut c1 = CommunityNode::new("Distinctcommunityalpha".into(), group.clone(), Utc::now());
    c1.name_embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    let mut c2 = CommunityNode::new("Otherbeta".into(), group.clone(), Utc::now());
    c2.name_embedding = Some(vec![0.0, 1.0, 0.0, 0.0]);
    d.save_community_nodes(&[c1.clone(), c2.clone()])
        .await
        .expect("save");

    let ft = d
        .community_fulltext_search("distinctcommunityalpha", std::slice::from_ref(&group), 10)
        .await
        .expect("fulltext");
    assert!(ft.iter().any(|c| c.uuid == c1.uuid));

    let sim = d
        .community_similarity_search(&[1.0, 0.0, 0.0, 0.0], std::slice::from_ref(&group), 10, 0.5)
        .await
        .expect("similarity");
    assert_eq!(sim.first().map(|c| c.uuid.clone()), Some(c1.uuid.clone()));

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn community_clusters_and_neighbor_vote_live() {
    let d = skip_without_db!();
    d.build_indices_and_constraints(false).await.expect("build");
    let group = unique_group();

    let a = EntityNode::new("A".into(), group.clone(), Utc::now());
    let b = EntityNode::new("B".into(), group.clone(), Utc::now());
    let x = EntityNode::new("X".into(), group.clone(), Utc::now());
    d.save_entity_nodes(&[a.clone(), b.clone(), x.clone()])
        .await
        .expect("save nodes");

    let mut ab = EntityEdge::new(
        a.uuid.clone(),
        b.uuid.clone(),
        "REL".into(),
        "a-b".into(),
        group.clone(),
    );
    ab.fact_embedding = Some(vec![0.1; 8]);
    let mut ax = EntityEdge::new(
        a.uuid.clone(),
        x.uuid.clone(),
        "REL".into(),
        "a-x".into(),
        group.clone(),
    );
    ax.fact_embedding = Some(vec![0.1; 8]);
    d.save_entity_edges(&[ab, ax]).await.expect("save edges");

    let clusters = d
        .get_community_clusters(std::slice::from_ref(&group))
        .await
        .expect("clusters");
    assert_eq!(clusters.len(), 1);
    let a_proj = clusters[0]
        .nodes
        .iter()
        .find(|n| n.node_uuid == a.uuid)
        .expect("a present");
    assert!(a_proj.neighbors.iter().any(|n| n.node_uuid == b.uuid));
    assert!(a_proj.neighbors.iter().any(|n| n.node_uuid == x.uuid));

    // a belongs to a community; x is a neighbour → neighbour-vote row
    let mut community = CommunityNode::new("C".into(), group.clone(), Utc::now());
    community.name_embedding = Some(vec![0.3; 8]);
    d.save_community_nodes(std::slice::from_ref(&community))
        .await
        .expect("save community");
    d.save_community_edges(&[CommunityEdge::new(
        community.uuid.clone(),
        a.uuid.clone(),
        group.clone(),
        Utc::now(),
    )])
    .await
    .expect("save member");

    let votes = d.neighbor_communities(&x.uuid).await.expect("votes");
    assert!(votes.iter().any(|c| c.uuid == community.uuid));

    // remove_communities clears them
    d.remove_communities().await.expect("remove");
    let after = d
        .get_community_nodes_by_group_ids(std::slice::from_ref(&group))
        .await
        .expect("after remove");
    assert!(after.is_empty());

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn saga_threading_and_contents_live() {
    let d = skip_without_db!();
    d.build_indices_and_constraints(false).await.expect("build");
    let group = unique_group();

    let v1 = Utc::now() - Duration::days(2);
    let v2 = Utc::now() - Duration::days(1);
    let mut e1 = EpisodicNode::new(
        "ep1".into(),
        group.clone(),
        EpisodeType::Message,
        "d".into(),
        "first content".into(),
        Utc::now(),
        v1,
    );
    e1.uuid = uuid::Uuid::new_v4().to_string();
    let mut e2 = EpisodicNode::new(
        "ep2".into(),
        group.clone(),
        EpisodeType::Message,
        "d".into(),
        "second content".into(),
        Utc::now(),
        v2,
    );
    e2.uuid = uuid::Uuid::new_v4().to_string();
    d.save_episode(&e1).await.expect("save e1");
    d.save_episode(&e2).await.expect("save e2");

    let saga = SagaNode::new("my-saga".into(), group.clone(), Utc::now());
    d.save_saga_node(&saga).await.expect("save saga");

    // get_or_create lookup finds it
    let found = d
        .get_saga_by_name("my-saga", &group)
        .await
        .expect("get saga");
    assert_eq!(found.map(|s| s.uuid), Some(saga.uuid.clone()));

    for ep in [&e1, &e2] {
        d.save_has_episode_edge(&HasEpisodeEdge::new(
            saga.uuid.clone(),
            ep.uuid.clone(),
            group.clone(),
            Utc::now(),
        ))
        .await
        .expect("has_episode");
    }
    d.save_next_episode_edge(&NextEpisodeEdge::new(
        e1.uuid.clone(),
        e2.uuid.clone(),
        group.clone(),
        Utc::now(),
    ))
    .await
    .expect("next_episode");

    // previous episode for e2 = e1
    let prev = d
        .saga_previous_episode_uuid(&saga.uuid, &e2.uuid)
        .await
        .expect("prev");
    assert_eq!(prev, Some(e1.uuid.clone()));

    // contents (no watermark) chronological
    let contents = d
        .saga_episode_contents(&saga.uuid, None, 200)
        .await
        .expect("contents");
    assert_eq!(contents.len(), 2);
    assert_eq!(contents[0].0, "first content");
    assert_eq!(contents[1].0, "second content");

    cleanup(&d, &group).await;
}

#[tokio::test]
async fn mentioned_nodes_and_cascade_deletes_live() {
    let d = skip_without_db!();
    d.build_indices_and_constraints(false).await.expect("build");
    let group = unique_group();

    let n1 = EntityNode::new("N1".into(), group.clone(), Utc::now());
    d.save_entity_nodes(std::slice::from_ref(&n1))
        .await
        .expect("save node");
    let mut ep = EpisodicNode::new(
        "ep".into(),
        group.clone(),
        EpisodeType::Message,
        "d".into(),
        "content".into(),
        Utc::now(),
        Utc::now(),
    );
    ep.uuid = uuid::Uuid::new_v4().to_string();
    d.save_episode(&ep).await.expect("save episode");
    d.save_episodic_edges(&[EpisodicEdge::new(
        ep.uuid.clone(),
        n1.uuid.clone(),
        group.clone(),
        Utc::now(),
    )])
    .await
    .expect("mention");

    let mentioned = d
        .get_mentioned_nodes(&[ep.uuid.clone()])
        .await
        .expect("mentioned");
    assert!(mentioned.iter().any(|n| n.uuid == n1.uuid));

    // delete cascade
    d.delete_entity_nodes_by_uuids(std::slice::from_ref(&n1.uuid))
        .await
        .expect("del node");
    assert!(d.get_entity_node(&n1.uuid).await.expect("get").is_none());
    d.delete_episode(&ep.uuid).await.expect("del episode");
    assert!(d.get_episode(&ep.uuid).await.expect("get ep").is_none());

    cleanup(&d, &group).await;
}
