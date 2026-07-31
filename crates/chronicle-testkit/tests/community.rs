// Driver-backed integration tests for chronicle_core::pipeline::community_ops.
//
// Placed in chronicle-testkit (not chronicle-core) to avoid the dev-dependency
// cycle chronicle-core → chronicle-testkit → chronicle-core (same pattern as
// node_ops.rs / edge_search.rs).
//
// MockLlm scripting note: build_community / update_community fan out summarize_pair
// under the shared Clients semaphore. We build Clients with `max_concurrency = 1`,
// which forces summarize_pair calls to run strictly sequentially (one permit), so the
// queued MockLlm responses are consumed in a deterministic order. summarize_pair
// responses use the `{"summary": ...}` shape (Summary); summary_description responses
// use `{"description": ...}` (SummaryDescription).

use std::sync::Arc;

use chronicle_core::driver::{CommunityOps, EntityEdgeOps, EntityNodeOps, GraphDriver};
use chronicle_core::pipeline::clients::Clients;
use chronicle_core::pipeline::community_ops::{
    build_community, determine_entity_community, rebuild_communities, update_community,
};
use chronicle_core::types::{CommunityEdge, CommunityNode, EntityEdge, EntityNode};

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

use chrono::Utc;

const EMB_DIM: usize = 8;

fn clients(driver: Arc<FakeDriver>, llm: Arc<MockLlm>, emb: Arc<MockEmbedder>) -> Clients {
    // max_concurrency = 1 forces deterministic sequential summarize_pair fan-out.
    Clients::new(driver as Arc<dyn GraphDriver>, llm as _, emb as _, 1)
}

fn entity(uuid: &str, summary: &str, group_id: &str) -> EntityNode {
    let mut n = EntityNode::new(format!("name-{uuid}"), group_id.to_string(), Utc::now());
    n.uuid = uuid.to_string();
    n.summary = summary.to_string();
    n
}

fn relates(source: &str, target: &str, group_id: &str) -> EntityEdge {
    let mut e = EntityEdge::new(
        source.to_string(),
        target.to_string(),
        "RELATES_TO".into(),
        "fact".into(),
        group_id.to_string(),
    );
    e.uuid = format!("e-{source}-{target}");
    e
}

fn summary_resp(s: &str) -> serde_json::Value {
    serde_json::json!({ "summary": s })
}

fn desc_resp(s: &str) -> serde_json::Value {
    serde_json::json!({ "description": s })
}

// ---------------------------------------------------------------------------
// build_community: odd (3) and even (4) member counts, scripted MockLlm
// ---------------------------------------------------------------------------

/// build_community over an EVEN cluster (4 members): round 1 = 2 summarize_pair
/// calls, round 2 = 1 summarize_pair call, then 1 summary_description call.
/// Total LLM calls = 4 (3 summarize_pair + 1 summary_description).
#[tokio::test]
async fn build_community_even_four_members() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![
        // Round 1 (2 pairs).
        summary_resp("merged AB"),
        summary_resp("merged CD"),
        // Round 2 (1 pair).
        summary_resp("merged ABCD."),
        // Name.
        desc_resp("Community of four."),
    ]));
    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let cluster = vec![
        entity("a", "A summary.", "g1"),
        entity("b", "B summary.", "g1"),
        entity("c", "C summary.", "g1"),
        entity("d", "D summary.", "g1"),
    ];

    let (node, edges) = build_community(&cl, &cluster).await.unwrap();

    assert_eq!(node.group_id, "g1");
    assert_eq!(node.name, "Community of four.");
    assert_eq!(node.summary, "merged ABCD.");
    assert_eq!(node.labels, vec!["Community"]);
    // One HAS_MEMBER edge per member.
    assert_eq!(edges.len(), 4);
    for e in &edges {
        assert_eq!(e.source_node_uuid, node.uuid);
        assert_eq!(e.group_id, "g1");
    }
    let targets: Vec<&str> = edges.iter().map(|e| e.target_node_uuid.as_str()).collect();
    assert!(["a", "b", "c", "d"].iter().all(|u| targets.contains(u)));

    // 3 summarize_pair + 1 summary_description.
    assert_eq!(llm.call_count(), 4);
    let requests = llm.requests.lock().unwrap();
    let pair_calls = requests
        .iter()
        .filter(|r| r.prompt_name.as_deref() == Some("summarize_nodes.summarize_pair"))
        .count();
    let desc_calls = requests
        .iter()
        .filter(|r| r.prompt_name.as_deref() == Some("summarize_nodes.summary_description"))
        .count();
    assert_eq!(pair_calls, 3, "even(4): 3 summarize_pair calls");
    assert_eq!(desc_calls, 1, "even(4): 1 summary_description call");
}

/// build_community over an ODD cluster (3 members): round 1 pops the odd member,
/// 1 summarize_pair call on the remaining pair, re-appends the odd → 2 summaries;
/// round 2 = 1 summarize_pair call; then 1 summary_description.
/// Total LLM calls = 3 (2 summarize_pair + 1 summary_description).
#[tokio::test]
async fn build_community_odd_three_members() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![
        // Round 1 (1 pair on first two; third carried).
        summary_resp("merged AB"),
        // Round 2 (1 pair: merged-AB + carried C).
        summary_resp("merged ABC."),
        // Name.
        desc_resp("Community of three."),
    ]));
    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let cluster = vec![
        entity("a", "A summary.", "g2"),
        entity("b", "B summary.", "g2"),
        entity("c", "C summary.", "g2"),
    ];

    let (node, edges) = build_community(&cl, &cluster).await.unwrap();

    assert_eq!(node.name, "Community of three.");
    assert_eq!(node.summary, "merged ABC.");
    assert_eq!(edges.len(), 3);

    assert_eq!(llm.call_count(), 3);
    let requests = llm.requests.lock().unwrap();
    let pair_calls = requests
        .iter()
        .filter(|r| r.prompt_name.as_deref() == Some("summarize_nodes.summarize_pair"))
        .count();
    assert_eq!(pair_calls, 2, "odd(3): 2 summarize_pair calls");
}

/// Single-member cluster: no summarize_pair, just the summary_description name.
#[tokio::test]
async fn build_community_single_member_skips_pairwise() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![desc_resp("Lone community.")]));
    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let cluster = vec![entity("solo", "Solo summary.", "g3")];
    let (node, edges) = build_community(&cl, &cluster).await.unwrap();

    assert_eq!(node.summary, "Solo summary.");
    assert_eq!(node.name, "Lone community.");
    assert_eq!(edges.len(), 1);
    assert_eq!(llm.call_count(), 1);
}

// ---------------------------------------------------------------------------
// determine_entity_community: 3 paths
// ---------------------------------------------------------------------------

/// Path (a): the entity already HAS_MEMBER a community → (community, is_new=false).
#[tokio::test]
async fn determine_entity_community_already_member() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![]));

    let mut community = CommunityNode::new("Existing".into(), "g1".into(), Utc::now());
    community.uuid = "c1".into();
    driver.save_community_nodes(&[community]).await.unwrap();
    driver
        .save_community_edges(&[CommunityEdge::new(
            "c1".into(),
            "ent".into(),
            "g1".into(),
            Utc::now(),
        )])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), llm, emb);
    let ent = entity("ent", "E.", "g1");
    let (community, is_new) = determine_entity_community(&cl, &ent).await.unwrap();
    assert!(!is_new, "already-member path is not new");
    assert_eq!(community.unwrap().uuid, "c1");
}

/// Path (b): neighbour-vote — the plurality community among RELATES_TO neighbours'
/// memberships → (community, is_new=true). Two neighbours in c1, one in c2.
#[tokio::test]
async fn determine_entity_community_neighbor_vote() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![]));

    // Communities c1, c2.
    let mut c1 = CommunityNode::new("C1".into(), "g1".into(), Utc::now());
    c1.uuid = "c1".into();
    let mut c2 = CommunityNode::new("C2".into(), "g1".into(), Utc::now());
    c2.uuid = "c2".into();
    driver.save_community_nodes(&[c1, c2]).await.unwrap();

    // Neighbours n1,n2 in c1; n3 in c2.
    driver
        .save_entity_nodes(&[
            entity("n1", "N1.", "g1"),
            entity("n2", "N2.", "g1"),
            entity("n3", "N3.", "g1"),
            entity("target", "T.", "g1"),
        ])
        .await
        .unwrap();
    driver
        .save_community_edges(&[
            CommunityEdge::new("c1".into(), "n1".into(), "g1".into(), Utc::now()),
            CommunityEdge::new("c1".into(), "n2".into(), "g1".into(), Utc::now()),
            CommunityEdge::new("c2".into(), "n3".into(), "g1".into(), Utc::now()),
        ])
        .await
        .unwrap();
    // target RELATES_TO n1, n2, n3.
    driver
        .save_entity_edges(&[
            relates("target", "n1", "g1"),
            relates("target", "n2", "g1"),
            relates("target", "n3", "g1"),
        ])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), llm, emb);
    let target = entity("target", "T.", "g1");
    let (community, is_new) = determine_entity_community(&cl, &target).await.unwrap();
    assert!(is_new, "neighbour-vote path is new");
    assert_eq!(community.unwrap().uuid, "c1", "plurality (2 vs 1) → c1");
}

/// Path (c): no membership and no community-bearing neighbours → (None, false).
#[tokio::test]
async fn determine_entity_community_none() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![]));

    driver
        .save_entity_nodes(&[entity("lonely", "L.", "g1")])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), llm, emb);
    let lonely = entity("lonely", "L.", "g1");
    let (community, is_new) = determine_entity_community(&cl, &lonely).await.unwrap();
    assert!(community.is_none());
    assert!(!is_new);
}

// ---------------------------------------------------------------------------
// update_community: is_new neighbour-vote path saves a HAS_MEMBER edge + summary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_community_is_new_saves_edge_and_summary() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    // summarize_pair (entity+community summaries) then summary_description (name).
    let llm = Arc::new(MockLlm::new(vec![
        summary_resp("folded summary."),
        desc_resp("Updated community name."),
    ]));

    let mut c1 = CommunityNode::new("Old".into(), "g1".into(), Utc::now());
    c1.uuid = "c1".into();
    c1.summary = "old community summary.".into();
    driver.save_community_nodes(&[c1]).await.unwrap();

    // Neighbour n1 in c1; target RELATES_TO n1.
    driver
        .save_entity_nodes(&[entity("n1", "N1.", "g1"), entity("target", "T.", "g1")])
        .await
        .unwrap();
    driver
        .save_community_edges(&[CommunityEdge::new(
            "c1".into(),
            "n1".into(),
            "g1".into(),
            Utc::now(),
        )])
        .await
        .unwrap();
    driver
        .save_entity_edges(&[relates("target", "n1", "g1")])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));
    let target = entity("target", "entity summary.", "g1");
    let (nodes, edges) = update_community(&cl, &target).await.unwrap();

    assert_eq!(nodes.len(), 1);
    let updated = &nodes[0];
    assert_eq!(updated.uuid, "c1");
    assert_eq!(updated.summary, "folded summary.");
    assert_eq!(updated.name, "Updated community name.");
    assert!(
        updated.name_embedding.is_some(),
        "name embedding regenerated"
    );

    // is_new → one HAS_MEMBER edge returned AND persisted.
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].source_node_uuid, "c1");
    assert_eq!(edges[0].target_node_uuid, "target");

    // The new HAS_MEMBER edge is now in the driver: target is a member of c1.
    let member = driver.community_of_member("target").await.unwrap();
    assert_eq!(member.unwrap().uuid, "c1");

    // The community node was persisted with the new summary.
    let persisted = driver
        .get_community_nodes_by_uuids(&["c1".into()])
        .await
        .unwrap();
    assert_eq!(persisted[0].summary, "folded summary.");
}

/// update_community no-op when the entity has no community.
#[tokio::test]
async fn update_community_noop_when_no_community() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![])); // any LLM call would fail.

    driver
        .save_entity_nodes(&[entity("lonely", "L.", "g1")])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), llm, emb);
    let lonely = entity("lonely", "L.", "g1");
    let (nodes, edges) = update_community(&cl, &lonely).await.unwrap();
    assert!(nodes.is_empty());
    assert!(edges.is_empty());
}

// ---------------------------------------------------------------------------
// rebuild_communities: small-graph end-to-end
// ---------------------------------------------------------------------------

/// Single triangle {a,b,c} in one group → one community with 3 HAS_MEMBER edges;
/// the node carries a regenerated name embedding and is persisted. Exercises the
/// full rebuild path: remove_communities → build_communities → name embeddings →
/// save nodes + edges.
///
/// A single cluster keeps the MockLlm queue deterministic: build_community over a
/// 3-member triangle issues exactly 2 summarize_pair calls (serialized via the
/// shared permit) then 1 summary_description call — consumed in queue order.
#[tokio::test]
async fn rebuild_communities_small_graph_e2e() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));

    let llm = Arc::new(MockLlm::new(vec![
        summary_resp("m1"),
        summary_resp("m2."),
        desc_resp("Triangle community."),
    ]));

    // Stale pre-existing community that rebuild must DETACH DELETE first.
    let mut stale = CommunityNode::new("Stale".into(), "g1".into(), Utc::now());
    stale.uuid = "stale".into();
    driver.save_community_nodes(&[stale]).await.unwrap();

    driver
        .save_entity_nodes(&[
            entity("a", "A.", "g1"),
            entity("b", "B.", "g1"),
            entity("c", "C.", "g1"),
        ])
        .await
        .unwrap();
    driver
        .save_entity_edges(&[
            relates("a", "b", "g1"),
            relates("b", "c", "g1"),
            relates("a", "c", "g1"),
        ])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));
    let (nodes, edges) = rebuild_communities(&cl, &["g1".into()]).await.unwrap();

    assert_eq!(nodes.len(), 1, "single triangle → one community");
    assert_eq!(edges.len(), 3, "3 HAS_MEMBER edges");
    let community = &nodes[0];
    assert_eq!(community.name, "Triangle community.");
    assert_eq!(community.summary, "m2.");
    assert!(
        community.name_embedding.is_some(),
        "name embedding regenerated"
    );
    assert_eq!(community.group_id, "g1");

    // Persisted: exactly the rebuilt community (stale one DETACH DELETEd).
    let saved = driver
        .get_community_nodes_by_group_ids(&["g1".into()])
        .await
        .unwrap();
    assert_eq!(saved.len(), 1, "stale community removed; one rebuilt");
    assert_eq!(saved[0].uuid, community.uuid);

    // All three entities are members of the rebuilt community.
    for u in ["a", "b", "c"] {
        let m = driver.community_of_member(u).await.unwrap();
        assert_eq!(m.unwrap().uuid, community.uuid);
    }
}
