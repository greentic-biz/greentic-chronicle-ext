// Phase-4 Task 6 acceptance gates for the maintenance + saga surface:
//   - chronicle_core::pipeline::maintenance::{add_triplet, remove_episode,
//     get_nodes_and_edges_by_episode}
//   - chronicle_core::pipeline::saga::summarize_saga
//   - saga threading + update_communities wired through add_episode
//
// Driver-backed tests over the in-memory FakeDriver + deterministic Mock
// embedder. Where the path issues LLM calls the order is documented inline and a
// scripted/keyed MockLlm replays them; the resolution-free fast paths use an
// empty MockLlm queue so any stray LLM call surfaces as a test failure.

use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::driver::{
    CommunityOps, EntityEdgeOps, EntityNodeOps, EpisodeOps, EpisodicEdgeOps, GraphDriver, SagaOps,
};
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::pipeline::clients::Clients;
use chronicle_core::pipeline::maintenance::{
    add_triplet, get_nodes_and_edges_by_episode, remove_episode,
};
use chronicle_core::pipeline::saga::summarize_saga;
use chronicle_core::types::{
    CommunityEdge, CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode,
    SagaNode,
};

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

use chrono::{DateTime, TimeZone, Utc};

const EMB_DIM: usize = 8;

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()
}

fn t1() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()
}

fn entity(uuid: &str, name: &str, group_id: &str) -> EntityNode {
    let mut n = EntityNode::new(name.to_string(), group_id.to_string(), Utc::now());
    n.uuid = uuid.to_string();
    n
}

fn clients(driver: Arc<FakeDriver>, llm: Arc<MockLlm>, emb: Arc<MockEmbedder>) -> Clients {
    Clients::new(driver as Arc<dyn GraphDriver>, llm as _, emb as _, 1)
}

// ---------------------------------------------------------------------------
// add_triplet
// ---------------------------------------------------------------------------

/// add_triplet over two PRE-EXISTING nodes (both resolve via get_by_uuid) into a
/// fresh graph: no node-dedupe LLM, empty search pools ⇒ edge resolves via the
/// both-empty fast path ⇒ ZERO LLM calls. The triplet is persisted with the edge
/// pointing at the resolved node uuids; NO episode / episodic edge is created.
#[tokio::test]
async fn add_triplet_existing_nodes_no_episode_zero_llm() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    // Empty queue: any LLM call would surface as EmptyResponse and fail.
    let llm = Arc::new(MockLlm::new(vec![]));

    // Pre-save the source + target so add_triplet resolves them via get_by_uuid.
    let source = entity("alice", "Alice", "g1");
    let target = entity("acme", "Acme", "g1");
    driver
        .save_entity_nodes(&[source.clone(), target.clone()])
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut edge = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme.".into(),
        "g1".into(),
    );
    edge.valid_at = Some(t0());

    let result = add_triplet(&cl, source, edge, target).await.unwrap();

    assert_eq!(result.nodes.len(), 2, "source + target resolved");
    assert_eq!(
        result.edges.len(),
        1,
        "single resolved edge, none invalidated"
    );
    let saved_edge = &result.edges[0];
    assert_eq!(saved_edge.source_node_uuid, "alice");
    assert_eq!(saved_edge.target_node_uuid, "acme");
    assert_eq!(saved_edge.fact, "Alice works at Acme.");

    // Persisted to the driver.
    let stored = driver
        .get_entity_edge(&saved_edge.uuid)
        .await
        .unwrap()
        .expect("edge persisted");
    assert_eq!(stored.name, "WORKS_AT");

    // No episode / episodic edge created by add_triplet.
    assert_eq!(
        driver.episodic_edge_count(),
        0,
        "add_triplet must not create MENTIONS edges"
    );

    assert_eq!(
        llm.call_count(),
        0,
        "resolved-node fast path issues no LLM call"
    );
}

/// add_triplet dedups against an EXISTING edge with the same endpoints + an
/// equivalent fact via the exact-duplicate fast path (no LLM): the result
/// resolves to the existing edge and appends no new edge.
#[tokio::test]
async fn add_triplet_dedups_to_existing_edge() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![]));

    let source = entity("alice", "Alice", "g1");
    let target = entity("acme", "Acme", "g1");
    driver
        .save_entity_nodes(&[source.clone(), target.clone()])
        .await
        .unwrap();

    // Existing edge between the same endpoints (so it lands in valid_edges and the
    // related pool); fact normalizes equal to the incoming one.
    let mut existing = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme".into(),
        "g1".into(),
    );
    existing.uuid = "existing-edge".into();
    existing.episodes = vec!["older-ep".into()];
    existing.fact_embedding = Some(emb.create("Alice works at Acme").await.unwrap());
    driver.save_entity_edges(&[existing]).await.unwrap();

    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut edge = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "  alice WORKS at acme ".into(), // normalizes to "alice works at acme"
        "g1".into(),
    );
    edge.valid_at = Some(t0());

    let result = add_triplet(&cl, source, edge, target).await.unwrap();

    // Resolves onto the existing edge (exact-duplicate fast path), nothing new.
    assert_eq!(result.edges.len(), 1);
    assert_eq!(
        result.edges[0].uuid, "existing-edge",
        "must resolve to the existing edge, not mint a new one"
    );
    assert_eq!(
        llm.call_count(),
        0,
        "exact-duplicate fast path issues no LLM call"
    );
}

// ---------------------------------------------------------------------------
// remove_episode cascade
// ---------------------------------------------------------------------------

/// Hand-builds a graph where one episode is the SOLE source of an edge and a node
/// it solely mentions, while a shared node is mentioned by a second episode.
/// remove_episode must delete: the primary-source edge, the single-mention node,
/// and the episode — but keep the multi-episode (survivor) node and any edge it
/// did not originate.
#[tokio::test]
async fn remove_episode_deletes_primary_edge_and_single_mention_node_only() {
    let driver = Arc::new(FakeDriver::new());

    // Episodes: ep1 (to remove) and ep2 (survivor witness). uuid pinned so the
    // edge/MENTIONS wiring and the remove_episode lookup all key on "ep1"/"ep2".
    let mut ep1 = EpisodicNode::new(
        "ep1".into(),
        "g1".into(),
        EpisodeType::Message,
        "d".into(),
        "Alice works at Acme.".into(),
        t0(),
        t0(),
    );
    ep1.uuid = "ep1".into();
    let mut ep2 = EpisodicNode::new(
        "ep2".into(),
        "g1".into(),
        EpisodeType::Message,
        "d".into(),
        "Alice is mentioned again.".into(),
        t1(),
        t1(),
    );
    ep2.uuid = "ep2".into();

    // Nodes:
    //  - alice: mentioned by ep1 AND ep2 → survives (multi-episode).
    //  - bob:   mentioned by ep2 only    → survives (not an ep1 node).
    //  - acme:  mentioned by ep1 only    → deleted (single-mention).
    let alice = entity("alice", "Alice", "g1");
    let bob = entity("bob", "Bob", "g1");
    let acme = entity("acme", "Acme", "g1");
    driver
        .save_entity_nodes(&[alice.clone(), bob.clone(), acme.clone()])
        .await
        .unwrap();

    // Edge originated by ep1 (episodes[0] == ep1) → deleted on ep1 removal.
    let mut edge_ep1 = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme.".into(),
        "g1".into(),
    );
    edge_ep1.uuid = "edge-ep1".into();
    edge_ep1.episodes = vec!["ep1".into()];

    // Edge between two SURVIVING nodes (alice→bob) whose primary source is ep2 but
    // which ep1 ALSO mentions → survives (episodes[0] != ep1), proving the
    // primary-source-only delete condition independent of node-detach cascade.
    let mut edge_shared = EntityEdge::new(
        "alice".into(),
        "bob".into(),
        "PARTNERS_WITH".into(),
        "Alice partners with Bob.".into(),
        "g1".into(),
    );
    edge_shared.uuid = "edge-shared".into();
    edge_shared.episodes = vec!["ep2".into(), "ep1".into()];

    driver
        .save_entity_edges(&[edge_ep1.clone(), edge_shared.clone()])
        .await
        .unwrap();

    // ep1.entity_edges references both edges it mentions.
    ep1.entity_edges = vec!["edge-ep1".into(), "edge-shared".into()];
    driver.save_episode(&ep1).await.unwrap();
    driver.save_episode(&ep2).await.unwrap();

    // MENTIONS edges: ep1 → {alice, acme}; ep2 → {alice, bob}.
    driver
        .save_episodic_edges(&[
            EpisodicEdge::new("ep1".into(), "alice".into(), "g1".into(), t0()),
            EpisodicEdge::new("ep1".into(), "acme".into(), "g1".into(), t0()),
            EpisodicEdge::new("ep2".into(), "alice".into(), "g1".into(), t1()),
            EpisodicEdge::new("ep2".into(), "bob".into(), "g1".into(), t1()),
        ])
        .await
        .unwrap();

    let llm = Arc::new(MockLlm::new(vec![]));
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let cl = clients(Arc::clone(&driver), llm, emb);

    remove_episode(&cl, "ep1").await.unwrap();

    // Primary-source edge deleted; shared edge (primary = ep2) survives.
    assert!(
        driver.get_entity_edge("edge-ep1").await.unwrap().is_none(),
        "ep1-originated edge must be deleted"
    );
    assert!(
        driver
            .get_entity_edge("edge-shared")
            .await
            .unwrap()
            .is_some(),
        "edge whose primary source is ep2 must survive"
    );

    // Single-mention node deleted; multi-episode node survives.
    assert!(
        driver.get_entity_node("acme").await.unwrap().is_none(),
        "single-mention node (acme) must be deleted"
    );
    assert!(
        driver.get_entity_node("alice").await.unwrap().is_some(),
        "multi-episode node (alice) must survive"
    );

    // Episode itself deleted.
    assert!(
        driver.get_episode("ep1").await.unwrap().is_none(),
        "episode must be deleted"
    );
    assert!(
        driver.get_episode("ep2").await.unwrap().is_some(),
        "the witness episode must survive"
    );
}

/// remove_episode on a missing UUID surfaces EpisodeNotFound.
#[tokio::test]
async fn remove_episode_missing_uuid_errors() {
    let driver = Arc::new(FakeDriver::new());
    let llm = Arc::new(MockLlm::new(vec![]));
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let cl = clients(Arc::clone(&driver), llm, emb);

    let err = remove_episode(&cl, "nope").await.unwrap_err();
    assert!(
        format!("{err}").contains("nope"),
        "error should reference the missing episode uuid: {err}"
    );
}

// ---------------------------------------------------------------------------
// get_nodes_and_edges_by_episode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_nodes_and_edges_by_episode_returns_mentioned_nodes_and_entity_edges() {
    let driver = Arc::new(FakeDriver::new());

    let alice = entity("alice", "Alice", "g1");
    let acme = entity("acme", "Acme", "g1");
    driver.save_entity_nodes(&[alice, acme]).await.unwrap();

    let mut edge = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme.".into(),
        "g1".into(),
    );
    edge.uuid = "edge-1".into();
    edge.episodes = vec!["ep1".into()];
    driver.save_entity_edges(&[edge]).await.unwrap();

    let mut ep1 = EpisodicNode::new(
        "ep1".into(),
        "g1".into(),
        EpisodeType::Message,
        "d".into(),
        "Alice works at Acme.".into(),
        t0(),
        t0(),
    );
    ep1.uuid = "ep1".into();
    ep1.entity_edges = vec!["edge-1".into()];
    driver.save_episode(&ep1).await.unwrap();

    driver
        .save_episodic_edges(&[
            EpisodicEdge::new("ep1".into(), "alice".into(), "g1".into(), t0()),
            EpisodicEdge::new("ep1".into(), "acme".into(), "g1".into(), t0()),
        ])
        .await
        .unwrap();

    let llm = Arc::new(MockLlm::new(vec![]));
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let cl = clients(Arc::clone(&driver), llm, emb);

    let results = get_nodes_and_edges_by_episode(&cl, "ep1").await.unwrap();

    assert_eq!(results.nodes.len(), 2, "both mentioned nodes returned");
    assert_eq!(results.edges.len(), 1, "the episode's entity edge returned");
    assert_eq!(results.edges[0].uuid, "edge-1");
    assert!(
        results.episodes.is_empty(),
        "episode scope is not populated"
    );
    assert!(results.communities.is_empty());
}

// ---------------------------------------------------------------------------
// summarize_saga watermarks
// ---------------------------------------------------------------------------

/// First summarize_saga run over two episodes advances both watermarks and stores
/// the LLM summary; the second run (no new episodes) is a no-op (no LLM call).
#[tokio::test]
async fn summarize_saga_advances_watermarks_and_skips_when_no_new_episodes() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    // One scripted SagaSummary for the single (first) run.
    let llm = Arc::new(MockLlm::new(vec![serde_json::json!({
        "summary": "Alice joined Acme then was promoted."
    })]));

    // Saga + two HAS_EPISODE-linked episodes (created_at t0/t1).
    let mut saga = SagaNode::new("career".into(), "g1".into(), t0());
    saga.uuid = "saga-1".into();
    driver.save_saga_node(&saga).await.unwrap();

    let ep1 = EpisodicNode::new(
        "ep1".into(),
        "g1".into(),
        EpisodeType::Message,
        "d".into(),
        "Alice joined Acme.".into(),
        t0(),
        t0(),
    );
    let ep2 = EpisodicNode::new(
        "ep2".into(),
        "g1".into(),
        EpisodeType::Message,
        "d".into(),
        "Alice was promoted.".into(),
        t1(),
        t1(),
    );
    let ep1_uuid = ep1.uuid.clone();
    let ep2_uuid = ep2.uuid.clone();
    driver.save_episode(&ep1).await.unwrap();
    driver.save_episode(&ep2).await.unwrap();
    driver
        .save_has_episode_edge(&chronicle_core::types::HasEpisodeEdge::new(
            "saga-1".into(),
            ep1_uuid.clone(),
            "g1".into(),
            t0(),
        ))
        .await
        .unwrap();
    driver
        .save_has_episode_edge(&chronicle_core::types::HasEpisodeEdge::new(
            "saga-1".into(),
            ep2_uuid.clone(),
            "g1".into(),
            t1(),
        ))
        .await
        .unwrap();

    let cl = clients(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let updated = summarize_saga(&cl, "saga-1").await.unwrap();
    assert_eq!(updated.summary, "Alice joined Acme then was promoted.");
    assert!(
        updated.last_summarized_at.is_some(),
        "wall-clock watermark set"
    );
    assert_eq!(
        updated.last_summarized_episode_valid_at,
        Some(t1()),
        "episode-time watermark advances to the latest valid_at"
    );
    assert_eq!(llm.call_count(), 1, "exactly one summarize_saga LLM call");

    // Second run: no episodes created after last_summarized_at → no-op, no LLM.
    let again = summarize_saga(&cl, "saga-1").await.unwrap();
    assert_eq!(again.summary, "Alice joined Acme then was promoted.");
    assert_eq!(
        llm.call_count(),
        1,
        "no new episodes ⇒ no additional LLM call"
    );
}

/// summarize_saga on a missing saga UUID surfaces NodeNotFound.
#[tokio::test]
async fn summarize_saga_missing_uuid_errors() {
    let driver = Arc::new(FakeDriver::new());
    let llm = Arc::new(MockLlm::new(vec![]));
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let cl = clients(Arc::clone(&driver), llm, emb);

    let err = summarize_saga(&cl, "nope").await.unwrap_err();
    assert!(
        format!("{err}").contains("nope"),
        "error should reference the missing saga uuid: {err}"
    );
}

// ---------------------------------------------------------------------------
// Saga threading + update_communities through add_episode (facade)
// ---------------------------------------------------------------------------

fn alice_only_keyed_llm() -> MockLlm {
    MockLlm::keyed(vec![
        (
            "extract_nodes.extract_message",
            serde_json::json!({
                "extracted_entities": [
                    {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]}
                ]
            }),
        ),
        ("extract_edges.edge", serde_json::json!({"edges": []})),
        (
            "summarize_nodes.summarize_context",
            serde_json::json!({"summary": "A person named Alice."}),
        ),
    ])
}

fn saga_req(
    name: &str,
    body: &str,
    reference_time: DateTime<Utc>,
    saga: &str,
) -> AddEpisodeRequest {
    AddEpisodeRequest {
        name: name.to_string(),
        episode_body: body.to_string(),
        source: EpisodeType::Message,
        source_description: "test".to_string(),
        reference_time,
        group_id: "g1".to_string(),
        saga: Some(saga.to_string()),
        ..Default::default()
    }
}

/// Two add_episode calls into the same saga chain NEXT_EPISODE (ep1 → ep2) and
/// attach HAS_EPISODE for both, advancing the saga's first/last pointers.
#[tokio::test]
async fn add_episode_saga_threading_chains_next_episode() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(alice_only_keyed_llm());

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&emb) as _,
        1,
    );

    let r1 = chronicle
        .add_episode(saga_req("ep1", "Alice joined.", t0(), "onboarding"))
        .await
        .unwrap();
    let r2 = chronicle
        .add_episode(saga_req("ep2", "Alice was promoted.", t1(), "onboarding"))
        .await
        .unwrap();

    let ep1_uuid = r1.episode.uuid.clone();
    let ep2_uuid = r2.episode.uuid.clone();

    // HAS_EPISODE points the saga at both episodes.
    let has_edges = driver.has_episode_edges();
    assert_eq!(has_edges.len(), 2, "one HAS_EPISODE per episode");

    // NEXT_EPISODE: single chain edge ep1 → ep2.
    let next_edges = driver.next_episode_edges();
    assert_eq!(next_edges.len(), 1, "single NEXT_EPISODE edge");
    assert_eq!(next_edges[0].source_node_uuid, ep1_uuid);
    assert_eq!(next_edges[0].target_node_uuid, ep2_uuid);

    // Saga node tracks first/last by ingest order.
    let saga = driver
        .get_saga_by_name("onboarding", "g1")
        .await
        .unwrap()
        .expect("saga created");
    assert_eq!(saga.first_episode_uuid.as_deref(), Some(ep1_uuid.as_str()));
    assert_eq!(saga.last_episode_uuid.as_deref(), Some(ep2_uuid.as_str()));
}

/// add_episode with update_communities=true attaches the resolved node to a
/// pre-existing community (via the neighbour-vote path) and surfaces the touched
/// community in the result.
#[tokio::test]
async fn add_episode_update_communities_attaches_node_to_community() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));

    // We exercise the update_communities WIRING through add_episode (the
    // community_ops internals are covered in community.rs). The episode's single
    // extracted "Alice" must resolve onto a PRE-EXISTING "alice" node that already
    // RELATES_TO a community member, so the post-persist update_community runs its
    // neighbour-vote path and attaches Alice to c1.
    //
    // For exact-name resolution to reuse uuid "alice", the existing node must
    // surface in the cosine candidate pool — so we seed its name_embedding.
    let mut c1 = CommunityNode::new("Team".into(), "g1".into(), Utc::now());
    c1.uuid = "c1".into();
    c1.summary = "Existing team.".into();
    driver.save_community_nodes(&[c1]).await.unwrap();

    let mut alice = entity("alice", "Alice", "g1");
    alice.name_embedding = Some(emb.create("Alice").await.unwrap());
    let member = entity("member", "Member", "g1");
    driver.save_entity_nodes(&[alice, member]).await.unwrap();
    driver
        .save_community_edges(&[CommunityEdge::new(
            "c1".into(),
            "member".into(),
            "g1".into(),
            Utc::now(),
        )])
        .await
        .unwrap();
    let mut rel = EntityEdge::new(
        "alice".into(),
        "member".into(),
        "RELATES_TO".into(),
        "fact".into(),
        "g1".into(),
    );
    rel.uuid = "e-alice-member".into();
    driver.save_entity_edges(&[rel]).await.unwrap();

    // LLM: extract Alice (resolves to the existing "alice" node by exact name),
    // no edges, summarize; then update_community fans out summarize_pair +
    // summary_description for the neighbour-vote-attached community.
    let llm = Arc::new(MockLlm::keyed(vec![
        (
            "extract_nodes.extract_message",
            serde_json::json!({
                "extracted_entities": [
                    {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]}
                ]
            }),
        ),
        ("extract_edges.edge", serde_json::json!({"edges": []})),
        (
            "summarize_nodes.summarize_context",
            serde_json::json!({"summary": "A person named Alice."}),
        ),
        (
            "summarize_nodes.summarize_pair",
            serde_json::json!({"summary": "Folded team summary."}),
        ),
        (
            "summarize_nodes.summary_description",
            serde_json::json!({"description": "The team."}),
        ),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&emb) as _,
        1,
    );

    let req = AddEpisodeRequest {
        name: "ep1".into(),
        episode_body: "Alice is around.".into(),
        source: EpisodeType::Message,
        source_description: "test".into(),
        reference_time: t0(),
        group_id: "g1".into(),
        update_communities: true,
        ..Default::default()
    };

    let result = chronicle.add_episode(req).await.unwrap();

    assert!(
        !result.communities.is_empty(),
        "update_communities must surface the touched community"
    );
    assert!(result.communities.iter().any(|c| c.uuid == "c1"));
    assert!(
        result
            .community_edges
            .iter()
            .any(|e| e.target_node_uuid == "alice"),
        "a HAS_MEMBER edge attaching Alice to the community must be returned"
    );

    // Alice is now a member of c1 in the driver.
    let member_of = driver.community_of_member("alice").await.unwrap();
    assert_eq!(member_of.unwrap().uuid, "c1");
}
