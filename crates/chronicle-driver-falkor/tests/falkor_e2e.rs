//! Phase-6 conformance gate: behavioural parity of the REAL FalkorDB driver with
//! the FakeDriver / Neo4j / SurrealDB reference, driven entirely through the
//! public `Chronicle` facade.
//!
//! The headline proof mirrors the Phase-1 add_episode e2e gate
//! (chronicle-testkit/tests/add_episode_e2e.rs) and the SurrealDB e2e
//! (chronicle-driver-surreal/tests/surreal_e2e.rs), but swaps the backend for a
//! REAL `FalkorDriver` connected to a live `falkordb/falkordb:latest` server. If
//! the FalkorDB driver is behaviourally interchangeable behind the GraphDriver
//! supertrait, the SAME scripted LLM sequence must produce the SAME bi-temporal
//! invalidation + search recall the other backends assert — through epoch-millis
//! datetimes, JSON-string attrs, the vector procedure and relationship-fulltext
//! DDL.
//!
//! These run ONLY when `FALKOR_TEST_URI` is set; otherwise each test prints a skip
//! notice and returns (mirrors the Task-1/2 live-test pattern). To run against a
//! disposable container:
//!
//! ```bash
//! docker run -d --rm --name chronicle-falkor-t3 -p 16379:6379 falkordb/falkordb:latest
//! FALKOR_TEST_URI=falkor://localhost:16379 \
//!     cargo test -p chronicle-driver-falkor --test falkor_e2e
//! docker stop chronicle-falkor-t3
//! ```
//!
//! Each test connects to a fresh, uniquely-named graph so state never leaks
//! across tests. A shared `tokio::sync::Mutex` serialises the setup+query phase
//! against the single shared server (the `falkordb` 0.2 crate runs its
//! `--compact` schema-refresh round-trips on a shared connection; concurrent
//! graph-build + vector-query cycles can otherwise race a just-built index — same
//! rationale as the search suite). The driver requires a multi-threaded runtime
//! (the crate's blocking schema refresh aborts on a single-threaded runtime), so
//! every test is `#[tokio::test(flavor = "multi_thread")]`.

use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::driver::{
    CommunityOps, EntityEdgeOps, EntityNodeOps, EpisodeOps, GraphDriver, SagaOps,
};
use chronicle_core::pipeline::bulk::RawEpisode;
use chronicle_core::search::{community_hybrid_search_rrf, edge_hybrid_search_rrf};
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType};

use chronicle_driver_falkor::FalkorDriver;
use chronicle_testkit::{MockEmbedder, MockLlm};

use chrono::{DateTime, TimeZone, Utc};

const EMB_DIM: usize = 8;

/// Serialise the setup+query phase against the shared FalkorDB server (see the
/// module doc). A `tokio::sync::Mutex` guard is `Send` and safe to hold across
/// `.await` points.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()
}

fn t1() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()
}

/// Connect to a fresh, uniquely-named graph (skips when `FALKOR_TEST_URI` unset).
async fn test_driver() -> Option<Arc<FalkorDriver>> {
    let uri = std::env::var("FALKOR_TEST_URI").ok()?;
    let graph = format!("chronicle_e2e_{}", uuid::Uuid::new_v4().simple());
    Some(Arc::new(
        FalkorDriver::connect(&uri, &graph, EMB_DIM)
            .await
            .expect("connect"),
    ))
}

/// Acquire the serial guard then connect; returns `(driver, guard)`. The guard
/// MUST be bound for the whole test body so serialisation holds across the
/// setup+query phase. Skips (returns early) when `FALKOR_TEST_URI` is unset.
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

fn req(name: &str, body: &str, reference_time: DateTime<Utc>) -> AddEpisodeRequest {
    AddEpisodeRequest {
        name: name.to_string(),
        episode_body: body.to_string(),
        source: EpisodeType::Message,
        source_description: "test".to_string(),
        reference_time,
        group_id: "g1".to_string(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Headline gate: add_episode bi-temporal invalidation + search through the
// real FalkorDriver (mirrors the FakeDriver / SurrealDB gates exactly).
// ---------------------------------------------------------------------------

/// Two episodes ingested through the `Chronicle` facade against the live FalkorDB
/// driver:
///   ep1 (t0): "Alice works at Acme."
///   ep2 (t1): "Alice now works at Globex."   (t1 > t0)
///
/// Gate: after ep2 the Acme fact is bi-temporally invalidated (invalid_at = t1,
/// expired_at set) but NOT deleted — it stays retrievable from FalkorDB storage
/// via `get_entity_edge` — and a search for where Alice works surfaces the Globex
/// fact. This is the strong proof that epoch-millis datetimes preserve the
/// bi-temporal comparison the invalidation logic depends on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_episode_two_episodes_invalidates_old_edge_and_keeps_it_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

    let alice_acme = serde_json::json!({
        "extracted_entities": [
            {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Acme", "entity_type_id": 0, "episode_indices": [0]}
        ]
    });
    let edge_acme = serde_json::json!({
        "edges": [{
            "source_entity_name": "Alice",
            "target_entity_name": "Acme",
            "relation_type": "WORKS_AT",
            "fact": "Alice works at Acme.",
            "valid_at": "2025-01-01T00:00:00Z",
            "invalid_at": null,
            "episode_indices": [0]
        }]
    });
    let summary = serde_json::json!({"summary": "Employment relationship."});

    let alice_globex = serde_json::json!({
        "extracted_entities": [
            {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Globex", "entity_type_id": 0, "episode_indices": [0]}
        ]
    });
    let edge_globex = serde_json::json!({
        "edges": [{
            "source_entity_name": "Alice",
            "target_entity_name": "Globex",
            "relation_type": "WORKS_AT",
            "fact": "Alice works at Globex.",
            "valid_at": "2025-06-01T00:00:00Z",
            "invalid_at": null,
            "episode_indices": [0]
        }]
    });
    let dedupe_globex = serde_json::json!({
        "entity_resolutions": [
            {"id": 0, "name": "Globex", "duplicate_candidate_id": -1}
        ]
    });
    let resolve_edge = serde_json::json!({
        "duplicate_facts": [],
        "contradicted_facts": [0]
    });

    let llm = Arc::new(MockLlm::new(vec![
        // ep1
        alice_acme,
        edge_acme,
        summary.clone(),
        summary.clone(),
        // ep2
        alice_globex,
        dedupe_globex,
        edge_globex,
        resolve_edge,
        summary.clone(),
        summary.clone(),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    // ----- Episode 1 ---------------------------------------------------------
    let r1 = chronicle
        .add_episode(req("ep1", "Alice works at Acme.", t0()))
        .await
        .expect("add_episode ep1");
    assert_eq!(r1.nodes.len(), 2, "ep1 should produce Alice + Acme nodes");
    assert_eq!(r1.edges.len(), 1, "ep1 should produce one WORKS_AT edge");
    assert_eq!(r1.edges[0].fact, "Alice works at Acme.");
    assert!(
        r1.edges[0].invalid_at.is_none() && r1.edges[0].expired_at.is_none(),
        "ep1 edge must be live (no invalidation)"
    );
    let acme_edge_uuid = r1.edges[0].uuid.clone();

    // ----- Episode 2 ---------------------------------------------------------
    let r2 = chronicle
        .add_episode(req("ep2", "Alice now works at Globex.", t1()))
        .await
        .expect("add_episode ep2");

    let invalidated: Vec<_> = r2
        .edges
        .iter()
        .filter(|e| e.invalid_at == Some(t1()) && e.expired_at.is_some())
        .collect();
    assert_eq!(
        invalidated.len(),
        1,
        "exactly one edge must be invalidated after ep2; got {:?}",
        r2.edges
            .iter()
            .map(|e| (&e.fact, e.invalid_at, e.expired_at))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        invalidated[0].fact, "Alice works at Acme.",
        "the invalidated edge must be the Acme fact"
    );
    assert_eq!(
        invalidated[0].uuid, acme_edge_uuid,
        "invalidation must mutate the SAME edge created in ep1, not a new one"
    );

    let globex = r2
        .edges
        .iter()
        .find(|e| e.fact == "Alice works at Globex.")
        .expect("ep2 should produce the Globex edge");
    assert!(
        globex.invalid_at.is_none() && globex.expired_at.is_none(),
        "the Globex edge must be live"
    );

    // ----- Storage: invalidated edge persists in FalkorDB, not deleted -------
    let stored_acme = driver
        .get_entity_edge(&acme_edge_uuid)
        .await
        .expect("driver get")
        .expect("invalidated Acme edge must remain in FalkorDB storage");
    assert_eq!(
        stored_acme.invalid_at,
        Some(t1()),
        "epoch-millis invalid_at must round-trip to t1"
    );
    assert!(stored_acme.expired_at.is_some());

    // ----- Search: Globex fact surfaces through the real driver --------------
    let results = chronicle
        .search(
            "where does Alice work",
            &["g1".to_string()],
            &edge_hybrid_search_rrf(),
        )
        .await
        .expect("search");
    assert!(
        results.iter().any(|e| e.fact == "Alice works at Globex."),
        "search must surface the Globex fact; got {:?}",
        results.iter().map(|e| &e.fact).collect::<Vec<_>>()
    );

    assert_eq!(
        llm.call_count(),
        10,
        "expected exactly 10 LLM calls across both episodes"
    );
}

// ---------------------------------------------------------------------------
// Phase-4 surface: communities (build + search) through the real driver.
// ---------------------------------------------------------------------------

/// Seed a small connected graph via add_episode, build communities over it (which
/// exercises get_community_clusters + save_community_*), then search the community
/// scope and assert the built community surfaces via the FalkorDB community FTS +
/// vector indices. Also asserts community_of_member sees the membership.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_communities_and_community_search_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

    let ingest_llm = Arc::new(MockLlm::keyed(vec![
        (
            "extract_nodes.extract_message",
            serde_json::json!({
                "extracted_entities": [
                    {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
                    {"name": "Acme", "entity_type_id": 0, "episode_indices": [0]}
                ]
            }),
        ),
        (
            "extract_edges.edge",
            serde_json::json!({
                "edges": [{
                    "source_entity_name": "Alice",
                    "target_entity_name": "Acme",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at Acme.",
                    "valid_at": "2025-01-01T00:00:00Z",
                    "invalid_at": null,
                    "episode_indices": [0]
                }]
            }),
        ),
        (
            "summarize_nodes.summarize_context",
            serde_json::json!({"summary": "Employment relationship."}),
        ),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&ingest_llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    chronicle
        .add_episode(req("ep1", "Alice works at Acme.", t0()))
        .await
        .expect("ingest ep1");

    let community_llm = Arc::new(MockLlm::keyed(vec![
        (
            "summarize_nodes.summarize_pair",
            serde_json::json!({"summary": "Employment cluster."}),
        ),
        (
            "summarize_nodes.summary_description",
            serde_json::json!({"description": "Acme employment community."}),
        ),
    ]));
    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&community_llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let (communities, edges) = chronicle
        .build_communities(&["g1".to_string()])
        .await
        .expect("build_communities");

    assert_eq!(
        communities.len(),
        1,
        "single connected group → one community"
    );
    assert_eq!(communities[0].name, "Acme employment community.");
    assert!(
        !edges.is_empty(),
        "HAS_MEMBER edges must connect the community to its members"
    );

    // Persisted in FalkorDB.
    let persisted = driver
        .get_community_nodes_by_group_ids(&["g1".to_string()])
        .await
        .expect("persisted communities");
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].uuid, communities[0].uuid);

    // Membership is queryable: at least one member resolves back to the community.
    let members = driver
        .get_entity_nodes_by_group_ids(&["g1".to_string()])
        .await
        .expect("members");
    let mut any_member_in_community = false;
    for m in &members {
        if let Some(c) = driver
            .community_of_member(&m.uuid)
            .await
            .expect("community_of_member")
        {
            assert_eq!(c.uuid, communities[0].uuid);
            any_member_in_community = true;
        }
    }
    assert!(
        any_member_in_community,
        "at least one entity must have a HAS_MEMBER edge to the built community"
    );

    // Community-scope search via the real community FTS + vector indices.
    let results = chronicle
        .search_(
            "Acme employment community",
            Some(&community_hybrid_search_rrf()),
            &["g1".to_string()],
            None,
            None,
            None,
        )
        .await
        .expect("community search");
    assert!(
        results
            .communities
            .iter()
            .any(|c| c.uuid == communities[0].uuid),
        "community search must surface the built community; got {:?}",
        results
            .communities
            .iter()
            .map(|c| &c.name)
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Phase-4 surface: saga ingest + summarize_saga through the real driver.
// ---------------------------------------------------------------------------

/// Two add_episode calls threaded into a named saga (NEXT_EPISODE chain +
/// HAS_EPISODE) then summarize_saga over the chain. Exercises
/// saga_previous_episode_uuid + saga_episode_contents (epoch-int ordering)
/// against the live FalkorDB saga / has_episode / next_episode edges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saga_ingest_and_summarize_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

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
            "summarize_sagas.summarize_saga",
            serde_json::json!({"summary": "Alice joined then was promoted."}),
        ),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let saga_req = |name: &str, body: &str, rt: DateTime<Utc>| AddEpisodeRequest {
        name: name.to_string(),
        episode_body: body.to_string(),
        source: EpisodeType::Message,
        source_description: "test".to_string(),
        reference_time: rt,
        group_id: "g1".to_string(),
        saga: Some("onboarding".to_string()),
        ..Default::default()
    };

    let r1 = chronicle
        .add_episode(saga_req("ep1", "Alice joined.", t0()))
        .await
        .expect("saga ep1");
    let r2 = chronicle
        .add_episode(saga_req("ep2", "Alice was promoted.", t1()))
        .await
        .expect("saga ep2");

    let ep1_uuid = r1.episode.uuid.clone();
    let ep2_uuid = r2.episode.uuid.clone();

    // Saga node tracks first/last episode by ingest order.
    let saga = driver
        .get_saga_by_name("onboarding", "g1")
        .await
        .expect("get saga")
        .expect("saga created");
    assert_eq!(saga.first_episode_uuid.as_deref(), Some(ep1_uuid.as_str()));
    assert_eq!(saga.last_episode_uuid.as_deref(), Some(ep2_uuid.as_str()));

    // NEXT_EPISODE chains ep1 -> ep2 (proved via saga_previous_episode_uuid;
    // epoch-int valid_at DESC ordering must put ep1 as the prior of ep2).
    let prev_of_ep2 = driver
        .saga_previous_episode_uuid(&saga.uuid, &ep2_uuid)
        .await
        .expect("previous episode");
    assert_eq!(
        prev_of_ep2.as_deref(),
        Some(ep1_uuid.as_str()),
        "ep1 must precede ep2 in the saga chain"
    );

    // summarize_saga advances watermarks and stores the LLM summary.
    let updated = chronicle
        .summarize_saga(&saga.uuid)
        .await
        .expect("summarize_saga");
    assert_eq!(updated.summary, "Alice joined then was promoted.");
    assert_eq!(
        updated.last_summarized_episode_valid_at,
        Some(t1()),
        "episode-time watermark advances to the latest valid_at"
    );

    // Second run over no new episodes is a no-op (saga_episode_contents since
    // watermark returns empty).
    let again = chronicle
        .summarize_saga(&saga.uuid)
        .await
        .expect("summarize_saga again");
    assert_eq!(again.summary, "Alice joined then was promoted.");
}

// ---------------------------------------------------------------------------
// Phase-4 surface: add_episode_bulk cross-episode dedup through the real driver.
// ---------------------------------------------------------------------------

/// Two episodes ingested in one bulk batch, both extracting Alice+Acme and the
/// same fact. Cross-episode dedup must collapse to two canonical nodes + a single
/// surviving WORKS_AT edge — mirrors the FakeDriver bulk gate against FalkorDB's
/// best-effort sequential save_all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_episode_bulk_cross_dedup_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

    let llm = Arc::new(MockLlm::keyed(vec![
        (
            "extract_nodes.extract_message",
            serde_json::json!({
                "extracted_entities": [
                    {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
                    {"name": "Acme", "entity_type_id": 0, "episode_indices": [0]}
                ]
            }),
        ),
        (
            "extract_edges.edge",
            serde_json::json!({
                "edges": [{
                    "source_entity_name": "Alice",
                    "target_entity_name": "Acme",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at Acme.",
                    "valid_at": "2025-01-01T00:00:00Z",
                    "invalid_at": null,
                    "episode_indices": [0]
                }]
            }),
        ),
        (
            "summarize_nodes.summarize_context",
            serde_json::json!({"summary": "Employment relationship."}),
        ),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let raw = |name: &str, content: &str, rt: DateTime<Utc>| RawEpisode {
        name: name.to_string(),
        uuid: None,
        content: content.to_string(),
        source_description: "test".to_string(),
        source: EpisodeType::Message,
        reference_time: rt,
    };

    let result = chronicle
        .add_episode_bulk(
            vec![
                raw("ep1", "Alice works at Acme.", t0()),
                raw("ep2", "Alice still works at Acme.", t1()),
            ],
            "g1",
            None,
        )
        .await
        .expect("add_episode_bulk");

    assert_eq!(result.episodes.len(), 2, "both episodes ingested");
    assert_eq!(
        result.nodes.len(),
        2,
        "Alice + Acme merged across both episodes; got {:?}",
        result.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    let names: std::collections::HashSet<&str> =
        result.nodes.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains("Alice") && names.contains("Acme"));
    assert!(
        result.edges.iter().any(|e| e.name == "WORKS_AT"),
        "a WORKS_AT edge must survive cross-episode dedup; got {:?}",
        result.edges.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // Communities are never populated in the bulk path (upstream parity).
    assert!(result.communities.is_empty());

    // Canonical nodes are persisted in FalkorDB.
    for n in &result.nodes {
        let stored = driver
            .get_entity_node(&n.uuid)
            .await
            .expect("driver get")
            .expect("canonical node persisted in FalkorDB");
        assert_eq!(stored.uuid, n.uuid);
    }
}

// ---------------------------------------------------------------------------
// Phase-4 surface: add_triplet through the real driver.
// ---------------------------------------------------------------------------

/// add_triplet over two pre-existing nodes resolves them via get_by_uuid (no
/// node-dedupe LLM) and persists the edge with no episode/episodic side effects.
/// Mirrors the FakeDriver add_triplet fast-path gate against FalkorDB.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_triplet_existing_nodes_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    // Empty queue: any LLM call would surface as EmptyResponse and fail.
    let llm = Arc::new(MockLlm::new(vec![]));

    let mut source = EntityNode::new("Alice".into(), "g1".into(), Utc::now());
    source.uuid = "alice".into();
    let mut target = EntityNode::new("Acme".into(), "g1".into(), Utc::now());
    target.uuid = "acme".into();
    driver
        .save_entity_nodes(&[source.clone(), target.clone()])
        .await
        .expect("seed nodes");

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let mut edge = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme.".into(),
        "g1".into(),
    );
    edge.valid_at = Some(t0());

    let result = chronicle
        .add_triplet(source, edge, target)
        .await
        .expect("add_triplet");

    assert_eq!(result.nodes.len(), 2, "source + target resolved");
    assert_eq!(result.edges.len(), 1, "single resolved edge");
    let saved_edge = &result.edges[0];
    assert_eq!(saved_edge.source_node_uuid, "alice");
    assert_eq!(saved_edge.target_node_uuid, "acme");

    // Persisted to FalkorDB.
    let stored = driver
        .get_entity_edge(&saved_edge.uuid)
        .await
        .expect("driver get")
        .expect("edge persisted in FalkorDB");
    assert_eq!(stored.name, "WORKS_AT");

    assert_eq!(
        llm.call_count(),
        0,
        "resolved-node fast path issues no LLM call"
    );
}

// ---------------------------------------------------------------------------
// Phase-4 surface: remove_episode cascade through the real driver.
// ---------------------------------------------------------------------------

/// Ingest two episodes (the second re-mentions Alice), then remove the first.
/// The single-mention node (Acme) + the ep1-originated edge must be deleted, the
/// episode removed (DETACH DELETE), while the multi-episode node (Alice)
/// survives. Mirrors the FakeDriver remove_episode cascade against FalkorDB's
/// detach-delete + get_mentioned_nodes path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_episode_cascade_falkor() {
    let (driver, _guard) = skip_if_no_env!();
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

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
            "dedupe_nodes.nodes",
            serde_json::json!({
                "entity_resolutions": [
                    {"id": 0, "name": "Alice", "duplicate_candidate_id": 0}
                ]
            }),
        ),
    ]));

    let chronicle = Chronicle::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let r1 = chronicle
        .add_episode(req("ep1", "Alice exists.", t0()))
        .await
        .expect("ep1");
    let ep1_uuid = r1.episode.uuid.clone();
    let alice_uuid = r1
        .nodes
        .iter()
        .find(|n| n.name == "Alice")
        .expect("Alice node")
        .uuid
        .clone();

    // ep2 re-mentions Alice (resolves onto the existing node via dedupe).
    let r2 = chronicle
        .add_episode(req("ep2", "Alice again.", t1()))
        .await
        .expect("ep2");
    assert!(
        r2.nodes.iter().any(|n| n.uuid == alice_uuid),
        "ep2 Alice must resolve onto the existing Alice node"
    );

    // Remove ep1.
    chronicle
        .remove_episode(&ep1_uuid)
        .await
        .expect("remove ep1");

    // Episode itself deleted from FalkorDB.
    assert!(
        driver
            .get_episode(&ep1_uuid)
            .await
            .expect("get episode")
            .is_none(),
        "removed episode must be gone from FalkorDB"
    );

    // Multi-episode Alice survives (mentioned by ep2 too).
    assert!(
        driver
            .get_entity_node(&alice_uuid)
            .await
            .expect("get node")
            .is_some(),
        "multi-episode node (Alice) must survive remove_episode"
    );

    // The witness episode survives.
    assert!(
        driver
            .get_episode(&r2.episode.uuid)
            .await
            .expect("get ep2")
            .is_some(),
        "witness episode ep2 must survive"
    );
}
