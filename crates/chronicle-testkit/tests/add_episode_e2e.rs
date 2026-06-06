// Phase-1 acceptance gate for the add_episode core loop.
//
// Two episodes ingested through the public `Chronicle` facade against the
// in-memory FakeDriver + deterministic Mock LLM/embedder:
//
//   ep1 (t0): "Alice works at Acme."
//   ep2 (t1): "Alice now works at Globex."   (t1 > t0)
//
// The gate: after ep2 the Acme fact is bi-temporally invalidated (invalid_at = t1,
// expired_at set) but NOT deleted, and a search for where Alice works surfaces the
// Globex fact while the invalidated Acme edge remains retrievable from storage.
//
// MockLlm replays one queued JSON value per `generate` call, in order, so the test
// must enqueue responses in the EXACT order the pipeline issues LLM calls. The
// ordered call sequence (derived from the implementation, concurrency pinned to 1)
// is documented inline below.
//
// Why concurrency = 1: `hydrate_node_summaries` fans out one summary call per node
// under the shared semaphore. With a single permit the calls are serialized, but
// the per-node order is still an implementation detail, so both summary responses
// per episode are made identical to decouple the queue from node ordering.

use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::driver::{EntityEdgeOps, GraphDriver};
use chronicle_core::pipeline::clients::Clients;
use chronicle_core::pipeline::edge_ops::resolve_extracted_edge;
use chronicle_core::search::edge_hybrid_search_rrf;
use chronicle_core::types::{EntityEdge, EpisodeType, EpisodicNode};

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

use chrono::{DateTime, TimeZone, Utc};

const EMB_DIM: usize = 8;

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()
}

fn t1() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()
}

/// Build an AddEpisodeRequest with all fields explicit (no Default).
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

#[tokio::test]
async fn add_episode_two_episodes_invalidates_old_edge_and_keeps_it() {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));

    // ----- Ordered LLM call sequence -----------------------------------------
    //
    // EPISODE 1 ("Alice works at Acme.", t0), fresh store:
    //   1. extract_nodes      → extract_message  → ExtractedEntities[Alice, Acme]
    //   (resolve_extracted_nodes: fresh store, no candidates ⇒ NO LLM dedupe call)
    //   2. extract_edges      → edge             → ExtractedEdges[Alice WORKS_AT Acme, valid_at=t0]
    //   (resolve_extracted_edges: related & existing both empty ⇒ both-empty fast
    //    path; edge already carries valid_at ⇒ extract_timestamps SKIPPED, no LLM)
    //   3. hydrate_node_summaries → summarize_context × 2 (Alice, Acme) → Summary
    //
    // EPISODE 2 ("Alice now works at Globex.", t1), store has Alice/Acme/WORKS_AT:
    //   4. extract_nodes      → extract_message  → ExtractedEntities[Alice, Globex]
    //   5. resolve_extracted_nodes → dedupe_nodes.nodes → NodeResolutions
    //      Alice resolves deterministically to the existing node (exact name); only
    //      Globex stays unresolved (its cosine candidate pool = the existing
    //      Alice/Acme nodes, but no fuzzy/exact match), so the LLM dedupe runs for
    //      Globex alone. relative id 0 = Globex, duplicate_candidate_id -1 = keep new.
    //   6. extract_edges      → edge             → ExtractedEdges[Alice WORKS_AT Globex, valid_at=t1]
    //   7. resolve_extracted_edge → dedupe_edges.resolve_edge →
    //        EdgeDuplicate{ duplicate_facts:[], contradicted_facts:[0] }
    //      (related pool empty, invalidation candidate idx 0 = the Acme edge found
    //       via cosine; new edge carries valid_at ⇒ extract_timestamps SKIPPED)
    //   8. hydrate_node_summaries → summarize_context × 2 (Alice, Globex) → Summary
    //
    // Total: ep1 = 4 calls, ep2 = 6 calls.
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
    // Globex (relative id 0) is the only unresolved node; keep it as new (-1).
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

    // Concurrency = 1 to serialize the summary fan-out.
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

    // Exactly one edge in the ep2 result is bi-temporally invalidated: the Acme
    // fact, with invalid_at == t1 and expired_at set (NOT deleted).
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

    // The Globex fact is present and live.
    let globex = r2
        .edges
        .iter()
        .find(|e| e.fact == "Alice works at Globex.")
        .expect("ep2 should produce the Globex edge");
    assert!(
        globex.invalid_at.is_none() && globex.expired_at.is_none(),
        "the Globex edge must be live"
    );

    // ----- Storage: invalidated edge persists, not deleted -------------------
    let stored_acme = driver
        .get_entity_edge(&acme_edge_uuid)
        .await
        .expect("driver get")
        .expect("invalidated Acme edge must remain in storage");
    assert_eq!(stored_acme.invalid_at, Some(t1()));
    assert!(stored_acme.expired_at.is_some());

    // ----- Search: Globex fact surfaces --------------------------------------
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

    // Sanity: all queued LLM responses consumed (4 + 6 = 10).
    assert_eq!(
        llm.call_count(),
        10,
        "expected exactly 10 LLM calls across both episodes"
    );
}

/// Focused test: `resolve_extracted_edge` takes the exact-duplicate fast path
/// (normalized fact + same endpoints already present in `related_edges`) and
/// returns the existing edge WITHOUT issuing any LLM call.
#[tokio::test]
async fn resolve_extracted_edge_exact_duplicate_fast_path_makes_no_llm_call() {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    // Empty queue: any LLM call would surface as EmptyResponse and fail the test.
    let llm = Arc::new(MockLlm::new(vec![]));
    let clients = Clients::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        4,
    );

    // Existing edge in the related (duplicate-candidate) pool. The extracted edge
    // carries the same endpoints and a fact that normalizes equal (case +
    // whitespace differences only).
    let mut existing = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "Alice works at Acme".into(),
        "g1".into(),
    );
    existing.uuid = "existing-edge".into();
    existing.episodes = vec!["older-ep".into()];

    let mut extracted = EntityEdge::new(
        "alice".into(),
        "acme".into(),
        "WORKS_AT".into(),
        "  alice   WORKS at  acme ".into(), // normalizes to "alice works at acme"
        "g1".into(),
    );
    extracted.uuid = "extracted-edge".into();

    let episode = EpisodicNode::new(
        "ep".into(),
        "g1".into(),
        EpisodeType::Message,
        "desc".into(),
        "Alice works at Acme.".into(),
        Utc::now(),
        t0(),
    );

    let (resolved, invalidated) =
        resolve_extracted_edge(&clients, extracted, vec![existing], vec![], &episode)
            .await
            .expect("resolve_extracted_edge");

    assert_eq!(
        llm.call_count(),
        0,
        "exact-duplicate fast path must not call the LLM"
    );
    assert_eq!(
        resolved.uuid, "existing-edge",
        "must resolve to the existing edge, not the extracted one"
    );
    assert!(
        resolved.episodes.contains(&episode.uuid),
        "this episode's uuid must be appended to the resolved edge"
    );
    assert!(
        resolved.episodes.contains(&"older-ep".to_string()),
        "pre-existing episode attribution must be preserved"
    );
    assert!(
        invalidated.is_empty(),
        "no invalidation on a pure duplicate"
    );
}
