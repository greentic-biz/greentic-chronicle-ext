// Phase-4 Task 5 acceptance gate for the bulk-ingestion pipeline
// (chronicle_core::pipeline::bulk::add_episode_bulk).
//
// Driver-backed end-to-end tests over the in-memory FakeDriver + deterministic
// Mock embedder + a PROMPT-KEYED MockLlm. Bulk fans out parallel LLM calls per
// episode (extract, resolve, summarize), so call ORDER is not deterministic;
// `MockLlm::keyed` routes responses by `prompt_name` instead, which is
// order-independent. Every episode in a given test therefore shares the same
// extraction shape so a single keyed response per prompt suffices.

use std::sync::Arc;

use chronicle_core::driver::{EntityNodeOps, GraphDriver, SagaOps};
use chronicle_core::pipeline::bulk::{RawEpisode, add_episode_bulk};
use chronicle_core::pipeline::clients::Clients;
use chronicle_core::types::EpisodeType;

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

use chrono::{DateTime, TimeZone, Utc};

const EMB_DIM: usize = 8;

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()
}

fn t1() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap()
}

fn raw(name: &str, content: &str, reference_time: DateTime<Utc>) -> RawEpisode {
    RawEpisode {
        name: name.to_string(),
        uuid: None,
        content: content.to_string(),
        source_description: "test".to_string(),
        source: EpisodeType::Message,
        reference_time,
    }
}

/// Keyed responses where both episodes extract a single shared entity "Alice"
/// and no edges. Pass-1 resolve (vs the empty graph) keeps each Alice as new;
/// the bulk intra-batch pass-2 then merges the two Alices by exact normalized
/// name. The result: exactly one canonical Alice node.
fn alice_only_llm() -> MockLlm {
    MockLlm::keyed(vec![
        (
            "extract_nodes.extract_message",
            serde_json::json!({
                "extracted_entities": [
                    {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]}
                ]
            }),
        ),
        // Single node → no edges to extract.
        ("extract_edges.edge", serde_json::json!({"edges": []})),
        (
            "summarize_nodes.summarize_context",
            serde_json::json!({"summary": "A person named Alice."}),
        ),
    ])
}

#[tokio::test]
async fn bulk_merges_same_entity_across_two_episodes() {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(alice_only_llm());

    let clients = Clients::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let episodes = vec![
        raw("ep1", "Alice joined the team.", t0()),
        raw("ep2", "Alice was promoted.", t1()),
    ];

    let result = add_episode_bulk(&clients, episodes, "g1", None, None, None)
        .await
        .expect("add_episode_bulk");

    // Two episodes ingested.
    assert_eq!(result.episodes.len(), 2, "both episodes returned");

    // Cross-episode dedup: the two "Alice" mentions collapse to a single node.
    assert_eq!(
        result.nodes.len(),
        1,
        "Alice must be merged to one canonical node across both episodes; got {:?}",
        result.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    assert_eq!(result.nodes[0].name, "Alice");

    // The single canonical Alice is persisted exactly once.
    let stored = driver
        .get_entity_node(&result.nodes[0].uuid)
        .await
        .expect("driver get")
        .expect("canonical Alice persisted");
    assert_eq!(stored.name, "Alice");

    // Communities are never populated in the bulk path (upstream parity).
    assert!(result.communities.is_empty());
    assert!(result.community_edges.is_empty());
}

/// Keyed responses where both episodes extract "Alice" + "Acme" and the fact
/// "Alice works at Acme.". Cross-episode dedup merges the two Alices, the two
/// Acmes, and (same endpoints + identical fact) the two edges.
fn alice_acme_llm() -> MockLlm {
    MockLlm::keyed(vec![
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
    ])
}

#[tokio::test]
async fn bulk_e2e_two_episodes_produce_merged_nodes_and_edge() {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(alice_acme_llm());

    let clients = Clients::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    let episodes = vec![
        raw("ep1", "Alice works at Acme.", t0()),
        raw("ep2", "Alice still works at Acme.", t1()),
    ];

    let result = add_episode_bulk(&clients, episodes, "g1", None, None, None)
        .await
        .expect("add_episode_bulk");

    assert_eq!(result.episodes.len(), 2);

    // Alice + Acme, merged across episodes → exactly two canonical nodes.
    assert_eq!(
        result.nodes.len(),
        2,
        "Alice + Acme merged across both episodes; got {:?}",
        result.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    let names: std::collections::HashSet<&str> =
        result.nodes.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains("Alice") && names.contains("Acme"));

    // At least one WORKS_AT edge connecting the two canonical nodes survives.
    assert!(
        result.edges.iter().any(|e| e.name == "WORKS_AT"),
        "a WORKS_AT edge must be present; got {:?}",
        result.edges.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // Episodic (MENTIONS) edges point at canonical node uuids only.
    let node_uuids: std::collections::HashSet<String> =
        result.nodes.iter().map(|n| n.uuid.clone()).collect();
    for ee in &result.episodic_edges {
        assert!(
            node_uuids.contains(&ee.target_node_uuid),
            "episodic edge target {} must be a canonical node uuid",
            ee.target_node_uuid
        );
    }

    assert!(result.communities.is_empty());
    assert!(result.community_edges.is_empty());
}

#[tokio::test]
async fn bulk_saga_association_chains_next_episode_by_valid_at() {
    let driver = Arc::new(FakeDriver::new());
    let embedder = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(alice_only_llm());

    let clients = Clients::new(
        Arc::clone(&driver) as Arc<dyn GraphDriver>,
        Arc::clone(&llm) as _,
        Arc::clone(&embedder) as _,
        1,
    );

    // Provide episodes OUT of valid_at order to prove the chain sorts by valid_at,
    // not by input order: ep_late (t1) precedes ep_early (t0) in the input vec.
    let episodes = vec![
        raw("ep_late", "Alice was promoted.", t1()),
        raw("ep_early", "Alice joined the team.", t0()),
    ];

    let result = add_episode_bulk(&clients, episodes, "g1", Some("onboarding"), None, None)
        .await
        .expect("add_episode_bulk");

    assert_eq!(result.episodes.len(), 2);

    // Map name -> uuid for the two episodes.
    let uuid_of = |name: &str| -> String {
        result
            .episodes
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.uuid.clone())
            .unwrap_or_else(|| panic!("episode {name} missing"))
    };
    let early_uuid = uuid_of("ep_early");
    let late_uuid = uuid_of("ep_late");

    // HAS_EPISODE: saga points at BOTH episodes.
    let has_edges = driver.has_episode_edges();
    assert_eq!(has_edges.len(), 2, "one HAS_EPISODE edge per episode");
    let has_targets: std::collections::HashSet<String> = has_edges
        .iter()
        .map(|e| e.target_node_uuid.clone())
        .collect();
    assert!(has_targets.contains(&early_uuid) && has_targets.contains(&late_uuid));

    // NEXT_EPISODE: exactly one chain edge, ep_early (t0) -> ep_late (t1),
    // proving valid_at ordering regardless of input order.
    let next_edges = driver.next_episode_edges();
    assert_eq!(
        next_edges.len(),
        1,
        "single NEXT_EPISODE edge for two episodes"
    );
    assert_eq!(
        next_edges[0].source_node_uuid, early_uuid,
        "chain must start at the earlier (t0) episode"
    );
    assert_eq!(
        next_edges[0].target_node_uuid, late_uuid,
        "chain must point at the later (t1) episode"
    );

    // Saga node tracks first/last episode by valid_at order.
    let saga = driver
        .get_saga_by_name("onboarding", "g1")
        .await
        .expect("get saga")
        .expect("saga created");
    assert_eq!(
        saga.first_episode_uuid.as_deref(),
        Some(early_uuid.as_str())
    );
    assert_eq!(saga.last_episode_uuid.as_deref(), Some(late_uuid.as_str()));
}
