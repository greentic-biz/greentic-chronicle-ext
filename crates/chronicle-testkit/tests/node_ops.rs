// Integration tests for chronicle_core::pipeline::node_ops.
//
// Placed in chronicle-testkit (not chronicle-core) to avoid the dev-dependency
// cycle chronicle-core → chronicle-testkit → chronicle-core. See edge_search.rs
// for the same pattern.

use std::sync::Arc;

use chronicle_core::driver::EntityNodeOps;
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::pipeline::clients::Clients;
use chronicle_core::pipeline::node_ops::{extract_nodes, resolve_extracted_nodes};
use chronicle_core::types::{EntityNode, EpisodeType, EpisodicNode};

use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

use chrono::Utc;

const EMB_DIM: usize = 8;

fn make_episode(content: &str, source: EpisodeType) -> EpisodicNode {
    EpisodicNode::new(
        "ep".into(),
        "g1".into(),
        source,
        "desc".into(),
        content.into(),
        Utc::now(),
        Utc::now(),
    )
}

fn clients_with(driver: Arc<FakeDriver>, llm: Arc<MockLlm>, emb: Arc<MockEmbedder>) -> Clients {
    Clients::new(driver, llm, emb, 4)
}

/// Test 1: extraction maps an ExtractedEntities LLM response into EntityNodes.
#[tokio::test]
async fn extract_nodes_builds_entities_from_llm_response() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![serde_json::json!({
        "extracted_entities": [
            {"name": "Alice Smith", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Acme Corp", "entity_type_id": 0, "episode_indices": [0]}
        ]
    })]));
    let clients = clients_with(driver, Arc::clone(&llm), emb);

    let episode = make_episode(
        "Mary: Alice Smith works at Acme Corp.",
        EpisodeType::Message,
    );

    let nodes = extract_nodes(&clients, &episode, &[], None, None)
        .await
        .expect("extract_nodes");

    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].name, "Alice Smith");
    assert_eq!(nodes[1].name, "Acme Corp");
    for n in &nodes {
        assert_eq!(n.group_id, "g1");
        assert!(n.labels.contains(&"Entity".to_string()));
    }

    // Exactly one LLM call, routed to the message-extraction prompt with the
    // ExtractedEntities schema attached.
    assert_eq!(llm.call_count(), 1);
    let requests = llm.requests.lock().unwrap();
    assert_eq!(
        requests[0].prompt_name.as_deref(),
        Some("extract_nodes.extract_message")
    );
    let schema = requests[0]
        .response_schema
        .as_ref()
        .expect("schema attached");
    assert_eq!(schema.name, "ExtractedEntities");
}

/// Test 2: an extracted node whose normalized name exactly matches a seeded
/// existing node resolves deterministically — no LLM call.
#[tokio::test]
async fn resolve_reuses_exact_name_match_without_llm() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    // No LLM responses queued — any LLM call would error / be observable.
    let llm = Arc::new(MockLlm::new(vec![]));

    // Seed an existing node "Alice Smith" with an embedding equal to the
    // extracted query ("alice  SMITH" with newlines→spaces is "alice  SMITH"),
    // so the cosine candidate search returns it.
    let query = "alice  SMITH".replace('\n', " ");
    let mut existing = EntityNode::new("Alice Smith".into(), "g1".into(), Utc::now());
    existing.uuid = "existing-alice".into();
    existing.name_embedding = Some(emb.create(&query).await.unwrap());
    driver.save_entity_nodes(&[existing.clone()]).await.unwrap();

    let clients = clients_with(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut extracted = EntityNode::new("alice  SMITH".into(), "g1".into(), Utc::now());
    extracted.uuid = "extracted-alice".into();
    let episode = make_episode("x", EpisodeType::Message);

    let outcome = resolve_extracted_nodes(&clients, vec![extracted.clone()], &episode, &[])
        .await
        .expect("resolve");

    assert_eq!(llm.call_count(), 0, "exact match must not call the LLM");
    assert_eq!(outcome.nodes.len(), 1);
    assert_eq!(outcome.nodes[0].uuid, "existing-alice");
    assert_eq!(outcome.duplicates.len(), 1);
    assert_eq!(
        outcome.uuid_map.get("extracted-alice").map(String::as_str),
        Some("existing-alice")
    );
}

/// Test 3: no deterministic match → escalate to LLM, which keeps the new node.
#[tokio::test]
async fn resolve_escalates_ambiguous_to_llm_keep_new() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![serde_json::json!({
        "entity_resolutions": [
            {"id": 0, "name": "Zephyr Quux", "duplicate_candidate_id": -1}
        ]
    })]));

    // Seed a candidate whose name does NOT exact-match and has low jaccard
    // overlap, but whose embedding matches the extracted query so it surfaces
    // as a semantic candidate (forcing the LLM pass to run).
    let query = "Zephyr Quux".replace('\n', " ");
    let mut candidate = EntityNode::new("Borodino Plateau".into(), "g1".into(), Utc::now());
    candidate.uuid = "cand-0".into();
    candidate.name_embedding = Some(emb.create(&query).await.unwrap());
    driver.save_entity_nodes(&[candidate]).await.unwrap();

    let clients = clients_with(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut extracted = EntityNode::new("Zephyr Quux".into(), "g1".into(), Utc::now());
    extracted.uuid = "extracted-0".into();
    let episode = make_episode("x", EpisodeType::Message);

    let outcome = resolve_extracted_nodes(&clients, vec![extracted.clone()], &episode, &[])
        .await
        .expect("resolve");

    assert_eq!(llm.call_count(), 1, "ambiguous node must escalate to LLM");
    assert_eq!(outcome.nodes.len(), 1);
    assert_eq!(outcome.nodes[0].uuid, "extracted-0", "kept as new");
    assert!(outcome.duplicates.is_empty());
    // self-mapping recorded for kept-as-new node
    assert_eq!(
        outcome.uuid_map.get("extracted-0").map(String::as_str),
        Some("extracted-0")
    );
}

/// Test 4: LLM resolves the extracted node to a seeded candidate (id 0); labels
/// are promoted from the extracted node onto the resolved candidate.
#[tokio::test]
async fn resolve_llm_resolves_to_candidate() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![serde_json::json!({
        "entity_resolutions": [
            {"id": 0, "name": "Zephyr Quux", "duplicate_candidate_id": 0}
        ]
    })]));

    let query = "Zephyr Quux".replace('\n', " ");
    let mut candidate = EntityNode::new("Borodino Plateau".into(), "g1".into(), Utc::now());
    candidate.uuid = "cand-0".into();
    // generic label so promotion from the extracted node's specific label fires
    candidate.labels = vec!["Entity".to_string()];
    candidate.name_embedding = Some(emb.create(&query).await.unwrap());
    driver.save_entity_nodes(&[candidate]).await.unwrap();

    let clients = clients_with(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut extracted = EntityNode::new("Zephyr Quux".into(), "g1".into(), Utc::now());
    extracted.uuid = "extracted-0".into();
    extracted.labels = vec!["Entity".to_string(), "Person".to_string()];
    let episode = make_episode("x", EpisodeType::Message);

    let outcome = resolve_extracted_nodes(&clients, vec![extracted.clone()], &episode, &[])
        .await
        .expect("resolve");

    assert_eq!(llm.call_count(), 1);
    assert_eq!(outcome.nodes.len(), 1);
    assert_eq!(outcome.nodes[0].uuid, "cand-0", "resolved to candidate");
    // label promoted from extracted ("Person") onto generic candidate
    assert!(outcome.nodes[0].labels.contains(&"Person".to_string()));
    assert_eq!(outcome.duplicates.len(), 1);
    assert_eq!(
        outcome.uuid_map.get("extracted-0").map(String::as_str),
        Some("cand-0")
    );
}

/// Test 5: an out-of-range duplicate_candidate_id is defensively ignored and
/// the extracted node is kept as new (mirrors the upstream guard).
#[tokio::test]
async fn resolve_out_of_range_candidate_id_keeps_extracted() {
    let driver = Arc::new(FakeDriver::new());
    let emb = Arc::new(MockEmbedder::new(EMB_DIM));
    let llm = Arc::new(MockLlm::new(vec![serde_json::json!({
        "entity_resolutions": [
            {"id": 0, "name": "Zephyr Quux", "duplicate_candidate_id": 99}
        ]
    })]));

    let query = "Zephyr Quux".replace('\n', " ");
    let mut candidate = EntityNode::new("Borodino Plateau".into(), "g1".into(), Utc::now());
    candidate.uuid = "cand-0".into();
    candidate.name_embedding = Some(emb.create(&query).await.unwrap());
    driver.save_entity_nodes(&[candidate]).await.unwrap();

    let clients = clients_with(Arc::clone(&driver), Arc::clone(&llm), Arc::clone(&emb));

    let mut extracted = EntityNode::new("Zephyr Quux".into(), "g1".into(), Utc::now());
    extracted.uuid = "extracted-0".into();
    let episode = make_episode("x", EpisodeType::Message);

    let outcome = resolve_extracted_nodes(&clients, vec![extracted.clone()], &episode, &[])
        .await
        .expect("resolve");

    assert_eq!(llm.call_count(), 1);
    assert_eq!(outcome.nodes.len(), 1);
    assert_eq!(
        outcome.nodes[0].uuid, "extracted-0",
        "out-of-range candidate id must keep the extracted node"
    );
    assert!(outcome.duplicates.is_empty());
}
