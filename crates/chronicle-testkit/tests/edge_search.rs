// Integration tests for chronicle_core::search::edge_search.
//
// Placed here (in chronicle-testkit) rather than in chronicle-core to avoid
// a Cargo dev-dependency cycle: chronicle-core → chronicle-testkit → chronicle-core.
// chronicle-testkit already depends on chronicle-core, so tests here get both.

use chronicle_core::driver::EntityEdgeOps;
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::search::config::{
    EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, SearchConfig, edge_hybrid_search_rrf,
};
use chronicle_core::search::edge_search::edge_search;
use chronicle_core::types::EntityEdge;

use chronicle_testkit::{FakeDriver, MockEmbedder};

fn make_edge_with_embedding(
    uuid: &str,
    fact: &str,
    group_id: &str,
    embedding: Option<Vec<f32>>,
) -> EntityEdge {
    let mut e = EntityEdge::new(
        "src".into(),
        "tgt".into(),
        "REL".into(),
        fact.into(),
        group_id.into(),
    );
    e.uuid = uuid.into();
    e.fact_embedding = embedding;
    e
}

/// Seed three edges:
///   e1: fact matches query text AND has embedding of query string → appears in both lists
///   e2: fact matches query text only (no embedding) → fulltext only
///   e3: embedding-only match (fact is unrelated text, has query-matching embedding) → similarity only
///
/// Expected: e1 ranks first (in both lists → highest RRF score), all three returned.
#[tokio::test]
async fn hybrid_search_e1_ranks_first_all_three_returned() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);

    let query = "alpha";

    // e1: matches both fulltext ("alpha" in fact) and cosine (embedding of "alpha")
    let e1 = make_edge_with_embedding(
        "e1",
        "alpha is present in the fact",
        "g1",
        Some(emb.create(query).await.unwrap()),
    );

    // e2: fulltext match only — fact contains "alpha", no embedding
    let e2 = make_edge_with_embedding("e2", "alpha appears here too", "g1", None);

    // e3: cosine match only — fact text is unrelated, embedding matches query
    let e3 = make_edge_with_embedding(
        "e3",
        "completely unrelated fact text zxqwerty",
        "g1",
        Some(emb.create(query).await.unwrap()),
    );

    driver.save_entity_edges(&[e1, e2, e3]).await.unwrap();

    let config = edge_hybrid_search_rrf();
    let results = edge_search(&driver, &emb, query, &["g1".to_string()], &config)
        .await
        .unwrap();

    assert_eq!(results.len(), 3, "all three edges should be returned");
    assert_eq!(
        results[0].uuid, "e1",
        "e1 must rank first (in both lists → highest RRF)"
    );

    let uuids: Vec<&str> = results.iter().map(|e| e.uuid.as_str()).collect();
    assert!(uuids.contains(&"e2"), "e2 (fulltext-only) must be present");
    assert!(uuids.contains(&"e3"), "e3 (cosine-only) must be present");
}

/// With limit=1, only e1 should be returned.
#[tokio::test]
async fn hybrid_search_limit_1_returns_only_e1() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);

    let query = "alpha";

    let e1 = make_edge_with_embedding(
        "e1",
        "alpha is present in the fact",
        "g1",
        Some(emb.create(query).await.unwrap()),
    );
    let e2 = make_edge_with_embedding("e2", "alpha appears here too", "g1", None);
    let e3 = make_edge_with_embedding(
        "e3",
        "completely unrelated fact text zxqwerty",
        "g1",
        Some(emb.create(query).await.unwrap()),
    );

    driver.save_entity_edges(&[e1, e2, e3]).await.unwrap();

    let config = SearchConfig {
        limit: 1,
        ..edge_hybrid_search_rrf()
    };
    let results = edge_search(&driver, &emb, query, &["g1".to_string()], &config)
        .await
        .unwrap();

    assert_eq!(results.len(), 1, "limit=1 must return only one edge");
    assert_eq!(results[0].uuid, "e1", "e1 must be the sole result");
}

/// When edge_config is None, edge_search returns an empty result immediately.
#[tokio::test]
async fn edge_search_no_edge_config_returns_empty() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);

    let config = SearchConfig {
        edge_config: None,
        node_config: None,
        episode_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    };
    let results = edge_search(&driver, &emb, "anything", &[], &config)
        .await
        .unwrap();
    assert!(results.is_empty());
}

/// BM25-only config: only fulltext search runs (no embedding required).
#[tokio::test]
async fn edge_search_bm25_only() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);

    let e1 = make_edge_with_embedding("e1", "alpha beta gamma", "g1", None);
    let e2 = make_edge_with_embedding("e2", "delta epsilon zeta", "g1", None);
    driver.save_entity_edges(&[e1, e2]).await.unwrap();

    let config = SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::Bm25],
            reranker: EdgeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        node_config: None,
        episode_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    };

    let results = edge_search(&driver, &emb, "alpha", &["g1".to_string()], &config)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].uuid, "e1");
}

/// Cosine-only config: only similarity search runs.
#[tokio::test]
async fn edge_search_cosine_only() {
    let driver = FakeDriver::new();
    let emb = MockEmbedder::new(8);

    let query = "alpha";
    let mut e1 = make_edge_with_embedding("e1", "unrelated text", "g1", None);
    e1.fact_embedding = Some(emb.create(query).await.unwrap());

    let e2 = make_edge_with_embedding("e2", "also unrelated", "g1", None);

    driver.save_entity_edges(&[e1, e2]).await.unwrap();

    let config = SearchConfig {
        edge_config: Some(EdgeSearchConfig {
            search_methods: vec![EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Rrf,
            sim_min_score: 0.0,
            mmr_lambda: 0.5,
            bfs_max_depth: 3,
        }),
        node_config: None,
        episode_config: None,

        limit: 10,
        reranker_min_score: 0.0,
    };

    let results = edge_search(&driver, &emb, query, &["g1".to_string()], &config)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].uuid, "e1");
}
