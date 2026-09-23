use chronicle_core::document_rag::{
    DocumentChunk, ingest_chunks_with_vectors, search_chunks_by_vector,
};
use chronicle_driver_surreal::SurrealDriver;

const DIMS: usize = 4;

fn unit(hot: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIMS];
    v[hot] = 1.0;
    v
}

fn chunk(doc: &str, index: usize, text: &str, embedding: Option<Vec<f32>>) -> DocumentChunk {
    DocumentChunk {
        doc_id: doc.to_string(),
        chunk_index: index,
        text: text.to_string(),
        metadata: serde_json::Map::new(),
        embedding,
    }
}

#[tokio::test]
async fn precomputed_ingest_then_vector_search_finds_the_matching_chunk() {
    let driver = SurrealDriver::connect_memory(DIMS).await.expect("driver");
    let group = "idx:acme:general:kb1";
    ingest_chunks_with_vectors(
        &driver,
        &[
            chunk(
                "refunds",
                0,
                "refunds are accepted within thirty days",
                Some(unit(0)),
            ),
            chunk(
                "shipping",
                0,
                "shipping takes five working days",
                Some(unit(1)),
            ),
        ],
        group,
        DIMS,
    )
    .await
    .expect("ingest");

    let hits = search_chunks_by_vector(&driver, "refunds", &unit(0), &[group.to_string()], 5)
        .await
        .expect("search");

    assert_eq!(
        hits.first().and_then(|h| h.doc_id.as_deref()),
        Some("refunds")
    );
}

#[tokio::test]
async fn a_chunk_without_a_vector_is_refused_and_nothing_is_written() {
    let driver = SurrealDriver::connect_memory(DIMS).await.expect("driver");
    let group = "idx:acme:general:kb1";
    let result = ingest_chunks_with_vectors(
        &driver,
        &[
            chunk("a", 0, "has a vector", Some(unit(0))),
            chunk("a", 1, "has none", None),
        ],
        group,
        DIMS,
    )
    .await;
    assert!(result.is_err());

    let hits = search_chunks_by_vector(&driver, "vector", &unit(0), &[group.to_string()], 5)
        .await
        .expect("search");
    assert!(hits.is_empty(), "validation must run before any write");
}

#[tokio::test]
async fn a_vector_of_the_wrong_dimension_is_refused() {
    let driver = SurrealDriver::connect_memory(DIMS).await.expect("driver");
    let result = ingest_chunks_with_vectors(
        &driver,
        &[chunk("a", 0, "short vector", Some(vec![1.0, 0.0]))],
        "idx:acme:general:kb1",
        DIMS,
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn search_never_crosses_group_ids() {
    let driver = SurrealDriver::connect_memory(DIMS).await.expect("driver");
    for group in ["idx:acme:general:kb1", "idx:other:general:kb1"] {
        ingest_chunks_with_vectors(
            &driver,
            &[chunk(
                "doc",
                0,
                "identical text in both tenants",
                Some(unit(2)),
            )],
            group,
            DIMS,
        )
        .await
        .expect("ingest");
    }

    let hits = search_chunks_by_vector(
        &driver,
        "identical text",
        &unit(2),
        &["idx:acme:general:kb1".to_string()],
        10,
    )
    .await
    .expect("search");

    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.group_id == "idx:acme:general:kb1"));
}

#[tokio::test]
async fn an_empty_query_or_zero_limit_returns_nothing() {
    let driver = SurrealDriver::connect_memory(DIMS).await.expect("driver");
    let groups = ["idx:acme:general:kb1".to_string()];
    assert!(
        search_chunks_by_vector(&driver, "  ", &unit(0), &groups, 5)
            .await
            .expect("search")
            .is_empty()
    );
    assert!(
        search_chunks_by_vector(&driver, "x", &unit(0), &groups, 0)
            .await
            .expect("search")
            .is_empty()
    );
}
