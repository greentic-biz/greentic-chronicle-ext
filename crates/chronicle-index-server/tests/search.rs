mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::json;

async fn seed(app: &axum::Router, h: &[(String, String)]) {
    create_index(app, h, "kb1").await;
    let docs = json!({"documents": [
        {"document_id": "refunds", "content_hash": "h1", "chunks": [
            {"chunk_index": 0, "text": "refunds are accepted within thirty days", "vector_b64": vec_b64(0)}]},
        {"document_id": "shipping", "content_hash": "h2", "chunks": [
            {"chunk_index": 0, "text": "shipping takes five working days", "vector_b64": vec_b64(1)}]}]});
    let (status, body) = call(app, "POST", "/v1/indexes/kb1/documents", h, Some(docs)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn search_returns_the_best_chunk_with_provenance() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    seed(&app, &h).await;
    let (status, body) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "refunds", "vector_b64": vec_b64(0), "limit": 1})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = body["chunks"].as_array().expect("chunks");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0]["document_id"], "refunds");
    assert_eq!(chunks[0]["chunk_index"], 0);
    assert!(chunks[0]["score"].is_number());
}

#[tokio::test]
async fn search_is_isolated_per_tenant() {
    let app = app().await;
    let acme = mint(&app, "acme", &["*"]).await;
    seed(&app, &tenant_headers(&acme, "acme", None)).await;

    let other = mint(&app, "other", &["*"]).await;
    let h_other = tenant_headers(&other, "other", None);
    create_index(&app, &h_other, "kb1").await;
    let (status, body) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h_other,
        Some(json!({"query": "refunds", "vector_b64": vec_b64(0), "limit": 10})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["chunks"],
        json!([]),
        "another tenant's same-named index is empty"
    );
}

#[tokio::test]
async fn search_refuses_a_wrong_dimension_vector_and_an_empty_query() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    seed(&app, &h).await;
    let short = chronicle_index_server::wire::encode_vector(&[1.0, 0.0]);
    let (status, body) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "refunds", "vector_b64": short})),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "dim_mismatch");

    let (status, _) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "  ", "vector_b64": vec_b64(0)})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn a_huge_limit_is_clamped_and_a_missing_index_is_404() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    let (status, _) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "x", "vector_b64": vec_b64(0)})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    seed(&app, &h).await;
    let (status, body) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "days", "vector_b64": vec_b64(0), "limit": 100000})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["chunks"].as_array().expect("chunks").len() <= 50);
}

fn unit_mix(primary: usize, secondary: usize, weight: f32) -> String {
    let mut v = [0.0_f32; DIMS];
    v[primary] = 1.0;
    v[secondary] += weight;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v: Vec<f32> = v.iter().map(|x| x / norm).collect();
    chronicle_index_server::wire::encode_vector(&v)
}

/// Every index with the same dimension shares one store. Another tenant's
/// index holding many chunks nearer to the query vector must not push this
/// index's chunk out of its own vector results. 400 crowding chunks is
/// more than both the driver's KNN over-fetch (30) and its HNSW EF (64);
/// SurrealDB 3 applies the `group_id` condition inside the KNN search, so
/// this passes today — it pins that behaviour against a regression.
#[tokio::test]
async fn the_vector_leg_is_not_starved_by_another_index_in_the_same_store() {
    let app = app().await;
    let a = tenant_headers(&mint(&app, "acme", &["*"]).await, "acme", None);
    let b = tenant_headers(&mint(&app, "other", &["*"]).await, "other", None);
    create_index(&app, &a, "kb1").await;
    create_index(&app, &b, "kb1").await;

    let docs = json!({"documents": [{"document_id": "only", "content_hash": "h", "chunks": [
        {"chunk_index": 0, "text": "alpha bravo", "vector_b64": unit_mix(0, 1, 0.3)}]}]});
    let (status, body) = call(&app, "POST", "/v1/indexes/kb1/documents", &a, Some(docs)).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Distinct vectors, every one much nearer to the query (axis 0) than
    // A's chunk.
    let chunks: Vec<_> = (0..400_u32)
        .map(|i| json!({"chunk_index": i, "text": format!("charlie delta {i}"),
            "vector_b64": unit_mix(0, 2, 0.0001 * f32::from(u16::try_from(i + 1).expect("small"))) }))
        .collect();
    let docs =
        json!({"documents": [{"document_id": "crowd", "content_hash": "h", "chunks": chunks}]});
    let (status, body) = call(&app, "POST", "/v1/indexes/kb1/documents", &b, Some(docs)).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &a,
        Some(json!({"query": "zulu", "vector_b64": vec_b64(0), "limit": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = body["chunks"].as_array().expect("chunks");
    assert_eq!(chunks.len(), 1, "{body}");
    assert_eq!(chunks[0]["document_id"], "only");
}
