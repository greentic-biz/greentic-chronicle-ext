mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::json;

async fn setup() -> (axum::Router, Vec<(String, String)>) {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    (app, tenant_headers(&key, "acme", None))
}

#[tokio::test]
async fn put_creates_then_asserts_idempotently() {
    let (app, h) = setup().await;
    let body = json!({"name":"Policies","embedding_model":"m","dims":DIMS,"chunk_size":1000,"chunk_overlap":100});
    let (first, _) = call(&app, "PUT", "/v1/indexes/kb1", &h, Some(body.clone())).await;
    assert_eq!(first, StatusCode::CREATED);
    let (second, view) = call(&app, "PUT", "/v1/indexes/kb1", &h, Some(body)).await;
    assert_eq!(second, StatusCode::OK);
    assert_eq!(view["dims"], DIMS);
}

#[tokio::test]
async fn put_with_another_model_is_refused() {
    let (app, h) = setup().await;
    create_index(&app, &h, "kb1").await;
    let (status, body) = call(&app, "PUT", "/v1/indexes/kb1", &h,
        Some(json!({"name":"x","embedding_model":"another","dims":DIMS,"chunk_size":1000,"chunk_overlap":100}))).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "model_mismatch");
    let (_, stats) = call(&app, "GET", "/v1/indexes/kb1/stats", &h, None).await;
    assert_eq!(
        stats["embedding_model"], "text-embedding-3-small",
        "a refused PUT changes nothing"
    );
}

#[tokio::test]
async fn put_with_other_dims_is_refused() {
    let (app, h) = setup().await;
    create_index(&app, &h, "kb1").await;
    let (status, body) = call(&app, "PUT", "/v1/indexes/kb1", &h,
        Some(json!({"name":"x","embedding_model":"text-embedding-3-small","dims":8,"chunk_size":1000,"chunk_overlap":100}))).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "dim_mismatch");
}

#[tokio::test]
async fn put_refuses_out_of_range_values() {
    let (app, h) = setup().await;
    for body in [
        json!({"name":"x","embedding_model":"m","dims":0,"chunk_size":1000,"chunk_overlap":0}),
        json!({"name":"x","embedding_model":"m","dims":9000,"chunk_size":1000,"chunk_overlap":0}),
        json!({"name":"x","embedding_model":"","dims":4,"chunk_size":1000,"chunk_overlap":0}),
        json!({"name":"x","embedding_model":"m","dims":4,"chunk_size":0,"chunk_overlap":0}),
    ] {
        let (status, _) = call(&app, "PUT", "/v1/indexes/kb1", &h, Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    let (status, _) = call(
        &app,
        "PUT",
        "/v1/indexes/kb:1",
        &h,
        Some(json!({"name":"x","embedding_model":"m","dims":4,"chunk_size":1,"chunk_overlap":0})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn stats_of_an_empty_index_and_of_a_missing_one() {
    let (app, h) = setup().await;
    let (status, body) = call(&app, "GET", "/v1/indexes/kb1/stats", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "index_not_found");

    create_index(&app, &h, "kb1").await;
    let (status, stats) = call(&app, "GET", "/v1/indexes/kb1/stats", &h, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stats["document_count"], 0);
    assert_eq!(stats["chunk_count"], 0);
    assert!(stats["updated_at"].is_string());
}

#[tokio::test]
async fn delete_removes_the_index_and_404s_the_second_time() {
    let (app, h) = setup().await;
    create_index(&app, &h, "kb1").await;
    let (status, _) = call(&app, "DELETE", "/v1/indexes/kb1", &h, None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = call(&app, "DELETE", "/v1/indexes/kb1", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = call(&app, "GET", "/v1/indexes/kb1/stats", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_same_index_id_in_two_teams_is_two_indexes() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    create_index(&app, &tenant_headers(&key, "acme", Some("sales")), "kb1").await;
    let (status, _) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers(&key, "acme", Some("support")),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
