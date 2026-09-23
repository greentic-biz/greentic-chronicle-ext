mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::json;

/// The designer syncs with: PUT (assert index) → POST documents → DELETE
/// removed documents → GET stats, comparing counts with its own records.
#[tokio::test]
async fn a_designer_sync_round_trip() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", Some("general"));

    let (status, _) = call(
        &app,
        "PUT",
        "/v1/indexes/kb_01J9",
        &h,
        Some(json!({
        "name": "Policies", "embedding_model": "text-embedding-3-small",
        "dims": DIMS, "chunk_size": 1000, "chunk_overlap": 100})),
    )
    .await;
    assert!(status.is_success());

    let (status, _) = call(
        &app,
        "POST",
        "/v1/indexes/kb_01J9/documents",
        &h,
        Some(json!({"documents": [
        {"document_id": "doc_a", "content_hash": "sha256:aa", "chunks": [
            {"chunk_index": 0, "text": "a0", "vector_b64": vec_b64(0)},
            {"chunk_index": 1, "text": "a1", "vector_b64": vec_b64(1)}]},
        {"document_id": "doc_b", "content_hash": "sha256:bb", "chunks": [
            {"chunk_index": 0, "text": "b0", "vector_b64": vec_b64(2)}]}]})),
    )
    .await;
    assert!(status.is_success());

    let (status, _) = call(
        &app,
        "DELETE",
        "/v1/indexes/kb_01J9/documents/doc_b",
        &h,
        None,
    )
    .await;
    assert!(status.is_success() || status == StatusCode::NOT_FOUND);

    let (status, stats) = call(&app, "GET", "/v1/indexes/kb_01J9/stats", &h, None).await;
    assert_eq!(status, StatusCode::OK);
    // Fields the designer's `IndexStats` decodes; extra fields are ignored there.
    assert_eq!(stats["document_count"], 1);
    assert_eq!(stats["chunk_count"], 2);
    assert_eq!(stats["embedding_model"], "text-embedding-3-small");
    assert_eq!(stats["dims"], DIMS);
}

#[tokio::test]
async fn deletes_of_absent_things_answer_404_which_the_designer_treats_as_success() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    let (status, _) = call(&app, "DELETE", "/v1/indexes/never", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    create_index(&app, &h, "kb1").await;
    let (status, _) = call(&app, "DELETE", "/v1/indexes/kb1/documents/never", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
