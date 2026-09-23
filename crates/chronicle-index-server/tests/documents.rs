mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

async fn setup() -> (axum::Router, Vec<(String, String)>) {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    create_index(&app, &h, "kb1").await;
    (app, h)
}

fn document(id: &str, hash: &str, chunks: &[(i64, &str, usize)]) -> Value {
    json!({"document_id": id, "content_hash": hash, "chunks": chunks.iter()
        .map(|(i, text, hot)| json!({"chunk_index": i, "text": text, "vector_b64": vec_b64(*hot)}))
        .collect::<Vec<_>>()})
}

async fn upsert(
    app: &axum::Router,
    h: &[(String, String)],
    docs: Vec<Value>,
) -> (StatusCode, Value) {
    call(
        app,
        "POST",
        "/v1/indexes/kb1/documents",
        h,
        Some(json!({"documents": docs})),
    )
    .await
}

async fn stats(app: &axum::Router, h: &[(String, String)]) -> Value {
    call(app, "GET", "/v1/indexes/kb1/stats", h, None).await.1
}

async fn search_texts(
    app: &axum::Router,
    h: &[(String, String)],
    query: &str,
    hot: usize,
) -> Vec<String> {
    let (_, body) = call(
        app,
        "POST",
        "/v1/indexes/kb1/search",
        h,
        Some(json!({"query": query, "vector_b64": vec_b64(hot), "limit": 20})),
    )
    .await;
    body["chunks"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| c["text"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn upsert_is_counted_in_stats() {
    let (app, h) = setup().await;
    let (status, body) = upsert(
        &app,
        &h,
        vec![document(
            "d1",
            "h1",
            &[(0, "alpha one", 0), (1, "alpha two", 0)],
        )],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"upserted": 1, "unchanged": 0}));
    let s = stats(&app, &h).await;
    assert_eq!(
        (s["document_count"].as_i64(), s["chunk_count"].as_i64()),
        (Some(1), Some(2))
    );
}

#[tokio::test]
async fn unchanged_hash_is_a_no_op() {
    let (app, h) = setup().await;
    let doc = document("d1", "h1", &[(0, "alpha", 0)]);
    upsert(&app, &h, vec![doc.clone()]).await;
    let (_, body) = upsert(&app, &h, vec![doc]).await;
    assert_eq!(body, json!({"upserted": 0, "unchanged": 1}));
    assert_eq!(stats(&app, &h).await["chunk_count"], 1);
}

#[tokio::test]
async fn a_shrunk_document_leaves_no_stale_chunks() {
    let (app, h) = setup().await;
    upsert(
        &app,
        &h,
        vec![document(
            "d1",
            "h1",
            &[(0, "keep this", 1), (1, "stale zebra", 1)],
        )],
    )
    .await;
    upsert(&app, &h, vec![document("d1", "h2", &[(0, "keep this", 1)])]).await;
    assert_eq!(stats(&app, &h).await["chunk_count"], 1);
    let texts = search_texts(&app, &h, "zebra", 1).await;
    assert!(
        !texts.iter().any(|t| t.contains("zebra")),
        "stale chunk still searchable: {texts:?}"
    );
}

#[tokio::test]
async fn a_wrong_dimension_vector_is_refused_before_anything_is_written() {
    let (app, h) = setup().await;
    let bad = json!({"document_id": "d2", "content_hash": "h", "chunks": [
        {"chunk_index": 0, "text": "short",
         "vector_b64": chronicle_index_server::wire::encode_vector(&[1.0, 0.0])}]});
    let (status, body) = upsert(&app, &h, vec![document("d1", "h1", &[(0, "fine", 0)]), bad]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "dim_mismatch");
    assert_eq!(
        stats(&app, &h).await["document_count"],
        0,
        "the valid document in the same batch must not land either"
    );
}

#[tokio::test]
async fn duplicate_or_negative_chunk_indexes_are_refused() {
    let (app, h) = setup().await;
    for chunks in [&[(0, "a", 0), (0, "b", 0)][..], &[(-1, "a", 0)][..]] {
        let (status, _) = upsert(&app, &h, vec![document("d1", "h1", chunks)]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn an_unsafe_document_id_is_refused_on_upsert_and_delete() {
    let (app, h) = setup().await;

    // A control character (here the same U+001F the meta store uses as its
    // own internal key separator) must never reach a document id.
    let (status, body) = upsert(&app, &h, vec![document("a\u{1f}b", "h1", &[(0, "x", 0)])]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // Empty / blank ids are refused too.
    let (status, body) = upsert(&app, &h, vec![document("   ", "h1", &[(0, "x", 0)])]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    assert_eq!(stats(&app, &h).await["document_count"], 0);

    // Same rule on the DELETE path, reached through a percent-encoded
    // control character in the URI.
    let (status, body) = call(&app, "DELETE", "/v1/indexes/kb1/documents/a%1Fb", &h, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn upsert_into_a_missing_index_is_404() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    let (status, body) = upsert(&app, &h, vec![document("d1", "h1", &[(0, "a", 0)])]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "index_not_found");
}

#[tokio::test]
async fn deleting_a_document_removes_its_chunks_and_404s_after() {
    let (app, h) = setup().await;
    upsert(
        &app,
        &h,
        vec![
            document("d1", "h1", &[(0, "gone soon", 2)]),
            document("d2", "h", &[(0, "stays", 3)]),
        ],
    )
    .await;
    let (status, _) = call(&app, "DELETE", "/v1/indexes/kb1/documents/d1", &h, None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = call(&app, "DELETE", "/v1/indexes/kb1/documents/d1", &h, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(stats(&app, &h).await["document_count"], 1);
    assert!(
        search_texts(&app, &h, "gone soon", 2)
            .await
            .iter()
            .all(|t| t != "gone soon")
    );
}

#[tokio::test]
async fn deleting_the_index_removes_its_chunks_from_search() {
    let (app, h) = setup().await;
    upsert(
        &app,
        &h,
        vec![document("d1", "h1", &[(0, "orphan check", 0)])],
    )
    .await;
    call(&app, "DELETE", "/v1/indexes/kb1", &h, None).await;
    create_index(&app, &h, "kb1").await;
    assert!(search_texts(&app, &h, "orphan check", 0).await.is_empty());
}

#[tokio::test]
async fn an_oversized_body_is_413() {
    let state = chronicle_index_server::state::AppState::in_memory(BOOTSTRAP)
        .await
        .expect("state");
    let small = chronicle_index_server::router(state, 1024);
    let key = mint(&small, "acme", &["*"]).await;
    let h = tenant_headers(&key, "acme", None);
    create_index(&small, &h, "kb1").await;
    let big = "x".repeat(4096);
    let (status, _) = upsert(&small, &h, vec![document("d1", "h1", &[(0, &big, 0)])]).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}
