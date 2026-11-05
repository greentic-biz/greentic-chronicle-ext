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

async fn upsert(app: &axum::Router, h: &[(String, String)], doc: Value) -> (StatusCode, Value) {
    call(
        app,
        "POST",
        "/v1/indexes/kb1/documents",
        h,
        Some(json!({"documents": [doc]})),
    )
    .await
}

fn doc(extra: Value, section: Option<&str>) -> Value {
    let mut chunk =
        json!({"chunk_index": 0, "text": "refunds within thirty days", "vector_b64": vec_b64(0)});
    if let Some(s) = section {
        chunk["section"] = json!(s);
    }
    let mut d = json!({"document_id": "d1", "content_hash": "h1", "chunks": [chunk]});
    for (k, v) in extra.as_object().expect("object") {
        d[k] = v.clone();
    }
    d
}

async fn first_hit(app: &axum::Router, h: &[(String, String)]) -> Value {
    let (status, body) = call(
        app,
        "POST",
        "/v1/indexes/kb1/search",
        h,
        Some(json!({"query": "refunds", "vector_b64": vec_b64(0), "limit": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["chunks"][0].clone()
}

#[tokio::test]
async fn citation_fields_are_stored_and_echoed_on_the_hit() {
    let (app, h) = setup().await;
    let d = doc(
        json!({"title": "Refund policy.pdf", "source_url": "https://drive.google.com/f/1"}),
        Some("Policy > Refunds"),
    );
    assert_eq!(upsert(&app, &h, d).await.0, StatusCode::OK);
    let hit = first_hit(&app, &h).await;
    assert_eq!(hit["title"], "Refund policy.pdf");
    assert_eq!(hit["source_url"], "https://drive.google.com/f/1");
    assert_eq!(hit["section"], "Policy > Refunds");
    assert_eq!(hit["document_id"], "d1");
}

#[tokio::test]
async fn a_document_without_citation_fields_answers_a_hit_without_them() {
    let (app, h) = setup().await;
    assert_eq!(
        upsert(&app, &h, doc(json!({}), None)).await.0,
        StatusCode::OK
    );
    let hit = first_hit(&app, &h).await;
    let obj = hit.as_object().expect("hit");
    for key in ["title", "source_url", "section"] {
        assert!(!obj.contains_key(key), "{key} must be absent, not null");
    }
}

#[tokio::test]
async fn an_unsafe_link_is_dropped_and_the_batch_still_lands() {
    let (app, h) = setup().await;
    let d = doc(
        json!({"title": "T", "source_url": "javascript:alert(1)"}),
        None,
    );
    let (status, body) = upsert(&app, &h, d).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let hit = first_hit(&app, &h).await;
    assert_eq!(hit["title"], "T");
    assert!(hit.get("source_url").is_none());
}

#[tokio::test]
async fn a_metadata_only_change_is_written_and_an_identical_resend_is_not() {
    let (app, h) = setup().await;
    let with = |url: &str| doc(json!({"title": "T", "source_url": url}), None);
    let (_, first) = upsert(&app, &h, with("https://e.com/a")).await;
    assert_eq!(first, json!({"upserted": 1, "unchanged": 0}));
    let (_, same) = upsert(&app, &h, with("https://e.com/a")).await;
    assert_eq!(same, json!({"upserted": 0, "unchanged": 1}));
    // Same content_hash, new link: re-written.
    let (_, moved) = upsert(&app, &h, with("https://e.com/b")).await;
    assert_eq!(moved, json!({"upserted": 1, "unchanged": 0}));
    assert_eq!(first_hit(&app, &h).await["source_url"], "https://e.com/b");
}

#[tokio::test]
async fn an_older_client_resend_keeps_the_stored_metadata() {
    let (app, h) = setup().await;
    upsert(&app, &h, doc(json!({"title": "T"}), Some("S"))).await;
    // No citation field at all, same hash: unchanged, metadata untouched.
    let (_, body) = upsert(&app, &h, doc(json!({}), None)).await;
    assert_eq!(body, json!({"upserted": 0, "unchanged": 1}));
    let hit = first_hit(&app, &h).await;
    assert_eq!(hit["title"], "T");
    assert_eq!(hit["section"], "S");
}

#[tokio::test]
async fn replacing_a_document_replaces_its_metadata() {
    let (app, h) = setup().await;
    upsert(&app, &h, doc(json!({"title": "Old"}), Some("Old > S"))).await;
    let mut next = doc(json!({"title": "New"}), None);
    next["content_hash"] = json!("h2");
    upsert(&app, &h, next).await;
    let hit = first_hit(&app, &h).await;
    assert_eq!(hit["title"], "New");
    assert!(
        hit.get("section").is_none(),
        "the old section must not linger"
    );
}
