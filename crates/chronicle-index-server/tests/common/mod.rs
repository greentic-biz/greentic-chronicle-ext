#![allow(dead_code)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chronicle_index_server::state::AppState;
use chronicle_index_server::{router, wire};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use tower::ServiceExt as _;

pub const BOOTSTRAP: &str = "bootstrap-key-for-tests-0123456789abcdef";
pub const DIMS: usize = 4;

pub async fn app() -> Router {
    let state = AppState::in_memory(BOOTSTRAP).await.expect("state");
    router(state, 16 * 1024 * 1024)
}

pub async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    headers: &[(String, String)],
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (k, v) in headers {
        builder = builder.header(k.as_str(), v.as_str());
    }
    let req = match body {
        Some(v) => builder
            .header("content-type", "application/json")
            .body(Body::from(v.to_string())),
        None => builder.body(Body::empty()),
    }
    .expect("request");
    let resp = app.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, json)
}

pub fn bootstrap_headers() -> Vec<(String, String)> {
    vec![("authorization".into(), format!("Bearer {BOOTSTRAP}"))]
}

pub fn tenant_headers(key: &str, tenant: &str, team: Option<&str>) -> Vec<(String, String)> {
    let mut h = vec![
        ("authorization".into(), format!("Bearer {key}")),
        ("x-greentic-tenant".into(), tenant.into()),
    ];
    if let Some(team) = team {
        h.push(("x-greentic-team".into(), team.into()));
    }
    h
}

pub async fn mint(app: &Router, tenant: &str, teams: &[&str]) -> String {
    let (status, body) = call(
        app,
        "POST",
        "/admin/v1/keys",
        &bootstrap_headers(),
        Some(json!({"tenant_slug": tenant, "teams": teams, "label": "test"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    body["api_key"].as_str().expect("api_key").to_string()
}

pub fn vec_b64(hot: usize) -> String {
    let mut v = vec![0.0_f32; DIMS];
    v[hot] = 1.0;
    wire::encode_vector(&v)
}

pub async fn create_index(app: &Router, headers: &[(String, String)], index_id: &str) {
    let (status, body) = call(
        app,
        "PUT",
        &format!("/v1/indexes/{index_id}"),
        headers,
        Some(
            json!({"name": "Policies", "embedding_model": "text-embedding-3-small",
                    "dims": DIMS, "chunk_size": 1000, "chunk_overlap": 100}),
        ),
    )
    .await;
    assert!(status.is_success(), "{status} {body}");
}
