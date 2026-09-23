mod common;

use axum::http::StatusCode;
use common::*;

#[tokio::test]
async fn healthz_needs_no_key() {
    let app = app().await;
    let (status, _) = call(&app, "GET", "/healthz", &[], None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn the_admin_api_refuses_anything_but_the_bootstrap_key() {
    let app = app().await;
    let (status, body) = call(&app, "GET", "/admin/v1/keys", &[], None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");
    let wrong = vec![(
        "authorization".to_string(),
        "Bearer not-the-bootstrap".to_string(),
    )];
    let (status, _) = call(&app, "GET", "/admin/v1/keys", &wrong, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn a_minted_key_is_listed_without_its_secret_and_is_revocable() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    assert!(key.starts_with("cix_"));

    let (_, listed) = call(&app, "GET", "/admin/v1/keys", &bootstrap_headers(), None).await;
    let entries = listed["keys"].as_array().expect("keys");
    assert_eq!(entries.len(), 1);
    assert!(
        !listed.to_string().contains(&key),
        "a listing must never carry a key"
    );

    let key_id = entries[0]["key_id"].as_str().expect("key_id").to_string();
    let (status, _) = call(
        &app,
        "DELETE",
        &format!("/admin/v1/keys/{key_id}"),
        &bootstrap_headers(),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (status, _) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers(&key, "acme", None),
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a revoked key must stop working at once"
    );
}

#[tokio::test]
async fn minting_refuses_bad_slugs_and_empty_team_lists() {
    let app = app().await;
    for body in [
        serde_json::json!({"tenant_slug": "ac:me", "teams": ["*"]}),
        serde_json::json!({"tenant_slug": "acme", "teams": []}),
        serde_json::json!({"tenant_slug": "acme", "teams": ["bad team"]}),
    ] {
        let (status, _) = call(
            &app,
            "POST",
            "/admin/v1/keys",
            &bootstrap_headers(),
            Some(body),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn an_unknown_key_is_401_and_a_missing_tenant_is_400() {
    let app = app().await;
    let (status, body) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers("cix_nope", "acme", None),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");

    let key = mint(&app, "acme", &["*"]).await;
    let no_tenant = vec![("authorization".to_string(), format!("Bearer {key}"))];
    let (status, _) = call(&app, "GET", "/v1/indexes/kb1/stats", &no_tenant, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn a_key_is_refused_for_another_tenant() {
    let app = app().await;
    let key = mint(&app, "acme", &["*"]).await;
    let (status, body) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers(&key, "other", None),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["code"], "unauthorized");
}

#[tokio::test]
async fn missing_team_means_general() {
    let app = app().await;
    let sales_only = mint(&app, "acme", &["sales"]).await;
    let (status, _) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers(&sales_only, "acme", None),
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "no team header is team `general`, which this key does not cover"
    );

    let general = mint(&app, "acme", &["general"]).await;
    let (status, _) = call(
        &app,
        "GET",
        "/v1/indexes/kb1/stats",
        &tenant_headers(&general, "acme", None),
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "authorised; the index simply does not exist yet"
    );
}
