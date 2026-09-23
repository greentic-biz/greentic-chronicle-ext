mod common;

use axum::http::StatusCode;
use chronicle_index_server::config::Config;
use chronicle_index_server::router;
use chronicle_index_server::state::AppState;
use common::*;
use serde_json::json;

fn config(dir: &std::path::Path) -> Config {
    Config::from_lookup(|name| match name {
        "CHRONICLE_INDEX_BOOTSTRAP_KEY" => Some(BOOTSTRAP.to_string()),
        "CHRONICLE_INDEX_DATA_DIR" => Some(dir.display().to_string()),
        "CHRONICLE_INDEX_ALLOWED_DIMS" => Some(DIMS.to_string()),
        _ => None,
    })
    .expect("config")
}

#[tokio::test]
async fn state_survives_reopen_on_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key = {
        let app = router(
            AppState::open(&config(dir.path())).await.expect("open"),
            1 << 20,
        );
        let key = mint(&app, "acme", &["*"]).await;
        let h = tenant_headers(&key, "acme", None);
        create_index(&app, &h, "kb1").await;
        let (status, _) = call(
            &app,
            "POST",
            "/v1/indexes/kb1/documents",
            &h,
            Some(json!({"documents": [
            {"document_id": "d1", "content_hash": "h1", "chunks": [
                {"chunk_index": 0, "text": "persisted text", "vector_b64": vec_b64(0)}]}]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        key
    }; // app and its RocksDB handles are dropped here

    // RocksDB handles from the first `AppState` are released asynchronously
    // (dropped, not awaited), so the reopen can race the lock file going
    // away. Retry briefly instead of weakening the assertions below.
    let mut attempt = 0;
    let state = loop {
        match AppState::open(&config(dir.path())).await {
            Ok(state) => break state,
            Err(err) if attempt < 10 => {
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let _ = err;
            }
            Err(err) => panic!("reopen: {err}"),
        }
    };
    let app = router(state, 1 << 20);
    let h = tenant_headers(&key, "acme", None);
    let (status, stats) = call(&app, "GET", "/v1/indexes/kb1/stats", &h, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the key and the index survive a restart"
    );
    assert_eq!(stats["chunk_count"], 1);
    let (_, found) = call(
        &app,
        "POST",
        "/v1/indexes/kb1/search",
        &h,
        Some(json!({"query": "persisted", "vector_b64": vec_b64(0)})),
    )
    .await;
    assert_eq!(found["chunks"][0]["document_id"], "d1");
}
