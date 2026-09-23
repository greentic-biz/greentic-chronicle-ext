#![forbid(unsafe_code)]
//! HTTP knowledge-index server: the `/v1/indexes` contract the Greentic
//! designer syncs into, a vector search route for runtimes, and a
//! tenant-bound API-key admin API. It never embeds: every document chunk and
//! every query arrives with its vector.

pub mod auth;
pub mod config;
pub mod error;
pub mod meta;
pub mod routes;
pub mod state;
pub mod wire;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{delete, get, post};

use crate::state::AppState;

fn tenant_routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/indexes/{index_id}",
            axum::routing::put(routes::indexes::put_index).delete(routes::indexes::delete_index),
        )
        .route("/v1/indexes/{index_id}/stats", get(routes::indexes::stats))
        .route(
            "/v1/indexes/{index_id}/documents",
            post(routes::documents::upsert),
        )
        .route(
            "/v1/indexes/{index_id}/documents/{document_id}",
            delete(routes::documents::delete_document),
        )
}

pub fn router(state: AppState, max_body_bytes: usize) -> Router {
    let tenant = tenant_routes().route_layer(middleware::from_fn_with_state(
        state.clone(),
        auth::require_tenant_key,
    ));
    let admin = Router::new()
        .route(
            "/admin/v1/keys",
            post(routes::keys::create).get(routes::keys::list),
        )
        .route("/admin/v1/keys/{key_id}", delete(routes::keys::revoke))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_bootstrap,
        ));
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .merge(tenant)
        .merge(admin)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}
