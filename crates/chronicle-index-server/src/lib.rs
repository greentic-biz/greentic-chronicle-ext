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
}

pub fn router(state: AppState, max_body_bytes: usize) -> Router {
    // `route_layer` panics if applied to a router with no routes yet
    // registered ("Adding a route_layer before any routes is a no-op").
    // `tenant_routes()` is empty until Task 5 adds the first `/v1` route, so
    // the guard is skipped until there is something for it to guard — an
    // unguarded empty router still answers every `/v1` request with 404,
    // which is exactly what the auth_keys tests expect before Task 5 lands.
    let tenant = tenant_routes();
    let tenant = if tenant.has_routes() {
        tenant.route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_tenant_key,
        ))
    } else {
        tenant
    };
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
