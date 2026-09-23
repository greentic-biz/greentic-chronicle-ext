use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::auth::{ALL_TEAMS, generate_api_key, hash_key};
use crate::error::ApiError;
use crate::meta::{KeyRecord, now_ms, rfc3339};
use crate::state::AppState;
use crate::wire::{CreateKeyRequest, CreateKeyResponse, KeyView, valid_slug};

pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<CreateKeyRequest>,
) -> Result<(StatusCode, Json<CreateKeyResponse>), ApiError> {
    if !valid_slug(&body.tenant_slug) {
        return Err(ApiError::bad_request("invalid tenant_slug"));
    }
    if body.teams.is_empty() || !body.teams.iter().all(|t| t == ALL_TEAMS || valid_slug(t)) {
        return Err(ApiError::bad_request(
            "teams must be [\"*\"] or a list of team slugs",
        ));
    }
    let api_key = generate_api_key();
    let record = KeyRecord {
        key_id: uuid::Uuid::new_v4().to_string(),
        hash: hash_key(&api_key),
        tenant_slug: body.tenant_slug,
        teams: body.teams,
        label: body.label,
        created_at_ms: now_ms(),
    };
    state.meta.insert_key(&record).await?;
    tracing::info!(key_id = %record.key_id, tenant = %record.tenant_slug, "api key minted");
    Ok((
        StatusCode::CREATED,
        Json(CreateKeyResponse {
            key_id: record.key_id,
            api_key,
        }),
    ))
}

pub async fn list(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let keys: Vec<KeyView> = state
        .meta
        .list_keys()
        .await?
        .into_iter()
        .map(|k| KeyView {
            key_id: k.key_id,
            tenant_slug: k.tenant_slug,
            teams: k.teams,
            label: k.label,
            created_at: rfc3339(k.created_at_ms),
        })
        .collect();
    Ok(Json(serde_json::json!({ "keys": keys })))
}

pub async fn revoke(
    State(state): State<AppState>,
    Path(key_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    if state.meta.delete_key(&key_id).await? {
        tracing::info!(%key_id, "api key revoked");
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "key_not_found",
            "no such key",
        ))
    }
}
