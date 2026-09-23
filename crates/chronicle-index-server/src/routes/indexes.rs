use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use chronicle_core::chunk_uuid;
use chronicle_core::driver::EntityNodeOps as _;

use crate::auth::Scope;
use crate::error::{ApiError, ApiJson};
use crate::meta::{IndexRecord, now_ms, rfc3339};
use crate::state::AppState;
use crate::wire::{IndexView, MAX_DIMS, PutIndexRequest, StatsResponse};

pub(crate) fn chunk_uuids(group_id: &str, document_id: &str, chunk_indexes: &[i64]) -> Vec<String> {
    chunk_indexes
        .iter()
        .filter_map(|i| usize::try_from(*i).ok())
        .map(|i| chunk_uuid(group_id, document_id, i))
        .collect()
}

pub(crate) fn dims_of(index: &IndexRecord) -> Result<usize, ApiError> {
    usize::try_from(index.dims).map_err(|_| ApiError::internal("stored dims out of range"))
}

fn view(index: &IndexRecord) -> IndexView {
    IndexView {
        index_id: index.index_id.clone(),
        name: index.name.clone(),
        embedding_model: index.embedding_model.clone(),
        dims: index.dims,
        chunk_size: index.chunk_size,
        chunk_overlap: index.chunk_overlap,
        updated_at: rfc3339(index.updated_at_ms),
    }
}

fn validate(body: &PutIndexRequest, state: &AppState) -> Result<(), ApiError> {
    if body.embedding_model.trim().is_empty() {
        return Err(ApiError::bad_request("embedding_model is required"));
    }
    if !(1..=MAX_DIMS).contains(&body.dims) {
        return Err(ApiError::bad_request(format!(
            "dims must be between 1 and {MAX_DIMS}"
        )));
    }
    if !state.allows_dims(body.dims) {
        return Err(ApiError::bad_request(format!(
            "dims {} is not allowed on this server",
            body.dims
        )));
    }
    if body.chunk_size < 1 || body.chunk_overlap < 0 {
        return Err(ApiError::bad_request(
            "chunk_size must be positive and chunk_overlap non-negative",
        ));
    }
    Ok(())
}

pub async fn put_index(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path(index_id): Path<String>,
    ApiJson(body): ApiJson<PutIndexRequest>,
) -> Result<(StatusCode, Json<IndexView>), ApiError> {
    let group_id = scope.group_id(&index_id)?;
    validate(&body, &state)?;
    let _guard = state.locks.lock(&group_id).await;
    let now = now_ms();
    match state.meta.get_index(&group_id).await? {
        Some(mut existing) => {
            if existing.embedding_model != body.embedding_model {
                return Err(ApiError::model_mismatch());
            }
            if existing.dims != body.dims {
                return Err(ApiError::dim_mismatch(StatusCode::CONFLICT));
            }
            existing.name = body.name;
            existing.chunk_size = body.chunk_size;
            existing.chunk_overlap = body.chunk_overlap;
            existing.updated_at_ms = now;
            state.meta.put_index(&existing).await?;
            Ok((StatusCode::OK, Json(view(&existing))))
        }
        None => {
            let created = IndexRecord {
                group_id,
                index_id,
                name: body.name,
                embedding_model: body.embedding_model,
                dims: body.dims,
                chunk_size: body.chunk_size,
                chunk_overlap: body.chunk_overlap,
                created_at_ms: now,
                updated_at_ms: now,
            };
            state.meta.put_index(&created).await?;
            Ok((StatusCode::CREATED, Json(view(&created))))
        }
    }
}

pub async fn stats(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path(index_id): Path<String>,
) -> Result<Json<StatsResponse>, ApiError> {
    let group_id = scope.group_id(&index_id)?;
    let index = state
        .meta
        .get_index(&group_id)
        .await?
        .ok_or_else(ApiError::index_not_found)?;
    let docs = state.meta.list_docs(&group_id).await?;
    let chunk_count: usize = docs.iter().map(|d| d.chunk_indexes.len()).sum();
    Ok(Json(StatsResponse {
        document_count: i64::try_from(docs.len()).unwrap_or(i64::MAX),
        chunk_count: i64::try_from(chunk_count).unwrap_or(i64::MAX),
        embedding_model: index.embedding_model,
        dims: index.dims,
        updated_at: rfc3339(index.updated_at_ms),
    }))
}

pub async fn delete_index(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path(index_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let group_id = scope.group_id(&index_id)?;
    let _guard = state.locks.lock(&group_id).await;
    let index = state
        .meta
        .get_index(&group_id)
        .await?
        .ok_or_else(ApiError::index_not_found)?;
    let uuids: Vec<String> = state
        .meta
        .list_docs(&group_id)
        .await?
        .iter()
        .flat_map(|d| chunk_uuids(&group_id, &d.document_id, &d.chunk_indexes))
        .collect();
    if !uuids.is_empty() {
        let driver = state.graphs.for_dims(dims_of(&index)?).await?;
        driver
            .delete_entity_nodes_by_uuids(&uuids)
            .await
            .map_err(ApiError::internal)?;
    }
    // Graph first, then meta: a crash in between leaves meta pointing at
    // chunks already gone, and a repeated DELETE is idempotent.
    state.meta.delete_index(&group_id).await?;
    Ok(StatusCode::NO_CONTENT)
}
