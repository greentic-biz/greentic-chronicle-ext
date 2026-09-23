use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use chronicle_core::search_chunks_by_vector;

use crate::auth::Scope;
use crate::error::{ApiError, ApiJson};
use crate::routes::indexes::dims_of;
use crate::state::AppState;
use crate::wire::{MAX_SEARCH_LIMIT, SearchHit, SearchRequest, SearchResponse, decode_vector};

pub async fn search(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path(index_id): Path<String>,
    ApiJson(body): ApiJson<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let group_id = scope.group_id(&index_id)?;
    let index = state
        .meta
        .get_index(&group_id)
        .await?
        .ok_or_else(ApiError::index_not_found)?;
    if body.query.trim().is_empty() {
        return Err(ApiError::bad_request("query is required"));
    }
    let dims = dims_of(&index)?;
    let vector = decode_vector(&body.vector_b64).map_err(ApiError::bad_request)?;
    if vector.len() != dims {
        return Err(ApiError::dim_mismatch(StatusCode::UNPROCESSABLE_ENTITY));
    }
    let limit = usize::try_from(body.limit.clamp(1, MAX_SEARCH_LIMIT)).unwrap_or(1);
    let driver = state.graphs.for_dims(dims).await?;
    let hits =
        search_chunks_by_vector(driver.as_ref(), &body.query, &vector, &[group_id], limit).await?;
    let chunks = hits
        .into_iter()
        .filter_map(|hit| {
            Some(SearchHit {
                document_id: hit.doc_id?,
                chunk_index: i64::try_from(hit.chunk_index?).ok()?,
                text: hit.text,
                score: hit.score,
            })
        })
        .collect();
    Ok(Json(SearchResponse { chunks }))
}
