use std::collections::HashSet;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use chronicle_core::DocumentChunk;
use chronicle_core::driver::EntityNodeOps as _;
use chronicle_core::ingest_chunks_with_vectors;

use crate::auth::Scope;
use crate::error::{ApiError, ApiJson};
use crate::meta::{DocRecord, now_ms};
use crate::routes::indexes::{chunk_uuids, dims_of};
use crate::state::AppState;
use crate::wire::{DocumentUpsert, UpsertRequest, UpsertResponse, decode_vector};

/// A document id must not be empty/blank and must not carry a control
/// character — in particular U+001F, the separator the meta store uses
/// inside its own record keys (`doc_rid`). Refusing it here keeps a
/// crafted id from ever reaching a store key or a chunk UUID.
fn valid_document_id(id: &str) -> bool {
    !id.trim().is_empty() && !id.chars().any(char::is_control)
}

struct PreparedDocument {
    document_id: String,
    content_hash: String,
    chunk_indexes: Vec<i64>,
    chunks: Vec<DocumentChunk>,
}

/// Validates one document's chunks — indexes, every vector — so the handler
/// can refuse a batch before writing any of it. Document-id validity is
/// checked earlier, in [`upsert`] itself, before the index is even looked
/// up: an unsafe id must answer 400 whether or not the index exists.
fn prepare(doc: DocumentUpsert, dims: usize) -> Result<PreparedDocument, ApiError> {
    let mut seen = HashSet::new();
    let mut chunk_indexes = Vec::with_capacity(doc.chunks.len());
    let mut chunks = Vec::with_capacity(doc.chunks.len());
    for chunk in doc.chunks {
        let index = usize::try_from(chunk.chunk_index)
            .map_err(|_| ApiError::bad_request("chunk_index must be non-negative"))?;
        if !seen.insert(chunk.chunk_index) {
            return Err(ApiError::bad_request(
                "duplicate chunk_index within a document",
            ));
        }
        let vector = decode_vector(&chunk.vector_b64).map_err(ApiError::bad_request)?;
        if vector.len() != dims {
            return Err(ApiError::dim_mismatch(StatusCode::UNPROCESSABLE_ENTITY));
        }
        chunk_indexes.push(chunk.chunk_index);
        chunks.push(DocumentChunk {
            doc_id: doc.document_id.clone(),
            chunk_index: index,
            text: chunk.text,
            metadata: serde_json::Map::new(),
            embedding: Some(vector),
        });
    }
    Ok(PreparedDocument {
        document_id: doc.document_id,
        content_hash: doc.content_hash,
        chunk_indexes,
        chunks,
    })
}

pub async fn upsert(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path(index_id): Path<String>,
    ApiJson(body): ApiJson<UpsertRequest>,
) -> Result<Json<UpsertResponse>, ApiError> {
    // Checked first thing, before the index is even looked up: an unsafe id
    // must answer 400 regardless of whether the index exists, never a 404
    // that leaks index existence past a bad request.
    for doc in &body.documents {
        if !valid_document_id(&doc.document_id) {
            return Err(ApiError::bad_request("document_id is invalid"));
        }
    }
    let group_id = scope.group_id(&index_id)?;
    let _guard = state.locks.lock(&group_id).await;
    let mut index = state
        .meta
        .get_index(&group_id)
        .await?
        .ok_or_else(ApiError::index_not_found)?;
    let dims = dims_of(&index)?;
    let prepared = body
        .documents
        .into_iter()
        .map(|doc| prepare(doc, dims))
        .collect::<Result<Vec<_>, _>>()?;
    let driver = state.graphs.for_dims(dims).await?;

    let (mut upserted, mut unchanged) = (0, 0);
    for doc in prepared {
        let previous = state.meta.get_doc(&group_id, &doc.document_id).await?;
        if previous
            .as_ref()
            .is_some_and(|p| p.content_hash == doc.content_hash)
        {
            unchanged += 1;
            continue;
        }

        // Crash safety for a changed document: nothing here may ever leave a
        // chunk that no later step knows to clean up. So the FIRST write is
        // an "intent" record — chunk_indexes = the union of the previous
        // record's indexes and the incoming ones (sorted, deduped), under
        // the OLD hash (or "" when there was no previous record). It names
        // every chunk that might come to exist for this document before a
        // single new chunk is ingested or a single stale one is dropped. A
        // crash at any point between here and the FINAL put_doc below
        // leaves a record whose hash disagrees with the incoming hash (so
        // the next sync redoes this document instead of skipping it) and
        // whose chunk_indexes cover every chunk that might exist (so a
        // retry, delete_document, and delete_index all reach them). Only
        // once the new chunks are ingested and every stale one is gone do
        // we write the FINAL record — the new hash, and only the new
        // indexes.
        let previous_hash = previous
            .as_ref()
            .map(|p| p.content_hash.clone())
            .unwrap_or_default();
        let mut union = previous
            .as_ref()
            .map(|p| p.chunk_indexes.clone())
            .unwrap_or_default();
        union.extend(doc.chunk_indexes.iter().copied());
        union.sort_unstable();
        union.dedup();
        state
            .meta
            .put_doc(&DocRecord {
                group_id: group_id.clone(),
                document_id: doc.document_id.clone(),
                content_hash: previous_hash,
                chunk_indexes: union.clone(),
            })
            .await?;

        ingest_chunks_with_vectors(driver.as_ref(), &doc.chunks, &group_id, dims).await?;

        let stale: Vec<i64> = union
            .into_iter()
            .filter(|i| !doc.chunk_indexes.contains(i))
            .collect();
        let stale_uuids = chunk_uuids(&group_id, &doc.document_id, &stale);
        if !stale_uuids.is_empty() {
            driver
                .delete_entity_nodes_by_uuids(&stale_uuids)
                .await
                .map_err(ApiError::internal)?;
        }

        state
            .meta
            .put_doc(&DocRecord {
                group_id: group_id.clone(),
                document_id: doc.document_id,
                content_hash: doc.content_hash,
                chunk_indexes: doc.chunk_indexes,
            })
            .await?;
        upserted += 1;
    }
    if upserted > 0 {
        index.updated_at_ms = now_ms();
        state.meta.put_index(&index).await?;
    }
    Ok(Json(UpsertResponse {
        upserted,
        unchanged,
    }))
}

pub async fn delete_document(
    State(state): State<AppState>,
    Extension(scope): Extension<Scope>,
    Path((index_id, document_id)): Path<(String, String)>,
) -> Result<StatusCode, ApiError> {
    if !valid_document_id(&document_id) {
        return Err(ApiError::bad_request("document_id is invalid"));
    }
    let group_id = scope.group_id(&index_id)?;
    let _guard = state.locks.lock(&group_id).await;
    let mut index = state
        .meta
        .get_index(&group_id)
        .await?
        .ok_or_else(ApiError::index_not_found)?;
    let doc = state
        .meta
        .get_doc(&group_id, &document_id)
        .await?
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "document_not_found",
                "no such document",
            )
        })?;
    let uuids = chunk_uuids(&group_id, &doc.document_id, &doc.chunk_indexes);
    if !uuids.is_empty() {
        let driver = state.graphs.for_dims(dims_of(&index)?).await?;
        driver
            .delete_entity_nodes_by_uuids(&uuids)
            .await
            .map_err(ApiError::internal)?;
    }
    state.meta.delete_doc(&group_id, &document_id).await?;
    index.updated_at_ms = now_ms();
    state.meta.put_index(&index).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_control_character_document_ids_are_invalid() {
        assert!(!valid_document_id(""));
        assert!(!valid_document_id("   "));
        assert!(!valid_document_id("a\u{1f}b"));
        assert!(!valid_document_id("a\u{7f}b"));
        assert!(valid_document_id("d1"));
    }
}
