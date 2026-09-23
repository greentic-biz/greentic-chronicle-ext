use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use chronicle_core::DOCUMENT_CHUNK_LABEL;
use chronicle_core::document_rag::node_to_chunk_hit;
use chronicle_core::driver::SearchOps as _;
use chronicle_core::search::{NodeSearchConfig, SearchFilters, rrf};
use chronicle_core::types::EntityNode;

use crate::auth::Scope;
use crate::error::{ApiError, ApiJson};
use crate::routes::indexes::dims_of;
use crate::state::AppState;
use crate::wire::{MAX_SEARCH_LIMIT, SearchHit, SearchRequest, SearchResponse, decode_vector};

/// RRF rank constant, as chronicle's own `node_search` uses it.
const RRF_RANK_CONST: usize = 1;

/// Fuse the two legs with reciprocal-rank fusion and keep the best `limit`
/// document chunks. `bm25` is ranked best first; `cosine` carries each
/// node's exact similarity, and a node below `min_similarity` does not vote
/// (chronicle's `node_search` drops it the same way).
fn fuse(
    bm25: Vec<EntityNode>,
    cosine: Vec<(EntityNode, f64)>,
    min_similarity: f64,
    limit: usize,
) -> Vec<SearchHit> {
    let cosine: Vec<EntityNode> = cosine
        .into_iter()
        .filter(|(_, similarity)| *similarity >= min_similarity)
        .map(|(node, _)| node)
        .collect();
    let ranked = [
        bm25.iter().map(|n| n.uuid.clone()).collect::<Vec<_>>(),
        cosine.iter().map(|n| n.uuid.clone()).collect::<Vec<_>>(),
    ];
    let (uuids, scores) = rrf(&ranked, RRF_RANK_CONST, 0.0);
    let mut nodes: std::collections::HashMap<String, EntityNode> = bm25
        .into_iter()
        .chain(cosine)
        .map(|n| (n.uuid.clone(), n))
        .collect();
    uuids
        .into_iter()
        .zip(scores)
        .filter_map(|(uuid, score)| {
            let node = nodes.remove(&uuid)?;
            if !node.labels.iter().any(|l| l == DOCUMENT_CHUNK_LABEL) {
                return None;
            }
            let hit = node_to_chunk_hit(node, score);
            Some(SearchHit {
                document_id: hit.doc_id?,
                chunk_index: i64::try_from(hit.chunk_index?).ok()?,
                text: hit.text,
                score: hit.score,
            })
        })
        .take(limit)
        .collect()
}

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
    // Each leg fetches twice the answer, as chronicle's `node_search` does,
    // so fusion has something to reorder.
    // An index nobody has synced into has nothing to rank, and opening its
    // dimension's store just to learn that would create one on disk.
    if state.meta.list_docs(&group_id).await?.is_empty() {
        return Ok(Json(SearchResponse { chunks: Vec::new() }));
    }
    let candidates = limit.saturating_mul(2);
    let groups = [group_id];
    let driver = state.graphs.for_dims(dims).await?;
    // BM25: the group filter is part of the SQL `WHERE`, not applied after.
    let bm25 = driver
        .node_fulltext_search(&body.query, &SearchFilters::default(), &groups, candidates)
        .await
        .map_err(ApiError::internal)?;
    // Cosine: an exact scan of this index's chunks only — never the shared
    // HNSW graph, whose answer depends on every other index in the store.
    let cosine = driver
        .document_chunks_by_cosine(&groups, &vector, candidates)
        .await
        .map_err(ApiError::internal)?;
    let min_similarity = f64::from(NodeSearchConfig::default().sim_min_score);
    let chunks = fuse(bm25, cosine, min_similarity, limit);
    Ok(Json(SearchResponse { chunks }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(doc: &str, index: usize) -> EntityNode {
        chronicle_core::document_rag::chunk_to_entity_node(
            &chronicle_core::DocumentChunk {
                doc_id: doc.into(),
                chunk_index: index,
                text: format!("{doc} {index}"),
                metadata: serde_json::Map::new(),
                embedding: None,
            },
            "g",
            vec![1.0],
            chrono::Utc::now(),
        )
    }

    #[test]
    fn a_chunk_found_by_both_legs_outranks_one_found_by_either() {
        let (both, text_only, vector_only) = (chunk("both", 0), chunk("text", 0), chunk("vec", 0));
        let hits = fuse(
            vec![text_only.clone(), both.clone()],
            vec![(both.clone(), 0.9), (vector_only.clone(), 0.8)],
            0.6,
            10,
        );
        let docs: Vec<_> = hits.iter().map(|h| h.document_id.as_str()).collect();
        // both: 1/2 (BM25) + 1/1 (cosine); the others collect one vote each.
        assert_eq!(docs, ["both", "text", "vec"]);
    }

    #[test]
    fn a_weak_vector_match_does_not_vote_and_the_limit_is_kept() {
        let hits = fuse(vec![], vec![(chunk("weak", 0), 0.1)], 0.6, 10);
        assert!(hits.is_empty());
        let many: Vec<_> = (0..5).map(|i| (chunk("d", i), 0.9)).collect();
        assert_eq!(fuse(vec![], many, 0.6, 2).len(), 2);
    }

    #[test]
    fn a_node_that_is_not_a_document_chunk_is_dropped() {
        let mut entity = chunk("d", 0);
        entity.labels = vec!["Entity".into()];
        assert!(fuse(vec![entity], vec![], 0.6, 10).is_empty());
    }
}
