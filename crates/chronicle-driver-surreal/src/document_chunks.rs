//! Exact, group-scoped cosine ranking of document chunks.
//!
//! The HNSW-backed `node_similarity_search` is approximate and shares one
//! graph across every group in the store, so how well it serves one small
//! group depends on everything else stored beside it. A knowledge index
//! wants the opposite guarantee: every chunk of *its* group is scored, and
//! nothing outside it is read. This scan costs time proportional to the
//! size of the groups asked for, never to the size of the store.

use chronicle_core::DOCUMENT_CHUNK_LABEL;
use chronicle_core::driver::DriverError;
use chronicle_core::types::EntityNode;
use surrealdb::types::SurrealValue as _;
use tracing::debug;

use crate::SurrealDriver;
use crate::convert::{ScoredEntityNodeRow, entity_node_from_row};

impl SurrealDriver {
    /// Every `DocumentChunk` node in `group_ids`, ranked by exact cosine
    /// similarity of its `name_embedding` to `vector`, best first, at most
    /// `limit`. An empty `group_ids` returns nothing — never the whole store.
    pub async fn document_chunks_by_cosine(
        &self,
        group_ids: &[String],
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(EntityNode, f64)>, DriverError> {
        debug!(
            limit,
            groups = group_ids.len(),
            "surreal document_chunks_by_cosine"
        );
        if group_ids.is_empty() || vector.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let sql = "SELECT *, vector::similarity::cosine(name_embedding, $vec) AS score \
                   FROM entity \
                   WHERE group_id IN $group_ids AND labels CONTAINS $label \
                   AND name_embedding != NONE AND name_embedding != NULL \
                   ORDER BY score DESC LIMIT $limit"
            .to_string();
        let params = vec![
            ("vec".to_string(), vector.to_vec().into_value()),
            ("group_ids".to_string(), group_ids.to_vec().into_value()),
            (
                "label".to_string(),
                DOCUMENT_CHUNK_LABEL.to_string().into_value(),
            ),
            (
                "limit".to_string(),
                i64::try_from(limit).unwrap_or(i64::MAX).into_value(),
            ),
        ];
        let rows: Vec<ScoredEntityNodeRow> = self
            .fetch_dyn(sql, params, "document_chunks_by_cosine")
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let (row, score) = row.into_parts();
                (entity_node_from_row(row), score)
            })
            .collect())
    }
}
