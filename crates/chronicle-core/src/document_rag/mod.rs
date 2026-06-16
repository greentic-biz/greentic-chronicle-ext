//! Lite document-RAG: chunk → embed → store as nodes (no LLM extraction) +
//! hybrid retrieval reusing existing driver BM25+HNSW indexes. Chunks are
//! stored as [`EntityNode`]s labelled `DocumentChunk` under a dedicated
//! knowledge `group_id`; retrieval is scoped to that group, cleanly isolated
//! from real graph entities. See docs/superpowers/plans/2026-06-16-chronicle-doc-rag-w2.md.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::ChronicleError;
use crate::pipeline::clients::Clients;
use crate::search::config::NodeSearchConfig;
use crate::search::filters::SearchFilters;
use crate::search::node_search::node_search;
use crate::types::node::EntityNode;

/// Label marking an EntityNode as a stored document chunk (in addition to "Entity").
pub const DOCUMENT_CHUNK_LABEL: &str = "DocumentChunk";

/// One pre-chunked unit of source text to ingest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub doc_id: String,
    pub chunk_index: usize,
    pub text: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// A retrieval hit: chunk text + relevance score + provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunkHit {
    pub text: String,
    pub score: f64,
    pub group_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Split text into overlapping windows, preferring whitespace boundaries.
/// Heuristic char-window (not a tokenizer). Empty text / `max_chars==0` → empty.
#[must_use]
pub fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() || max_chars == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return vec![text.trim().to_string()];
    }
    let overlap = overlap.min(max_chars - 1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let hard_end = (start + max_chars).min(chars.len());
        let mut end = hard_end;
        if end < chars.len()
            && let Some(ws) = (start + 1..end).rev().find(|&i| chars[i].is_whitespace())
        {
            end = ws;
        }
        let piece: String = chars[start..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if !piece.is_empty() {
            chunks.push(piece);
        }
        if end >= chars.len() {
            break;
        }
        // guarantee forward progress even with large overlap
        let next = end.saturating_sub(overlap);
        start = if next > start { next } else { end };
    }
    chunks
}

use blake2::{Blake2b512, Digest};

/// Deterministic chunk UUID (stable across re-ingest → idempotent UPSERT).
#[must_use]
pub fn chunk_uuid(group_id: &str, doc_id: &str, chunk_index: usize) -> String {
    let mut hasher = Blake2b512::new();
    hasher.update(group_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(doc_id.as_bytes());
    hasher.update([0x1f]);
    hasher.update(chunk_index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Uuid::from_bytes(bytes).to_string()
}

/// Build an EntityNode for a chunk: text in `name` (→ name_embedding), labelled
/// `["Entity","DocumentChunk"]`, provenance in `attributes`. Keeps the "Entity"
/// label so the existing entity-table BM25/HNSW search finds it.
#[must_use]
pub fn chunk_to_entity_node(
    chunk: &DocumentChunk,
    group_id: &str,
    embedding: Vec<f32>,
    created_at: DateTime<Utc>,
) -> EntityNode {
    let mut node = EntityNode::new(chunk.text.clone(), group_id.to_string(), created_at);
    node.uuid = chunk_uuid(group_id, &chunk.doc_id, chunk.chunk_index);
    node.labels = vec!["Entity".to_string(), DOCUMENT_CHUNK_LABEL.to_string()];
    node.name_embedding = Some(embedding);
    node.attributes
        .insert("doc_id".to_string(), Value::String(chunk.doc_id.clone()));
    node.attributes
        .insert("chunk_index".to_string(), Value::from(chunk.chunk_index));
    for (k, v) in &chunk.metadata {
        node.attributes
            .entry(k.clone())
            .or_insert_with(|| v.clone());
    }
    node
}

/// Map a retrieved node + score into a hit, pulling provenance from attributes.
#[must_use]
pub fn node_to_chunk_hit(node: EntityNode, score: f64) -> DocumentChunkHit {
    let doc_id = node
        .attributes
        .get("doc_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let chunk_index = node
        .attributes
        .get("chunk_index")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    DocumentChunkHit {
        text: node.name,
        score,
        group_id: node.group_id,
        doc_id,
        chunk_index,
        metadata: node.attributes,
    }
}

/// Ingest pre-chunked text: batch-embed, build nodes, persist. Returns chunk UUIDs.
/// Idempotent: deterministic UUIDs + driver UPSERT. No LLM extraction.
pub async fn ingest_chunks(
    clients: &Clients,
    chunks: Vec<DocumentChunk>,
    group_id: &str,
) -> Result<Vec<String>, ChronicleError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = clients.embedder.create_batch(&texts).await?;
    if embeddings.len() != chunks.len() {
        return Err(ChronicleError::InvalidInput(format!(
            "embedder returned {} vectors for {} chunks",
            embeddings.len(),
            chunks.len()
        )));
    }
    let created_at = Utc::now();
    let mut nodes = Vec::with_capacity(chunks.len());
    let mut uuids = Vec::with_capacity(chunks.len());
    for (chunk, embedding) in chunks.into_iter().zip(embeddings) {
        let node = chunk_to_entity_node(&chunk, group_id, embedding, created_at);
        uuids.push(node.uuid.clone());
        nodes.push(node);
    }
    clients.driver.save_entity_nodes(&nodes).await?;
    Ok(uuids)
}

/// Retrieve top-k document chunks for a query via hybrid BM25+cosine (RRF),
/// scoped to the given knowledge group_id(s). Returns mapped hits, newest-API
/// node ordering preserved.
pub async fn search_chunks(
    clients: &Clients,
    query: &str,
    group_ids: &[String],
    limit: usize,
) -> Result<Vec<DocumentChunkHit>, ChronicleError> {
    if query.trim().is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let query_vector = clients.embedder.create(&query.replace('\n', " ")).await?;
    let config = NodeSearchConfig::default(); // BM25 + Cosine, RRF
    let (nodes, scores) = node_search(
        clients.driver.as_ref(),
        None, // no cross-encoder in the lite path
        query,
        &query_vector,
        group_ids,
        Some(&config),
        &SearchFilters::default(),
        None,
        None,
        limit,
        0.0,
    )
    .await?;
    let hits = nodes
        .into_iter()
        .zip(scores)
        .filter(|(node, _)| node.labels.iter().any(|l| l == DOCUMENT_CHUNK_LABEL))
        .map(|(node, score)| node_to_chunk_hit(node, score))
        .collect();
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_empty_or_zero_returns_empty() {
        assert!(chunk_text("   ", 100, 10).is_empty());
        assert!(chunk_text("hello", 0, 0).is_empty());
    }

    #[test]
    fn chunk_text_short_returns_single_trimmed() {
        assert_eq!(
            chunk_text("  hello world  ", 100, 10),
            vec!["hello world".to_string()]
        );
    }

    #[test]
    fn chunk_text_splits_with_overlap_and_terminates() {
        let text = "aaaa bbbb cccc dddd eeee ffff"; // 29 chars
        let chunks = chunk_text(text, 10, 4);
        assert!(chunks.len() >= 3); // multiple windows
        assert!(chunks.iter().all(|c| c.chars().count() <= 10));
        assert!(chunks.iter().all(|c| !c.is_empty())); // no empties, no infinite loop
    }

    #[test]
    fn chunk_text_prefers_whitespace_boundary() {
        let chunks = chunk_text("alpha beta gamma", 8, 0);
        assert_eq!(chunks[0], "alpha"); // broke at space, not mid-word
    }

    use chrono::Utc;

    fn sample_chunk() -> DocumentChunk {
        let mut md = serde_json::Map::new();
        md.insert("source".into(), serde_json::json!("kb.pdf"));
        DocumentChunk {
            doc_id: "doc1".into(),
            chunk_index: 2,
            text: "hello world".into(),
            metadata: md,
        }
    }

    #[test]
    fn chunk_uuid_is_deterministic_and_stable() {
        let a = chunk_uuid("g1", "doc1", 2);
        let b = chunk_uuid("g1", "doc1", 2);
        let c = chunk_uuid("g1", "doc1", 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 36); // hyphenated uuid
    }

    #[test]
    fn build_node_sets_text_embedding_label_attrs() {
        let node = chunk_to_entity_node(&sample_chunk(), "g1", vec![0.1, 0.2, 0.3], Utc::now());
        assert_eq!(node.name, "hello world");
        assert_eq!(node.group_id, "g1");
        assert_eq!(node.name_embedding, Some(vec![0.1, 0.2, 0.3]));
        assert!(node.labels.contains(&"Entity".to_string()));
        assert!(node.labels.contains(&DOCUMENT_CHUNK_LABEL.to_string()));
        assert_eq!(
            node.attributes.get("doc_id").and_then(|v| v.as_str()),
            Some("doc1")
        );
        assert_eq!(
            node.attributes.get("chunk_index").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            node.attributes.get("source").and_then(|v| v.as_str()),
            Some("kb.pdf")
        );
        assert_eq!(node.uuid, chunk_uuid("g1", "doc1", 2)); // deterministic
    }

    #[test]
    fn map_hit_extracts_provenance() {
        let node = chunk_to_entity_node(&sample_chunk(), "g1", vec![0.0; 3], Utc::now());
        let hit = node_to_chunk_hit(node, 0.87);
        assert_eq!(hit.text, "hello world");
        assert_eq!(hit.score, 0.87);
        assert_eq!(hit.doc_id.as_deref(), Some("doc1"));
        assert_eq!(hit.chunk_index, Some(2));
        assert_eq!(hit.group_id, "g1");
    }
}
