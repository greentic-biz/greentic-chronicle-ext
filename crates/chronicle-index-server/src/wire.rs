use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};

pub const MAX_SEARCH_LIMIT: u32 = 50;
pub const MAX_DIMS: i64 = 8192;

fn default_search_limit() -> u32 {
    5
}

#[derive(Debug, Deserialize)]
pub struct PutIndexRequest {
    pub name: String,
    pub embedding_model: String,
    pub dims: i64,
    pub chunk_size: i64,
    pub chunk_overlap: i64,
}

#[derive(Debug, Serialize)]
pub struct IndexView {
    pub index_id: String,
    pub name: String,
    pub embedding_model: String,
    pub dims: i64,
    pub chunk_size: i64,
    pub chunk_overlap: i64,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub documents: Vec<DocumentUpsert>,
}

#[derive(Debug, Deserialize)]
pub struct DocumentUpsert {
    pub document_id: String,
    pub content_hash: String,
    pub chunks: Vec<ChunkUpsert>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkUpsert {
    pub chunk_index: i64,
    pub text: String,
    pub vector_b64: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UpsertResponse {
    pub upserted: usize,
    pub unchanged: usize,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub document_count: i64,
    pub chunk_count: i64,
    pub embedding_model: String,
    pub dims: i64,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub vector_b64: String,
    #[serde(default = "default_search_limit")]
    pub limit: u32,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub chunks: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub text: String,
    pub score: f64,
    pub document_id: String,
    pub chunk_index: i64,
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    pub tenant_slug: String,
    pub teams: Vec<String>,
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Serialize)]
pub struct CreateKeyResponse {
    pub key_id: String,
    pub api_key: String,
}

#[derive(Debug, Serialize)]
pub struct KeyView {
    pub key_id: String,
    pub tenant_slug: String,
    pub teams: Vec<String>,
    pub label: String,
    pub created_at: String,
}

/// Standard padded base64 of little-endian `f32` bytes — the designer's
/// `vector_b64` encoding.
pub fn decode_vector(b64: &str) -> Result<Vec<f32>, String> {
    let bytes = STANDARD
        .decode(b64)
        .map_err(|e| format!("vector_b64 is not base64: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "vector_b64 decodes to {} bytes, not a multiple of 4",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

pub fn encode_vector(vector: &[f32]) -> String {
    let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
    STANDARD.encode(bytes)
}

pub fn valid_slug(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

pub fn valid_index_id(s: &str) -> bool {
    (1..=128).contains(&s.len())
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_round_trip_as_little_endian_f32() {
        let v = vec![0.5_f32, -1.25, 3.0];
        assert_eq!(decode_vector(&encode_vector(&v)).expect("decode"), v);
    }

    #[test]
    fn a_vector_whose_byte_length_is_not_a_multiple_of_four_is_refused() {
        use base64::Engine as _;
        let three_bytes = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        assert!(decode_vector(&three_bytes).is_err());
        assert!(decode_vector("%%%").is_err());
    }

    #[test]
    fn slugs_and_index_ids_exclude_the_group_separator() {
        assert!(valid_slug("acme-corp_1"));
        assert!(!valid_slug("acme:corp"));
        assert!(!valid_slug(""));
        assert!(valid_index_id("kb_01J.abc-1"));
        assert!(!valid_index_id("kb:1"));
        assert!(!valid_index_id(&"a".repeat(129)));
    }

    #[test]
    fn the_designer_upsert_body_deserializes() {
        let body = serde_json::json!({"documents":[{"document_id":"d1","content_hash":"h1",
            "chunks":[{"chunk_index":0,"text":"t","vector_b64":"AAAAAA=="}]}]});
        let parsed: UpsertRequest = serde_json::from_value(body).expect("parse");
        assert_eq!(parsed.documents[0].chunks[0].chunk_index, 0);
    }
}
