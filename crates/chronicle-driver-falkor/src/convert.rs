//! Value conversion for the FalkorDB backend.
//!
//! FalkorDB is openCypher over Redis. The `falkordb` 0.2 crate binds parameters
//! as `CYPHER key=<literal> <query>` — i.e. every parameter *value* is a textual
//! **Cypher literal expression**, not a typed wire value (unlike neo4rs Bolt).
//! This module therefore centres on two jobs:
//!
//! 1. **Cypher-literal encoding** ([`lit_str`], [`lit_int`], [`lit_f64`],
//!    [`lit_string_list`], [`lit_vecf32`], [`opt_lit`]) — every dynamic value
//!    crosses the boundary as a properly-escaped literal so user data can NEVER
//!    break out of the literal (no string interpolation of raw values).
//!
//! 2. **Result extraction** ([`node_props`], [`edge_props`], the `read_*`
//!    helpers, [`*_from_props`]) — pulling typed domain values out of
//!    `FalkorValue` (`Node` / `Edge` properties, scalars, lists, `Vec32`).
//!
//! ## Dialect deltas vs the Neo4j driver (verified against falkordb 0.2.1 +
//!    live `falkordb/falkordb:latest`)
//!
//! - **Datetime → epoch millis `i64`.** FalkorDB has no native datetime type.
//!   All temporal fields (`created_at`, `valid_at`, `expired_at`, `invalid_at`,
//!   the saga watermarks) are stored as epoch-millisecond integers; bi-temporal
//!   comparisons are integer comparisons. [`datetime_to_millis`] /
//!   [`millis_to_datetime`] are the only boundary.
//! - **Embeddings → `vecf32([...])`.** Vectors are written wrapped in the
//!   `vecf32([..])` constructor and read back as `FalkorValue::Vec32`.
//! - **Attributes → JSON string.** FalkorDB rejects nested-map properties
//!   ("Property values can only be of primitive types or arrays of primitive
//!   types"). The flat-or-nested `attributes` map is serialised to a single JSON
//!   **string** property `attrs_json` and parsed back on read. This is a
//!   documented deviation from the Neo4j backend (which stores flat attributes as
//!   real node properties).

use chronicle_core::driver::DriverError;
use chronicle_core::types::{
    CommunityNode, EntityEdge, EntityNode, EpisodeType, EpisodicNode, SagaNode,
};
use chrono::{DateTime, TimeZone, Utc};
use falkordb::FalkorValue;
use serde_json::{Map, Value};
use std::collections::HashMap;

// =====================================================================
// Cypher-literal encoders (write side)
// =====================================================================

/// Encode a string as a single-quoted Cypher string literal, escaping
/// backslashes and single quotes so the value cannot break out of the literal.
pub fn lit_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// Encode an `i64` as a Cypher integer literal.
pub fn lit_int(v: i64) -> String {
    v.to_string()
}

/// Encode an `f64` as a Cypher float literal. Non-finite values are clamped to
/// `0.0` (FalkorDB has no NaN/Inf literal); embeddings are finite in practice.
pub fn lit_f64(v: f64) -> String {
    if v.is_finite() {
        // Always include a decimal point so the value parses as a float.
        let s = format!("{v:?}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        "0.0".to_string()
    }
}

/// Encode a `Vec<String>` as a Cypher list-of-strings literal.
pub fn lit_string_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&lit_str(s));
    }
    out.push(']');
    out
}

/// Encode an embedding as a `vecf32([..])` constructor literal (the FalkorDB
/// vector-write form). Both stored vectors and query vectors use this wrapper.
pub fn lit_vecf32(embedding: &[f32]) -> String {
    let mut out = String::from("vecf32([");
    for (i, f) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&lit_f64(*f as f64));
    }
    out.push_str("])");
    out
}

/// `DateTime<Utc>` → epoch-millis integer literal (THE temporal boundary).
pub fn lit_datetime(dt: DateTime<Utc>) -> String {
    lit_int(datetime_to_millis(dt))
}

/// `DateTime<Utc>` → epoch milliseconds.
pub fn datetime_to_millis(dt: DateTime<Utc>) -> i64 {
    dt.timestamp_millis()
}

/// Epoch milliseconds → `DateTime<Utc>`.
pub fn millis_to_datetime(millis: i64) -> Result<DateTime<Utc>, DriverError> {
    match Utc.timestamp_millis_opt(millis) {
        chrono::LocalResult::Single(dt) => Ok(dt),
        _ => Err(DriverError::Decode(format!(
            "epoch-millis {millis} is out of range for a UTC datetime"
        ))),
    }
}

/// Encode the `attributes` map as a JSON-string Cypher literal (deviation: nested
/// maps are unsupported as FalkorDB properties, so attributes live in a single
/// `attrs_json` string property).
pub fn lit_attrs(attributes: &Map<String, Value>) -> Result<String, DriverError> {
    let json = serde_json::to_string(&Value::Object(attributes.clone()))
        .map_err(|e| DriverError::Decode(format!("serialize attributes: {e}")))?;
    Ok(lit_str(&json))
}

/// Helper for an optional literal: `None` becomes the Cypher `null` literal.
pub fn opt_lit(value: Option<String>) -> String {
    value.unwrap_or_else(|| "null".to_string())
}

// =====================================================================
// FalkorValue extraction (read side)
// =====================================================================

/// Borrow the `HashMap` of properties from a `FalkorValue::Node`.
pub fn node_props(value: &FalkorValue) -> Result<&HashMap<String, FalkorValue>, DriverError> {
    value
        .as_node()
        .map(|n| &n.properties)
        .ok_or_else(|| DriverError::Decode("expected a Node value".to_string()))
}

/// Borrow the `HashMap` of properties from a `FalkorValue::Edge`.
pub fn edge_props(value: &FalkorValue) -> Result<&HashMap<String, FalkorValue>, DriverError> {
    value
        .as_edge()
        .map(|e| &e.properties)
        .ok_or_else(|| DriverError::Decode("expected an Edge value".to_string()))
}

/// Read a required string property.
pub fn read_string(props: &HashMap<String, FalkorValue>, key: &str) -> Result<String, DriverError> {
    match props.get(key) {
        Some(FalkorValue::String(s)) => Ok(s.clone()),
        Some(other) => Err(DriverError::Decode(format!(
            "property '{key}' is not a string: {other:?}"
        ))),
        None => Err(DriverError::Decode(format!("missing property '{key}'"))),
    }
}

/// Read a required i64 property.
pub fn read_i64(props: &HashMap<String, FalkorValue>, key: &str) -> Result<i64, DriverError> {
    match props.get(key) {
        Some(v) => v
            .to_i64()
            .ok_or_else(|| DriverError::Decode(format!("property '{key}' is not an integer"))),
        None => Err(DriverError::Decode(format!("missing property '{key}'"))),
    }
}

/// Read a required datetime property stored as epoch millis.
pub fn read_datetime(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<DateTime<Utc>, DriverError> {
    millis_to_datetime(read_i64(props, key)?)
}

/// Read an optional datetime property (absent or `None` → `None`).
pub fn read_opt_datetime(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<Option<DateTime<Utc>>, DriverError> {
    match props.get(key) {
        None | Some(FalkorValue::None) => Ok(None),
        Some(v) => {
            let millis = v.to_i64().ok_or_else(|| {
                DriverError::Decode(format!("property '{key}' is not an integer"))
            })?;
            Ok(Some(millis_to_datetime(millis)?))
        }
    }
}

/// Read an optional string property (absent or `None` → `None`).
pub fn read_opt_string(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<Option<String>, DriverError> {
    match props.get(key) {
        None | Some(FalkorValue::None) => Ok(None),
        Some(FalkorValue::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(DriverError::Decode(format!(
            "property '{key}' is not a string: {other:?}"
        ))),
    }
}

/// Read a `Vec<String>` property (absent or `None` → empty vec).
pub fn read_string_list(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<Vec<String>, DriverError> {
    match props.get(key) {
        None | Some(FalkorValue::None) => Ok(Vec::new()),
        Some(FalkorValue::Array(items)) => items
            .iter()
            .map(|x| match x {
                FalkorValue::String(s) => Ok(s.clone()),
                other => Err(DriverError::Decode(format!(
                    "expected string in list '{key}', got {other:?}"
                ))),
            })
            .collect(),
        Some(other) => Err(DriverError::Decode(format!(
            "property '{key}' is not a list: {other:?}"
        ))),
    }
}

/// Read an optional embedding property (`Vec32`, or an array of numbers, or
/// absent/`None`). FalkorDB returns stored `vecf32` values as `Vec32`.
pub fn read_opt_embedding(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<Option<Vec<f32>>, DriverError> {
    match props.get(key) {
        None | Some(FalkorValue::None) => Ok(None),
        Some(FalkorValue::Vec32(v)) => Ok(Some(v.values.clone())),
        Some(FalkorValue::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for x in items {
                match x {
                    FalkorValue::F64(f) => out.push(*f as f32),
                    FalkorValue::I64(i) => out.push(*i as f32),
                    other => {
                        return Err(DriverError::Decode(format!(
                            "expected number in embedding '{key}', got {other:?}"
                        )));
                    }
                }
            }
            Ok(Some(out))
        }
        Some(other) => Err(DriverError::Decode(format!(
            "property '{key}' is not a vector: {other:?}"
        ))),
    }
}

/// Read the `attrs_json` string property and parse it back into a flat-or-nested
/// attribute map (absent → empty map).
pub fn read_attributes(
    props: &HashMap<String, FalkorValue>,
    key: &str,
) -> Result<Map<String, Value>, DriverError> {
    let raw = match props.get(key) {
        None | Some(FalkorValue::None) => return Ok(Map::new()),
        Some(FalkorValue::String(s)) => s,
        Some(other) => {
            return Err(DriverError::Decode(format!(
                "attributes property '{key}' is not a string: {other:?}"
            )));
        }
    };
    if raw.is_empty() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(raw)
        .map_err(|e| DriverError::Decode(format!("parse attributes JSON: {e}")))?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(DriverError::Decode(format!(
            "attributes JSON is not an object: {other}"
        ))),
    }
}

/// Parse a `Vec<f32>` embedding directly from a positional column value
/// (`Vec32`, or a numeric `Array`). Used by the embedding loaders, which RETURN
/// the bare vector column rather than a whole Node. `None`/absent is an error
/// here — callers guard against null embeddings in the query (`IS NOT NULL`).
pub fn embedding_from_value(value: &FalkorValue) -> Result<Vec<f32>, DriverError> {
    match value {
        FalkorValue::Vec32(v) => Ok(v.values.clone()),
        FalkorValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for x in items {
                match x {
                    FalkorValue::F64(f) => out.push(*f as f32),
                    FalkorValue::I64(i) => out.push(*i as f32),
                    other => {
                        return Err(DriverError::Decode(format!(
                            "expected number in embedding, got {other:?}"
                        )));
                    }
                }
            }
            Ok(out)
        }
        other => Err(DriverError::Decode(format!(
            "expected a vector value, got {other:?}"
        ))),
    }
}

/// Read a required string directly from a positional column value.
pub fn string_from_value(value: &FalkorValue) -> Result<String, DriverError> {
    match value {
        FalkorValue::String(s) => Ok(s.clone()),
        other => Err(DriverError::Decode(format!(
            "expected a string value, got {other:?}"
        ))),
    }
}

/// Read a required i64 directly from a positional column value.
pub fn i64_from_value(value: &FalkorValue) -> Result<i64, DriverError> {
    value
        .to_i64()
        .ok_or_else(|| DriverError::Decode(format!("expected an integer value, got {value:?}")))
}

/// Read a required f64 (e.g. a vector/fulltext `score`) from a positional column
/// value. Accepts both `F64` and `I64` (a 0/1 score may arrive as an integer).
pub fn f64_from_value(value: &FalkorValue) -> Result<f64, DriverError> {
    match value {
        FalkorValue::F64(f) => Ok(*f),
        FalkorValue::I64(i) => Ok(*i as f64),
        other => Err(DriverError::Decode(format!(
            "expected a number value, got {other:?}"
        ))),
    }
}

// =====================================================================
// EpisodeType <-> source column
// =====================================================================

/// Map `EpisodeType` to the stored `source` string (lowercase, matching the
/// Neo4j backend so behaviour parity holds).
pub fn episode_type_to_str(t: EpisodeType) -> &'static str {
    match t {
        EpisodeType::Message => "message",
        EpisodeType::Text => "text",
        EpisodeType::Json => "json",
    }
}

/// Parse the stored `source` string back to `EpisodeType`.
pub fn episode_type_from_str(s: &str) -> Result<EpisodeType, DriverError> {
    match s {
        "message" => Ok(EpisodeType::Message),
        "text" => Ok(EpisodeType::Text),
        "json" => Ok(EpisodeType::Json),
        other => Err(DriverError::Decode(format!(
            "unknown episode source '{other}'"
        ))),
    }
}

// =====================================================================
// Domain reads from a Node/Edge property map
// =====================================================================

/// Parse an `EntityNode` from a `Node`'s property map.
pub fn entity_node_from_props(
    props: &HashMap<String, FalkorValue>,
) -> Result<EntityNode, DriverError> {
    Ok(EntityNode {
        uuid: read_string(props, "uuid")?,
        name: read_string(props, "name")?,
        group_id: read_string(props, "group_id")?,
        labels: read_string_list(props, "labels")?,
        created_at: read_datetime(props, "created_at")?,
        summary: read_string(props, "summary")?,
        attributes: read_attributes(props, "attrs_json")?,
        name_embedding: read_opt_embedding(props, "name_embedding")?,
    })
}

/// Parse an `EntityEdge` from an `Edge`'s property map.
pub fn entity_edge_from_props(
    props: &HashMap<String, FalkorValue>,
) -> Result<EntityEdge, DriverError> {
    Ok(EntityEdge {
        uuid: read_string(props, "uuid")?,
        source_node_uuid: read_string(props, "source_node_uuid")?,
        target_node_uuid: read_string(props, "target_node_uuid")?,
        name: read_string(props, "name")?,
        fact: read_string(props, "fact")?,
        group_id: read_string(props, "group_id")?,
        episodes: read_string_list(props, "episodes")?,
        created_at: read_datetime(props, "created_at")?,
        expired_at: read_opt_datetime(props, "expired_at")?,
        valid_at: read_opt_datetime(props, "valid_at")?,
        invalid_at: read_opt_datetime(props, "invalid_at")?,
        attributes: read_attributes(props, "attrs_json")?,
        fact_embedding: read_opt_embedding(props, "fact_embedding")?,
    })
}

/// Parse an `EpisodicNode` from a `Node`'s property map.
pub fn episodic_node_from_props(
    props: &HashMap<String, FalkorValue>,
) -> Result<EpisodicNode, DriverError> {
    Ok(EpisodicNode {
        uuid: read_string(props, "uuid")?,
        name: read_string(props, "name")?,
        group_id: read_string(props, "group_id")?,
        labels: Vec::new(),
        source: episode_type_from_str(&read_string(props, "source")?)?,
        source_description: read_string(props, "source_description")?,
        content: read_string(props, "content")?,
        entity_edges: read_string_list(props, "entity_edges")?,
        created_at: read_datetime(props, "created_at")?,
        valid_at: read_datetime(props, "valid_at")?,
    })
}

/// Parse a `CommunityNode` from a `Node`'s property map.
pub fn community_node_from_props(
    props: &HashMap<String, FalkorValue>,
) -> Result<CommunityNode, DriverError> {
    Ok(CommunityNode {
        uuid: read_string(props, "uuid")?,
        name: read_string(props, "name")?,
        group_id: read_string(props, "group_id")?,
        labels: vec!["Community".to_string()],
        created_at: read_datetime(props, "created_at")?,
        summary: read_string(props, "summary")?,
        name_embedding: read_opt_embedding(props, "name_embedding")?,
    })
}

/// Parse a `SagaNode` from a `Node`'s property map.
pub fn saga_node_from_props(props: &HashMap<String, FalkorValue>) -> Result<SagaNode, DriverError> {
    Ok(SagaNode {
        uuid: read_string(props, "uuid")?,
        name: read_string(props, "name")?,
        group_id: read_string(props, "group_id")?,
        labels: vec!["Saga".to_string()],
        created_at: read_datetime(props, "created_at")?,
        summary: read_string(props, "summary")?,
        first_episode_uuid: read_opt_string(props, "first_episode_uuid")?,
        last_episode_uuid: read_opt_string(props, "last_episode_uuid")?,
        last_summarized_at: read_opt_datetime(props, "last_summarized_at")?,
        last_summarized_episode_valid_at: read_opt_datetime(
            props,
            "last_summarized_episode_valid_at",
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lit_str_escapes_quotes_and_backslashes() {
        assert_eq!(lit_str("O'Brien"), "'O\\'Brien'");
        assert_eq!(lit_str("a\\b"), "'a\\\\b'");
        assert_eq!(lit_str("plain"), "'plain'");
    }

    #[test]
    fn lit_f64_always_has_decimal() {
        assert_eq!(lit_f64(1.0), "1.0");
        assert!(lit_f64(0.5).contains('.'));
        assert_eq!(lit_f64(f64::NAN), "0.0");
        assert_eq!(lit_f64(f64::INFINITY), "0.0");
    }

    #[test]
    fn lit_vecf32_wraps_constructor() {
        let v = lit_vecf32(&[1.0, 0.0, 2.5]);
        assert!(v.starts_with("vecf32(["));
        assert!(v.ends_with("])"));
        assert!(v.contains("1.0"));
        assert!(v.contains("2.5"));
    }

    #[test]
    fn lit_string_list_encodes_each() {
        assert_eq!(lit_string_list(&[]), "[]");
        assert_eq!(lit_string_list(&["a".into(), "b".into()]), "['a','b']");
    }

    #[test]
    fn datetime_millis_roundtrip() {
        let dt = Utc.timestamp_millis_opt(1_700_000_000_123).unwrap();
        let millis = datetime_to_millis(dt);
        assert_eq!(millis, 1_700_000_000_123);
        assert_eq!(millis_to_datetime(millis).unwrap(), dt);
    }

    #[test]
    fn lit_attrs_roundtrips_via_json() {
        let mut attrs = Map::new();
        attrs.insert("count".into(), json!(3));
        attrs.insert("label".into(), json!("x"));
        let lit = lit_attrs(&attrs).unwrap();
        // Strip the surrounding single-quote literal wrapper for reparse.
        assert!(lit.starts_with('\'') && lit.ends_with('\''));
    }

    #[test]
    fn read_attributes_parses_json_string() {
        let mut props = HashMap::new();
        props.insert(
            "attrs_json".to_string(),
            FalkorValue::String(r#"{"a":1,"b":"x"}"#.to_string()),
        );
        let attrs = read_attributes(&props, "attrs_json").unwrap();
        assert_eq!(attrs.get("a"), Some(&json!(1)));
        assert_eq!(attrs.get("b"), Some(&json!("x")));
    }

    #[test]
    fn read_attributes_absent_is_empty() {
        let props = HashMap::new();
        assert!(read_attributes(&props, "attrs_json").unwrap().is_empty());
    }

    #[test]
    fn episode_type_str_roundtrip() {
        for t in [EpisodeType::Message, EpisodeType::Text, EpisodeType::Json] {
            assert_eq!(episode_type_from_str(episode_type_to_str(t)).unwrap(), t);
        }
        assert!(episode_type_from_str("bogus").is_err());
    }

    #[test]
    fn read_opt_embedding_handles_array_and_none() {
        // `Vec32` is the live readback shape (exercised by the integration tests);
        // here we cover the `None` and numeric-`Array` fallback paths the unit
        // suite can construct without the crate-private `Vec32` constructor.
        let mut props = HashMap::new();
        assert!(read_opt_embedding(&props, "e").unwrap().is_none());
        props.insert("e".to_string(), FalkorValue::None);
        assert!(read_opt_embedding(&props, "e").unwrap().is_none());
        props.insert(
            "e".to_string(),
            FalkorValue::Array(vec![FalkorValue::F64(1.0), FalkorValue::I64(2)]),
        );
        assert_eq!(
            read_opt_embedding(&props, "e").unwrap().unwrap(),
            vec![1.0_f32, 2.0]
        );
    }

    #[test]
    fn embedding_from_value_parses_numeric_array() {
        let v = FalkorValue::Array(vec![FalkorValue::F64(1.0), FalkorValue::I64(2)]);
        assert_eq!(embedding_from_value(&v).unwrap(), vec![1.0_f32, 2.0]);
        // A non-vector value is a hard decode error (loaders guard null in-query).
        assert!(embedding_from_value(&FalkorValue::I64(5)).is_err());
    }

    #[test]
    fn scalar_from_value_helpers() {
        assert_eq!(
            string_from_value(&FalkorValue::String("x".into())).unwrap(),
            "x"
        );
        assert!(string_from_value(&FalkorValue::I64(1)).is_err());

        assert_eq!(i64_from_value(&FalkorValue::I64(7)).unwrap(), 7);
        assert!(i64_from_value(&FalkorValue::String("7".into())).is_err());

        // f64 reader accepts both F64 and I64 (a 0/1 score may arrive as an int).
        assert_eq!(f64_from_value(&FalkorValue::F64(0.5)).unwrap(), 0.5);
        assert_eq!(f64_from_value(&FalkorValue::I64(1)).unwrap(), 1.0);
        assert!(f64_from_value(&FalkorValue::String("x".into())).is_err());
    }
}
