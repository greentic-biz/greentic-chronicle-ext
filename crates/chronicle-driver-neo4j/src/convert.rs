//! Record <-> domain-type conversion for the Neo4j backend.
//!
//! Mirrors the upstream record parsers in
//! `graphiti_core/driver/record_parsers.py` (attribute-stripping logic) and the
//! save-data dict construction in
//! `graphiti_core/driver/neo4j/operations/{entity_node_ops,entity_edge_ops}.py`.
//!
//! neo4rs 0.8 notes:
//! - Neo4j temporal `datetime()` properties round-trip as `BoltType::DateTime`,
//!   which converts to `chrono::DateTime<FixedOffset>`. We store `DateTime<Utc>`
//!   in the domain types, so params go Utc -> FixedOffset (via `.fixed_offset()`)
//!   and reads go FixedOffset -> Utc (via `.with_timezone(&Utc)`).
//! - Embeddings: `Vec<f32>` <-> `BoltType::List` of `BoltType::Float` (f64 on the
//!   wire). We widen f32 -> f64 on write and narrow f64 -> f32 on read.
//! - Flat `attributes` (serde_json primitives) <-> `BoltType::Map`. Only flat
//!   primitive values are supported (mirrors upstream, which stores attributes as
//!   flat scalar node/edge properties). A non-primitive (nested object / array)
//!   attribute is a `DriverError::Decode` on write.

use chronicle_core::driver::DriverError;
use chronicle_core::types::{EntityEdge, EntityNode, EpisodeType, EpisodicEdge, EpisodicNode};
use chrono::{DateTime, FixedOffset, Utc};
use neo4rs::{BoltList, BoltMap, BoltNull, BoltString, BoltType, Row};
use serde_json::{Map, Number, Value};

// ---------------------------------------------------------------------
// Scalar BoltType helpers
// ---------------------------------------------------------------------

/// `DateTime<Utc>` -> a `BoltType::DateTime` param value.
///
/// Exposed `pub(crate)` so the filter-fragment builder in `queries.rs` can bind
/// `DateFilter` dates as the SAME `BoltType::DateTime` wire form used everywhere
/// else (so `e.valid_at <op> $p0` compares datetime-to-datetime, not against a
/// string).
pub(crate) fn datetime_to_bolt(dt: DateTime<Utc>) -> BoltType {
    let fixed: DateTime<FixedOffset> = dt.fixed_offset();
    BoltType::from(fixed)
}

/// `Option<DateTime<Utc>>` -> param value (Null when None).
fn opt_datetime_to_bolt(dt: Option<DateTime<Utc>>) -> BoltType {
    match dt {
        Some(d) => datetime_to_bolt(d),
        None => BoltType::Null(BoltNull),
    }
}

/// `Vec<f32>` -> `BoltType::List` of floats (widened to f64 on the wire).
fn embedding_to_bolt(embedding: &[f32]) -> BoltType {
    let list: Vec<BoltType> = embedding
        .iter()
        .map(|f| BoltType::from(*f as f64))
        .collect();
    BoltType::List(BoltList::from(list))
}

/// `Option<Vec<f32>>` -> param value (Null when None).
fn opt_embedding_to_bolt(embedding: Option<&Vec<f32>>) -> BoltType {
    match embedding {
        Some(e) => embedding_to_bolt(e),
        None => BoltType::Null(BoltNull),
    }
}

/// `Vec<String>` -> `BoltType::List` of strings.
fn string_list_to_bolt(items: &[String]) -> BoltType {
    let list: Vec<BoltType> = items.iter().map(|s| BoltType::from(s.as_str())).collect();
    BoltType::List(BoltList::from(list))
}

/// A flat serde_json primitive -> BoltType. Errors on non-primitive values.
fn json_primitive_to_bolt(key: &str, value: &Value) -> Result<BoltType, DriverError> {
    match value {
        Value::Null => Ok(BoltType::Null(BoltNull)),
        Value::Bool(b) => Ok(BoltType::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(BoltType::from(i))
            } else if let Some(f) = n.as_f64() {
                Ok(BoltType::from(f))
            } else {
                Err(DriverError::Decode(format!(
                    "attribute '{key}' has an unrepresentable number"
                )))
            }
        }
        Value::String(s) => Ok(BoltType::from(s.as_str())),
        Value::Array(_) | Value::Object(_) => Err(DriverError::Decode(format!(
            "attribute '{key}' is non-primitive (nested arrays/objects are not supported; \
             upstream stores flat attributes only)"
        ))),
    }
}

/// Build a `BoltType::Map` of flat attributes from a serde_json object map.
fn attributes_to_bolt(attributes: &Map<String, Value>) -> Result<BoltMap, DriverError> {
    let mut map = BoltMap::new();
    for (k, v) in attributes {
        map.put(BoltString::new(k), json_primitive_to_bolt(k, v)?);
    }
    Ok(map)
}

// ---------------------------------------------------------------------
// BoltType -> domain reads
// ---------------------------------------------------------------------

/// Read a required `DateTime<Utc>` from a row column.
fn read_datetime(row: &Row, col: &str) -> Result<DateTime<Utc>, DriverError> {
    let bolt: BoltType = row
        .get(col)
        .map_err(|e| DriverError::Decode(format!("missing/invalid datetime '{col}': {e}")))?;
    bolt_to_datetime(col, &bolt)?
        .ok_or_else(|| DriverError::Decode(format!("datetime '{col}' was null")))
}

/// Read an optional `DateTime<Utc>` from a row column.
fn read_opt_datetime(row: &Row, col: &str) -> Result<Option<DateTime<Utc>>, DriverError> {
    let bolt: BoltType = match row.get(col) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    bolt_to_datetime(col, &bolt)
}

fn bolt_to_datetime(col: &str, bolt: &BoltType) -> Result<Option<DateTime<Utc>>, DriverError> {
    match bolt {
        BoltType::Null(_) => Ok(None),
        BoltType::DateTime(_) => {
            let fixed: DateTime<FixedOffset> = bolt
                .clone()
                .try_into()
                .map_err(|e| DriverError::Decode(format!("datetime '{col}': {e}")))?;
            Ok(Some(fixed.with_timezone(&Utc)))
        }
        other => Err(DriverError::Decode(format!(
            "expected datetime for '{col}', got {other:?}"
        ))),
    }
}

/// Read a required String column.
fn read_string(row: &Row, col: &str) -> Result<String, DriverError> {
    row.get(col)
        .map_err(|e| DriverError::Decode(format!("missing/invalid string '{col}': {e}")))
}

/// Read a `Vec<String>` column (used for labels / episodes / entity_edges).
fn read_string_list(row: &Row, col: &str) -> Result<Vec<String>, DriverError> {
    let bolt: BoltType = match row.get(col) {
        Ok(b) => b,
        Err(_) => return Ok(Vec::new()),
    };
    match bolt {
        BoltType::Null(_) => Ok(Vec::new()),
        BoltType::List(l) => l
            .value
            .iter()
            .map(|x| match x {
                BoltType::String(s) => Ok(s.value.clone()),
                other => Err(DriverError::Decode(format!(
                    "expected string in list '{col}', got {other:?}"
                ))),
            })
            .collect(),
        other => Err(DriverError::Decode(format!(
            "expected list for '{col}', got {other:?}"
        ))),
    }
}

/// Read an optional embedding (`Vec<f32>`) column.
fn read_opt_embedding(row: &Row, col: &str) -> Result<Option<Vec<f32>>, DriverError> {
    let bolt: BoltType = match row.get(col) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };
    match bolt {
        BoltType::Null(_) => Ok(None),
        BoltType::List(l) => {
            let mut out = Vec::with_capacity(l.value.len());
            for x in &l.value {
                match x {
                    BoltType::Float(f) => out.push(f.value as f32),
                    BoltType::Integer(i) => out.push(i.value as f32),
                    other => {
                        return Err(DriverError::Decode(format!(
                            "expected float in embedding '{col}', got {other:?}"
                        )));
                    }
                }
            }
            Ok(Some(out))
        }
        other => Err(DriverError::Decode(format!(
            "expected list for embedding '{col}', got {other:?}"
        ))),
    }
}

/// Read a `(uuid, embedding)` pair from an embeddings-loader row
/// (`GET_NODE_EMBEDDINGS` / `GET_EDGE_EMBEDDINGS`). The Cypher already guards
/// `embedding IS NOT NULL`, so a row here is expected to carry a non-null vector;
/// a null/absent embedding maps to `None` (the caller then omits the uuid),
/// preserving the upstream "omit-when-missing" contract even if a backend returns
/// a stray null.
pub(crate) fn embedding_row(row: &Row) -> Result<Option<(String, Vec<f32>)>, DriverError> {
    let uuid: String = row
        .get("uuid")
        .map_err(|e| DriverError::Decode(format!("embedding row missing uuid: {e}")))?;
    match read_opt_embedding(row, "embedding")? {
        Some(emb) => Ok(Some((uuid, emb))),
        None => Ok(None),
    }
}

/// A single bolt primitive -> serde_json::Value (for the attributes remainder).
fn bolt_primitive_to_json(bolt: &BoltType) -> Option<Value> {
    match bolt {
        BoltType::Null(_) => Some(Value::Null),
        BoltType::Boolean(b) => Some(Value::Bool(b.value)),
        BoltType::Integer(i) => Some(Value::Number(Number::from(i.value))),
        BoltType::Float(f) => Number::from_f64(f.value).map(Value::Number),
        BoltType::String(s) => Some(Value::String(s.value.clone())),
        // Non-primitive bolt types are not part of the flat attribute contract;
        // they are dropped from the attributes remainder (they correspond to
        // core typed fields like name_embedding, or to list columns handled
        // explicitly above).
        _ => None,
    }
}

/// Read `properties(n)` / `properties(e)` map, strip the listed core keys, and
/// return the remainder as flat serde_json attributes.
fn read_attributes(row: &Row, strip: &[&str]) -> Result<Map<String, Value>, DriverError> {
    let bolt: BoltType = row
        .get("attributes")
        .map_err(|e| DriverError::Decode(format!("missing attributes map: {e}")))?;
    let map = match bolt {
        BoltType::Map(m) => m,
        BoltType::Null(_) => return Ok(Map::new()),
        other => {
            return Err(DriverError::Decode(format!(
                "expected map for attributes, got {other:?}"
            )));
        }
    };

    let mut out = Map::new();
    for (k, v) in &map.value {
        let key = &k.value;
        if strip.contains(&key.as_str()) {
            continue;
        }
        if let Some(json) = bolt_primitive_to_json(v) {
            out.insert(key.clone(), json);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// EpisodeType <-> source column
// ---------------------------------------------------------------------

/// Map `EpisodeType` to the stored `source` string. Upstream stores the Python
/// enum *value* (lowercase `message` / `text` / `json`) via `source.value`, and
/// the source-filter param uses `source.name` (uppercase). The stored property
/// is the value, so we use the lowercase form for both storage and the
/// retrieve_episodes filter param (consistent with what is written).
pub fn episode_type_to_str(t: EpisodeType) -> &'static str {
    match t {
        EpisodeType::Message => "message",
        EpisodeType::Text => "text",
        EpisodeType::Json => "json",
    }
}

fn episode_type_from_str(s: &str) -> Result<EpisodeType, DriverError> {
    match s {
        "message" => Ok(EpisodeType::Message),
        "text" => Ok(EpisodeType::Text),
        "json" => Ok(EpisodeType::Json),
        other => Err(DriverError::Decode(format!(
            "unknown episode source '{other}'"
        ))),
    }
}

// ---------------------------------------------------------------------
// Save-data map builders (props maps for the UNWIND save queries)
// ---------------------------------------------------------------------

/// Build the `{uuid, labels, props, name_embedding}` map for one entity node.
///
/// `props` mirrors upstream `entity_data` (core scalar fields + flattened
/// attributes), but EXCLUDES `name_embedding` (set separately via the vector
/// procedure) and `labels` (applied via `SET n:$(node.labels)`).
pub fn entity_node_to_bolt(node: &EntityNode) -> Result<BoltType, DriverError> {
    let mut props = attributes_to_bolt(&node.attributes)?;
    props.put(BoltString::new("uuid"), BoltType::from(node.uuid.as_str()));
    props.put(BoltString::new("name"), BoltType::from(node.name.as_str()));
    props.put(
        BoltString::new("group_id"),
        BoltType::from(node.group_id.as_str()),
    );
    props.put(
        BoltString::new("summary"),
        BoltType::from(node.summary.as_str()),
    );
    props.put(
        BoltString::new("created_at"),
        datetime_to_bolt(node.created_at),
    );

    // Upstream always includes Entity in the label set.
    let mut labels = node.labels.clone();
    if !labels.iter().any(|l| l == "Entity") {
        labels.push("Entity".to_string());
    }

    let mut outer = BoltMap::new();
    outer.put(BoltString::new("uuid"), BoltType::from(node.uuid.as_str()));
    outer.put(BoltString::new("labels"), string_list_to_bolt(&labels));
    outer.put(BoltString::new("props"), BoltType::Map(props));
    outer.put(
        BoltString::new("name_embedding"),
        opt_embedding_to_bolt(node.name_embedding.as_ref()),
    );
    Ok(BoltType::Map(outer))
}

/// Build the `{uuid, props}` map for one episode. `props` mirrors the upstream
/// Neo4j episode save map exactly.
pub fn episode_to_bolt(episode: &EpisodicNode) -> BoltType {
    let mut props = BoltMap::new();
    props.put(
        BoltString::new("uuid"),
        BoltType::from(episode.uuid.as_str()),
    );
    props.put(
        BoltString::new("name"),
        BoltType::from(episode.name.as_str()),
    );
    props.put(
        BoltString::new("group_id"),
        BoltType::from(episode.group_id.as_str()),
    );
    props.put(
        BoltString::new("source_description"),
        BoltType::from(episode.source_description.as_str()),
    );
    props.put(
        BoltString::new("source"),
        BoltType::from(episode_type_to_str(episode.source)),
    );
    props.put(
        BoltString::new("content"),
        BoltType::from(episode.content.as_str()),
    );
    props.put(
        BoltString::new("entity_edges"),
        string_list_to_bolt(&episode.entity_edges),
    );
    props.put(
        BoltString::new("created_at"),
        datetime_to_bolt(episode.created_at),
    );
    props.put(
        BoltString::new("valid_at"),
        datetime_to_bolt(episode.valid_at),
    );

    let mut outer = BoltMap::new();
    outer.put(
        BoltString::new("uuid"),
        BoltType::from(episode.uuid.as_str()),
    );
    outer.put(BoltString::new("props"), BoltType::Map(props));
    BoltType::Map(outer)
}

/// Build the `{source_node_uuid, target_node_uuid, uuid, props, fact_embedding}`
/// map for one entity edge. `props` mirrors upstream `edge_data` (core fields +
/// flattened attributes), EXCLUDING `fact_embedding` (vector procedure).
pub fn entity_edge_to_bolt(edge: &EntityEdge) -> Result<BoltType, DriverError> {
    let mut props = attributes_to_bolt(&edge.attributes)?;
    props.put(BoltString::new("uuid"), BoltType::from(edge.uuid.as_str()));
    props.put(
        BoltString::new("source_node_uuid"),
        BoltType::from(edge.source_node_uuid.as_str()),
    );
    props.put(
        BoltString::new("target_node_uuid"),
        BoltType::from(edge.target_node_uuid.as_str()),
    );
    props.put(BoltString::new("name"), BoltType::from(edge.name.as_str()));
    props.put(BoltString::new("fact"), BoltType::from(edge.fact.as_str()));
    props.put(
        BoltString::new("group_id"),
        BoltType::from(edge.group_id.as_str()),
    );
    props.put(
        BoltString::new("episodes"),
        string_list_to_bolt(&edge.episodes),
    );
    props.put(
        BoltString::new("created_at"),
        datetime_to_bolt(edge.created_at),
    );
    props.put(
        BoltString::new("expired_at"),
        opt_datetime_to_bolt(edge.expired_at),
    );
    props.put(
        BoltString::new("valid_at"),
        opt_datetime_to_bolt(edge.valid_at),
    );
    props.put(
        BoltString::new("invalid_at"),
        opt_datetime_to_bolt(edge.invalid_at),
    );

    let mut outer = BoltMap::new();
    outer.put(
        BoltString::new("source_node_uuid"),
        BoltType::from(edge.source_node_uuid.as_str()),
    );
    outer.put(
        BoltString::new("target_node_uuid"),
        BoltType::from(edge.target_node_uuid.as_str()),
    );
    outer.put(BoltString::new("uuid"), BoltType::from(edge.uuid.as_str()));
    outer.put(BoltString::new("props"), BoltType::Map(props));
    outer.put(
        BoltString::new("fact_embedding"),
        opt_embedding_to_bolt(edge.fact_embedding.as_ref()),
    );
    Ok(BoltType::Map(outer))
}

/// Build the map for one episodic (MENTIONS) edge.
pub fn episodic_edge_to_bolt(edge: &EpisodicEdge) -> BoltType {
    let mut map = BoltMap::new();
    map.put(BoltString::new("uuid"), BoltType::from(edge.uuid.as_str()));
    map.put(
        BoltString::new("source_node_uuid"),
        BoltType::from(edge.source_node_uuid.as_str()),
    );
    map.put(
        BoltString::new("target_node_uuid"),
        BoltType::from(edge.target_node_uuid.as_str()),
    );
    map.put(
        BoltString::new("group_id"),
        BoltType::from(edge.group_id.as_str()),
    );
    map.put(
        BoltString::new("created_at"),
        datetime_to_bolt(edge.created_at),
    );
    BoltType::Map(map)
}

// ---------------------------------------------------------------------
// Record -> domain reads
// ---------------------------------------------------------------------

/// Core attribute keys stripped for entity nodes (mirrors
/// `entity_node_from_record`).
const ENTITY_NODE_STRIP: &[&str] = &[
    "uuid",
    "name",
    "group_id",
    "name_embedding",
    "summary",
    "created_at",
    "labels",
];

/// Core attribute keys stripped for entity edges (mirrors
/// `entity_edge_from_record`).
const ENTITY_EDGE_STRIP: &[&str] = &[
    "uuid",
    "source_node_uuid",
    "target_node_uuid",
    "fact",
    "fact_embedding",
    "name",
    "group_id",
    "episodes",
    "created_at",
    "expired_at",
    "valid_at",
    "invalid_at",
    "reference_time",
];

/// Parse an entity node from a row. Mirrors `entity_node_from_record`, including
/// removal of the dynamic `Entity_<group_id-without-hyphens>` label.
pub fn entity_node_from_row(row: &Row) -> Result<EntityNode, DriverError> {
    let group_id = read_string(row, "group_id")?;
    let attributes = read_attributes(row, ENTITY_NODE_STRIP)?;

    let mut labels = read_string_list(row, "labels")?;
    let dynamic_label = format!("Entity_{}", group_id.replace('-', ""));
    labels.retain(|l| l != &dynamic_label);

    Ok(EntityNode {
        uuid: read_string(row, "uuid")?,
        name: read_string(row, "name")?,
        group_id,
        labels,
        created_at: read_datetime(row, "created_at")?,
        summary: read_string(row, "summary")?,
        attributes,
        name_embedding: read_opt_embedding(row, "name_embedding")?,
    })
}

/// Parse an entity edge from a row. Mirrors `entity_edge_from_record`.
pub fn entity_edge_from_row(row: &Row) -> Result<EntityEdge, DriverError> {
    let attributes = read_attributes(row, ENTITY_EDGE_STRIP)?;
    Ok(EntityEdge {
        uuid: read_string(row, "uuid")?,
        source_node_uuid: read_string(row, "source_node_uuid")?,
        target_node_uuid: read_string(row, "target_node_uuid")?,
        name: read_string(row, "name")?,
        fact: read_string(row, "fact")?,
        group_id: read_string(row, "group_id")?,
        episodes: read_string_list(row, "episodes")?,
        created_at: read_datetime(row, "created_at")?,
        expired_at: read_opt_datetime(row, "expired_at")?,
        valid_at: read_opt_datetime(row, "valid_at")?,
        invalid_at: read_opt_datetime(row, "invalid_at")?,
        attributes,
        fact_embedding: read_opt_embedding(row, "fact_embedding")?,
    })
}

/// Parse an episodic node from a row. Mirrors `episodic_node_from_record`.
pub fn episodic_node_from_row(row: &Row) -> Result<EpisodicNode, DriverError> {
    let source = episode_type_from_str(&read_string(row, "source")?)?;
    Ok(EpisodicNode {
        uuid: read_string(row, "uuid")?,
        name: read_string(row, "name")?,
        group_id: read_string(row, "group_id")?,
        labels: Vec::new(),
        source,
        source_description: read_string(row, "source_description")?,
        content: read_string(row, "content")?,
        entity_edges: read_string_list(row, "entity_edges")?,
        created_at: read_datetime(row, "created_at")?,
        valid_at: read_datetime(row, "valid_at")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_primitive_conversions() {
        assert!(matches!(
            json_primitive_to_bolt("k", &json!(true)).unwrap(),
            BoltType::Boolean(_)
        ));
        assert!(matches!(
            json_primitive_to_bolt("k", &json!(42)).unwrap(),
            BoltType::Integer(_)
        ));
        assert!(matches!(
            json_primitive_to_bolt("k", &json!(1.5)).unwrap(),
            BoltType::Float(_)
        ));
        assert!(matches!(
            json_primitive_to_bolt("k", &json!("hi")).unwrap(),
            BoltType::String(_)
        ));
        assert!(matches!(
            json_primitive_to_bolt("k", &json!(null)).unwrap(),
            BoltType::Null(_)
        ));
    }

    #[test]
    fn nested_attribute_is_decode_error() {
        assert!(json_primitive_to_bolt("k", &json!({"a": 1})).is_err());
        assert!(json_primitive_to_bolt("k", &json!([1, 2, 3])).is_err());
    }

    #[test]
    fn attributes_map_excludes_nothing_but_errors_on_nesting() {
        let mut attrs = Map::new();
        attrs.insert("count".to_string(), json!(3));
        attrs.insert("label".to_string(), json!("x"));
        let bolt = attributes_to_bolt(&attrs).unwrap();
        assert_eq!(bolt.value.len(), 2);

        let mut bad = Map::new();
        bad.insert("nested".to_string(), json!({"x": 1}));
        assert!(attributes_to_bolt(&bad).is_err());
    }

    #[test]
    fn bolt_primitive_roundtrip_to_json() {
        assert_eq!(
            bolt_primitive_to_json(&BoltType::from(7_i64)),
            Some(json!(7))
        );
        assert_eq!(
            bolt_primitive_to_json(&BoltType::from("a")),
            Some(json!("a"))
        );
        assert_eq!(
            bolt_primitive_to_json(&BoltType::from(true)),
            Some(json!(true))
        );
        // A list is not a flat primitive -> dropped from attributes remainder.
        assert_eq!(
            bolt_primitive_to_json(&BoltType::List(BoltList::new())),
            None
        );
    }

    #[test]
    fn embedding_roundtrip_widen_narrow() {
        let emb = vec![0.1_f32, 0.2, 0.3];
        let bolt = embedding_to_bolt(&emb);
        let BoltType::List(list) = bolt else {
            panic!("expected list");
        };
        assert_eq!(list.value.len(), 3);
        for x in &list.value {
            assert!(matches!(x, BoltType::Float(_)));
        }
    }

    #[test]
    fn episode_type_str_roundtrip() {
        for t in [EpisodeType::Message, EpisodeType::Text, EpisodeType::Json] {
            let s = episode_type_to_str(t);
            assert_eq!(episode_type_from_str(s).unwrap(), t);
        }
        assert!(episode_type_from_str("bogus").is_err());
    }
}
