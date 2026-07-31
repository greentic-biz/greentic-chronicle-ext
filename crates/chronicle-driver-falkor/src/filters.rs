//! `SearchFilters` → FalkorDB Cypher `WHERE`-fragment builder.
//!
//! This mirrors the Neo4j driver's `edge_filter_fragments` /
//! `node_filter_fragments` semantics (see
//! `chronicle-driver-neo4j/src/queries.rs`), re-expressed for the FalkorDB
//! backend. The single structural difference from the Neo4j builder is the
//! parameter model: the `falkordb` 0.2 crate binds every parameter as a
//! **textual Cypher literal**, so this builder returns ONLY a `Vec<String>` of
//! WHERE fragments — there is no `(name, value)` param list. Every user value is
//! inlined via the escaping encoders in [`crate::convert`] (no raw
//! interpolation), exactly as the rest of the driver does.
//!
//! | SearchFilters field | Neo4j (Cypher)                  | FalkorDB (here)                         |
//! |---------------------|---------------------------------|-----------------------------------------|
//! | `edge_types`        | `e.name IN $filter_edge_types`  | `e.name IN ['..','..']`                 |
//! | `edge_uuids`        | `e.uuid IN $filter_edge_uuids`  | `e.uuid IN ['..']`                      |
//! | `node_labels`       | `n:L1\|L2 AND m:L1\|L2`          | `ANY(l IN n.labels WHERE l IN [..])`     |
//! | date OR-of-ANDs     | `((f op $p0 AND ..) OR (..))`    | INT compares (epoch millis), inlined    |
//!
//! ## Datetime as INT
//!
//! FalkorDB has no native datetime type — chronicle stores all four temporal
//! fields as epoch-millisecond integers (see [`crate::convert`]). Date filters
//! therefore become **integer comparisons**: a value operator renders as
//! `(e.valid_at >= 1700000000000)` with the millis inlined via
//! [`crate::convert::lit_int`].
//!
//! ## `IS NULL` / `IS NOT NULL`
//!
//! FalkorDB / openCypher supports `field IS NULL` / `field IS NOT NULL`
//! verbatim (verified against `falkordb/falkordb:latest`). A null-class operator
//! renders as `(e.expired_at IS NULL)` and inlines no value.
//!
//! ## Node-label storage deviation
//!
//! Task-1 persistence stores a node's labels in a `labels` **string-array
//! property** (the extra labels beyond `:Entity` are NOT applied as Cypher
//! labels). The label filter therefore tests membership against that array —
//! `ANY(l IN n.labels WHERE l IN ['L1','L2'])` — rather than the Neo4j
//! `n:L1|L2` label-pattern form. This matches how the data is actually stored
//! (and mirrors the SurrealDB driver's `labels CONTAINSANY [..]` predicate).
//!
//! ## Injection safety
//!
//! Every user-supplied VALUE crosses the boundary through the literal encoders
//! in [`crate::convert`]. Node labels are additionally sanitised against
//! `^[A-Za-z0-9_]+$` before being inlined as literals — an invalid label is a
//! hard [`DriverError::Query`], never a silent drop — matching the Neo4j and
//! SurrealDB drivers' policy.

use chronicle_core::driver::DriverError;
use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};

use crate::convert;

/// Strict label sanitiser: `^[A-Za-z0-9_]+$` (matches the Neo4j / SurrealDB
/// drivers). An invalid label is a hard `DriverError::Query`, never a silent drop.
fn sanitize_label(label: &str) -> Result<&str, DriverError> {
    if !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(label)
    } else {
        Err(DriverError::Query(format!(
            "node label \"{label}\" must match ^[A-Za-z0-9_]+$ (labels cannot be parameterized)"
        )))
    }
}

/// Build a `labels`-array membership predicate over the given alias, e.g.
/// `ANY(l IN n.labels WHERE l IN ['L1','L2'])`. Each label is sanitised, then
/// encoded as a literal via [`convert::lit_str`] (defence in depth — sanitised
/// labels are already word-only). `node_labels` must be non-empty.
fn labels_predicate(alias: &str, node_labels: &[String]) -> Result<String, DriverError> {
    let mut sanitized = Vec::with_capacity(node_labels.len());
    for l in node_labels {
        sanitized.push(sanitize_label(l)?.to_string());
    }
    Ok(format!(
        "ANY(l IN {alias}.labels WHERE l IN {})",
        convert::lit_string_list(&sanitized)
    ))
}

/// Render one date condition into a parenthesised Cypher fragment over `field`
/// (e.g. `e.valid_at`). Value operators compare against the inlined epoch-millis
/// integer; `IS NULL` / `IS NOT NULL` inline no value.
///
/// A value operator with a `None` date is a caller error; it renders as a
/// comparison against `null`, which is well-formed and simply never matches
/// (parity with the Neo4j / SurrealDB drivers — no panic).
fn date_condition(field: &str, df: &DateFilter) -> String {
    match df.comparison_operator {
        ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => {
            format!("({} {})", field, df.comparison_operator.as_cypher())
        }
        op => {
            let value = match df.date {
                Some(d) => convert::lit_int(convert::datetime_to_millis(d)),
                None => "null".to_string(),
            };
            format!("({} {} {})", field, op.as_cypher(), value)
        }
    }
}

/// Build the OR-of-ANDs fragment for one date field. Outer groups joined by
/// ` OR `, inner conditions by ` AND `, wrapped in one paren pair:
/// `((c1 AND c2) OR (c3))`. Empty outer vec → `None`.
fn date_field(field: &str, groups: &[Vec<DateFilter>]) -> Option<String> {
    if groups.is_empty() {
        return None;
    }
    let group_frags: Vec<String> = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|df| date_condition(field, df))
                .collect::<Vec<_>>()
                .join(" AND ")
        })
        .collect();
    Some(format!("({})", group_frags.join(" OR ")))
}

/// Edge-side filter fragments (mirrors `edge_filter_fragments`). The fragments
/// are qualified against the edge alias `e` and the endpoint aliases `n`/`m`, so
/// callers must run them where `e`, `n` and `m` are bound. Order matches
/// upstream: `edge_types`, `edge_uuids`, `node_labels` (both endpoints), then
/// `valid_at` / `invalid_at` / `created_at` / `expired_at`.
pub fn edge_filter_fragments(filters: &SearchFilters) -> Result<Vec<String>, DriverError> {
    let mut fragments: Vec<String> = Vec::new();

    if let Some(edge_types) = &filters.edge_types {
        fragments.push(format!(
            "e.name IN {}",
            convert::lit_string_list(edge_types)
        ));
    }

    if let Some(edge_uuids) = &filters.edge_uuids {
        fragments.push(format!(
            "e.uuid IN {}",
            convert::lit_string_list(edge_uuids)
        ));
    }

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        // Both endpoints must carry at least one of the requested labels.
        let n_pred = labels_predicate("n", node_labels)?;
        let m_pred = labels_predicate("m", node_labels)?;
        fragments.push(format!("{n_pred} AND {m_pred}"));
    }

    if let Some(groups) = &filters.valid_at
        && let Some(frag) = date_field("e.valid_at", groups)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.invalid_at
        && let Some(frag) = date_field("e.invalid_at", groups)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.created_at
        && let Some(frag) = date_field("e.created_at", groups)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.expired_at
        && let Some(frag) = date_field("e.expired_at", groups)
    {
        fragments.push(frag);
    }

    Ok(fragments)
}

/// Node-side filter fragments (mirrors `node_filter_fragments`): ONLY
/// `node_labels` applies on the node scope (`n:L1|L2`); edge_types / edge_uuids /
/// date fields are edge-only and ignored here, exactly as upstream does. The
/// fragment is qualified against the node alias `n`.
pub fn node_filter_fragments(filters: &SearchFilters) -> Result<Vec<String>, DriverError> {
    let mut fragments: Vec<String> = Vec::new();

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        fragments.push(labels_predicate("n", node_labels)?);
    }

    Ok(fragments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    #[test]
    fn edge_types_and_uuids_inline_literal_lists() {
        let f = SearchFilters {
            edge_types: Some(vec!["KNOWS".into(), "LIKES".into()]),
            edge_uuids: Some(vec!["u1".into()]),
            ..Default::default()
        };
        let frags = edge_filter_fragments(&f).unwrap();
        assert!(frags.contains(&"e.name IN ['KNOWS','LIKES']".to_string()));
        assert!(frags.contains(&"e.uuid IN ['u1']".to_string()));
    }

    #[test]
    fn node_labels_inlined_on_both_endpoints_for_edges() {
        let f = SearchFilters {
            node_labels: Some(vec!["Entity".into(), "Person".into()]),
            ..Default::default()
        };
        let frags = edge_filter_fragments(&f).unwrap();
        let joined = frags.join(" ");
        assert!(joined.contains("ANY(l IN n.labels WHERE l IN ['Entity','Person'])"));
        assert!(joined.contains("ANY(l IN m.labels WHERE l IN ['Entity','Person'])"));
    }

    #[test]
    fn node_filter_only_labels() {
        let f = SearchFilters {
            node_labels: Some(vec!["Person".into()]),
            edge_types: Some(vec!["X".into()]),
            ..Default::default()
        };
        let frags = node_filter_fragments(&f).unwrap();
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0], "ANY(l IN n.labels WHERE l IN ['Person'])");
    }

    #[test]
    fn invalid_label_is_hard_error() {
        let f = SearchFilters {
            node_labels: Some(vec!["bad-label".into()]),
            ..Default::default()
        };
        assert!(node_filter_fragments(&f).is_err());
        assert!(edge_filter_fragments(&f).is_err());
    }

    #[test]
    fn date_or_of_ands_inlines_int_millis() {
        let t0 = Utc.timestamp_millis_opt(1_700_000_000_000).unwrap();
        let t1 = t0 + Duration::hours(1);
        let t2 = t0 + Duration::hours(2);
        let f = SearchFilters {
            valid_at: Some(vec![
                vec![
                    DateFilter {
                        date: Some(t0),
                        comparison_operator: ComparisonOperator::Gte,
                    },
                    DateFilter {
                        date: Some(t1),
                        comparison_operator: ComparisonOperator::Lt,
                    },
                ],
                vec![DateFilter {
                    date: Some(t2),
                    comparison_operator: ComparisonOperator::Gte,
                }],
            ]),
            ..Default::default()
        };
        let frags = edge_filter_fragments(&f).unwrap();
        let frag = frags.iter().find(|s| s.contains("valid_at")).unwrap();
        // ((e.valid_at >= 1700000000000 AND e.valid_at < 1700003600000) OR (e.valid_at >= 1700007200000))
        assert!(frag.contains("e.valid_at >= 1700000000000"));
        assert!(frag.contains("e.valid_at < 1700003600000"));
        assert!(frag.contains("e.valid_at >= 1700007200000"));
        assert!(frag.contains(" OR "));
        assert!(frag.contains(" AND "));
    }

    #[test]
    fn is_null_maps_to_cypher_null_ops() {
        let f = SearchFilters {
            invalid_at: Some(vec![vec![DateFilter {
                date: None,
                comparison_operator: ComparisonOperator::IsNull,
            }]]),
            expired_at: Some(vec![vec![DateFilter {
                date: None,
                comparison_operator: ComparisonOperator::IsNotNull,
            }]]),
            ..Default::default()
        };
        let frags = edge_filter_fragments(&f).unwrap();
        let joined = frags.join(" ");
        assert!(joined.contains("(e.invalid_at IS NULL)"));
        assert!(joined.contains("(e.expired_at IS NOT NULL)"));
    }
}
