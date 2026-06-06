//! `SearchFilters` → SurrealQL `WHERE`-fragment builder for the embedded backend.
//!
//! This mirrors the Neo4j driver's `edge_filter_fragments` /
//! `node_filter_fragments` semantics (see
//! `chronicle-driver-neo4j/src/queries.rs`), re-expressed in SurrealQL:
//!
//! | SearchFilters field | Neo4j (Cypher)                  | SurrealQL (here)                       |
//! |---------------------|---------------------------------|----------------------------------------|
//! | `edge_types`        | `e.name IN $filter_edge_types`  | `relation_type IN $filter_edge_types`  |
//! | `edge_uuids`        | `e.uuid IN $filter_edge_uuids`  | `uuid IN $filter_edge_uuids`           |
//! | `node_labels`       | `n:L1\|L2 AND m:L1\|L2`          | `$l IN labels` (CONTAINSANY)           |
//! | date OR-of-ANDs     | `((f op $p0 AND ..) OR (..))`    | same shape, `= NONE` / `!= NONE` nulls |
//!
//! ## Field naming
//!
//! Neo4j stores the edge's relation label as the `name` property and filters
//! `e.name IN $edge_types`. The chronicle `EntityEdge.name` is that same relation
//! label, stored here in the `name` column — so `edge_types` filters on `name`,
//! matching upstream exactly. (The amendment's table sketches `relation_type`; the
//! actual stored column is `name`, which is what we filter, preserving parity.)
//!
//! ## Parameterisation & injection safety
//!
//! Every user-supplied VALUE is bound via a `$param` (never interpolated). The
//! only inlined user strings are node LABELS, which are sanitised against
//! `^[A-Za-z0-9_]+$` (an invalid label is a hard error, never a silent drop) —
//! exactly the Neo4j driver's policy. Date params get globally-unique `$p{n}`
//! names from a single counter so OR-groups across all four date fields never
//! collide.
//!
//! ## `IS NULL` / `IS NOT NULL`
//!
//! SurrealQL has no `IS NULL`; an absent/option field reads as `NONE`. Verified
//! against surrealdb 3.1.3: `field = NONE` / `field != NONE` are the correct
//! predicates (and `IS NOT NONE` parses too). We emit `= NONE` / `!= NONE`.

use surrealdb::types::{SurrealValue, Value};

use chronicle_core::driver::DriverError;
use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};

use crate::convert::to_dt;

/// A built set of WHERE fragments plus the params they reference.
///
/// `.0` = SurrealQL boolean fragments (the caller joins them with ` AND ` and
/// prefixes the right connective); `.1` = `(param_name, Value)` pairs to `.bind()`.
pub struct FilterFragments {
    pub fragments: Vec<String>,
    pub params: Vec<(String, Value)>,
}

/// Strict label sanitiser: `^[A-Za-z0-9_]+$` (matches the Neo4j driver). Labels
/// cannot be parameterised, so an invalid label is a hard `DriverError::Query`.
fn sanitize_label(label: &str) -> Result<&str, DriverError> {
    if !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(label)
    } else {
        Err(DriverError::Query(format!(
            "node label \"{label}\" must match ^[A-Za-z0-9_]+$ (labels cannot be parameterized)"
        )))
    }
}

/// SurrealQL operator string for a value comparison operator. `IsNull` /
/// `IsNotNull` are handled separately (they take no param and render as
/// `= NONE` / `!= NONE`).
fn value_op(op: ComparisonOperator) -> &'static str {
    match op {
        ComparisonOperator::Eq => "=",
        ComparisonOperator::Neq => "!=",
        ComparisonOperator::Gt => ">",
        ComparisonOperator::Lt => "<",
        ComparisonOperator::Gte => ">=",
        ComparisonOperator::Lte => "<=",
        // Unreachable for value ops; kept total for exhaustiveness.
        ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => "=",
    }
}

/// Render one date condition into a parenthesised SurrealQL fragment over `field`.
///
/// Value operators append `$p{n}` and bind the date (as a surreal `Datetime`);
/// `IS NULL` / `IS NOT NULL` map to `field = NONE` / `field != NONE` and bind
/// nothing (mirroring the Neo4j null-op handling).
fn date_condition(
    field: &str,
    df: &DateFilter,
    counter: &mut usize,
    params: &mut Vec<(String, Value)>,
) -> String {
    match df.comparison_operator {
        ComparisonOperator::IsNull => format!("({field} = NONE)"),
        ComparisonOperator::IsNotNull => format!("({field} != NONE)"),
        op => {
            let name = format!("p{}", *counter);
            *counter += 1;
            // A value op with a None date is a caller error; bind NONE so the
            // comparison stays well-formed (and simply never matches), never
            // panicking — same tolerance as the Neo4j driver.
            let value = match df.date {
                Some(d) => to_dt(d).into_value(),
                None => Value::None,
            };
            params.push((name.clone(), value));
            format!("({field} {} ${name})", value_op(op))
        }
    }
}

/// Build the OR-of-ANDs fragment for one date field. Outer groups joined by
/// ` OR `, inner conditions by ` AND `, wrapped in one paren pair:
/// `((c1 AND c2) OR (c3))`. Empty outer vec → `None`.
fn date_field(
    field: &str,
    groups: &[Vec<DateFilter>],
    counter: &mut usize,
    params: &mut Vec<(String, Value)>,
) -> Option<String> {
    if groups.is_empty() {
        return None;
    }
    let group_frags: Vec<String> = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|df| date_condition(field, df, counter, params))
                .collect::<Vec<_>>()
                .join(" AND ")
        })
        .collect();
    Some(format!("({})", group_frags.join(" OR ")))
}

/// Edge-side filter fragments + params (mirrors `edge_filter_fragments`).
///
/// Order: `edge_types`, `edge_uuids`, `node_labels` (applied to BOTH endpoint
/// label arrays `in.labels` / `out.labels`), then the four date fields. The
/// fragments are unqualified (they apply to the `relates_to` row in scope), so
/// callers must run them in a context where `name` / `uuid` / `valid_at` etc.
/// resolve to the edge row.
pub fn edge_filter_fragments(filters: &SearchFilters) -> Result<FilterFragments, DriverError> {
    let mut fragments: Vec<String> = Vec::new();
    let mut params: Vec<(String, Value)> = Vec::new();
    let mut counter: usize = 0;

    if let Some(edge_types) = &filters.edge_types {
        fragments.push("name IN $filter_edge_types".to_string());
        params.push((
            "filter_edge_types".to_string(),
            edge_types.clone().into_value(),
        ));
    }

    if let Some(edge_uuids) = &filters.edge_uuids {
        fragments.push("uuid IN $filter_edge_uuids".to_string());
        params.push((
            "filter_edge_uuids".to_string(),
            edge_uuids.clone().into_value(),
        ));
    }

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        for l in node_labels {
            sanitize_label(l)?;
        }
        // Both endpoints must carry at least one of the requested labels. The
        // endpoint records (`in`/`out`) expose their `labels` array; we require
        // the requested set to intersect each (CONTAINSANY).
        let arr = labels_array_literal(node_labels);
        fragments.push(format!(
            "(in.labels CONTAINSANY {arr} AND out.labels CONTAINSANY {arr})"
        ));
    }

    if let Some(groups) = &filters.valid_at
        && let Some(frag) = date_field("valid_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.invalid_at
        && let Some(frag) = date_field("invalid_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.created_at
        && let Some(frag) = date_field("created_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.expired_at
        && let Some(frag) = date_field("expired_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }

    Ok(FilterFragments { fragments, params })
}

/// Node-side filter fragments + params (mirrors `node_filter_fragments`): ONLY
/// `node_labels` applies on the node scope; edge_types / edge_uuids / date fields
/// are edge-only and ignored here, exactly as upstream does. The label predicate
/// is `labels CONTAINSANY [..]` over the entity row in scope.
pub fn node_filter_fragments(filters: &SearchFilters) -> Result<FilterFragments, DriverError> {
    let mut fragments: Vec<String> = Vec::new();
    let params: Vec<(String, Value)> = Vec::new();

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        for l in node_labels {
            sanitize_label(l)?;
        }
        let arr = labels_array_literal(node_labels);
        fragments.push(format!("labels CONTAINSANY {arr}"));
    }

    Ok(FilterFragments { fragments, params })
}

/// Render a list of ALREADY-SANITISED labels as a SurrealQL string-array literal
/// (`['A', 'B']`). Each label has passed [`sanitize_label`] (word chars only), so
/// there is nothing to escape — no quote, backslash, or bracket can appear.
fn labels_array_literal(labels: &[String]) -> String {
    let items: Vec<String> = labels.iter().map(|l| format!("'{l}'")).collect();
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    #[test]
    fn edge_types_and_uuids_bind_params() {
        let f = SearchFilters {
            edge_types: Some(vec!["KNOWS".into(), "LIKES".into()]),
            edge_uuids: Some(vec!["u1".into()]),
            ..Default::default()
        };
        let built = edge_filter_fragments(&f).unwrap();
        assert!(
            built
                .fragments
                .contains(&"name IN $filter_edge_types".to_string())
        );
        assert!(
            built
                .fragments
                .contains(&"uuid IN $filter_edge_uuids".to_string())
        );
        let names: Vec<&str> = built.params.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"filter_edge_types"));
        assert!(names.contains(&"filter_edge_uuids"));
    }

    #[test]
    fn node_labels_inlined_on_both_endpoints_for_edges() {
        let f = SearchFilters {
            node_labels: Some(vec!["Entity".into(), "Person".into()]),
            ..Default::default()
        };
        let built = edge_filter_fragments(&f).unwrap();
        let frag = built.fragments.join(" ");
        assert!(frag.contains("in.labels CONTAINSANY ['Entity', 'Person']"));
        assert!(frag.contains("out.labels CONTAINSANY ['Entity', 'Person']"));
    }

    #[test]
    fn node_filter_only_labels() {
        let f = SearchFilters {
            node_labels: Some(vec!["Person".into()]),
            edge_types: Some(vec!["X".into()]),
            ..Default::default()
        };
        let built = node_filter_fragments(&f).unwrap();
        assert_eq!(built.fragments.len(), 1);
        assert!(built.fragments[0].contains("labels CONTAINSANY ['Person']"));
        assert!(built.params.is_empty());
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
    fn date_or_of_ands_globally_unique_params() {
        let now = Utc::now();
        let f = SearchFilters {
            valid_at: Some(vec![
                vec![
                    DateFilter {
                        date: Some(now),
                        comparison_operator: ComparisonOperator::Gte,
                    },
                    DateFilter {
                        date: Some(now + Duration::hours(1)),
                        comparison_operator: ComparisonOperator::Lt,
                    },
                ],
                vec![DateFilter {
                    date: Some(now + Duration::hours(2)),
                    comparison_operator: ComparisonOperator::Gte,
                }],
            ]),
            ..Default::default()
        };
        let built = edge_filter_fragments(&f).unwrap();
        let frag = built
            .fragments
            .iter()
            .find(|s| s.contains("valid_at"))
            .unwrap();
        // ((valid_at >= $p0 AND valid_at < $p1) OR (valid_at >= $p2))
        assert!(frag.contains("$p0"));
        assert!(frag.contains("$p1"));
        assert!(frag.contains("$p2"));
        assert!(frag.contains(" OR "));
        assert!(frag.contains(" AND "));
        assert_eq!(built.params.len(), 3);
    }

    #[test]
    fn is_null_maps_to_none_no_param() {
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
        let built = edge_filter_fragments(&f).unwrap();
        let frag = built.fragments.join(" ");
        assert!(frag.contains("(invalid_at = NONE)"));
        assert!(frag.contains("(expired_at != NONE)"));
        assert!(built.params.is_empty(), "null ops bind no params");
    }
}
