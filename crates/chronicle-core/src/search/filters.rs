// Ported from graphiti_core/search/search_filters.py @ 34f56e65.
//
// Upstream structural decisions verified against source:
//   - SearchFilters.valid_at / invalid_at / created_at / expired_at are
//     `list[list[DateFilter]]` in Python → `Vec<Vec<DateFilter>>` here.
//     OUTER vec = OR groups; INNER vec = AND conditions within each group.
//     This models `((c1 AND c2) OR (c3))` when building WHERE clauses.
//   - property_filters field is ported but NOT wired into query construction.
//     Upstream: the field exists on SearchFilters at this commit but is not
//     referenced in any WHERE builder function. Ported for structural parity;
//     a WHERE builder will be added once upstream activates the field.
//   - DateFilter.date is `datetime | None` in Python → `Option<DateTime<Utc>>`.
//     Only IS NULL / IS NOT NULL operators use a None date; value operators require
//     Some. The caller is responsible for supplying a valid date for value operators.
//   - PropertyFilter.property_value is `str | int | float | None` in Python.
//     Represented as `serde_json::Value` here for open-ended support (covers all
//     JSON scalar types including null).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Comparison operators for date and property filters.
///
/// Upstream: `ComparisonOperator` enum in `search_filters.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOperator {
    /// `=` — exact match.
    Eq,
    /// `<>` — not equal.
    Neq,
    /// `>` — strictly greater.
    Gt,
    /// `<` — strictly less.
    Lt,
    /// `>=` — greater than or equal.
    Gte,
    /// `<=` — less than or equal.
    Lte,
    /// `IS NULL` — field has no value.
    IsNull,
    /// `IS NOT NULL` — field has a value.
    IsNotNull,
}

impl ComparisonOperator {
    /// Returns the exact Cypher operator string used in WHERE clauses.
    ///
    /// These strings match the upstream `ComparisonOperator.value` enum strings
    /// verbatim (search_filters.py). They are intentionally returned as `&str`
    /// (not `String`) since they are compile-time constants.
    pub fn as_cypher(&self) -> &'static str {
        match self {
            ComparisonOperator::Eq => "=",
            ComparisonOperator::Neq => "<>",
            ComparisonOperator::Gt => ">",
            ComparisonOperator::Lt => "<",
            ComparisonOperator::Gte => ">=",
            ComparisonOperator::Lte => "<=",
            ComparisonOperator::IsNull => "IS NULL",
            ComparisonOperator::IsNotNull => "IS NOT NULL",
        }
    }
}

/// Single date condition within a filter group.
///
/// Upstream: `DateFilter` model in `search_filters.py`.
/// `date` is `None` when the operator is `IS NULL` or `IS NOT NULL`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateFilter {
    /// The timestamp to compare against. Must be `Some` for value operators
    /// (`Eq`, `Neq`, `Gt`, `Lt`, `Gte`, `Lte`); may be `None` for `IS NULL` /
    /// `IS NOT NULL`.
    pub date: Option<DateTime<Utc>>,
    /// The comparison operator to apply.
    pub comparison_operator: ComparisonOperator,
}

/// Property-level predicate filter.
///
/// Upstream: `PropertyFilter` model in `search_filters.py`.
///
/// # Porting note
/// This field is ported from upstream for structural parity but is **not applied**
/// in query construction at this commit (34f56e65). Upstream's WHERE builder
/// functions (`edge_search_filter_query_constructor`, `node_search_filter_query_constructor`)
/// do not reference `property_filters`. This field will be wired up once upstream
/// activates the corresponding WHERE clause logic.
// ported, not applied in query construction — upstream dead field at 34f56e65
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFilter {
    /// Name of the graph property to filter on.
    pub property_name: String,
    /// Value to compare. Uses `serde_json::Value` to cover `str | int | float | null`
    /// (upstream `str | int | float | None`).
    pub property_value: JsonValue,
    /// Comparison operator.
    pub comparison_operator: ComparisonOperator,
}

/// Composite filter applied to node or edge search queries.
///
/// Upstream: `SearchFilters` model in `search_filters.py` @ 34f56e65.
///
/// ## Date filter groups (OR-of-ANDs)
/// Fields `valid_at`, `invalid_at`, `created_at`, `expired_at` use a two-level
/// `Vec<Vec<DateFilter>>` structure:
/// - Outer vec: OR groups — at least one group must match.
/// - Inner vec: AND conditions — all conditions in the group must match.
///
/// Example: `[[gte(t1), lt(t2)], [gte(t3)]]` → `((e.valid_at >= t1 AND e.valid_at < t2) OR e.valid_at >= t3)`.
///
/// ## Default
/// All fields default to `None` (no filtering applied).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Filter entities to those matching at least one of these node labels.
    /// For edges: applied to both source (`n`) and target (`m`) nodes.
    /// Upstream: validated with `validate_node_labels` in Python; sanitization
    /// responsibility is on the Cypher query builder.
    pub node_labels: Option<Vec<String>>,

    /// Filter edges to those whose `name` property is in this list.
    /// Upstream: `e.name IN $edge_types`.
    pub edge_types: Option<Vec<String>>,

    /// Date filter on the edge `valid_at` field.
    /// Outer = OR groups; inner = AND conditions (see struct-level doc).
    pub valid_at: Option<Vec<Vec<DateFilter>>>,

    /// Date filter on the edge `invalid_at` field.
    /// Outer = OR groups; inner = AND conditions (see struct-level doc).
    pub invalid_at: Option<Vec<Vec<DateFilter>>>,

    /// Date filter on the edge `created_at` field.
    /// Outer = OR groups; inner = AND conditions (see struct-level doc).
    pub created_at: Option<Vec<Vec<DateFilter>>>,

    /// Date filter on the edge `expired_at` field.
    /// Outer = OR groups; inner = AND conditions (see struct-level doc).
    pub expired_at: Option<Vec<Vec<DateFilter>>>,

    /// Restrict results to edges whose UUID is in this list.
    /// Upstream: `e.uuid IN $edge_uuids`.
    pub edge_uuids: Option<Vec<String>>,

    /// Property-level predicates.
    ///
    // ported, not applied in query construction — upstream dead field at 34f56e65
    pub property_filters: Option<Vec<PropertyFilter>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ComparisonOperator::as_cypher exact Cypher strings ──────────────────

    #[test]
    fn comparison_operator_as_cypher_eq() {
        assert_eq!(ComparisonOperator::Eq.as_cypher(), "=");
    }

    #[test]
    fn comparison_operator_as_cypher_neq() {
        assert_eq!(ComparisonOperator::Neq.as_cypher(), "<>");
    }

    #[test]
    fn comparison_operator_as_cypher_gt() {
        assert_eq!(ComparisonOperator::Gt.as_cypher(), ">");
    }

    #[test]
    fn comparison_operator_as_cypher_lt() {
        assert_eq!(ComparisonOperator::Lt.as_cypher(), "<");
    }

    #[test]
    fn comparison_operator_as_cypher_gte() {
        assert_eq!(ComparisonOperator::Gte.as_cypher(), ">=");
    }

    #[test]
    fn comparison_operator_as_cypher_lte() {
        assert_eq!(ComparisonOperator::Lte.as_cypher(), "<=");
    }

    #[test]
    fn comparison_operator_as_cypher_is_null() {
        assert_eq!(ComparisonOperator::IsNull.as_cypher(), "IS NULL");
    }

    #[test]
    fn comparison_operator_as_cypher_is_not_null() {
        assert_eq!(ComparisonOperator::IsNotNull.as_cypher(), "IS NOT NULL");
    }

    // ── SearchFilters default is all-None ───────────────────────────────────

    #[test]
    fn search_filters_default_all_none() {
        let f = SearchFilters::default();
        assert!(f.node_labels.is_none());
        assert!(f.edge_types.is_none());
        assert!(f.valid_at.is_none());
        assert!(f.invalid_at.is_none());
        assert!(f.created_at.is_none());
        assert!(f.expired_at.is_none());
        assert!(f.edge_uuids.is_none());
        assert!(f.property_filters.is_none());
    }

    // ── DateFilter construction ─────────────────────────────────────────────

    #[test]
    fn date_filter_with_null_operator_has_no_date() {
        let f = DateFilter {
            date: None,
            comparison_operator: ComparisonOperator::IsNull,
        };
        assert!(f.date.is_none());
        assert_eq!(f.comparison_operator.as_cypher(), "IS NULL");
    }

    #[test]
    fn date_filter_with_value_operator_has_date() {
        let now = Utc::now();
        let f = DateFilter {
            date: Some(now),
            comparison_operator: ComparisonOperator::Gte,
        };
        assert!(f.date.is_some());
        assert_eq!(f.comparison_operator.as_cypher(), ">=");
    }

    // ── PropertyFilter construction ─────────────────────────────────────────

    #[test]
    fn property_filter_accepts_json_value_types() {
        let string_filter = PropertyFilter {
            property_name: "status".to_string(),
            property_value: serde_json::json!("active"),
            comparison_operator: ComparisonOperator::Eq,
        };
        assert_eq!(string_filter.property_name, "status");

        let numeric_filter = PropertyFilter {
            property_name: "score".to_string(),
            property_value: serde_json::json!(42),
            comparison_operator: ComparisonOperator::Gt,
        };
        assert_eq!(numeric_filter.comparison_operator.as_cypher(), ">");

        let null_filter = PropertyFilter {
            property_name: "deleted_at".to_string(),
            property_value: serde_json::Value::Null,
            comparison_operator: ComparisonOperator::IsNull,
        };
        assert!(null_filter.property_value.is_null());
    }

    // ── SearchFilters OR-of-ANDs structure round-trip via serde ────────────

    #[test]
    fn search_filters_date_groups_round_trip_serde() {
        let now = Utc::now();
        let f = SearchFilters {
            valid_at: Some(vec![
                // AND group: gte(now) AND lt(now+1h)
                vec![
                    DateFilter {
                        date: Some(now),
                        comparison_operator: ComparisonOperator::Gte,
                    },
                    DateFilter {
                        date: Some(now + chrono::Duration::hours(1)),
                        comparison_operator: ComparisonOperator::Lt,
                    },
                ],
                // OR group: single gte
                vec![DateFilter {
                    date: Some(now + chrono::Duration::hours(2)),
                    comparison_operator: ComparisonOperator::Gte,
                }],
            ]),
            ..Default::default()
        };
        let json = serde_json::to_string(&f).expect("serialize");
        let back: SearchFilters = serde_json::from_str(&json).expect("deserialize");
        let groups = back.valid_at.unwrap();
        assert_eq!(groups.len(), 2, "two OR groups");
        assert_eq!(groups[0].len(), 2, "first group has two AND conditions");
        assert_eq!(groups[1].len(), 1, "second group has one condition");
    }
}
