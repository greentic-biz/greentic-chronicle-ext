//! Cypher query constants ported from upstream graphiti @ 34f56e65 (v0.29.1).
//!
//! Sources:
//! - `graphiti_core/models/nodes/node_db_queries.py` (entity/episode save + return)
//! - `graphiti_core/models/edges/edge_db_queries.py` (entity/episodic edge save + return)
//! - `graphiti_core/graph_queries.py` (index DDL, fulltext/vector helpers)
//! - `graphiti_core/search/search_utils.py` (fulltext/similarity search assembly)
//! - `graphiti_core/utils/maintenance/graph_data_operations.py` (retrieve_episodes)
//! - `graphiti_core/edges.py` / `graphiti_core/nodes.py` (get_by_uuid / get_between_nodes)
//! - `graphiti_core/helpers.py::lucene_sanitize` (lucene escaping)
//! - `graphiti_core/helpers.py::validate_group_id` / `validate_group_ids` (input validation)
//!
//! ## Fidelity notes (Task 17 ledger)
//!
//! ### OR-precedence quirk in multi-group lucene scoping (bug-for-bug reproduction)
//!
//! `build_fulltext_query` joins group-scope clauses with bare ` OR ` and appends
//! ` AND (terms)` at the end. For two groups this produces:
//!
//! ```text
//! group_id:"g1" OR group_id:"g2" AND (terms)
//! ```
//!
//! Lucene operator precedence makes `AND` bind tighter than `OR`, so the query is
//! effectively `group_id:"g1" OR (group_id:"g2" AND (terms))`. The intent is
//! `(group_id:"g1" OR group_id:"g2") AND (terms)`, but upstream
//! `search_utils.py::fulltext_query` emits the same unparenthesised form
//! (commit 34f56e65). We reproduce this verbatim for fidelity; the correct fix
//! would parenthesise the OR group — tracked in Task 17 and left for the upstream
//! fix cycle.
//!
//! ### `fact_triple` read-compat gap
//!
//! Upstream v0.29.1 no longer stores the `fact_triple` property on `RELATES_TO`
//! edges, but older graphs may have it. Our RETURN clause omits `e.fact_triple`
//! (matching current upstream schema), meaning older edges with the property will
//! silently drop it on round-trip read. Applications that need `fact_triple`
//! back-compat must project it explicitly or migrate the data. Tracked in Task 17.
//!
//! DEVIATION (props-map adaptation): upstream Neo4j save uses `SET n = $entity_data`
//! where `$entity_data` is a flat property map (core fields + flattened attributes),
//! and sets the vector property in the same statement via
//! `db.create.setNodeVectorProperty`. We pass the same flat map under `node.props`
//! (built in `convert.rs`) and use `SET n += node.props` instead of `SET n = ...`.
//!
//! Why `+=` and not `=`: with `MERGE ... {uuid}` followed by `SET n = node.props`,
//! the props map MUST contain `uuid` (it does — see convert.rs) for the merged node
//! to keep its key. We retain `uuid` in the map so `=` would also be correct, but
//! `+=` is used for the bulk UNWIND form to keep label assignment (`SET n:$(...)`)
//! and the embedding guard composable without clobbering the dynamic labels Neo4j
//! attaches. This matches upstream observable behaviour (a MERGE-by-uuid upsert that
//! overwrites scalar properties and (re)sets the embedding vector).
//!
//! EMBEDDING GUARD: upstream always sets the vector at save time. We must tolerate
//! nodes/edges without an embedding (Phase-1 may save before embedding). The vector
//! procedures `db.create.setNodeVectorProperty` /
//! `db.create.setRelationshipVectorProperty` cannot run inside `FOREACH`, so we guard
//! with a post-`WITH` filter: `WITH n, node WHERE node.name_embedding IS NOT NULL
//! CALL ...`. This drops rows whose embedding is null from the embedding sub-pass
//! (the node/edge is already MERGEd by then, so it is still persisted) and runs the
//! vector call only for rows that have an embedding.

// =====================================================================
// SAVE queries
// =====================================================================

/// Save entity nodes. Ported from `get_entity_node_save_bulk_query` (Neo4j branch).
///
/// Upstream:
/// ```cypher
/// UNWIND $nodes AS node
/// MERGE (n:Entity {uuid: node.uuid})
/// SET n:$(node.labels)
/// SET n = node
/// WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
/// RETURN n.uuid AS uuid
/// ```
///
/// We split into two passes so absent embeddings are tolerated and so the
/// dynamic per-node label list is applied with `apoc`-free Cypher.
/// `node.labels` is applied via the Cypher 5 dynamic-label syntax `SET n:$(node.labels)`.
pub const SAVE_ENTITY_NODES: &str = r#"
    UNWIND $nodes AS node
    MERGE (n:Entity {uuid: node.uuid})
    SET n:$(node.labels)
    SET n += node.props
    WITH n, node
    WHERE node.name_embedding IS NOT NULL
    CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
    RETURN count(*) AS c
"#;

/// Save episodes. Ported from `get_episode_node_save_bulk_query` (Neo4j branch).
///
/// Upstream sets the full property set with `SET n = {...}`; we pass the same
/// fields under `episode.props` and use `SET n += episode.props`.
pub const SAVE_EPISODES: &str = r#"
    UNWIND $episodes AS episode
    MERGE (n:Episodic {uuid: episode.uuid})
    SET n += episode.props
    RETURN n.uuid AS uuid
"#;

/// Save entity edges. Ported from `get_entity_edge_save_bulk_query` (Neo4j branch).
///
/// Upstream:
/// ```cypher
/// UNWIND $entity_edges AS edge
/// MATCH (source:Entity {uuid: edge.source_node_uuid})
/// MATCH (target:Entity {uuid: edge.target_node_uuid})
/// MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
/// SET e = edge
/// WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
/// RETURN edge.uuid AS uuid
/// ```
pub const SAVE_ENTITY_EDGES: &str = r#"
    UNWIND $edges AS edge
    MATCH (source:Entity {uuid: edge.source_node_uuid})
    MATCH (target:Entity {uuid: edge.target_node_uuid})
    MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
    SET e += edge.props
    WITH e, edge
    WHERE edge.fact_embedding IS NOT NULL
    CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
    RETURN count(*) AS c
"#;

/// Save episodic (MENTIONS) edges. Ported from `get_episodic_edge_save_bulk_query`
/// (Neo4j branch).
pub const SAVE_EPISODIC_EDGES: &str = r#"
    UNWIND $episodic_edges AS edge
    MATCH (episode:Episodic {uuid: edge.source_node_uuid})
    MATCH (node:Entity {uuid: edge.target_node_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
    SET
        e.group_id = edge.group_id,
        e.created_at = edge.created_at
    RETURN e.uuid AS uuid
"#;

// =====================================================================
// GET / RETURN queries
// =====================================================================
//
// Column lists are VERBATIM from upstream `get_entity_node_return_query`,
// `EPISODIC_NODE_RETURN`, and `get_entity_edge_return_query` (Neo4j branch).
// `name_embedding` / `fact_embedding` are NOT in the default RETURN upstream
// (loaded separately via `load_*_embedding`); we add them explicitly so a single
// round-trip returns the embedding for roundtrip fidelity in the typed API.

/// Entity-node RETURN clause (verbatim columns from `get_entity_node_return_query`),
/// plus `n.name_embedding`.
pub const ENTITY_NODE_RETURN: &str = r#"
    n.uuid AS uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at,
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes,
    n.name_embedding AS name_embedding
"#;

pub const GET_ENTITY_NODE: &str = r#"
    MATCH (n:Entity {uuid: $uuid})
    RETURN
"#;

pub const GET_ENTITY_NODES_BY_UUIDS: &str = r#"
    MATCH (n:Entity)
    WHERE n.uuid IN $uuids
    RETURN
"#;

/// Episodic-node RETURN clause (verbatim from `EPISODIC_NODE_RETURN`).
pub const EPISODIC_NODE_RETURN: &str = r#"
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
    e.entity_edges AS entity_edges
"#;

pub const GET_EPISODE: &str = r#"
    MATCH (e:Episodic {uuid: $uuid})
    RETURN
"#;

pub const GET_EPISODES_BY_UUIDS: &str = r#"
    MATCH (e:Episodic)
    WHERE e.uuid IN $uuids
    RETURN DISTINCT
"#;

/// Entity-edge RETURN clause (verbatim columns from `get_entity_edge_return_query`,
/// Neo4j branch), plus `e.fact_embedding`.
pub const ENTITY_EDGE_RETURN: &str = r#"
    e.uuid AS uuid,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.name AS name,
    e.fact AS fact,
    e.episodes AS episodes,
    e.expired_at AS expired_at,
    e.valid_at AS valid_at,
    e.invalid_at AS invalid_at,
    properties(e) AS attributes,
    e.fact_embedding AS fact_embedding
"#;

/// Ported from `EntityEdge.get_by_uuid` (Neo4j branch).
pub const GET_ENTITY_EDGE: &str = r#"
    MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
    RETURN
"#;

/// Ported from `EntityEdge.get_between_nodes` (Neo4j branch). Directed
/// (source)-[e:RELATES_TO]->(target) per the amended trait doc.
pub const GET_EDGES_BETWEEN_NODES: &str = r#"
    MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
    RETURN
"#;

// =====================================================================
// RETRIEVE_EPISODES
// =====================================================================
//
// Ported from `retrieve_episodes` (graph_data_operations.py, Neo4j/default branch).
// `$source` is the EpisodeType .name (Python enum name); our convert layer maps
// the Rust enum to the same uppercase-name string the column stores.
// Result is ORDER BY e.valid_at DESC LIMIT $num_episodes; caller reverses to
// chronological order in Rust.

pub const RETRIEVE_EPISODES_BASE: &str = r#"
    MATCH (e:Episodic)
    WHERE e.valid_at <= $reference_time
"#;

pub const RETRIEVE_EPISODES_GROUP_FILTER: &str = "\nAND e.group_id IN $group_ids";
pub const RETRIEVE_EPISODES_SOURCE_FILTER: &str = "\nAND e.source = $source";

pub const RETRIEVE_EPISODES_TAIL: &str = r#"
    ORDER BY e.valid_at DESC
    LIMIT $num_episodes
"#;

// =====================================================================
// SEARCH queries
// =====================================================================

/// Edge fulltext search. Ported from `edge_fulltext_search` (Neo4j branch):
/// `get_relationships_query('edge_name_and_fact', ...)` +
/// `YIELD relationship AS rel, score MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)`
/// + group filter + entity_edge return + ORDER BY score DESC LIMIT.
pub const EDGE_FULLTEXT_SEARCH: &str = r#"
    CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $query, {limit: $limit})
    YIELD relationship AS rel, score
    MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
    WHERE e.group_id IN $group_ids
    WITH e, score, n, m
    RETURN
    e.uuid AS uuid,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.name AS name,
    e.fact AS fact,
    e.episodes AS episodes,
    e.expired_at AS expired_at,
    e.valid_at AS valid_at,
    e.invalid_at AS invalid_at,
    properties(e) AS attributes,
    e.fact_embedding AS fact_embedding
    ORDER BY score DESC
    LIMIT $limit
"#;

/// Edge similarity search. Ported from `edge_similarity_search` (Neo4j branch):
/// `MATCH (n)-[e:RELATES_TO]->(m)` + group filter +
/// `WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $search_vector) AS score
///  WHERE score > $min_score`.
/// Added `e.fact_embedding IS NOT NULL` guard before the cosine call so null
/// embeddings (Phase-1 partial saves) do not raise.
pub const EDGE_SIMILARITY_SEARCH: &str = r#"
    MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
    WHERE e.group_id IN $group_ids AND e.fact_embedding IS NOT NULL
    WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $search_vector) AS score
    WHERE score > $min_score
    RETURN
    e.uuid AS uuid,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.name AS name,
    e.fact AS fact,
    e.episodes AS episodes,
    e.expired_at AS expired_at,
    e.valid_at AS valid_at,
    e.invalid_at AS invalid_at,
    properties(e) AS attributes,
    e.fact_embedding AS fact_embedding
    ORDER BY score DESC
    LIMIT $limit
"#;

/// Node fulltext search. Ported from `node_fulltext_search` (Neo4j branch):
/// `get_nodes_query('node_name_and_summary', '$query', ...)` +
/// `YIELD node AS n, score` + group filter +
/// `WITH n, score ORDER BY score DESC LIMIT $limit RETURN <entity_node_return>`.
pub const NODE_FULLTEXT_SEARCH: &str = r#"
    CALL db.index.fulltext.queryNodes("node_name_and_summary", $query, {limit: $limit})
    YIELD node AS n, score
    WHERE n.group_id IN $group_ids
    WITH n, score
    ORDER BY score DESC
    LIMIT $limit
    RETURN
    n.uuid AS uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at,
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes,
    n.name_embedding AS name_embedding
"#;

/// Node similarity search. Ported from `node_similarity_search` (Neo4j branch):
/// `MATCH (n:Entity)` + group filter +
/// `WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
///  WHERE score > $min_score`.
/// Added `n.name_embedding IS NOT NULL` guard for partial saves.
pub const NODE_SIMILARITY_SEARCH: &str = r#"
    MATCH (n:Entity)
    WHERE n.group_id IN $group_ids AND n.name_embedding IS NOT NULL
    WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
    WHERE score > $min_score
    RETURN
    n.uuid AS uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at,
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes,
    n.name_embedding AS name_embedding
    ORDER BY score DESC
    LIMIT $limit
"#;

// =====================================================================
// INDEX DDL  (Phase-1 subset of graph_queries.py, Neo4j branch)
// =====================================================================

/// Range indices: uuid + group_id range indices for Entity/Episodic/RELATES_TO/
/// MENTIONS, `name_entity_index`, created/valid/expired/invalid_at edge indices,
/// valid_at/created_at episodic. Verbatim names + targets from
/// `get_range_indices` (Neo4j branch), filtered to the Phase-1 label set
/// (Community / Saga / HAS_MEMBER / HAS_EPISODE / NEXT_EPISODE excluded — those
/// node/edge kinds are Phase-4+ and have no save path yet).
pub const RANGE_INDICES: &[&str] = &[
    "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)",
    "CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)",
    "CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)",
    "CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)",
    "CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)",
    "CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)",
    "CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)",
    "CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)",
    "CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
    "CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)",
    "CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)",
    "CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)",
    "CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)",
    "CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)",
    "CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)",
    "CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)",
];

/// Fulltext indices (verbatim from `get_fulltext_indices`, Neo4j branch),
/// limited to the two Phase-1 search surfaces (`node_name_and_summary`,
/// `edge_name_and_fact`). `episode_content` / `community_name` are excluded
/// (no fulltext search path on those yet).
pub const FULLTEXT_INDICES: &[&str] = &[
    "CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS \
     FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]",
    "CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS \
     FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]",
];

/// DROP statements used when `delete_existing = true`. Upstream calls
/// `CALL db.indexes() YIELD name DROP INDEX name` (drop ALL). That procedure is
/// removed in Neo4j 5, so we issue targeted `DROP INDEX <name> IF EXISTS` for the
/// indices we manage instead.
pub fn drop_index_statements() -> Vec<String> {
    let mut out = Vec::new();
    for name in [
        "entity_uuid",
        "episode_uuid",
        "relation_uuid",
        "mention_uuid",
        "entity_group_id",
        "episode_group_id",
        "relation_group_id",
        "mention_group_id",
        "name_entity_index",
        "created_at_entity_index",
        "created_at_episodic_index",
        "valid_at_episodic_index",
        "name_edge_index",
        "created_at_edge_index",
        "expired_at_edge_index",
        "valid_at_edge_index",
        "invalid_at_edge_index",
        "node_name_and_summary",
        "edge_name_and_fact",
    ] {
        out.push(format!("DROP INDEX {name} IF EXISTS"));
    }
    out
}

// =====================================================================
// LUCENE FULLTEXT QUERY BUILDER
// =====================================================================

use chronicle_core::driver::DriverError;

/// Validate a single `group_id` against the upstream-allowed character set.
///
/// Port of `graphiti_core/helpers.py::validate_group_id` @ 34f56e65 (v0.29.1).
/// Upstream pattern: `^[a-zA-Z0-9_-]+$` (ASCII alphanumeric, dash, underscore).
/// Empty / `None` (represented here as an empty string) is treated as valid by
/// upstream (`if not group_id: return True`) — we mirror that by accepting an
/// empty slice element, but in practice callers never produce empty `group_id`
/// strings in normal use.
///
/// On invalid input, upstream raises `GroupIdValidationError` with the message:
/// `group_id "{id}" must contain only alphanumeric characters, dashes, or underscores`.
/// We surface the same message via `DriverError::Query`.
fn validate_group_id(group_id: &str) -> Result<(), DriverError> {
    // Upstream: `if not group_id: return True`
    if group_id.is_empty() {
        return Ok(());
    }
    if group_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(DriverError::Query(format!(
            "group_id \"{group_id}\" must contain only alphanumeric characters, dashes, or underscores"
        )))
    }
}

/// Validate a list of `group_id` values before building a lucene fulltext query.
///
/// Port of `graphiti_core/helpers.py::validate_group_ids` @ 34f56e65 (v0.29.1).
/// `None` (represented here as an empty slice) is accepted without checking.
/// Each individual id is validated via [`validate_group_id`]; the first invalid
/// id short-circuits with `DriverError::Query`.
///
/// This guard must be called before any interpolation of `group_ids` into a
/// lucene query string (see [`build_fulltext_query`]) to prevent lucene-injection.
pub fn validate_group_ids(group_ids: &[String]) -> Result<(), DriverError> {
    for id in group_ids {
        validate_group_id(id)?;
    }
    Ok(())
}

/// Lucene special-character escaping, ported from
/// `graphiti_core/helpers.py::lucene_sanitize`.
///
/// Upstream escapes (each prefixed with a single backslash):
///   `+ - & | ! ( ) { } [ ] ^ " ~ * ? : \ /`
/// plus the single uppercase letters `O R N T A D` (to neutralise the Lucene
/// boolean operators AND / OR / NOT and the range keyword TO by breaking any
/// uppercase token that could form them). This is a deliberate, somewhat blunt
/// upstream choice — we reproduce it verbatim for fidelity.
fn lucene_sanitize(query: &str) -> String {
    let mut out = String::with_capacity(query.len() * 2);
    for c in query.chars() {
        match c {
            '+' | '-' | '&' | '|' | '!' | '(' | ')' | '{' | '}' | '[' | ']' | '^' | '"' | '~'
            | '*' | '?' | ':' | '\\' | '/' | 'O' | 'R' | 'N' | 'T' | 'A' | 'D' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Maximum token count, ported from `MAX_QUERY_LENGTH` in `search_utils.py`.
pub const MAX_QUERY_LENGTH: usize = 128;

/// Build the lucene fulltext query string with group scoping.
///
/// Ported from `fulltext_query` (search_utils.py, Neo4j/default branch):
/// - group scope is `group_id:"<gid>"` clauses joined with ` OR `, then ` AND `
///   prepended before the sanitized term group: `<scope> AND (<lucene>)`.
/// - if `len(lucene.split(' ')) + len(group_ids) >= MAX_QUERY_LENGTH`, returns ""
///   (empty string) — the caller then short-circuits to no results.
///
/// Returns `Ok(None)` for the empty-query short-circuit so the caller can return
/// an empty result set without issuing a Cypher call (matches upstream `if
/// fuzzy_query == '': return []`).
///
/// Returns `Err(DriverError::Query)` if any `group_id` contains characters
/// outside `[a-zA-Z0-9_-]` (mirrors upstream `GroupIdValidationError`).
/// This guard must run before any interpolation of `group_ids` into the lucene
/// string to close the lucene-injection path.
pub fn build_fulltext_query(
    query: &str,
    group_ids: &[String],
) -> Result<Option<String>, DriverError> {
    // Validate before any interpolation — mirrors upstream validate_group_ids()
    // call site in graphiti_core/helpers.py::validate_group_ids @ 34f56e65.
    validate_group_ids(group_ids)?;

    let mut group_ids_filter = String::new();
    for g in group_ids {
        let clause = format!("group_id:\"{g}\"");
        if group_ids_filter.is_empty() {
            group_ids_filter = clause;
        } else {
            group_ids_filter = format!("{group_ids_filter} OR {clause}");
        }
    }
    if !group_ids_filter.is_empty() {
        group_ids_filter.push_str(" AND ");
    }

    let lucene_query = lucene_sanitize(query);

    // Upstream: len(lucene.split(' ')) + len(group_ids) >= MAX_QUERY_LENGTH -> ''
    let token_count = lucene_query.split(' ').count();
    if token_count + group_ids.len() >= MAX_QUERY_LENGTH {
        return Ok(None);
    }

    Ok(Some(format!("{group_ids_filter}({lucene_query})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // lucene_sanitize
    // -----------------------------------------------------------------

    #[test]
    fn lucene_escapes_special_chars() {
        // Each listed special char must be backslash-prefixed.
        assert_eq!(lucene_sanitize("a+b"), r"a\+b");
        assert_eq!(lucene_sanitize("(x)"), r"\(x\)");
        assert_eq!(lucene_sanitize("a:b"), r"a\:b");
        assert_eq!(lucene_sanitize("a/b"), r"a\/b");
        assert_eq!(lucene_sanitize(r"a\b"), r"a\\b");
        assert_eq!(lucene_sanitize("a*b?c"), r"a\*b\?c");
    }

    #[test]
    fn lucene_escapes_boolean_operator_letters() {
        // Upstream escapes the bare uppercase letters O R N T A D.
        assert_eq!(lucene_sanitize("OR"), r"\O\R");
        assert_eq!(lucene_sanitize("AND"), r"\A\N\D");
        assert_eq!(lucene_sanitize("NOT"), r"\N\O\T");
        assert_eq!(lucene_sanitize("TO"), r"\T\O");
        // Lowercase letters are untouched.
        assert_eq!(lucene_sanitize("android"), "android");
    }

    // -----------------------------------------------------------------
    // validate_group_ids / validate_group_id
    // -----------------------------------------------------------------

    #[test]
    fn validate_group_ids_empty_list_passes() {
        // Empty list == upstream `if group_ids is None: return True`.
        assert!(validate_group_ids(&[]).is_ok());
    }

    #[test]
    fn validate_group_ids_valid_ids_pass() {
        let ids: Vec<String> = vec![
            "abc".to_string(),
            "my-group".to_string(),
            "group_1".to_string(),
            "ABC123".to_string(),
            "a-B_9".to_string(),
        ];
        assert!(validate_group_ids(&ids).is_ok());
    }

    #[test]
    fn validate_group_id_with_space_is_rejected() {
        let ids = vec!["bad group".to_string()];
        let err = validate_group_ids(&ids).unwrap_err();
        assert!(
            err.to_string().contains("bad group"),
            "error message should contain the offending id"
        );
        assert!(
            err.to_string().contains("alphanumeric"),
            "error message should mention allowed chars"
        );
    }

    #[test]
    fn validate_group_id_with_quote_is_rejected() {
        // Double-quote is a lucene injection vector inside group_id:"<id>".
        let ids = vec!["g1\" OR group_id:\"g2".to_string()];
        let err = validate_group_ids(&ids).unwrap_err();
        assert!(err.to_string().contains("alphanumeric"));
    }

    #[test]
    fn validate_group_id_with_lucene_special_chars_rejected() {
        for bad in &["g+1", "g:1", "g(1", "g*1", "g?1", "g!1", "g[1", "g^1"] {
            let ids = vec![bad.to_string()];
            assert!(
                validate_group_ids(&ids).is_err(),
                "expected rejection for: {bad}"
            );
        }
    }

    #[test]
    fn validate_group_id_empty_string_passes() {
        // Mirrors upstream `if not group_id: return True`.
        assert!(validate_group_id("").is_ok());
    }

    // -----------------------------------------------------------------
    // build_fulltext_query
    // -----------------------------------------------------------------

    #[test]
    fn build_fulltext_query_no_groups() {
        let q = build_fulltext_query("hello world", &[])
            .expect("valid")
            .unwrap();
        assert_eq!(q, "(hello world)");
    }

    #[test]
    fn build_fulltext_query_single_group() {
        let q = build_fulltext_query("hello", &["g1".to_string()])
            .expect("valid")
            .unwrap();
        assert_eq!(q, "group_id:\"g1\" AND (hello)");
    }

    #[test]
    fn build_fulltext_query_multiple_groups_or_chain() {
        // Reproduces the upstream OR-precedence quirk bug-for-bug (see module fidelity note).
        let q = build_fulltext_query("hi", &["g1".to_string(), "g2".to_string()])
            .expect("valid")
            .unwrap();
        assert_eq!(q, "group_id:\"g1\" OR group_id:\"g2\" AND (hi)");
    }

    #[test]
    fn build_fulltext_query_too_long_returns_none() {
        // 128 single-char tokens -> token_count == 128 >= MAX_QUERY_LENGTH (128).
        let query = (0..128).map(|_| "x").collect::<Vec<_>>().join(" ");
        assert!(build_fulltext_query(&query, &[]).expect("valid").is_none());
    }

    #[test]
    fn build_fulltext_query_special_chars_are_sanitized_inside() {
        let q = build_fulltext_query("a+b", &["g".to_string()])
            .expect("valid")
            .unwrap();
        assert_eq!(q, r#"group_id:"g" AND (a\+b)"#);
    }

    #[test]
    fn build_fulltext_query_rejects_invalid_group_id() {
        let err = build_fulltext_query("hello", &["bad group".to_string()]).unwrap_err();
        assert!(err.to_string().contains("bad group"));
    }

    #[test]
    fn build_fulltext_query_rejects_group_id_with_injection_chars() {
        // Quoted injection attempt — must be rejected before interpolation.
        let err = build_fulltext_query("search", &["legit\" OR group_id:\"other".to_string()])
            .unwrap_err();
        assert!(err.to_string().contains("alphanumeric"));
    }
}
