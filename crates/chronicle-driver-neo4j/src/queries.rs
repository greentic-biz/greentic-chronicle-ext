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
    WHERE e.group_id IN $group_ids{filters}
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
    WHERE e.group_id IN $group_ids AND e.fact_embedding IS NOT NULL{filters}
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
    WHERE n.group_id IN $group_ids{filters}
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
    WHERE n.group_id IN $group_ids AND n.name_embedding IS NOT NULL{filters}
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
// PHASE-2 SEARCH PRIMITIVES (BFS / episode fulltext / embeddings loaders /
//                           node-distance / episode-mentions)
// =====================================================================
//
// Ported from `graphiti_core/search/search_utils.py` (Neo4j/default branch) @
// 34f56e65. Each constant or builder documents the upstream source block.

/// Episode fulltext search. Ported from `episode_fulltext_search` (Neo4j branch):
/// `get_nodes_query('episode_content', '$query', ...)` +
/// `YIELD node AS episode, score MATCH (e:Episodic) WHERE e.uuid = episode.uuid`
/// + optional group filter + `EPISODIC_NODE_RETURN` + ORDER BY score DESC LIMIT.
///
/// The `{group_filter}` placeholder is replaced at call time with
/// `\nAND e.group_id IN $group_ids` (only when group_ids is non-empty), exactly
/// mirroring upstream's `group_filter_query` concatenation. Note the YIELD-node
/// then MATCH-rejoin pattern: the fulltext hit yields an `Episodic` node, then we
/// re-`MATCH` it by uuid so the RETURN can project the canonical column list.
pub const EPISODE_FULLTEXT_SEARCH_HEAD: &str = r#"
    CALL db.index.fulltext.queryNodes("episode_content", $query, {limit: $limit})
    YIELD node AS episode, score
    MATCH (e:Episodic)
    WHERE e.uuid = episode.uuid"#;

pub const EPISODE_FULLTEXT_GROUP_FILTER: &str = "\nAND e.group_id IN $group_ids";

pub const EPISODE_FULLTEXT_SEARCH_TAIL: &str = r#"
    RETURN
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
    e.entity_edges AS entity_edges
    ORDER BY score DESC
    LIMIT $limit
"#;

/// Node-embedding loader. Ported from `get_embeddings_for_nodes` (Neo4j/default
/// branch). Upstream omits the `IS NOT NULL` guard and post-filters null rows in
/// Python (`if uuid is not None and embedding is not None`); we push the null
/// guard into Cypher so absent-embedding nodes are simply not returned — the
/// observable result (omit-when-missing) is identical, with one less row over the
/// wire.
pub const GET_NODE_EMBEDDINGS: &str = r#"
    MATCH (n:Entity)
    WHERE n.uuid IN $uuids AND n.name_embedding IS NOT NULL
    RETURN DISTINCT n.uuid AS uuid, n.name_embedding AS embedding
"#;

/// Edge-embedding loader. Ported from `get_embeddings_for_edges` (Neo4j/default
/// branch): UNDIRECTED `(n:Entity)-[e:RELATES_TO]-(m:Entity)` match. Same null
/// guard rationale as [`GET_NODE_EMBEDDINGS`].
pub const GET_EDGE_EMBEDDINGS: &str = r#"
    MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
    WHERE e.uuid IN $uuids AND e.fact_embedding IS NOT NULL
    RETURN DISTINCT e.uuid AS uuid, e.fact_embedding AS embedding
"#;

/// 1-HOP UNDIRECTED adjacency to a center node (R7 / `node_distance_reranker`
/// Cypher). Verbatim from upstream: `UNWIND $node_uuids AS node_uuid MATCH
/// (center:Entity {uuid:$center_uuid})-[:RELATES_TO]-(n:Entity {uuid:node_uuid})
/// RETURN 1 AS score, node_uuid AS uuid`. We only need the adjacent uuids back
/// (the reranker maths — inf handling, center prepend, 1/score — lives in
/// chronicle-core::search::rerank), so we return just `node_uuid AS uuid`.
pub const NODES_CONNECTED_TO_CENTER: &str = r#"
    UNWIND $node_uuids AS node_uuid
    MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
    RETURN node_uuid AS uuid
"#;

/// MENTIONS in-degree per node uuid (R8 / `episode_mentions_reranker` Cypher).
/// Verbatim from upstream: `UNWIND $node_uuids AS node_uuid MATCH
/// (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid:node_uuid}) RETURN count(*)
/// AS score, n.uuid AS uuid`. Nodes with zero mentions produce no row (upstream
/// fills them with `inf` in Python); the caller (rerank) handles the missing →
/// inf mapping.
pub const EPISODE_MENTION_COUNTS: &str = r#"
    UNWIND $node_uuids AS node_uuid
    MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
    RETURN count(*) AS score, n.uuid AS uuid
"#;

/// Node-BFS depth bounds. Upstream var-length pattern is
/// `[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]`; the `{bfs_max_depth}` is inlined
/// into the query STRING (Cypher does not allow a `$param` as a var-length upper
/// bound — `*1..$depth` is a syntax error). We therefore clamp the requested
/// depth into `[MIN_BFS_DEPTH, MAX_BFS_DEPTH]` and format it as a literal integer
/// in [`node_bfs_query`] / [`edge_bfs_query`]. The clamp is the injection guard:
/// a usize that has passed through `MIN..=MAX` can only render as digits.
pub const MIN_BFS_DEPTH: usize = 1;
pub const MAX_BFS_DEPTH: usize = 10;

/// Clamp a requested BFS depth into the inline-safe range `[1, 10]`.
fn clamp_bfs_depth(depth: usize) -> usize {
    depth.clamp(MIN_BFS_DEPTH, MAX_BFS_DEPTH)
}

/// Build the node-BFS Cypher (R9 / `node_bfs_search`, Neo4j/default branch).
///
/// Upstream shape:
/// ```cypher
/// UNWIND $bfs_origin_node_uuids AS origin_uuid
/// MATCH (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{depth}]->(n:Entity)
/// WHERE n.group_id = origin.group_id {filters}
/// RETURN <entity_node_return> LIMIT $limit
/// ```
/// The `origin` node is label-free (it may be an `Entity` or an `Episodic`, so a
/// MENTIONS hop from an episode origin is included). Filter fragments and the
/// optional `n.group_id IN $group_ids AND origin.group_id IN $group_ids` clauses
/// are appended with a leading ` AND ` (the base WHERE always has
/// `n.group_id = origin.group_id`).
pub fn node_bfs_query(depth: usize, filter_fragments: &[String], with_group_ids: bool) -> String {
    let depth = clamp_bfs_depth(depth);
    let mut tail = String::new();
    if with_group_ids {
        tail.push_str(" AND n.group_id IN $group_ids AND origin.group_id IN $group_ids");
    }
    for frag in filter_fragments {
        tail.push_str(" AND ");
        tail.push_str(frag);
    }
    format!(
        r#"
    UNWIND $bfs_origin_node_uuids AS origin_uuid
    MATCH (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{depth}]->(n:Entity)
    WHERE n.group_id = origin.group_id{tail}
    RETURN
    n.uuid AS uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at,
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes,
    n.name_embedding AS name_embedding
    LIMIT $limit
"#
    )
}

/// Build the edge-BFS Cypher (R9 / `edge_bfs_search`, Neo4j/default branch).
///
/// Upstream shape:
/// ```cypher
/// UNWIND $bfs_origin_node_uuids AS origin_uuid
/// MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{depth}]->(:Entity)
/// UNWIND relationships(path) AS rel
/// MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)
/// {filters}
/// RETURN DISTINCT <entity_edge_return> LIMIT $limit
/// ```
/// The second MATCH is UNDIRECTED (`-[e:RELATES_TO]-`), so `n`/`m` are not
/// necessarily source/target; the RETURN still projects `n.uuid` /`m.uuid` as
/// source/target exactly as upstream `get_entity_edge_return_query` does. Filter
/// fragments (incl. the optional `e.group_id IN $group_ids`) are joined into a
/// single ` WHERE ... AND ...` block, matching upstream's
/// `' WHERE ' + ' AND '.join(filter_queries)`.
pub fn edge_bfs_query(depth: usize, filter_fragments: &[String]) -> String {
    let depth = clamp_bfs_depth(depth);
    let mut where_block = String::new();
    if !filter_fragments.is_empty() {
        where_block.push_str("\n    WHERE ");
        where_block.push_str(&filter_fragments.join(" AND "));
    }
    format!(
        r#"
    UNWIND $bfs_origin_node_uuids AS origin_uuid
    MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{depth}]->(:Entity)
    UNWIND relationships(path) AS rel
    MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity){where_block}
    RETURN DISTINCT
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
    LIMIT $limit
"#
    )
}

// =====================================================================
// SEARCHFILTERS → Cypher WHERE fragments  (R10 / search_filters.py)
// =====================================================================
//
// Ported from `graphiti_core/search/search_filters.py`
// (`edge_search_filter_query_constructor`, `node_search_filter_query_constructor`,
// `date_filter_query_constructor`) @ 34f56e65, Neo4j/default branch.
//
// Two deliberate divergences from upstream, both documented:
//
//   1. Node-label sanitization. Upstream relies on a pydantic field-validator
//      (`validate_node_labels`) running at construction time and inlines the
//      labels unchecked in the WHERE builder. Our `SearchFilters` is a plain
//      struct with no validator, so we sanitize EACH label here against the
//      strict `^[A-Za-z0-9_]+$` set before inlining (labels CANNOT be Cypher
//      parameters). An invalid label is a hard `DriverError::Query`, not a silent
//      drop — a bad label must never reach the query string.
//
//   2. Date-param naming. Upstream names date params by the INNER index only
//      (`$valid_at_{j}`), which COLLIDES across OR-groups (group 1's `valid_at_0`
//      is overwritten by group 2's `valid_at_0`) AND across fields when the same
//      driver call carries more than one date field. Upstream gets away with it
//      because its test fixtures never combine OR-groups with repeated inner
//      positions. The plan (R10) requires correct OR-of-ANDs windows with
//      multiple groups, so we assign every value-bearing date condition a GLOBALLY
//      unique param name `$p{n}` from a single monotonic counter shared across all
//      four date fields. This fixes the upstream collision while emitting the
//      identical Cypher SHAPE (`((e.valid_at >= $p0 AND e.valid_at < $p1) OR (...))`).

use neo4rs::BoltType;

use crate::convert::datetime_to_bolt;
use chronicle_core::search::filters::{ComparisonOperator, DateFilter, SearchFilters};

/// WHERE fragments plus their bound params, as produced by the filter builders.
/// `.0` = Cypher fragment strings (joined with ` AND ` by the caller);
/// `.1` = `(param_name, bolt_value)` pairs to bind onto the query.
pub type FilterFragments = (Vec<String>, Vec<(String, BoltType)>);

/// Strict label sanitizer: `^[A-Za-z0-9_]+$` (plan R10). Returns the label
/// unchanged when valid, else a `DriverError::Query`.
///
/// Divergence note: upstream `SAFE_CYPHER_IDENTIFIER_PATTERN` is
/// `^[A-Za-z_][A-Za-z0-9_]*$` (a leading digit is rejected). The plan pins the
/// slightly looser `^[A-Za-z0-9_]+$` (leading digit allowed). We follow the plan;
/// the looser set is still injection-safe (only word characters survive, so no
/// `:`, space, `|`, `}`, backtick, etc. can break out of the `n:Label` position),
/// and a leading-digit label is harmless in a `n:0Foo` expression.
fn sanitize_label(label: &str) -> Result<&str, DriverError> {
    if !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(label)
    } else {
        Err(DriverError::Query(format!(
            "node label \"{label}\" must match ^[A-Za-z0-9_]+$ (labels cannot be parameterized)"
        )))
    }
}

/// Render one date condition into a parenthesised Cypher fragment, mirroring
/// upstream `date_filter_query_constructor`.
///
/// For value operators it appends `<param_name>` and pushes the bound date onto
/// `params` under that name; for `IS NULL` / `IS NOT NULL` it takes NO param
/// (matching upstream, which skips `filter_params['..'] = ..` for the null ops).
/// `counter` is the shared monotonic source of globally-unique `$p{n}` names.
fn date_condition_fragment(
    value_name: &str,
    df: &DateFilter,
    counter: &mut usize,
    params: &mut Vec<(String, BoltType)>,
) -> String {
    match df.comparison_operator {
        ComparisonOperator::IsNull | ComparisonOperator::IsNotNull => {
            format!("({} {})", value_name, df.comparison_operator.as_cypher())
        }
        op => {
            let param_name = format!("p{}", *counter);
            *counter += 1;
            // A value operator with a None date is a caller error; bind NULL so the
            // comparison is well-formed (it will simply never match), rather than
            // panicking. Upstream would bind Python `None` here too.
            let bolt = match df.date {
                Some(d) => datetime_to_bolt(d),
                None => BoltType::Null(neo4rs::BoltNull),
            };
            params.push((param_name.clone(), bolt));
            format!("({} {} ${})", value_name, op.as_cypher(), param_name)
        }
    }
}

/// Build the OR-of-ANDs fragment for one date field (`valid_at`, `invalid_at`,
/// `created_at`, `expired_at`). Mirrors upstream's nested loop exactly:
/// outer groups joined by ` OR `, inner conditions joined by ` AND `, the whole
/// thing wrapped in a single pair of parentheses:
/// `((c1 AND c2) OR (c3))`.
///
/// Empty outer vec → no fragment (returns None). `counter` threads through so
/// param names stay globally unique across ALL date fields in the same call.
fn date_field_fragment(
    value_name: &str,
    groups: &[Vec<DateFilter>],
    counter: &mut usize,
    params: &mut Vec<(String, BoltType)>,
) -> Option<String> {
    if groups.is_empty() {
        return None;
    }
    let mut group_frags = Vec::with_capacity(groups.len());
    for group in groups {
        let and_frags: Vec<String> = group
            .iter()
            .map(|df| date_condition_fragment(value_name, df, counter, params))
            .collect();
        group_frags.push(and_frags.join(" AND "));
    }
    Some(format!("({})", group_frags.join(" OR ")))
}

/// Build the edge-side WHERE fragments + bound params from a [`SearchFilters`]
/// (R10). Order matches upstream `edge_search_filter_query_constructor`:
/// `edge_types`, `edge_uuids`, `node_labels` (both endpoints), then
/// `valid_at` / `invalid_at` / `created_at` / `expired_at`.
///
/// `edge_types` / `edge_uuids` bind list params (`$filter_edge_types`,
/// `$filter_edge_uuids`); node labels are inlined after [`sanitize_label`]; date
/// fields use globally-unique numbered params (`$p0`, `$p1`, ...). Returns the
/// fragments (caller joins with ` AND ` into a WHERE block) and the params.
pub fn edge_filter_fragments(filters: &SearchFilters) -> Result<FilterFragments, DriverError> {
    let mut fragments: Vec<String> = Vec::new();
    let mut params: Vec<(String, BoltType)> = Vec::new();
    let mut counter: usize = 0;

    if let Some(edge_types) = &filters.edge_types {
        fragments.push("e.name IN $filter_edge_types".to_string());
        let list: Vec<BoltType> = edge_types
            .iter()
            .map(|s| BoltType::from(s.as_str()))
            .collect();
        params.push((
            "filter_edge_types".to_string(),
            BoltType::List(neo4rs::BoltList::from(list)),
        ));
    }

    if let Some(edge_uuids) = &filters.edge_uuids {
        fragments.push("e.uuid IN $filter_edge_uuids".to_string());
        let list: Vec<BoltType> = edge_uuids
            .iter()
            .map(|s| BoltType::from(s.as_str()))
            .collect();
        params.push((
            "filter_edge_uuids".to_string(),
            BoltType::List(neo4rs::BoltList::from(list)),
        ));
    }

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        let mut sanitized = Vec::with_capacity(node_labels.len());
        for l in node_labels {
            sanitized.push(sanitize_label(l)?);
        }
        let joined = sanitized.join("|");
        fragments.push(format!("n:{joined} AND m:{joined}"));
    }

    if let Some(groups) = &filters.valid_at
        && let Some(frag) = date_field_fragment("e.valid_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.invalid_at
        && let Some(frag) = date_field_fragment("e.invalid_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.created_at
        && let Some(frag) = date_field_fragment("e.created_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }
    if let Some(groups) = &filters.expired_at
        && let Some(frag) = date_field_fragment("e.expired_at", groups, &mut counter, &mut params)
    {
        fragments.push(frag);
    }

    Ok((fragments, params))
}

/// Build the node-side WHERE fragments + params (R10). Mirrors upstream
/// `node_search_filter_query_constructor`: ONLY `node_labels` applies on the node
/// scope (`n:L1|L2`); edge_types / edge_uuids / date fields are edge-only and are
/// ignored here, exactly as upstream does.
pub fn node_filter_fragments(filters: &SearchFilters) -> Result<FilterFragments, DriverError> {
    let mut fragments: Vec<String> = Vec::new();
    let params: Vec<(String, BoltType)> = Vec::new();

    if let Some(node_labels) = &filters.node_labels
        && !node_labels.is_empty()
    {
        let mut sanitized = Vec::with_capacity(node_labels.len());
        for l in node_labels {
            sanitized.push(sanitize_label(l)?);
        }
        fragments.push(format!("n:{}", sanitized.join("|")));
    }

    Ok((fragments, params))
}

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

/// Fulltext indices (verbatim from `get_fulltext_indices`, Neo4j branch).
/// Phase-2 adds `episode_content` (drives [`EPISODE_FULLTEXT_SEARCH_HEAD`]).
/// `community_name` remains excluded (community scope is Phase-4).
pub const FULLTEXT_INDICES: &[&str] = &[
    "CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS \
     FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]",
    "CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS \
     FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]",
    "CREATE FULLTEXT INDEX episode_content IF NOT EXISTS \
     FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]",
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
        "episode_content",
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

    // -----------------------------------------------------------------
    // SearchFilters → WHERE fragment builders (R10)
    // -----------------------------------------------------------------

    use chronicle_core::search::filters::{ComparisonOperator, DateFilter};
    use chrono::{DateTime, Utc};

    fn df(op: ComparisonOperator, date: Option<DateTime<Utc>>) -> DateFilter {
        DateFilter {
            date,
            comparison_operator: op,
        }
    }

    #[test]
    fn edge_filter_empty_is_empty() {
        let (frags, params) = edge_filter_fragments(&SearchFilters::default()).expect("ok");
        assert!(frags.is_empty());
        assert!(params.is_empty());
    }

    #[test]
    fn edge_filter_edge_types_binds_list_param() {
        let f = SearchFilters {
            edge_types: Some(vec!["WORKS_AT".into(), "KNOWS".into()]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["e.name IN $filter_edge_types".to_string()]);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "filter_edge_types");
    }

    #[test]
    fn edge_filter_edge_uuids_binds_list_param() {
        let f = SearchFilters {
            edge_uuids: Some(vec!["u1".into()]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["e.uuid IN $filter_edge_uuids".to_string()]);
        assert_eq!(params[0].0, "filter_edge_uuids");
    }

    #[test]
    fn edge_filter_node_labels_both_endpoints_inlined() {
        let f = SearchFilters {
            node_labels: Some(vec!["Person".into(), "Company".into()]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(
            frags,
            vec!["n:Person|Company AND m:Person|Company".to_string()]
        );
        assert!(params.is_empty(), "labels are inlined, never params");
    }

    #[test]
    fn node_filter_node_labels_single_side() {
        let f = SearchFilters {
            node_labels: Some(vec!["Person".into()]),
            ..Default::default()
        };
        let (frags, params) = node_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["n:Person".to_string()]);
        assert!(params.is_empty());
    }

    #[test]
    fn node_filter_ignores_edge_only_fields() {
        // edge_types / edge_uuids / dates are edge-only and must not appear on the
        // node scope (matches upstream node_search_filter_query_constructor).
        let f = SearchFilters {
            edge_types: Some(vec!["X".into()]),
            edge_uuids: Some(vec!["u".into()]),
            valid_at: Some(vec![vec![df(ComparisonOperator::Gte, Some(Utc::now()))]]),
            ..Default::default()
        };
        let (frags, params) = node_filter_fragments(&f).expect("ok");
        assert!(frags.is_empty());
        assert!(params.is_empty());
    }

    #[test]
    fn label_sanitizer_rejects_injection() {
        for bad in &[
            "Per son", "Foo|Bar", "Foo:Bar", "Foo}", "Foo`", "", "Foo-Bar",
        ] {
            let f = SearchFilters {
                node_labels: Some(vec![(*bad).into()]),
                ..Default::default()
            };
            assert!(
                node_filter_fragments(&f).is_err(),
                "expected rejection for label: {bad:?}"
            );
        }
    }

    #[test]
    fn label_sanitizer_accepts_wordchars_and_leading_digit() {
        // Plan R10 set ^[A-Za-z0-9_]+$ allows a leading digit (looser than upstream).
        let f = SearchFilters {
            node_labels: Some(vec!["Foo_Bar9".into(), "0Leading".into()]),
            ..Default::default()
        };
        let (frags, _) = node_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["n:Foo_Bar9|0Leading".to_string()]);
    }

    #[test]
    fn date_single_value_op_emits_param() {
        let now = Utc::now();
        let f = SearchFilters {
            valid_at: Some(vec![vec![df(ComparisonOperator::Gte, Some(now))]]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["((e.valid_at >= $p0))".to_string()]);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "p0");
    }

    #[test]
    fn date_is_null_takes_no_param() {
        let f = SearchFilters {
            invalid_at: Some(vec![vec![df(ComparisonOperator::IsNull, None)]]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(frags, vec!["((e.invalid_at IS NULL))".to_string()]);
        assert!(params.is_empty(), "IS NULL binds no param");
    }

    #[test]
    fn date_or_of_ands_two_groups_distinct_params() {
        // ((>= p0 AND < p1) OR (>= p2)) — three value conditions across two OR
        // groups must get THREE distinct param names (the upstream collision bug:
        // upstream would name them valid_at_0, valid_at_1, valid_at_0 — the third
        // clobbering the first). Pin the fix here.
        let t0 = Utc::now();
        let t1 = t0 + chrono::Duration::hours(1);
        let t2 = t0 + chrono::Duration::hours(2);
        let f = SearchFilters {
            valid_at: Some(vec![
                vec![
                    df(ComparisonOperator::Gte, Some(t0)),
                    df(ComparisonOperator::Lt, Some(t1)),
                ],
                vec![df(ComparisonOperator::Gte, Some(t2))],
            ]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        // Each condition is individually parenthesised (upstream
        // date_filter_query_constructor wraps every condition in '(...)'), then
        // joined by AND within a group and by OR across groups.
        assert_eq!(
            frags,
            vec!["((e.valid_at >= $p0) AND (e.valid_at < $p1) OR (e.valid_at >= $p2))".to_string()]
        );
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["p0", "p1", "p2"],
            "globally unique, no collision"
        );
    }

    #[test]
    fn date_params_unique_across_multiple_fields() {
        // valid_at AND created_at in the SAME call: the monotonic counter must not
        // restart per field, else valid_at_0 and created_at_0 would both be "p0".
        let now = Utc::now();
        let f = SearchFilters {
            valid_at: Some(vec![vec![df(ComparisonOperator::Gte, Some(now))]]),
            created_at: Some(vec![vec![df(ComparisonOperator::Lte, Some(now))]]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0], "((e.valid_at >= $p0))");
        assert_eq!(frags[1], "((e.created_at <= $p1))");
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["p0", "p1"]);
    }

    #[test]
    fn date_mixed_null_and_value_in_same_group() {
        // (IS NOT NULL AND >= p0) — null op contributes no param, value op does.
        let now = Utc::now();
        let f = SearchFilters {
            expired_at: Some(vec![vec![
                df(ComparisonOperator::IsNotNull, None),
                df(ComparisonOperator::Gte, Some(now)),
            ]]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(
            frags,
            vec!["((e.expired_at IS NOT NULL) AND (e.expired_at >= $p0))".to_string()]
        );
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "p0");
    }

    #[test]
    fn edge_filter_full_combo_order_and_params() {
        // edge_types, edge_uuids, node_labels, valid_at — fragment ORDER must match
        // upstream edge_search_filter_query_constructor.
        let now = Utc::now();
        let f = SearchFilters {
            edge_types: Some(vec!["REL".into()]),
            edge_uuids: Some(vec!["u1".into()]),
            node_labels: Some(vec!["Person".into()]),
            valid_at: Some(vec![vec![df(ComparisonOperator::Gte, Some(now))]]),
            ..Default::default()
        };
        let (frags, params) = edge_filter_fragments(&f).expect("ok");
        assert_eq!(
            frags,
            vec![
                "e.name IN $filter_edge_types".to_string(),
                "e.uuid IN $filter_edge_uuids".to_string(),
                "n:Person AND m:Person".to_string(),
                "((e.valid_at >= $p0))".to_string(),
            ]
        );
        // params: edge_types list, edge_uuids list, p0 date (labels inlined).
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["filter_edge_types", "filter_edge_uuids", "p0"]);
    }

    // -----------------------------------------------------------------
    // BFS query builders (R9)
    // -----------------------------------------------------------------

    #[test]
    fn node_bfs_depth_is_clamped_and_inlined() {
        let q = node_bfs_query(99, &[], false);
        assert!(
            q.contains("*1..10"),
            "depth clamped to MAX and inlined: {q}"
        );
        let q0 = node_bfs_query(0, &[], false);
        assert!(q0.contains("*1..1"), "depth clamped to MIN: {q0}");
    }

    #[test]
    fn node_bfs_group_and_filter_clauses_appended_with_and() {
        let q = node_bfs_query(2, &["n:Person".to_string()], true);
        assert!(q.contains("WHERE n.group_id = origin.group_id"));
        assert!(q.contains("AND n.group_id IN $group_ids AND origin.group_id IN $group_ids"));
        assert!(q.contains("AND n:Person"));
    }

    #[test]
    fn edge_bfs_undirected_match_and_where_block() {
        let q = edge_bfs_query(3, &["e.group_id IN $group_ids".to_string()]);
        assert!(q.contains("*1..3"));
        assert!(q.contains("-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)"));
        assert!(q.contains("WHERE e.group_id IN $group_ids"));
        assert!(q.contains("RETURN DISTINCT"));
    }

    #[test]
    fn edge_bfs_no_filters_has_no_where() {
        let q = edge_bfs_query(1, &[]);
        assert!(!q.contains("WHERE"), "no filters => no WHERE block: {q}");
    }
}
