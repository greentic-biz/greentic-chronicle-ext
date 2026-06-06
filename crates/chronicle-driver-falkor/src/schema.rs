//! Index DDL for the FalkorDB backend.
//!
//! Three index families, all verified against live `falkordb/falkordb:latest`:
//!
//! 1. **Range indices** on `uuid` + `group_id` per label — fast point lookups and
//!    group-scoped scans. `CREATE INDEX FOR (n:Label) ON (n.prop)`.
//! 2. **Vector indices** (HNSW, cosine) on the three embedding properties —
//!    `entity.name_embedding`, `relates_to.fact_embedding`,
//!    `community.name_embedding`. `dimension` is the driver's configured
//!    `embedding_dim`. Query side uses `db.idx.vector.queryNodes` (Task 2).
//! 3. **Fulltext indices** — node fulltext via
//!    `db.idx.fulltext.createNodeIndex('Label', field...)` for `Entity`
//!    (name+summary), `Episodic` (content) and `Community` (name).
//!
//!    **Relationship fulltext** (chronicle indexes the edge `fact`) IS supported,
//!    but NOT via `createNodeIndex` (which silently indexes nothing for a
//!    relationship type — verified empirically). The working path is the DDL
//!    form `CREATE FULLTEXT INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact)`, queried
//!    with `db.idx.fulltext.queryRelationships('RELATES_TO', $q)` in Task 2.
//!
//! All DDL is idempotent: a re-create on an existing index returns an
//! "already indexed" error that [`already_exists`] recognises and the driver
//! swallows (FalkorDB has no `IF NOT EXISTS` for these forms).

/// Range indices: `(label, property)` pairs. One `CREATE INDEX` each.
pub const RANGE_INDICES: &[(&str, &str)] = &[
    ("Entity", "uuid"),
    ("Entity", "group_id"),
    ("Episodic", "uuid"),
    ("Episodic", "group_id"),
    ("Community", "uuid"),
    ("Community", "group_id"),
    ("Saga", "uuid"),
    ("Saga", "group_id"),
];

/// Node fulltext indices: `(label, [fields])`. Created via
/// `db.idx.fulltext.createNodeIndex`.
pub const NODE_FULLTEXT_INDICES: &[(&str, &[&str])] = &[
    ("Entity", &["name", "summary"]),
    ("Episodic", &["content"]),
    ("Community", &["name"]),
];

/// Vector indices: `(label, property)`. The `dimension` is supplied at runtime
/// from the driver's `embedding_dim`. Cosine similarity (HNSW defaults).
pub const VECTOR_INDICES: &[(&str, &str)] = &[
    ("Entity", "name_embedding"),
    ("RELATES_TO", "fact_embedding"),
    ("Community", "name_embedding"),
];

/// Build a range-index DDL statement.
pub fn range_index_ddl(label: &str, property: &str) -> String {
    format!("CREATE INDEX FOR (n:{label}) ON (n.{property})")
}

/// Build a node-fulltext create procedure call. Fields are passed as quoted
/// literal arguments (`createNodeIndex('Label','f1','f2')`).
pub fn node_fulltext_ddl(label: &str, fields: &[&str]) -> String {
    let quoted: Vec<String> = std::iter::once(format!("'{label}'"))
        .chain(fields.iter().map(|f| format!("'{f}'")))
        .collect();
    format!("CALL db.idx.fulltext.createNodeIndex({})", quoted.join(","))
}

/// Build the relationship-fulltext DDL (the working path for edge fulltext).
pub fn relationship_fulltext_ddl(rel_type: &str, property: &str) -> String {
    format!("CREATE FULLTEXT INDEX FOR ()-[r:{rel_type}]-() ON (r.{property})")
}

/// Build a node vector-index DDL with the configured dimension.
pub fn node_vector_index_ddl(label: &str, property: &str, dimension: usize) -> String {
    format!(
        "CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property}) \
         OPTIONS {{dimension:{dimension}, similarityFunction:'cosine'}}"
    )
}

/// Build a relationship vector-index DDL with the configured dimension.
pub fn relationship_vector_index_ddl(rel_type: &str, property: &str, dimension: usize) -> String {
    format!(
        "CREATE VECTOR INDEX FOR ()-[r:{rel_type}]-() ON (r.{property}) \
         OPTIONS {{dimension:{dimension}, similarityFunction:'cosine'}}"
    )
}

/// Edge-fulltext (relationship) is indexed on `RELATES_TO.fact`.
pub const RELATIONSHIP_FULLTEXT: (&str, &str) = ("RELATES_TO", "fact");

/// Recognise the "index already exists" class of FalkorDB errors so idempotent
/// re-builds succeed.
pub fn already_exists(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("already indexed")
        || e.contains("already exists")
        || e.contains("attribute is already indexed")
        || e.contains("already a fulltext")
        || e.contains("there already exists")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_ddl_shape() {
        assert_eq!(
            range_index_ddl("Entity", "uuid"),
            "CREATE INDEX FOR (n:Entity) ON (n.uuid)"
        );
    }

    #[test]
    fn node_fulltext_ddl_multi_field() {
        assert_eq!(
            node_fulltext_ddl("Entity", &["name", "summary"]),
            "CALL db.idx.fulltext.createNodeIndex('Entity','name','summary')"
        );
    }

    #[test]
    fn relationship_fulltext_uses_ddl_form() {
        assert_eq!(
            relationship_fulltext_ddl("RELATES_TO", "fact"),
            "CREATE FULLTEXT INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact)"
        );
    }

    #[test]
    fn vector_ddl_includes_dimension() {
        let ddl = node_vector_index_ddl("Entity", "name_embedding", 8);
        assert!(ddl.contains("dimension:8"));
        assert!(ddl.contains("similarityFunction:'cosine'"));
    }

    #[test]
    fn already_exists_matches_common_messages() {
        assert!(already_exists("Attribute 'uuid' is already indexed"));
        assert!(already_exists("Index already exists"));
        assert!(!already_exists("syntax error"));
    }
}
