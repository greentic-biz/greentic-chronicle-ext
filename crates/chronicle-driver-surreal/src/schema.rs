//! Idempotent SurrealQL schema DDL for the chronicle embedded backend.
//!
//! Design decisions (Phase-3 spec amendment §3/§4):
//!
//! - **Separate tables per node kind** — `entity`, `episodic`, `community`,
//!   `saga` — instead of one polymorphic table with a `label` discriminator.
//!   Typed queries (e.g. "all entity nodes in group X") stay simple, and the
//!   relation-table `IN`/`OUT` constraints document the graph shape.
//! - **Record id = the chronicle uuid** (`entity:⟨uuid⟩`). This gives O(1)
//!   point lookups (`SELECT * FROM entity:⟨uuid⟩`) and lets `RELATE` reference
//!   endpoints by uuid directly. The uuid is ALSO stored as a plain `uuid`
//!   string field so row structs deserialize cleanly without parsing the
//!   record id key back out.
//! - **`OVERWRITE` everywhere** (not `IF NOT EXISTS`): re-running the DDL is a
//!   clean redefinition, which is exactly what `build_indices_and_constraints`
//!   needs (idempotent connect + `delete_existing` rebuild). Verified against
//!   surrealdb 3.1.3.
//! - **Tables are `SCHEMALESS`**: attributes are open-ended JSON, so we only
//!   `DEFINE FIELD` the columns that need a declared type for indexing
//!   (embeddings as `option<array<float>>`, group_id as `string`, datetimes).
//! - **HNSW** vector indexes carry `TYPE F32 DIST COSINE` (chronicle stores
//!   `Vec<f32>`; the SurrealDB default is F64). `DIMENSION` is configurable.
//! - **Full-text** uses the 3.x `FULLTEXT ANALYZER ... BM25` form (the pre-3.0
//!   `SEARCH ANALYZER` keyword was removed). One index per field.

/// Namespace used by the embedded driver.
pub const NAMESPACE: &str = "chronicle";
/// Database used by the embedded driver.
pub const DATABASE: &str = "graph";

/// Analyzer name shared by every full-text index.
pub const ANALYZER: &str = "chronicle_ascii";

/// Build the full schema DDL as a single multi-statement SurrealQL string.
///
/// `embedding_dim` parameterises the HNSW vector-index `DIMENSION`. The whole
/// string is idempotent (`OVERWRITE`), so it is safe to run on every connect and
/// to re-run for `build_indices_and_constraints`.
pub fn schema_ddl(embedding_dim: usize) -> String {
    let mut ddl = String::new();

    // ── Node tables ──────────────────────────────────────────────────────
    for table in ["entity", "episodic", "community", "saga"] {
        ddl.push_str(&format!("DEFINE TABLE OVERWRITE {table} SCHEMALESS;\n"));
        // group_id is present on every node and is the dominant filter key.
        ddl.push_str(&format!(
            "DEFINE FIELD OVERWRITE group_id ON {table} TYPE string;\n"
        ));
        ddl.push_str(&format!(
            "DEFINE INDEX OVERWRITE {table}_group_id ON {table} FIELDS group_id;\n"
        ));
    }

    // Embedding fields (declared typed so the HNSW index can attach).
    ddl.push_str("DEFINE FIELD OVERWRITE name_embedding ON entity TYPE option<array<float>>;\n");
    ddl.push_str("DEFINE FIELD OVERWRITE name_embedding ON community TYPE option<array<float>>;\n");

    // ── Edge / relation tables ───────────────────────────────────────────
    // relates_to: entity -> entity (carries the fact + fact_embedding).
    ddl.push_str(
        "DEFINE TABLE OVERWRITE relates_to TYPE RELATION IN entity OUT entity SCHEMALESS;\n",
    );
    ddl.push_str("DEFINE FIELD OVERWRITE group_id ON relates_to TYPE string;\n");
    ddl.push_str("DEFINE INDEX OVERWRITE relates_to_group_id ON relates_to FIELDS group_id;\n");
    ddl.push_str("DEFINE INDEX OVERWRITE relates_to_uuid ON relates_to FIELDS uuid;\n");
    ddl.push_str(
        "DEFINE FIELD OVERWRITE fact_embedding ON relates_to TYPE option<array<float>>;\n",
    );

    // mentions: episodic -> entity.
    ddl.push_str(
        "DEFINE TABLE OVERWRITE mentions TYPE RELATION IN episodic OUT entity SCHEMALESS;\n",
    );
    ddl.push_str("DEFINE INDEX OVERWRITE mentions_uuid ON mentions FIELDS uuid;\n");

    // has_member: community -> entity | community.
    ddl.push_str(
        "DEFINE TABLE OVERWRITE has_member TYPE RELATION IN community OUT entity|community SCHEMALESS;\n",
    );
    ddl.push_str("DEFINE INDEX OVERWRITE has_member_uuid ON has_member FIELDS uuid;\n");

    // has_episode: saga -> episodic.
    ddl.push_str(
        "DEFINE TABLE OVERWRITE has_episode TYPE RELATION IN saga OUT episodic SCHEMALESS;\n",
    );
    ddl.push_str("DEFINE INDEX OVERWRITE has_episode_uuid ON has_episode FIELDS uuid;\n");

    // next_episode: episodic -> episodic.
    ddl.push_str(
        "DEFINE TABLE OVERWRITE next_episode TYPE RELATION IN episodic OUT episodic SCHEMALESS;\n",
    );
    ddl.push_str("DEFINE INDEX OVERWRITE next_episode_uuid ON next_episode FIELDS uuid;\n");

    // ── HNSW vector indexes (TYPE F32 DIST COSINE) ───────────────────────
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE entity_name_emb_hnsw ON entity FIELDS name_embedding \
         HNSW DIMENSION {embedding_dim} TYPE F32 DIST COSINE;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE community_name_emb_hnsw ON community FIELDS name_embedding \
         HNSW DIMENSION {embedding_dim} TYPE F32 DIST COSINE;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE relates_to_fact_emb_hnsw ON relates_to FIELDS fact_embedding \
         HNSW DIMENSION {embedding_dim} TYPE F32 DIST COSINE;\n"
    ));

    // ── Full-text (BM25) analyzer + per-field indexes ────────────────────
    ddl.push_str(&format!(
        "DEFINE ANALYZER OVERWRITE {ANALYZER} TOKENIZERS blank,class FILTERS lowercase,ascii;\n"
    ));
    // One index per searchable field (locked decision #5).
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE entity_name_fts ON entity FIELDS name \
         FULLTEXT ANALYZER {ANALYZER} BM25 HIGHLIGHTS;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE entity_summary_fts ON entity FIELDS summary \
         FULLTEXT ANALYZER {ANALYZER} BM25 HIGHLIGHTS;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE relates_to_fact_fts ON relates_to FIELDS fact \
         FULLTEXT ANALYZER {ANALYZER} BM25 HIGHLIGHTS;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE episodic_content_fts ON episodic FIELDS content \
         FULLTEXT ANALYZER {ANALYZER} BM25 HIGHLIGHTS;\n"
    ));
    ddl.push_str(&format!(
        "DEFINE INDEX OVERWRITE community_name_fts ON community FIELDS name \
         FULLTEXT ANALYZER {ANALYZER} BM25 HIGHLIGHTS;\n"
    ));

    ddl
}

/// `REMOVE` statements for every index/analyzer/table defined above, used by
/// `build_indices_and_constraints(delete_existing = true)`. Each carries
/// `IF EXISTS` so a fresh database (nothing to drop) is a clean no-op.
pub fn drop_ddl() -> String {
    let mut ddl = String::new();
    // Indexes first, then analyzers, then tables (which cascades the data).
    let index_drops = [
        ("entity", "entity_group_id"),
        ("entity", "entity_name_emb_hnsw"),
        ("entity", "entity_name_fts"),
        ("entity", "entity_summary_fts"),
        ("episodic", "episodic_group_id"),
        ("episodic", "episodic_content_fts"),
        ("community", "community_group_id"),
        ("community", "community_name_emb_hnsw"),
        ("community", "community_name_fts"),
        ("saga", "saga_group_id"),
        ("relates_to", "relates_to_group_id"),
        ("relates_to", "relates_to_uuid"),
        ("relates_to", "relates_to_fact_emb_hnsw"),
        ("relates_to", "relates_to_fact_fts"),
        ("mentions", "mentions_uuid"),
        ("has_member", "has_member_uuid"),
        ("has_episode", "has_episode_uuid"),
        ("next_episode", "next_episode_uuid"),
    ];
    for (table, index) in index_drops {
        ddl.push_str(&format!("REMOVE INDEX IF EXISTS {index} ON {table};\n"));
    }
    ddl.push_str(&format!("REMOVE ANALYZER IF EXISTS {ANALYZER};\n"));
    for table in [
        "relates_to",
        "mentions",
        "has_member",
        "has_episode",
        "next_episode",
        "entity",
        "episodic",
        "community",
        "saga",
    ] {
        ddl.push_str(&format!("REMOVE TABLE IF EXISTS {table};\n"));
    }
    ddl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddl_mentions_every_table_and_index_family() {
        let ddl = schema_ddl(1024);
        for table in ["entity", "episodic", "community", "saga"] {
            assert!(ddl.contains(&format!("DEFINE TABLE OVERWRITE {table}")));
        }
        for rel in [
            "relates_to",
            "mentions",
            "has_member",
            "has_episode",
            "next_episode",
        ] {
            assert!(ddl.contains(&format!("DEFINE TABLE OVERWRITE {rel} TYPE RELATION")));
        }
        assert!(ddl.contains("HNSW DIMENSION 1024 TYPE F32 DIST COSINE"));
        assert!(ddl.contains("FULLTEXT ANALYZER chronicle_ascii BM25"));
    }

    #[test]
    fn embedding_dim_is_parameterised() {
        assert!(schema_ddl(768).contains("DIMENSION 768"));
        assert!(schema_ddl(256).contains("DIMENSION 256"));
    }

    #[test]
    fn drop_ddl_uses_if_exists() {
        let ddl = drop_ddl();
        assert!(ddl.contains("REMOVE TABLE IF EXISTS entity"));
        assert!(ddl.contains("REMOVE ANALYZER IF EXISTS chronicle_ascii"));
    }
}
