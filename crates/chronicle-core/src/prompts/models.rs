// Ported from (verbatim field descriptions copied from each upstream Pydantic Field):
//   - graphiti_core/prompts/extract_nodes.py  @ 34f56e65 (v0.29.1)
//   - graphiti_core/prompts/extract_edges.py  @ 34f56e65 (v0.29.1)
//   - graphiti_core/prompts/dedupe_nodes.py   @ 34f56e65 (v0.29.1)
//   - graphiti_core/prompts/dedupe_edges.py   @ 34f56e65 (v0.29.1)
//   - graphiti_core/prompts/summarize_nodes.py @ 34f56e65 (v0.29.1)
//
// Serde contract: NO `deny_unknown_fields` anywhere in this module — LLMs may
// return extra fields; serde silently ignoring them is intentional tolerance.
// Do not "harden" these structs with deny_unknown_fields.
// Fields without a #[schemars(description)] mirror upstream fields that have
// no Field(description=...) — do not invent descriptions (verbatim rule).

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

fn default_episode_indices() -> Vec<i64> {
    vec![0]
}

// --------------------------------------------------------------------------
// extract_nodes.py
// --------------------------------------------------------------------------

/// Extracted entity from a conversational episode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExtractedEntity {
    #[schemars(description = "Name of the extracted entity")]
    pub name: String,

    #[schemars(
        description = "ID of the classified entity type. Must be one of the provided entity_type_id integers."
    )]
    pub entity_type_id: i64,

    /// Upstream default_factory=lambda: [0]
    #[serde(default = "default_episode_indices")]
    #[schemars(
        description = "List of episode numbers (0-indexed) this entity was extracted from. When processing a single episode, this should be [0]."
    )]
    pub episode_indices: Vec<i64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExtractedEntities {
    #[schemars(description = "List of extracted entities")]
    pub extracted_entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct EntitySummary {
    #[schemars(description = "Summary of the entity")]
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SummarizedEntity {
    #[schemars(description = "Name of the entity being summarized")]
    pub name: String,

    #[schemars(description = "Updated summary for the entity")]
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SummarizedEntities {
    #[schemars(
        description = "List of entity summaries. Only include entities that need summary updates."
    )]
    pub summaries: Vec<SummarizedEntity>,
}

// --------------------------------------------------------------------------
// extract_edges.py
// --------------------------------------------------------------------------
// NOTE: The upstream Python class is named `Edge`. Renamed to `ExtractedEdge`
// here to avoid a name clash with the graph domain type `Edge` that will live
// in chronicle-core::types. The rename is intentional and documented.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExtractedEdge {
    #[schemars(description = "The name of the source entity from the ENTITIES list")]
    pub source_entity_name: String,

    #[schemars(description = "The name of the target entity from the ENTITIES list")]
    pub target_entity_name: String,

    #[schemars(
        description = "The type of relationship between the entities, in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH)"
    )]
    pub relation_type: String,

    #[schemars(
        description = "A natural language description of the relationship between the entities, paraphrased from the source text"
    )]
    pub fact: String,

    #[schemars(
        description = "The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)"
    )]
    pub valid_at: Option<String>,

    #[schemars(
        description = "The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)"
    )]
    pub invalid_at: Option<String>,

    /// Upstream default_factory=lambda: [0]
    #[serde(default = "default_episode_indices")]
    #[schemars(
        description = "List of episode numbers (0-indexed) that this fact was derived from. When processing a single episode, this should be [0]."
    )]
    pub episode_indices: Vec<i64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExtractedEdges {
    pub edges: Vec<ExtractedEdge>,
}

/// Temporal bounds extracted from a fact.
/// Upstream: class EdgeTimestamps(BaseModel)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct EdgeTimestamps {
    #[schemars(
        description = "When the fact became true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
    )]
    pub valid_at: Option<String>,

    #[schemars(
        description = "When the fact stopped being true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
    )]
    pub invalid_at: Option<String>,
}

/// Temporal bounds for a batch of facts.
/// Upstream: class BatchEdgeTimestamps(BaseModel) — present in extract_edges.py @ 34f56e65
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BatchEdgeTimestamps {
    #[schemars(description = "Timestamps for each fact, in the same order as the input facts")]
    pub timestamps: Vec<EdgeTimestamps>,
}

// --------------------------------------------------------------------------
// dedupe_nodes.py
// --------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct NodeDuplicate {
    #[schemars(description = "integer id of the entity")]
    pub id: i64,

    #[schemars(
        description = "Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}."
    )]
    pub name: String,

    #[schemars(
        description = "candidate_id of the matching EXISTING ENTITY, or -1 if no duplicate exists."
    )]
    pub duplicate_candidate_id: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct NodeResolutions {
    #[schemars(description = "List of resolved nodes")]
    pub entity_resolutions: Vec<NodeDuplicate>,
}

// --------------------------------------------------------------------------
// dedupe_edges.py
// --------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct EdgeDuplicate {
    #[schemars(
        description = "List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none."
    )]
    pub duplicate_facts: Vec<i64>,

    #[schemars(
        description = "List of idx values of contradicted facts (from full idx range). Empty list if none."
    )]
    pub contradicted_facts: Vec<i64>,
}

// --------------------------------------------------------------------------
// summarize_nodes.py
// --------------------------------------------------------------------------
// NOTE: The upstream Summary.summary description contains the f-string
//   f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters'
// where MAX_SUMMARY_CHARS = 1000 (graphiti_core/utils/text_utils.py).
// The literal string is resolved to "1000" here and noted in the comment below.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Summary {
    /// Upstream description resolves the f-string: MAX_SUMMARY_CHARS = 1000
    #[schemars(
        description = "Summary containing the important information about the entity. Under 1000 characters"
    )]
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SummaryDescription {
    #[schemars(description = "One sentence description of the provided summary")]
    pub description: String,
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Step 1: required fidelity test (from task spec) ----

    #[test]
    fn extracted_entity_schema_has_upstream_descriptions() {
        let schema = serde_json::to_value(schemars::schema_for!(ExtractedEntity)).unwrap();
        // schemars may inline the root type's properties directly or place them under definitions.
        let props = schema
            .get("definitions")
            .and_then(|d| d.get("ExtractedEntity"))
            .and_then(|n| n.get("properties"))
            .or_else(|| schema.get("properties"))
            .expect("properties not found in ExtractedEntity schema");
        assert!(
            props["name"]["description"]
                .as_str()
                .unwrap()
                .contains("Name of the extracted entity"),
            "name description mismatch: {:?}",
            props["name"]["description"]
        );
        assert!(
            props["entity_type_id"]["description"]
                .as_str()
                .unwrap()
                .contains("ID of the classified entity type"),
            "entity_type_id description mismatch: {:?}",
            props["entity_type_id"]["description"]
        );
    }

    // ---- Step 4: per-family fidelity tests ----

    #[test]
    fn node_duplicate_schema_has_upstream_descriptions() {
        let schema = serde_json::to_value(schemars::schema_for!(NodeDuplicate)).unwrap();
        // Schema may be inlined or under definitions depending on schemars version.
        let root = &schema;
        let props = root
            .get("definitions")
            .and_then(|d| d.get("NodeDuplicate"))
            .and_then(|n| n.get("properties"))
            .or_else(|| root.get("properties"))
            .expect("properties not found in NodeDuplicate schema");

        assert!(
            props["duplicate_candidate_id"]["description"]
                .as_str()
                .unwrap()
                .contains("candidate_id of the matching EXISTING ENTITY"),
            "duplicate_candidate_id description mismatch: {:?}",
            props["duplicate_candidate_id"]["description"]
        );
    }

    #[test]
    fn edge_duplicate_schema_has_upstream_descriptions() {
        let schema = serde_json::to_value(schemars::schema_for!(EdgeDuplicate)).unwrap();
        let root = &schema;
        let props = root
            .get("definitions")
            .and_then(|d| d.get("EdgeDuplicate"))
            .and_then(|n| n.get("properties"))
            .or_else(|| root.get("properties"))
            .expect("properties not found in EdgeDuplicate schema");

        assert!(
            props["duplicate_facts"]["description"]
                .as_str()
                .unwrap()
                .contains("List of idx values of duplicate facts"),
            "duplicate_facts description mismatch: {:?}",
            props["duplicate_facts"]["description"]
        );
    }

    #[test]
    fn extracted_edge_schema_has_upstream_descriptions() {
        let schema = serde_json::to_value(schemars::schema_for!(ExtractedEdge)).unwrap();
        let root = &schema;
        let props = root
            .get("definitions")
            .and_then(|d| d.get("ExtractedEdge"))
            .and_then(|n| n.get("properties"))
            .or_else(|| root.get("properties"))
            .expect("properties not found in ExtractedEdge schema");

        assert!(
            props["relation_type"]["description"]
                .as_str()
                .unwrap()
                .contains("The type of relationship between the entities"),
            "relation_type description mismatch: {:?}",
            props["relation_type"]["description"]
        );
    }

    // ---- Step 4: deserialization tests ----

    #[test]
    fn edge_duplicate_deserializes_correctly() {
        let json = r#"{"duplicate_facts":[1],"contradicted_facts":[]}"#;
        let ed: EdgeDuplicate = serde_json::from_str(json).expect("EdgeDuplicate deserialize");
        assert_eq!(ed.duplicate_facts, vec![1]);
        assert!(ed.contradicted_facts.is_empty());
    }

    #[test]
    fn extracted_entity_default_episode_indices() {
        let json = r#"{"name":"Alice","entity_type_id":2}"#;
        let entity: ExtractedEntity = serde_json::from_str(json)
            .expect("ExtractedEntity deserialize without episode_indices");
        assert_eq!(
            entity.episode_indices,
            vec![0],
            "default episode_indices should be [0]"
        );
    }
}
