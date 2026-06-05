// Ported from graphiti_core/prompts/extract_edges.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE is appended to every system prompt (VersionWrapper behaviour).
// - `edge`: the upstream user f-string's closing triple-quote is indented by 8 spaces
//   (`        """,`), so the rendered user content ends with "\n        " (newline + 8
//   spaces) AFTER the final DATETIME rule line. We preserve that trailing whitespace
//   exactly via a raw string literal — see the literal tail in `edge`.
// - `edge_types_section` is conditional: empty string when context has no edge_types,
//   otherwise the <FACT_TYPES>…</FACT_TYPES> block (itself starting and ending with a
//   newline, rendered via to_prompt_json). The upstream template places the
//   `{edge_types_section}` placeholder on its OWN line between `</REFERENCE_TIME>` and
//   `# TASK`, so when the section is empty the output is `</REFERENCE_TIME>\n\n# TASK`
//   (blank line) and when present it is `</FACT_TYPES>\n\n# TASK`. Replicated exactly.
// - `nodes`, `previous_episodes`, `edge_types` are rendered via to_prompt_json.
// - `episode_content`, `reference_time`, `custom_extraction_instructions` interpolate raw.

use crate::llm::Message;
use crate::prompts::helpers::{DO_NOT_ESCAPE_UNICODE, to_prompt_json};

/// Context for [`edge`].
pub struct EdgeContext<'a> {
    /// Upstream `context['previous_episodes']` — a JSON array of
    /// `{"content": ..., "timestamp": ...}` objects (see
    /// `pipeline::node_ops::previous_episodes_context`), rendered via to_prompt_json.
    pub previous_episodes: &'a serde_json::Value,
    /// Upstream `context['episode_content']`.
    pub episode_content: &'a str,
    /// Upstream `context['nodes']`, rendered via to_prompt_json.
    pub nodes: &'a serde_json::Value,
    /// Upstream `context['reference_time']`.
    pub reference_time: &'a str,
    /// Upstream `context.get('edge_types')` — optional; renders the <FACT_TYPES> block
    /// when present, otherwise the section is empty.
    pub edge_types: Option<&'a serde_json::Value>,
    /// Upstream `context['custom_extraction_instructions']` (resolved to '' when absent).
    pub custom_extraction_instructions: &'a str,
}

/// Verbatim port of upstream `edge`.
pub fn edge(ctx: &EdgeContext<'_>) -> Vec<Message> {
    // Upstream: edge_types_section = '' unless context.get('edge_types') is truthy.
    let edge_types_section = match ctx.edge_types {
        Some(edge_types) if !is_falsy(edge_types) => {
            let edge_types = to_prompt_json(edge_types);
            format!(
                r#"
<FACT_TYPES>
{edge_types}
</FACT_TYPES>
"#
            )
        }
        _ => String::new(),
    };

    let sys_prompt = format!(
        "You are an expert fact extractor that extracts fact triples from text. \
1. Extracted fact triples should also be extracted with relevant date information. \
2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. \
Use each episode's timestamp to resolve temporal references within that episode. \
REFERENCE_TIME is a fallback for when no per-episode timestamp is available.{DO_NOT_ESCAPE_UNICODE}"
    );

    let previous_episodes = to_prompt_json(ctx.previous_episodes);
    let episode_content = ctx.episode_content;
    let nodes = to_prompt_json(ctx.nodes);
    let reference_time = ctx.reference_time;
    let custom_extraction_instructions = ctx.custom_extraction_instructions;

    // NOTE: trailing "\n        " (8 spaces) after the last DATETIME rule is intentional
    // and mirrors the upstream f-string's indented closing triple-quote.
    let user_prompt = format!(
        r#"
<PREVIOUS_MESSAGES>
{previous_episodes}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{episode_content}
</CURRENT_MESSAGE>

<ENTITIES>
{nodes}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
{edge_types_section}
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{custom_extraction_instructions}

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities — `source_entity_name` and `target_entity_name` NEVER refer to the same entity.
3. Prefer facts that involve two distinct entities from the ENTITIES list. When a sentence describes a specific, concrete detail about a single entity (a brand name, a specific item, a physical description, a quantity, a location, a named activity), do NOT drop it. Instead, look for a second entity in the ENTITIES list that the detail relates to and form a proper triple (e.g., Entity -> OWNS -> item-entity, Entity -> LIVES_IN -> place-entity, Entity -> HAS_ATTRIBUTE -> detail-entity). Only skip the fact when no second entity in the ENTITIES list can anchor the detail.
   - BAD: "Alice feels happy" (vague single-entity state with no concrete detail — what is Alice happy about?)
   - GOOD: "Alice feels happy about Bob's promotion" → Alice -> FEELS_HAPPY_ABOUT -> Bob's promotion
   - GOOD: "Nate plays games on a Gamecube" → Nate -> PLAYS_GAMES_ON -> Gamecube (when "Gamecube" is in ENTITIES)
   - GOOD: "Alice congratulated Bob" (relationship between two entities), "Alice lives in Paris" (relationship between entity and place)
4. Do not emit semantically redundant facts, even across episodes within the CURRENT_MESSAGE. However, if a later episode adds specific details to a previously stated fact (e.g., adding a brand name, a count, a color, a location, or any concrete attribute), extract the more detailed version as a NEW fact — it is NOT a duplicate. Only treat facts as duplicates when they convey the same specificity.
   - NOT a duplicate: "user plays video games" (Episode 0) vs. "user plays games on a Gamecube" (Episode 1) → extract the second, more detailed fact.
   - IS a duplicate: "user plays games on a Gamecube" (Episode 0) vs. "user plays Gamecube games" (Episode 1) → extract once, list both episodes in `episode_indices`.
5. The `fact` MUST preserve all specific details from the source text: proper nouns, brand names, product names, model numbers, quantities, counts, colors, materials, physical descriptions, specific items, named locations, and named activities. Paraphrase the sentence structure but NEVER generalize:
   - NEVER generalize "Gamecube" to "gaming console", "Ford Mustang" to "car", "wool coat" to "coat", "red and purple lighting" to "lighting", "cracked windshield" to "car damage", or "three screenplays" to "several screenplays".
   - Do not verbatim quote the original text, but every concrete noun, number, and descriptor in the source should survive into the `fact`.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week"). When the CURRENT_MESSAGE contains multiple episodes with per-episode timestamps, prefer the timestamp of the specific episode the fact originates from.
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
        "#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`extract_timestamps`].
pub struct ExtractTimestampsContext<'a> {
    /// Upstream `context['fact']`.
    pub fact: &'a str,
    /// Upstream `context['reference_time']`.
    pub reference_time: &'a str,
}

/// Verbatim port of upstream `extract_timestamps`.
pub fn extract_timestamps(ctx: &ExtractTimestampsContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You extract temporal bounds from facts. NEVER hallucinate dates.{DO_NOT_ESCAPE_UNICODE}"
    );

    let fact = ctx.fact;
    let reference_time = ctx.reference_time;

    let user_prompt = format!(
        r#"Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

<FACT>
{fact}
</FACT>

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Replicates Python truthiness for `context.get('edge_types')`: an empty list / empty
/// object / null / false / empty string is falsy and suppresses the FACT_TYPES section.
fn is_falsy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Bool(b) => !b,
        serde_json::Value::String(s) => s.is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Object(o) => o.is_empty(),
        // Python: bool(0) == False, bool(non-zero) == True.
        serde_json::Value::Number(n) => n.as_f64().is_some_and(|v| v == 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

    #[test]
    fn edge_renders_verbatim_with_fact_types() {
        let prev =
            serde_json::json!([{"content": "Alice: hi", "timestamp": "2025-04-29T00:00:00+00:00"}]);
        let nodes = serde_json::json!([{"name": "Alice"}, {"name": "Acme Corp"}]);
        let edge_types = serde_json::json!([{"fact_type_name": "WORKS_AT"}]);
        let ctx = EdgeContext {
            previous_episodes: &prev,
            episode_content: "Alice works at Acme Corp.",
            nodes: &nodes,
            reference_time: "2025-04-30T00:00:00Z",
            edge_types: Some(&edge_types),
            custom_extraction_instructions: "",
        };
        let msgs = edge(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an expert fact extractor that extracts fact triples from text.")
        );
        let u = &msgs[1].content;
        assert!(u.contains("# TASK\nExtract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE."));
        assert!(u.contains("# EXTRACTION RULES"));
        assert!(u.contains(
            "2. Each fact must involve two **distinct** entities — `source_entity_name` and `target_entity_name` NEVER refer to the same entity."
        ));
        assert!(u.contains("Alice works at Acme Corp."));
        // FACT_TYPES section present
        assert!(u.contains("<FACT_TYPES>"));
        // trailing whitespace preserved
        assert!(u.ends_with("If only a year is mentioned, use January 1st at 00:00:00.\n        "));
    }

    #[test]
    fn edge_omits_fact_types_when_absent() {
        let prev = serde_json::json!([]);
        let nodes = serde_json::json!([{"name": "Alice"}]);
        let ctx = EdgeContext {
            previous_episodes: &prev,
            episode_content: "Alice exists.",
            nodes: &nodes,
            reference_time: "2025-04-30T00:00:00Z",
            edge_types: None,
            custom_extraction_instructions: "",
        };
        let msgs = edge(&ctx);
        let u = &msgs[1].content;
        assert!(!u.contains("<FACT_TYPES>"));
        // empty section: the {edge_types_section} placeholder sits on its own line, so the
        // REFERENCE_TIME block is followed by a blank line then # TASK (matches upstream).
        assert!(u.contains("</REFERENCE_TIME>\n\n# TASK"));
    }

    #[test]
    fn edge_omits_fact_types_when_empty_list() {
        let prev = serde_json::json!([]);
        let nodes = serde_json::json!([{"name": "Alice"}]);
        let empty = serde_json::json!([]);
        let ctx = EdgeContext {
            previous_episodes: &prev,
            episode_content: "Alice exists.",
            nodes: &nodes,
            reference_time: "2025-04-30T00:00:00Z",
            edge_types: Some(&empty),
            custom_extraction_instructions: "",
        };
        let msgs = edge(&ctx);
        assert!(!msgs[1].content.contains("<FACT_TYPES>"));
    }

    #[test]
    fn zero_number_suppresses_fact_types_block() {
        // Python truthiness: bool(0) == False → FACT_TYPES should be omitted.
        let prev = serde_json::json!([]);
        let nodes = serde_json::json!([{"name": "Alice"}]);
        let zero = serde_json::json!(0);
        let ctx = EdgeContext {
            previous_episodes: &prev,
            episode_content: "Alice exists.",
            nodes: &nodes,
            reference_time: "2025-04-30T00:00:00Z",
            edge_types: Some(&zero),
            custom_extraction_instructions: "",
        };
        let msgs = edge(&ctx);
        assert!(!msgs[1].content.contains("<FACT_TYPES>"));
    }

    #[test]
    fn nonzero_number_keeps_fact_types_block() {
        // Python truthiness: bool(1) == True → FACT_TYPES should be present.
        let prev = serde_json::json!([]);
        let nodes = serde_json::json!([{"name": "Alice"}]);
        let one = serde_json::json!(1);
        let ctx = EdgeContext {
            previous_episodes: &prev,
            episode_content: "Alice exists.",
            nodes: &nodes,
            reference_time: "2025-04-30T00:00:00Z",
            edge_types: Some(&one),
            custom_extraction_instructions: "",
        };
        let msgs = edge(&ctx);
        assert!(msgs[1].content.contains("<FACT_TYPES>"));
    }

    #[test]
    fn extract_timestamps_renders_verbatim() {
        let ctx = ExtractTimestampsContext {
            fact: "Alice joined Acme in 2020.",
            reference_time: "2025-04-30T00:00:00Z",
        };
        let msgs = extract_timestamps(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You extract temporal bounds from facts. NEVER hallucinate dates.")
        );
        let u = &msgs[1].content;
        assert!(u.starts_with(
            "Given a FACT and its REFERENCE TIME, determine when the fact became true"
        ));
        assert!(u.contains("- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)."));
        assert!(u.contains("- Do NOT hallucinate or infer dates from unrelated events."));
        assert!(u.contains("Alice joined Acme in 2020."));
    }
}
