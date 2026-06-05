// Ported from graphiti_core/prompts/dedupe_nodes.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE appended to the system prompt (VersionWrapper behaviour).
// - Only the `nodes` function is ported in the Phase 1 set.
// - `extracted_nodes`, `previous_episodes`, `existing_nodes` rendered via to_prompt_json.
// - `episode_content` interpolated raw.
// - Upstream interpolates `len(context['extracted_nodes'])` and
//   `len(context['extracted_nodes']) - 1` three times. We derive the count from the
//   slice length so it stays consistent with the rendered ENTITIES list.
// - Literal `{}` braces appear in the prose ("such as {}" / JSON examples). Inside the
//   format! raw string those are escaped as `{{` / `}}` so the rendered output contains
//   single braces, byte-identical to upstream.

use crate::llm::Message;
use crate::prompts::helpers::{DO_NOT_ESCAPE_UNICODE, to_prompt_json};

/// Context for [`nodes`].
pub struct NodesContext<'a> {
    /// Upstream `context['previous_episodes']`, rendered via to_prompt_json.
    pub previous_episodes: &'a [String],
    /// Upstream `context['episode_content']`.
    pub episode_content: &'a str,
    /// Upstream `context['extracted_nodes']`, rendered via to_prompt_json. The slice
    /// length supplies `len(context['extracted_nodes'])` interpolations.
    pub extracted_nodes: &'a serde_json::Value,
    /// Upstream `context['existing_nodes']`, rendered via to_prompt_json.
    pub existing_nodes: &'a serde_json::Value,
}

/// Verbatim port of upstream `nodes`.
pub fn nodes(ctx: &NodesContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are an entity deduplication assistant. \
NEVER fabricate entity names or mark distinct entities as duplicates.{DO_NOT_ESCAPE_UNICODE}"
    );

    let previous_episodes = to_prompt_json(&ctx.previous_episodes);
    let episode_content = ctx.episode_content;
    let extracted_nodes = to_prompt_json(ctx.extracted_nodes);
    let existing_nodes = to_prompt_json(ctx.existing_nodes);

    // len(context['extracted_nodes']) — only arrays carry a length upstream.
    // Upstream never calls this with an empty list; the debug_assert documents that
    // invariant and guards against count underflow in the subtraction below.
    let count = ctx
        .extracted_nodes
        .as_array()
        .map(|a| a.len() as i64)
        .unwrap_or(0);
    debug_assert!(
        count > 0,
        "nodes() called with empty extracted_nodes - prompt will be malformed"
    );
    let count_minus_one = count.saturating_sub(1);

    let user_prompt = format!(
        r#"
<PREVIOUS MESSAGES>
{previous_episodes}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{episode_content}
</CURRENT MESSAGE>

<ENTITIES>
{extracted_nodes}
</ENTITIES>

<EXISTING ENTITIES>
{existing_nodes}
</EXISTING ENTITIES>

Each of the above ENTITIES was extracted from the CURRENT MESSAGE.
For each entity, determine if it is a duplicate of any EXISTING ENTITY.
Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains {count} entities with IDs 0 through {count_minus_one}.
Your response MUST include EXACTLY {count} resolutions with IDs 0 through {count_minus_one}. Do not skip or add IDs.

For every entity, provide:
- `id`: integer id from ENTITIES
- `name`: the best full name for the entity (preserve the original name unless a duplicate has a more complete name)
- `duplicate_candidate_id`: the `candidate_id` of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate

<EXAMPLE>
ENTITY: "Sam" (Person)
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}}]
Result: duplicate_candidate_id = 0 (same person referenced in conversation)

ENTITY: "NYC"
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "New York City", "entity_types": ["Location"]}}, {{"candidate_id": 1, "name": "New York Knicks", "entity_types": ["Organization"]}}]
Result: duplicate_candidate_id = 0 (same location, abbreviated name)

ENTITY: "Java" (programming language)
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Java", "entity_types": ["Location"], "summary": "An island in Indonesia"}}]
Result: duplicate_candidate_id = -1 (same name but distinct real-world things)

ENTITY: "Marco's car"
EXISTING ENTITIES: [{{"candidate_id": 0, "name": "Marco's vehicle", "entity_types": ["Entity"], "summary": "Marco drives a red sedan."}}]
Result: duplicate_candidate_id = 0 (synonym — "car" and "vehicle" refer to the same thing, same possessor)
</EXAMPLE>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

    #[test]
    fn nodes_renders_verbatim() {
        let prev = vec!["Sam: hi".to_string()];
        let extracted = serde_json::json!([{"id": 0, "name": "Sam"}, {"id": 1, "name": "NYC"}]);
        let existing = serde_json::json!([{"candidate_id": 0, "name": "Sam"}]);
        let ctx = NodesContext {
            previous_episodes: &prev,
            episode_content: "Sam visited NYC.",
            extracted_nodes: &extracted,
            existing_nodes: &existing,
        };
        let msgs = nodes(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are an entity deduplication assistant.")
        );
        let u = &msgs[1].content;
        assert!(u.contains("Each of the above ENTITIES was extracted from the CURRENT MESSAGE."));
        assert!(u.contains(
            "Entities should only be considered duplicates if they refer to the *same real-world object or concept*."
        ));
        assert!(u.contains("- They are related but distinct."));
        // count interpolation
        assert!(u.contains("ENTITIES contains 2 entities with IDs 0 through 1."));
        assert!(u.contains("Your response MUST include EXACTLY 2 resolutions with IDs 0 through 1. Do not skip or add IDs."));
        // literal braces in example rendered as single braces
        assert!(u.contains(r#"EXISTING ENTITIES: [{"candidate_id": 0, "name": "Sam", "entity_types": ["Person"], "summary": "Sam enjoys hiking and photography"}]"#));
        // raw episode content interpolated
        assert!(u.contains("Sam visited NYC."));
    }
}
