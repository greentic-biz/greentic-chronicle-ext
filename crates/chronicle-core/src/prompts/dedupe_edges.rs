// Ported from graphiti_core/prompts/dedupe_edges.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE appended to the system prompt (VersionWrapper behaviour).
// - `existing_edges`, `edge_invalidation_candidates`, `new_edge` are interpolated RAW
//   upstream (no to_prompt_json). They carry already-formatted text, so the context
//   fields are `&str`.
// - Literal `{}` only appears inside the JSON-ish EXAMPLE block as `[...]` etc; there are
//   no f-string braces to escape in the example list values here, but the format! call
//   still treats any stray `{`/`}` as placeholders — this prompt contains none in its
//   static text, so no extra escaping is needed beyond the interpolation placeholders.

use crate::llm::Message;
use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

/// Context for [`resolve_edge`].
pub struct ResolveEdgeContext<'a> {
    /// Upstream `context['existing_edges']` (interpolated raw).
    pub existing_edges: &'a str,
    /// Upstream `context['edge_invalidation_candidates']` (interpolated raw).
    pub edge_invalidation_candidates: &'a str,
    /// Upstream `context['new_edge']` (interpolated raw).
    pub new_edge: &'a str,
}

/// Verbatim port of upstream `resolve_edge`.
pub fn resolve_edge(ctx: &ResolveEdgeContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are a fact deduplication assistant. \
NEVER mark facts with key differences as duplicates.{DO_NOT_ESCAPE_UNICODE}"
    );

    let existing_edges = ctx.existing_edges;
    let edge_invalidation_candidates = ctx.edge_invalidation_candidates;
    let new_edge = ctx.new_edge;

    let user_prompt = format!(
        r#"
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{edge_invalidation_candidates}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
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
    fn resolve_edge_renders_verbatim() {
        let ctx = ResolveEdgeContext {
            existing_edges: "idx=0, \"Alice joined Acme Corp in 2020\"",
            edge_invalidation_candidates: "idx=1, \"Alice works at Acme Corp\"",
            new_edge: "Alice joined Acme Corp in 2020",
        };
        let msgs = resolve_edge(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(
            msgs[0]
                .content
                .contains("You are a fact deduplication assistant.")
        );
        let u = &msgs[1].content;
        assert!(u.contains(
            "NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers."
        ));
        assert!(u.contains("1. DUPLICATE DETECTION:"));
        assert!(u.contains(
            "You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists."
        ));
        assert!(u.contains("idx=0, \"Alice joined Acme Corp in 2020\""));
        assert!(u.contains("Alice joined Acme Corp in 2020"));
    }
}
