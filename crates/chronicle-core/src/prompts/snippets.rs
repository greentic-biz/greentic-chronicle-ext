// Ported from graphiti_core/prompts/snippets.py @ 34f56e65 (v0.29.1)
//
// Upstream `summary_instructions` is an f-string interpolating MAX_SUMMARY_CHARS.
// MAX_SUMMARY_CHARS = 1000 (graphiti_core/utils/text_utils.py @ 34f56e65, verified).
// The f-string is resolved to the literal "1000" here, byte-for-byte; the rest of
// the text (including the leading indentation on each line, which upstream carries
// because the f-string is indented inside the module) is preserved verbatim.

/// Verbatim port of upstream `summary_instructions` with MAX_SUMMARY_CHARS resolved
/// to 1000. The leading whitespace on each line mirrors the upstream f-string exactly.
pub const SUMMARY_INSTRUCTIONS: &str = r#"Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER 1000 CHARACTERS.
        4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
        5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
        6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
        9. Include temporal anchors when the messages provide them and they help ground the fact.
        10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

        Example summary:
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
        "#;
