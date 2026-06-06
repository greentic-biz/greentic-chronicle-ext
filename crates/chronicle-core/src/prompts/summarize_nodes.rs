// Ported from graphiti_core/prompts/summarize_nodes.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE appended to the system prompt (VersionWrapper behaviour).
// - Phase 1 ported `summarize_context`; Phase 4 (Task 3) adds `summarize_pair`
//   and `summary_description` (community build/update).
// - The upstream user f-string is indented 8 spaces per line (it lives inside the function
//   body) and the prompt text carries that indentation verbatim. We reproduce every line's
//   leading 8 spaces. The content starts with "\n        Given the MESSAGES…" and ends with
//   "\n        " (the indented closing triple-quote).
// - `{summary_instructions}` interpolates the ported SUMMARY_INSTRUCTIONS constant from
//   prompts::snippets, which itself carries the upstream 8-space indentation.
// - `previous_episodes`, `episode_content`, `attributes` rendered via to_prompt_json.
// - `node_name`, `node_summary` interpolated raw.
// - MAX_SUMMARY_CHARS = 1000 (graphiti_core/utils/text_utils.py @ 34f56e65, verified);
//   the f-string `{MAX_SUMMARY_CHARS}` is resolved to the literal "1000" byte-for-byte.

use crate::llm::Message;
use crate::prompts::helpers::{DO_NOT_ESCAPE_UNICODE, to_prompt_json};
use crate::prompts::snippets::SUMMARY_INSTRUCTIONS;

/// Context for [`summarize_context`].
pub struct SummarizeContext<'a> {
    /// Upstream `context['previous_episodes']` — a JSON array of
    /// `{"content": ..., "timestamp": ...}` objects (see
    /// `pipeline::node_ops::previous_episodes_context`), rendered via to_prompt_json.
    pub previous_episodes: &'a serde_json::Value,
    /// Upstream `context['episode_content']`, rendered via to_prompt_json.
    pub episode_content: &'a serde_json::Value,
    /// Upstream `context['node_name']` (interpolated raw).
    pub node_name: &'a str,
    /// Upstream `context['node_summary']` (interpolated raw).
    pub node_summary: &'a str,
    /// Upstream `context['attributes']`, rendered via to_prompt_json.
    pub attributes: &'a serde_json::Value,
}

/// Verbatim port of upstream `summarize_context`.
pub fn summarize_context(ctx: &SummarizeContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text.{DO_NOT_ESCAPE_UNICODE}"
    );

    let summary_instructions = SUMMARY_INSTRUCTIONS;
    let previous_episodes = to_prompt_json(ctx.previous_episodes);
    let episode_content = to_prompt_json(ctx.episode_content);
    let node_name = ctx.node_name;
    let node_summary = ctx.node_summary;
    let attributes = to_prompt_json(ctx.attributes);

    let user_prompt = format!(
        r#"
        Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use
        information from the provided MESSAGES. Your summary should also only contain information relevant to the
        provided ENTITY.

        In addition, extract any values for the provided entity properties based on their descriptions.
        If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None.

        {summary_instructions}

        <MESSAGES>
        {previous_episodes}
        {episode_content}
        </MESSAGES>

        <ENTITY>
        {node_name}
        </ENTITY>

        <ENTITY CONTEXT>
        {node_summary}
        </ENTITY CONTEXT>

        <ATTRIBUTES>
        {attributes}
        </ATTRIBUTES>
        "#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`summarize_pair`].
///
/// Upstream `context['node_summaries']` is `[{'summary': s} for s in pair]` —
/// a two-element JSON array of `{"summary": ...}` objects, rendered via
/// `to_prompt_json`. We hold the two raw summary strings and build that array
/// shape at render time so the JSON is byte-identical to upstream.
pub struct SummarizePairContext<'a> {
    /// Left/first summary of the pair (upstream `summary_pair[0]`).
    pub left_summary: &'a str,
    /// Right/second summary of the pair (upstream `summary_pair[1]`).
    pub right_summary: &'a str,
}

/// Verbatim port of upstream `summarize_pair`.
pub fn summarize_pair(ctx: &SummarizePairContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are a helpful assistant that combines summaries into a single dense factual summary.{DO_NOT_ESCAPE_UNICODE}"
    );

    // Upstream: context['node_summaries'] = [{'summary': s} for s in summary_pair].
    let node_summaries = serde_json::json!([
        {"summary": ctx.left_summary},
        {"summary": ctx.right_summary},
    ]);
    let node_summaries = to_prompt_json(&node_summaries);

    let user_prompt = format!(
        r#"
        Synthesize the information from the following two summaries into a single information-dense summary.

        IMPORTANT:
        - Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
        - Prefer compact factual sentences over vague thematic phrasing.
        - When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        - Use communication verbs only when the act of speaking, asking, sharing, presenting, or announcing is itself the important fact.
        - Avoid filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", and "indicated" unless the communication act itself matters.
        - SUMMARIES MUST BE LESS THAN 1000 CHARACTERS.

        Summaries:
        {node_summaries}
        "#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

/// Context for [`summary_description`].
pub struct SummaryDescriptionContext<'a> {
    /// Upstream `context['summary']`, rendered via `to_prompt_json`.
    pub summary: &'a str,
}

/// Verbatim port of upstream `summary_description`.
pub fn summary_description(ctx: &SummaryDescriptionContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are a helpful assistant that describes provided contents in a single sentence.{DO_NOT_ESCAPE_UNICODE}"
    );

    // Upstream interpolates context['summary'] through to_prompt_json — for a
    // plain string this yields the JSON-quoted form (e.g. "foo").
    let summary = to_prompt_json(&ctx.summary);

    let user_prompt = format!(
        r#"
        Create a short one sentence description of the summary that explains what kind of information is summarized.
        Summaries must be under 1000 characters.

        Summary:
        {summary}
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
    fn summarize_context_renders_verbatim() {
        let prev =
            serde_json::json!([{"content": "Mina: hi", "timestamp": "2025-04-29T00:00:00+00:00"}]);
        let episode = serde_json::json!("Jordan presented a ceramics workshop.");
        let attributes = serde_json::json!({"role": "instructor"});
        let ctx = SummarizeContext {
            previous_episodes: &prev,
            episode_content: &episode,
            node_name: "Jordan Lee",
            node_summary: "Jordan Lee works at Belmont Arts Center.",
            attributes: &attributes,
        };
        let msgs = summarize_context(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(msgs[0].content.contains(
            "You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text."
        ));
        let u = &msgs[1].content;
        assert!(u.contains(
            "Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use"
        ));
        assert!(u.contains(
            "If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None."
        ));
        // summary_instructions snippet present
        assert!(u.contains(
            "1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints."
        ));
        assert!(u.contains("STATE FACTS DIRECTLY IN UNDER 1000 CHARACTERS."));
        // raw node_name interpolated
        assert!(u.contains("Jordan Lee"));
        assert!(u.contains("Jordan Lee works at Belmont Arts Center."));
    }

    #[test]
    fn summarize_pair_renders_verbatim() {
        let ctx = SummarizePairContext {
            left_summary: "Acme makes ceramics.",
            right_summary: "Acme is based in Belmont.",
        };
        let msgs = summarize_pair(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(msgs[0].content.contains(
            "You are a helpful assistant that combines summaries into a single dense factual summary."
        ));
        let u = &msgs[1].content;
        assert!(u.contains(
            "Synthesize the information from the following two summaries into a single information-dense summary."
        ));
        assert!(u.contains("SUMMARIES MUST BE LESS THAN 1000 CHARACTERS."));
        // node_summaries rendered as a JSON array of {"summary": ...} objects,
        // byte-identical to upstream to_prompt_json (Python default separators).
        assert!(u.contains(
            r#"[{"summary": "Acme makes ceramics."}, {"summary": "Acme is based in Belmont."}]"#
        ));
    }

    #[test]
    fn summary_description_renders_verbatim() {
        let ctx = SummaryDescriptionContext {
            summary: "Acme makes ceramics in Belmont.",
        };
        let msgs = summary_description(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(msgs[0].content.contains(
            "You are a helpful assistant that describes provided contents in a single sentence."
        ));
        let u = &msgs[1].content;
        assert!(u.contains(
            "Create a short one sentence description of the summary that explains what kind of information is summarized."
        ));
        assert!(u.contains("Summaries must be under 1000 characters."));
        // to_prompt_json on a string yields the JSON-quoted form.
        assert!(u.contains(r#""Acme makes ceramics in Belmont.""#));
    }
}
