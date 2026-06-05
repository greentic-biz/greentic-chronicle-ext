// Ported from graphiti_core/prompts/summarize_nodes.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE appended to the system prompt (VersionWrapper behaviour).
// - Only `summarize_context` is ported in the Phase 1 set.
// - The upstream user f-string is indented 8 spaces per line (it lives inside the function
//   body) and the prompt text carries that indentation verbatim. We reproduce every line's
//   leading 8 spaces. The content starts with "\n        Given the MESSAGES…" and ends with
//   "\n        " (the indented closing triple-quote).
// - `{summary_instructions}` interpolates the ported SUMMARY_INSTRUCTIONS constant from
//   prompts::snippets, which itself carries the upstream 8-space indentation.
// - `previous_episodes`, `episode_content`, `attributes` rendered via to_prompt_json.
// - `node_name`, `node_summary` interpolated raw.

use crate::llm::Message;
use crate::prompts::helpers::{DO_NOT_ESCAPE_UNICODE, to_prompt_json};
use crate::prompts::snippets::SUMMARY_INSTRUCTIONS;

/// Context for [`summarize_context`].
pub struct SummarizeContextContext<'a> {
    /// Upstream `context['previous_episodes']`, rendered via to_prompt_json.
    pub previous_episodes: &'a [String],
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
pub fn summarize_context(ctx: &SummarizeContextContext<'_>) -> Vec<Message> {
    let sys_prompt = format!(
        "You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text.{DO_NOT_ESCAPE_UNICODE}"
    );

    let summary_instructions = SUMMARY_INSTRUCTIONS;
    let previous_episodes = to_prompt_json(&ctx.previous_episodes);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

    #[test]
    fn summarize_context_renders_verbatim() {
        let prev = vec!["Mina: hi".to_string()];
        let episode = serde_json::json!("Jordan presented a ceramics workshop.");
        let attributes = serde_json::json!({"role": "instructor"});
        let ctx = SummarizeContextContext {
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
}
