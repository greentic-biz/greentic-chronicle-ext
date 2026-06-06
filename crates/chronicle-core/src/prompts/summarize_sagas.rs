// Ported from graphiti_core/prompts/summarize_sagas.py @ 34f56e65 (v0.29.1)
//
// Fidelity notes:
// - DO_NOT_ESCAPE_UNICODE appended to the system prompt (VersionWrapper behaviour).
// - Upstream's user f-string is a top-level function f-string (NOT a class-method
//   body), so it carries NO per-line leading indentation. The content begins at
//   "NEVER use meta-language verbs:" and we reproduce every line with no leading
//   whitespace, byte-for-byte.
// - `episodes` are joined with the literal separator "\n---\n" (upstream
//   `'\n---\n'.join(episodes)`); an empty list renders "(no messages)".
// - `existing_summary` is conditional: when non-empty, the <EXISTING_KNOWLEDGE>
//   block plus the merge instruction are injected verbatim where upstream
//   interpolates `{existing_summary_section}`; when empty, that interpolation is
//   an empty string (the surrounding blank line collapses, exactly as upstream).
// - MAX_SUMMARY_CHARS = 1000 (graphiti_core/utils/text_utils.py @ 34f56e65); the
//   f-string `{MAX_SUMMARY_CHARS}` is resolved to the literal "1000" byte-for-byte.
// - `{saga_name}` interpolated raw (upstream reads it directly into the f-string).

use crate::llm::Message;
use crate::prompts::helpers::DO_NOT_ESCAPE_UNICODE;

/// Context for [`summarize_saga`].
///
/// Mirrors upstream `context = {'saga_name', 'existing_summary', 'episodes'}`.
pub struct SummarizeSagaContext<'a> {
    /// Upstream `context['saga_name']` (interpolated raw, defaults to "Unknown"
    /// upstream when absent; the caller always supplies the saga's name here).
    pub saga_name: &'a str,
    /// Upstream `context['existing_summary']`. When empty, the
    /// <EXISTING_KNOWLEDGE> section is omitted (upstream conditional).
    pub existing_summary: &'a str,
    /// Upstream `context['episodes']` — the raw episode contents. Joined with
    /// "\n---\n"; an empty slice renders "(no messages)".
    pub episodes: &'a [String],
}

/// Verbatim port of upstream `summarize_saga`.
pub fn summarize_saga(ctx: &SummarizeSagaContext<'_>) -> Vec<Message> {
    let saga_name = ctx.saga_name;

    // Upstream: episodes_text = '\n---\n'.join(episodes) if episodes else '(no messages)'.
    let episodes_text = if ctx.episodes.is_empty() {
        "(no messages)".to_string()
    } else {
        ctx.episodes.join("\n---\n")
    };

    // Upstream existing_summary_section: empty string when existing_summary is
    // falsy, else the <EXISTING_KNOWLEDGE> block + merge instruction.
    let existing_summary_section = if ctx.existing_summary.is_empty() {
        String::new()
    } else {
        let existing_summary = ctx.existing_summary;
        format!(
            "\n<EXISTING_KNOWLEDGE>\n{existing_summary}\n</EXISTING_KNOWLEDGE>\nThe EXISTING_KNOWLEDGE contains previously extracted facts. Merge any new facts from MESSAGES into it. When newer messages contradict older facts, prefer the newer fact. If MESSAGES add no new durable facts, return the existing knowledge unchanged.\n"
        )
    };

    let sys_prompt = format!(
        "You extract durable knowledge from message threads. Output a factual knowledge brief — facts, decisions, preferences, plans, entities, and relationships — that stands alone without reference to the original messages. Stay under 1000 characters.{DO_NOT_ESCAPE_UNICODE}"
    );

    let user_prompt = format!(
        r#"NEVER use meta-language verbs: "mentioned", "discussed", "noted", "stated", "described", "referenced", "indicated", "reported", "talked about", "brought up" — these describe conversational dynamics, not knowledge. State facts directly instead.
NEVER refer to the messages, conversation, thread, or participants' communicative acts. The output must read as if no conversation happened — only the facts matter.
NEVER begin with "This conversation", "The thread", "In this thread", or "The discussion".
NEVER infer preferences or habits from a single passing mention. When a person explicitly states a preference ("I prefer X", "I love X", "I always do X"), capture it as a stated preference attributed to that person.

Your task: extract all durable knowledge from the MESSAGES below and produce a factual knowledge brief for the topic "{saga_name}".

Capture explicitly stated:
- Facts and concrete details (names, dates, numbers, locations)
- Decisions and their outcomes
- Preferences and requirements (when a person explicitly claims them)
- Plans, next steps, and commitments
- Relationships between entities (who works where, who owns what)
- State changes (what was X, now is Y)

Write 2-6 dense sentences. Use third person. Preserve all names, dates, counts, and temporal qualifiers. Lead with the most important fact or decision.
{existing_summary_section}
<MESSAGES>
{episodes_text}
</MESSAGES>

<EXAMPLES>
MESSAGES: "Jordan: We decided to move the deployment to March 15 instead of March 8. The staging environment isn't ready.\n---\nPriya: Agreed. I'll update the client timeline. We also need to switch from PostgreSQL to CockroachDB for the multi-region requirement."
GOOD: "Deployment moved from March 8 to March 15 because the staging environment is not ready. Priya owns updating the client timeline. The database is switching from PostgreSQL to CockroachDB to support the multi-region requirement."
BAD: "Jordan mentioned moving the deployment date. Priya discussed updating the timeline and talked about switching databases. The team noted staging issues."
</EXAMPLES>

<EXAMPLES>
MESSAGES: "Alex: I tried the new Thai place on Elm Street last night — the pad see ew was incredible. Definitely going back.\n---\nMina: Oh nice, I've been wanting to try that. Is it the one next to the bookstore?\n---\nAlex: Yeah, Siam Kitchen. They're open until 11 PM on weekends."
GOOD: "Siam Kitchen is a Thai restaurant on Elm Street, next to a bookstore, open until 11 PM on weekends. Alex considers the pad see ew excellent."
BAD: "Alex mentioned trying a new Thai place and discussed the pad see ew. Mina asked about the location. Alex noted it was Siam Kitchen and stated the weekend hours."
</EXAMPLES>

<EXAMPLES>
MESSAGES: "Sam: I really prefer working in the mornings — I'm way more productive before noon.\n---\nDana: Same. I've been blocking 9-11 AM for deep work. Also, I can't stand Jira — can we move the tracker to Linear?\n---\nSam: Fine by me. I'll set up the workspace."
GOOD: "Sam prefers morning work and reports higher productivity before noon. Dana blocks 9-11 AM for deep work. Dana prefers Linear over Jira for issue tracking. Sam is setting up the Linear workspace."
BAD: "Sam and Dana discussed their work preferences. They talked about morning productivity and mentioned switching from Jira to Linear."
</EXAMPLES>
"#
    );

    vec![Message::system(sys_prompt), Message::user(user_prompt)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;

    #[test]
    fn summarize_saga_renders_verbatim_without_existing_summary() {
        let episodes = vec![
            "Jordan: moving deploy to March 15.".to_string(),
            "Priya: I'll update the timeline.".to_string(),
        ];
        let ctx = SummarizeSagaContext {
            saga_name: "deployment-planning",
            existing_summary: "",
            episodes: &episodes,
        };
        let msgs = summarize_saga(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);

        // System: verbatim sentence + DO_NOT_ESCAPE_UNICODE suffix.
        assert!(msgs[0].content.ends_with(DO_NOT_ESCAPE_UNICODE));
        assert!(msgs[0].content.contains(
            "You extract durable knowledge from message threads. Output a factual knowledge brief"
        ));
        assert!(msgs[0].content.contains("Stay under 1000 characters."));

        let u = &msgs[1].content;
        // Verbatim leading sentence.
        assert!(u.starts_with(
            "NEVER use meta-language verbs: \"mentioned\", \"discussed\", \"noted\", \"stated\","
        ));
        assert!(
            u.contains("produce a factual knowledge brief for the topic \"deployment-planning\".")
        );
        // Episodes joined with the literal "\n---\n" separator.
        assert!(
            u.contains("Jordan: moving deploy to March 15.\n---\nPriya: I'll update the timeline.")
        );
        // No existing-knowledge block when summary is empty.
        assert!(!u.contains("<EXISTING_KNOWLEDGE>"));
        // Annotated examples present (verbatim GOOD/BAD).
        assert!(u.contains("Siam Kitchen is a Thai restaurant on Elm Street"));
        assert!(
            u.contains("Sam prefers morning work and reports higher productivity before noon.")
        );
    }

    #[test]
    fn summarize_saga_injects_existing_summary_branch() {
        let episodes = vec!["Sam: new fact today.".to_string()];
        let ctx = SummarizeSagaContext {
            saga_name: "support",
            existing_summary: "Sam owns the deployment.",
            episodes: &episodes,
        };
        let msgs = summarize_saga(&ctx);
        let u = &msgs[1].content;
        assert!(
            u.contains("<EXISTING_KNOWLEDGE>\nSam owns the deployment.\n</EXISTING_KNOWLEDGE>")
        );
        assert!(u.contains(
            "The EXISTING_KNOWLEDGE contains previously extracted facts. Merge any new facts from MESSAGES into it."
        ));
        assert!(u.contains(
            "If MESSAGES add no new durable facts, return the existing knowledge unchanged."
        ));
    }

    #[test]
    fn summarize_saga_empty_episodes_renders_placeholder() {
        let ctx = SummarizeSagaContext {
            saga_name: "empty",
            existing_summary: "",
            episodes: &[],
        };
        let msgs = summarize_saga(&ctx);
        assert!(
            msgs[1]
                .content
                .contains("<MESSAGES>\n(no messages)\n</MESSAGES>")
        );
    }
}
