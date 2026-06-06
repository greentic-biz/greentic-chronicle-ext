// Ported from graphiti_core/cross_encoder/openai_reranker_client.py
// @ 34f56e65 (v0.29.1).
//
// Boolean-logprob cross-encoder. For each passage we run a one-token chat
// completion that the model is forced (via `logit_bias`) to answer with either
// " True" or " False"; the log-probability of the chosen top token is converted
// into a relevance score. All passages are scored CONCURRENTLY and the results
// are returned sorted by score, descending.
//
// LOGPROB PATH CONFIRMED (no fallback): async-openai 0.40.3 exposes every
// request field upstream needs on the typed builder —
//   * `logit_bias: Option<HashMap<String, i8>>`   (chat_.rs:862)
//   * `logprobs:   Option<bool>`                  (chat_.rs:867)
//   * `top_logprobs: Option<u8>`                  (chat_.rs:813)
// and the response carries `choices[0].logprobs.content[0].top_logprobs[0]`
// with `{ token: String, logprob: f32 }` (chat_.rs ChatChoiceLogprobs /
// ChatCompletionTokenLogprob / TopLogprobs). The Gemini-style numeric-rating
// fallback documented in the plan (R12 risk row) is therefore NOT used.
//
// VERBATIM upstream prompt: the system + user message strings below (including
// the user message's leading newline and 27-space indentation) reproduce the
// upstream f-string byte-for-byte so the model sees an identical prompt.
//
// ERROR SEMANTICS (faithful port): upstream zips passages with the collected
// scores using `strict=True`. A passage whose response carries no
// top-logprobs hits the `continue` branch and is silently dropped from the
// score list, which makes the score list shorter than the passage list and
// raises `ValueError` from `zip(..., strict=True)` — failing the WHOLE rank
// call. We replicate that: a passage with empty/absent top-logprobs is logged
// via `tracing::warn!` and surfaced as `LlmError::EmptyResponse`, aborting the
// batch rather than silently returning a partial ranking.

use std::collections::HashMap;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use async_trait::async_trait;
use chronicle_core::cross_encoder::CrossEncoderClient;
use chronicle_core::llm::{LlmConfig, LlmError};
use futures::future::try_join_all;

use crate::llm::map_openai_error;

/// Upstream `DEFAULT_MODEL` (openai_reranker_client.py line 31).
pub const DEFAULT_MODEL: &str = "gpt-4.1-nano";

/// Upstream system prompt (openai_reranker_client.py).
const SYSTEM_PROMPT: &str =
    "You are an expert tasked with determining whether the passage is relevant to the query";

/// OpenAI-backed [`CrossEncoderClient`] using boolean log-probabilities.
///
/// Mirrors upstream `OpenAIRerankerClient`. Construction follows the same
/// `LlmConfig` plumbing as [`crate::OpenAiLlm`]: when `api_key` is `None` the
/// async-openai client falls back to the `OPENAI_API_KEY` environment variable.
pub struct OpenAiReranker {
    client: Client<OpenAIConfig>,
    config: LlmConfig,
}

impl OpenAiReranker {
    /// Build a reranker from an [`LlmConfig`].
    ///
    /// API-key / base-URL handling matches [`crate::OpenAiLlm::new`].
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let mut openai_config = OpenAIConfig::new();
        if let Some(api_key) = &config.api_key {
            openai_config = openai_config.with_api_key(api_key.clone());
        }
        if let Some(base_url) = &config.base_url {
            openai_config = openai_config.with_api_base(base_url.clone());
        }
        let client = Client::with_config(openai_config);
        Ok(Self { client, config })
    }

    /// Resolve the model name, applying the upstream `gpt-4.1-nano` default.
    fn model(&self) -> String {
        self.config
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }

    /// Build the verbatim system + user message pair for a single passage.
    ///
    /// The user content reproduces the upstream f-string byte-for-byte,
    /// including the leading newline and the 27-space indentation on each line.
    fn messages(query: &str, passage: &str) -> Result<Vec<ChatCompletionRequestMessage>, LlmError> {
        let build_err = |e: async_openai::error::OpenAIError| {
            LlmError::Transport(format!("building rerank message: {e}"))
        };
        let user_content = format!(
            "\n                           Respond with \"True\" if PASSAGE is relevant to QUERY and \"False\" otherwise.\n                           <PASSAGE>\n                           {passage}\n                           </PASSAGE>\n                           <QUERY>\n                           {query}\n                           </QUERY>\n                           "
        );
        let system = ChatCompletionRequestSystemMessageArgs::default()
            .content(SYSTEM_PROMPT)
            .build()
            .map_err(build_err)?
            .into();
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_content)
            .build()
            .map_err(build_err)?
            .into();
        Ok(vec![system, user])
    }

    /// The upstream `logit_bias` map: `{"6432": 1, "7983": 1}` — the token ids
    /// for " True" / " False", nudged so the one generated token is one of them.
    fn logit_bias() -> HashMap<String, i8> {
        let mut bias = HashMap::new();
        bias.insert("6432".to_string(), 1);
        bias.insert("7983".to_string(), 1);
        bias
    }
}

/// Convert a top log-probability `(token, logprob)` into a relevance score.
///
/// Pure port of the upstream scoring branch (openai_reranker_client.py
/// lines 110-114):
///
/// ```text
/// norm_logprobs = exp(top_logprobs[0].logprob)
/// if top_logprobs[0].token.strip().split(' ')[0].lower() == 'true':
///     score = norm_logprobs
/// else:
///     score = 1 - norm_logprobs
/// ```
///
/// `token.strip().split(' ')[0]` takes the first whitespace-delimited word of
/// the trimmed token (so `" True"`, `"True"`, `"true\n"` all map to `"true"`).
fn score_from_top_logprob(token: &str, logprob: f64) -> f64 {
    let prob = logprob.exp();
    let first_word = token.trim().split(' ').next().unwrap_or("").to_lowercase();
    if first_word == "true" {
        prob
    } else {
        1.0 - prob
    }
}

#[async_trait]
impl CrossEncoderClient for OpenAiReranker {
    async fn rank(&self, query: &str, passages: &[String]) -> Result<Vec<(String, f64)>, LlmError> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.model();
        let logit_bias = Self::logit_bias();

        // One concurrent request per passage. `try_join_all` short-circuits on
        // the first error, matching upstream's fail-the-batch behavior.
        let futures = passages.iter().map(|passage| {
            let client = &self.client;
            let model = model.clone();
            let logit_bias = logit_bias.clone();
            let query = query.to_string();
            let passage = passage.clone();
            async move {
                let messages = Self::messages(&query, &passage)?;
                #[allow(deprecated)]
                let request = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(messages)
                    .temperature(0.0_f32)
                    .max_tokens(1_u32)
                    .logit_bias(logit_bias)
                    .logprobs(true)
                    .top_logprobs(2_u8)
                    .build()
                    .map_err(|e| LlmError::Transport(format!("building rerank request: {e}")))?;

                let response = client
                    .chat()
                    .create(request)
                    .await
                    .map_err(map_openai_error)?;

                // choices[0].logprobs.content[0].top_logprobs[0]
                let top = response
                    .choices
                    .first()
                    .and_then(|c| c.logprobs.as_ref())
                    .and_then(|lp| lp.content.as_ref())
                    .and_then(|content| content.first())
                    .map(|tok| &tok.top_logprobs)
                    .and_then(|tops| tops.first());

                match top {
                    Some(tl) => Ok(score_from_top_logprob(&tl.token, tl.logprob as f64)),
                    None => {
                        // Upstream `continue` → score list shrinks → zip(strict)
                        // raises. We surface that as a batch-level error.
                        tracing::warn!(
                            passage = %passage,
                            "openai reranker: response carried no top-logprobs; failing rank batch"
                        );
                        Err(LlmError::EmptyResponse)
                    }
                }
            }
        });

        let scores: Vec<f64> = try_join_all(futures).await?;

        let mut results: Vec<(String, f64)> = passages.iter().cloned().zip(scores).collect();
        // Descending by score; stable sort preserves input order on ties.
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_true_token_returns_prob() {
        // logprob 0 → prob 1.0; "True" → score = prob.
        let s = score_from_top_logprob("True", 0.0);
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn score_false_token_returns_one_minus_prob() {
        // logprob 0 → prob 1.0; non-true → score = 1 - prob = 0.0.
        let s = score_from_top_logprob("False", 0.0);
        assert!(s.abs() < 1e-12);
    }

    #[test]
    fn score_handles_leading_space_token() {
        // OpenAI tokenizes the biased tokens as " True" / " False".
        let logprob = -0.1_f64;
        let prob = logprob.exp();
        assert!((score_from_top_logprob(" True", logprob) - prob).abs() < 1e-12);
        assert!((score_from_top_logprob(" False", logprob) - (1.0 - prob)).abs() < 1e-12);
    }

    #[test]
    fn score_is_case_insensitive_and_trims() {
        let logprob = -0.5_f64;
        let prob = logprob.exp();
        for tok in ["true", "TRUE", "  tRuE  ", "true\n", "\tTrue"] {
            assert!(
                (score_from_top_logprob(tok, logprob) - prob).abs() < 1e-12,
                "token {tok:?} should map to true-score"
            );
        }
    }

    #[test]
    fn score_first_word_only() {
        // strip().split(' ')[0] → only the first whitespace-delimited word.
        let logprob = -0.2_f64;
        let prob = logprob.exp();
        // "true false" → first word "true" → true-score.
        assert!((score_from_top_logprob("true false", logprob) - prob).abs() < 1e-12);
        // "false true" → first word "false" → false-score.
        assert!((score_from_top_logprob("false true", logprob) - (1.0 - prob)).abs() < 1e-12);
    }

    #[test]
    fn score_exp_math_for_typical_logprob() {
        // A realistic high-confidence "True": logprob -0.0001 → prob ~0.9999.
        let logprob = -0.0001_f64;
        let expected = logprob.exp();
        let s = score_from_top_logprob(" True", logprob);
        assert!((s - expected).abs() < 1e-12);
        assert!(s > 0.999 && s <= 1.0);
    }

    #[test]
    fn empty_token_treated_as_false() {
        // Defensive: empty/blank token is not "true" → false-branch.
        let s = score_from_top_logprob("", 0.0);
        assert!(s.abs() < 1e-12);
    }

    #[test]
    fn logit_bias_matches_upstream() {
        let bias = OpenAiReranker::logit_bias();
        assert_eq!(bias.get("6432"), Some(&1));
        assert_eq!(bias.get("7983"), Some(&1));
        assert_eq!(bias.len(), 2);
    }

    #[test]
    fn default_model_is_nano() {
        let r = OpenAiReranker::new(LlmConfig::default()).unwrap();
        assert_eq!(r.model(), DEFAULT_MODEL);
    }

    #[test]
    fn model_override_honored() {
        let config = LlmConfig {
            model: Some("custom-rerank".to_string()),
            ..LlmConfig::default()
        };
        let r = OpenAiReranker::new(config).unwrap();
        assert_eq!(r.model(), "custom-rerank");
    }

    #[test]
    fn user_message_is_verbatim_upstream() {
        // Verify the exact prompt bytes (including indentation) the model sees.
        let msgs = OpenAiReranker::messages("my query", "my passage").unwrap();
        assert_eq!(msgs.len(), 2);
        let user = match &msgs[1] {
            ChatCompletionRequestMessage::User(u) => u,
            _ => panic!("expected user message"),
        };
        let content = match &user.content {
            async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text(t) => t,
            _ => panic!("expected text content"),
        };
        let expected = "\n                           Respond with \"True\" if PASSAGE is relevant to QUERY and \"False\" otherwise.\n                           <PASSAGE>\n                           my passage\n                           </PASSAGE>\n                           <QUERY>\n                           my query\n                           </QUERY>\n                           ";
        assert_eq!(content, expected);
    }

    // Live API path — requires a real OPENAI_API_KEY. Not run in CI.
    #[tokio::test]
    #[ignore = "live API; needs OPENAI_API_KEY"]
    async fn live_rank_orders_relevant_first() {
        let reranker = OpenAiReranker::new(LlmConfig::default()).unwrap();
        let passages = vec![
            "The Eiffel Tower is located in Paris, France.".to_string(),
            "Bananas are a good source of potassium.".to_string(),
        ];
        let ranked = reranker
            .rank("Where is the Eiffel Tower?", &passages)
            .await
            .unwrap();
        assert_eq!(ranked.len(), 2);
        // The geography passage should rank above the nutrition passage.
        assert!(ranked[0].0.contains("Eiffel"));
        assert!(ranked[0].1 >= ranked[1].1);
    }
}
