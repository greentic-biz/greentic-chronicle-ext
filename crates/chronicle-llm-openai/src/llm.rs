// Ported from graphiti_core/llm_client/openai_generic_client.py and
// openai_base_client.py @ 34f56e65 (v0.29.1).
//
// Defaults carried over verbatim from upstream openai_base_client.py:
//   DEFAULT_MODEL       = "gpt-4.1-mini"
//   DEFAULT_SMALL_MODEL = "gpt-4.1-nano"
//   temperature         = 0   (chronicle_core::llm::DEFAULT_TEMPERATURE)
//   max_tokens          = 16384 (chronicle_core::llm::DEFAULT_MAX_TOKENS)
//
// response_format: upstream openai_generic_client builds
//   {"type": "json_schema", "json_schema": {"name", "schema"}} and does NOT
//   set `strict`. We mirror that exactly: `strict` stays `None` (omitted from
//   the wire payload). Without a `response_model` upstream falls back to
//   {"type": "json_object"}; here we leave `response_format` unset when no
//   schema is supplied (the json_object fallback is unnecessary because the
//   chronicle pipeline always provides a schema via `generate_typed`).
//
// Retry / typing layering (carried-forward design note from Task 4 review):
//   * `with_retry` wraps ONLY the provider call inside `generate()`. Provider-
//     malformed JSON (the API returns non-JSON text) maps to
//     `LlmError::InvalidJson`, which `is_retryable()` reports as retryable —
//     the upstream `json.JSONDecodeError` analog, correctly inside the retry
//     boundary.
//   * serde *struct* deserialization happens in
//     `chronicle_core::llm::generate_typed`, which lives OUTSIDE this retry
//     loop. A struct mismatch (upstream `ValidationError`, non-retryable)
//     therefore never reaches `with_retry`. This client never deserializes
//     into a concrete struct, so it cannot leak a struct-mismatch error into
//     the retry path.

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
};
use async_trait::async_trait;
use chronicle_core::llm::retry::with_retry;
use chronicle_core::llm::{
    LlmClient, LlmConfig, LlmError, LlmRequest, Message, ModelSize, ResponseSchema, Role,
};

/// Upstream `DEFAULT_MODEL` (openai_base_client.py).
pub const DEFAULT_MODEL: &str = "gpt-4.1-mini";
/// Upstream `DEFAULT_SMALL_MODEL` (openai_base_client.py).
pub const DEFAULT_SMALL_MODEL: &str = "gpt-4.1-nano";

/// OpenAI-compatible [`LlmClient`] backed by `async-openai`.
///
/// Works against the OpenAI API or any OpenAI-compatible endpoint via
/// `base_url` (upstream `LLMConfig.base_url`).
pub struct OpenAiLlm {
    client: Client<OpenAIConfig>,
    config: LlmConfig,
}

impl OpenAiLlm {
    /// Build a client from an [`LlmConfig`].
    ///
    /// API-key handling mirrors upstream: upstream passes `config.api_key`
    /// straight to `AsyncOpenAI(api_key=...)`, and the SDK falls back to the
    /// `OPENAI_API_KEY` environment variable when it is `None`. We replicate
    /// that — when `config.api_key` is `None` we leave the async-openai config
    /// at its default, which reads `OPENAI_API_KEY` from the environment.
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

    /// Resolve the model name for the requested size, applying upstream
    /// `_get_model_for_size` fallbacks.
    fn model_for_size(&self, size: ModelSize) -> String {
        match size {
            ModelSize::Small => self
                .config
                .small_model
                .clone()
                .unwrap_or_else(|| DEFAULT_SMALL_MODEL.to_string()),
            ModelSize::Medium => self
                .config
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        }
    }
}

/// Convert a chronicle [`Message`] into an async-openai request message.
///
/// Upstream only forwards `system` and `user` roles; we additionally map
/// `assistant` for completeness (chronicle prompts are system+user today).
fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, LlmError> {
    let build_err = |e: OpenAIError| LlmError::Transport(format!("building message: {e}"));
    let out = match message.role {
        Role::System => ChatCompletionRequestSystemMessageArgs::default()
            .content(message.content.clone())
            .build()
            .map_err(build_err)?
            .into(),
        Role::User => ChatCompletionRequestUserMessageArgs::default()
            .content(message.content.clone())
            .build()
            .map_err(build_err)?
            .into(),
        Role::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
            .content(message.content.clone())
            .build()
            .map_err(build_err)?
            .into(),
    };
    Ok(out)
}

/// Build the `response_format` payload from a chronicle [`ResponseSchema`].
///
/// Mirrors upstream openai_generic_client: `name` + `schema`, `strict` left
/// unset (`None` → omitted on the wire).
fn to_response_format(schema: &ResponseSchema) -> ResponseFormat {
    ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: schema.name.clone(),
            schema: schema.schema.clone(),
            strict: None,
        },
    }
}

/// Map an async-openai error into an [`LlmError`], preserving retryability.
///
/// * HTTP 429 → [`LlmError::RateLimit`] (retryable).
/// * HTTP 5xx → [`LlmError::Server`] (retryable).
/// * other API errors → [`LlmError::Transport`] (non-retryable).
/// * transport/deserialize/other SDK errors → [`LlmError::Transport`].
fn map_openai_error(err: OpenAIError) -> LlmError {
    match err {
        OpenAIError::ApiError(_) => {
            // The `_api` feature exposes the HTTP status via ApiErrorResponse,
            // but async-openai's public `OpenAIError::ApiError` carries the
            // status only when that feature is active. Classify by status code
            // through the dedicated helper so the mapping is unit-testable.
            map_api_status(status_of(&err).unwrap_or(0), err.to_string())
        }
        other => LlmError::Transport(other.to_string()),
    }
}

/// Extract the HTTP status code from an async-openai `ApiError`, if present.
fn status_of(err: &OpenAIError) -> Option<u16> {
    if let OpenAIError::ApiError(resp) = err {
        Some(resp.status_code.as_u16())
    } else {
        None
    }
}

/// Map an HTTP status code (plus message) to an [`LlmError`].
///
/// Factored out so the status → variant decision is testable without
/// constructing a live async-openai error.
fn map_api_status(status: u16, message: String) -> LlmError {
    match status {
        429 => LlmError::RateLimit,
        500..=599 => LlmError::Server { status, message },
        _ => LlmError::Transport(message),
    }
}

#[async_trait]
impl LlmClient for OpenAiLlm {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        let model = self.model_for_size(request.model_size);
        let max_tokens = request.max_tokens.unwrap_or(self.config.max_tokens);
        let temperature = self.config.temperature;

        tracing::debug!(
            prompt_name = request.prompt_name.as_deref().unwrap_or("<unnamed>"),
            model = %model,
            model_size = ?request.model_size,
            "openai generate"
        );

        let messages: Vec<ChatCompletionRequestMessage> = request
            .messages
            .iter()
            .map(to_openai_message)
            .collect::<Result<_, _>>()?;

        let response_format = request.response_schema.as_ref().map(to_response_format);

        // Retry wraps ONLY the provider call. Provider-malformed JSON maps to
        // InvalidJson (retryable); struct deserialization happens in
        // generate_typed, outside this boundary.
        with_retry(|| {
            let client = &self.client;
            let messages = messages.clone();
            let response_format = response_format.clone();
            let model = model.clone();
            async move {
                let mut builder = CreateChatCompletionRequestArgs::default();
                builder
                    .model(model)
                    .messages(messages)
                    .temperature(temperature)
                    .max_tokens(max_tokens);
                if let Some(rf) = response_format {
                    builder.response_format(rf);
                }
                let api_request = builder
                    .build()
                    .map_err(|e| LlmError::Transport(format!("building request: {e}")))?;

                let response = client
                    .chat()
                    .create(api_request)
                    .await
                    .map_err(map_openai_error)?;

                let choice = response
                    .choices
                    .into_iter()
                    .next()
                    .ok_or(LlmError::EmptyResponse)?;

                // Upstream `_handle_structured_response`: refusal is checked
                // before treating the response as content.
                if let Some(refusal) = choice.message.refusal {
                    return Err(LlmError::Refusal(refusal));
                }

                let content = choice.message.content.unwrap_or_default();
                if content.is_empty() {
                    return Err(LlmError::EmptyResponse);
                }

                // Provider-malformed JSON → InvalidJson (retryable). This is
                // the upstream json.JSONDecodeError analog, correctly inside
                // the retry boundary.
                let value: serde_json::Value = serde_json::from_str(&content)?;
                Ok(value)
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_core::llm::ResponseSchema;

    #[test]
    fn model_for_size_uses_defaults_when_unset() {
        let llm = OpenAiLlm::new(LlmConfig::default()).unwrap();
        assert_eq!(llm.model_for_size(ModelSize::Medium), DEFAULT_MODEL);
        assert_eq!(llm.model_for_size(ModelSize::Small), DEFAULT_SMALL_MODEL);
    }

    #[test]
    fn model_for_size_honors_config_overrides() {
        let config = LlmConfig {
            model: Some("custom-medium".to_string()),
            small_model: Some("custom-small".to_string()),
            ..LlmConfig::default()
        };
        let llm = OpenAiLlm::new(config).unwrap();
        assert_eq!(llm.model_for_size(ModelSize::Medium), "custom-medium");
        assert_eq!(llm.model_for_size(ModelSize::Small), "custom-small");
    }

    #[test]
    fn to_openai_message_maps_all_roles() {
        for msg in [
            Message {
                role: Role::System,
                content: "s".into(),
            },
            Message {
                role: Role::User,
                content: "u".into(),
            },
            Message {
                role: Role::Assistant,
                content: "a".into(),
            },
        ] {
            // Each role must build without error.
            to_openai_message(&msg).unwrap();
        }
    }

    #[test]
    fn response_format_carries_name_schema_and_omits_strict() {
        let schema = ResponseSchema {
            name: "Answer".to_string(),
            schema: serde_json::json!({"type": "object"}),
        };
        match to_response_format(&schema) {
            ResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "Answer");
                assert_eq!(json_schema.schema, serde_json::json!({"type": "object"}));
                // Upstream openai_generic_client never sets strict.
                assert_eq!(json_schema.strict, None);
                assert_eq!(json_schema.description, None);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn map_api_status_classifies_retryable_codes() {
        assert!(matches!(
            map_api_status(429, "slow down".into()),
            LlmError::RateLimit
        ));
        assert!(matches!(
            map_api_status(503, "overloaded".into()),
            LlmError::Server { status: 503, .. }
        ));
        assert!(matches!(
            map_api_status(500, "boom".into()),
            LlmError::Server { status: 500, .. }
        ));
        assert!(matches!(
            map_api_status(400, "bad request".into()),
            LlmError::Transport(_)
        ));
        assert!(matches!(
            map_api_status(401, "unauthorized".into()),
            LlmError::Transport(_)
        ));
    }

    #[test]
    fn map_api_status_retryability_matches_core_predicate() {
        assert!(map_api_status(429, "x".into()).is_retryable());
        assert!(map_api_status(500, "x".into()).is_retryable());
        assert!(!map_api_status(400, "x".into()).is_retryable());
    }

    #[test]
    fn map_openai_error_maps_invalid_argument_to_transport() {
        let err = OpenAIError::InvalidArgument("nope".into());
        assert!(matches!(map_openai_error(err), LlmError::Transport(_)));
    }

    // Live API roundtrip — requires a real OPENAI_API_KEY. Not run in CI.
    #[tokio::test]
    #[ignore = "live API; needs OPENAI_API_KEY"]
    async fn live_structured_output_roundtrip() {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct Answer {
            answer: String,
        }

        let llm = OpenAiLlm::new(LlmConfig::default()).unwrap();
        let req = LlmRequest::new(vec![
            Message::system("Reply with JSON containing an `answer` field."),
            Message::user("What is 2 + 2? Answer in one word."),
        ]);
        let out: Answer = chronicle_core::llm::generate_typed(&llm, req)
            .await
            .unwrap();
        assert!(!out.answer.is_empty());
    }
}
