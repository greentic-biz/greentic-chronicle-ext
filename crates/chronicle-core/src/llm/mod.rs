// Ported from graphiti_core/llm_client/client.py, config.py, errors.py
// @ 34f56e65 (v0.29.1)

pub mod config;
pub mod message;
pub mod retry;

pub use config::{DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, LlmConfig};
pub use message::{Message, Role};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use thiserror::Error;

// ---------------------------------------------------------------------------
// ModelSize
// ---------------------------------------------------------------------------

/// Upstream `ModelSize` enum (`graphiti_core/llm_client/config.py`).
/// `Medium` is the default and corresponds to the primary model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSize {
    Small,
    #[default]
    Medium,
}

// ---------------------------------------------------------------------------
// LlmError
// ---------------------------------------------------------------------------

/// Errors that an [`LlmClient`] implementation may return.
///
/// Mirrors upstream error taxonomy from `graphiti_core/llm_client/errors.py`
/// and the retry predicate `is_server_or_retry_error` in `client.py`.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Provider rate-limit (upstream `RateLimitError`).
    #[error("rate limit exceeded")]
    RateLimit,

    /// Model refused to produce a response (upstream `RefusalError`).
    #[error("refusal: {0}")]
    Refusal(String),

    /// Provider returned an empty response (upstream `EmptyResponseError`).
    #[error("empty response from LLM")]
    EmptyResponse,

    /// HTTP-level server error (5xx), maps to upstream `httpx.HTTPStatusError`
    /// with `500 <= status < 600`.
    #[error("server error {status}: {message}")]
    Server { status: u16, message: String },

    /// JSON deserialization failed (upstream `json.JSONDecodeError`).
    #[error("invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    /// Low-level transport / network error.
    #[error("transport error: {0}")]
    Transport(String),
}

impl LlmError {
    /// Returns `true` for errors the retry policy should retry.
    ///
    /// Mirrors upstream `is_server_or_retry_error` in `client.py`:
    /// retries on `RateLimitError`, `json.JSONDecodeError`, and
    /// `httpx.HTTPStatusError` with a 5xx status code.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            LlmError::RateLimit | LlmError::InvalidJson(_) | LlmError::Server { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// ResponseSchema
// ---------------------------------------------------------------------------

/// Structured-output schema descriptor passed to the LLM.
///
/// The schema is generated via `schemars` and serialized to JSON so that
/// provider implementations can inject it into the prompt or the
/// `response_format` parameter.
pub struct ResponseSchema {
    pub name: String,
    pub schema: serde_json::Value,
}

impl ResponseSchema {
    /// Build a `ResponseSchema` for type `T`.
    ///
    /// The name is derived from `std::any::type_name::<T>()` — specifically
    /// the last `::` segment — falling back to `"structured_response"` if
    /// the name cannot be determined.
    ///
    /// schemars 1.x API note: `schema_for!(T)` returns a `Schema` newtype
    /// wrapping `serde_json::Value`; `serde_json::to_value` serializes it
    /// directly (no `.schema` field indirection as in 0.8.x).
    pub fn of<T: JsonSchema>() -> Self {
        let raw_name = std::any::type_name::<T>();
        let name = raw_name
            .rsplit("::")
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("structured_response")
            .to_string();

        let schema =
            serde_json::to_value(schemars::schema_for!(T)).unwrap_or(serde_json::Value::Null);

        Self { name, schema }
    }
}

// ---------------------------------------------------------------------------
// LlmRequest
// ---------------------------------------------------------------------------

/// A request to an LLM, carrying messages plus optional structured-output
/// hints and routing metadata.
pub struct LlmRequest {
    pub messages: Vec<Message>,
    pub response_schema: Option<ResponseSchema>,
    pub max_tokens: Option<u32>,
    pub model_size: ModelSize,
    pub prompt_name: Option<String>,
}

impl LlmRequest {
    /// Create a new request with defaults: no schema, `ModelSize::Medium`, no name.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            response_schema: None,
            max_tokens: None,
            model_size: ModelSize::Medium,
            prompt_name: None,
        }
    }

    /// Attach a structured-output schema.
    pub fn with_schema(mut self, schema: ResponseSchema) -> Self {
        self.response_schema = Some(schema);
        self
    }

    /// Route this request to the small model.
    pub fn small(mut self) -> Self {
        self.model_size = ModelSize::Small;
        self
    }

    /// Attach a prompt name for observability / tracing.
    pub fn named(mut self, name: &str) -> Self {
        self.prompt_name = Some(name.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// LlmClient trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError>;
}

// ---------------------------------------------------------------------------
// generate_typed helper
// ---------------------------------------------------------------------------

/// Call `client.generate` with a structured-output schema for `T` attached,
/// then deserialize the JSON response into `T`.
pub async fn generate_typed<T: DeserializeOwned + JsonSchema>(
    client: &dyn LlmClient,
    request: LlmRequest,
) -> Result<T, LlmError> {
    let schema = ResponseSchema::of::<T>();
    let request = request.with_schema(schema);
    let value = client.generate(request).await?;
    let typed = serde_json::from_value(value)?;
    Ok(typed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticLlm(serde_json::Value);

    #[async_trait]
    impl LlmClient for StaticLlm {
        async fn generate(&self, _request: LlmRequest) -> Result<serde_json::Value, LlmError> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn generate_typed_deserializes_structured_response() {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct Out {
            answer: String,
        }

        let mock = StaticLlm(serde_json::json!({"answer": "42"}));
        let out: Out = generate_typed(
            &mock,
            LlmRequest::new(vec![Message::system("s"), Message::user("u")]),
        )
        .await
        .unwrap();
        assert_eq!(out.answer, "42");
    }

    #[test]
    fn rate_limit_is_retryable() {
        assert!(LlmError::RateLimit.is_retryable());
    }

    #[test]
    fn invalid_json_is_retryable() {
        let err: serde_json::Error = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        assert!(LlmError::InvalidJson(err).is_retryable());
    }

    #[test]
    fn server_5xx_is_retryable() {
        assert!(
            LlmError::Server {
                status: 503,
                message: "overloaded".into()
            }
            .is_retryable()
        );
    }

    #[test]
    fn refusal_is_not_retryable() {
        assert!(!LlmError::Refusal("nope".into()).is_retryable());
    }

    #[test]
    fn empty_response_is_not_retryable() {
        assert!(!LlmError::EmptyResponse.is_retryable());
    }

    #[test]
    fn transport_is_not_retryable() {
        assert!(!LlmError::Transport("timeout".into()).is_retryable());
    }

    #[test]
    fn response_schema_of_extracts_short_name() {
        #[derive(schemars::JsonSchema)]
        struct MyPayload {
            #[allow(dead_code)]
            x: i32,
        }
        let rs = ResponseSchema::of::<MyPayload>();
        assert_eq!(rs.name, "MyPayload");
        assert!(rs.schema.is_object());
    }

    #[test]
    fn llm_request_builder_methods_work() {
        let req = LlmRequest::new(vec![Message::user("hi")])
            .small()
            .named("test-prompt");
        assert_eq!(req.model_size, ModelSize::Small);
        assert_eq!(req.prompt_name.as_deref(), Some("test-prompt"));
        assert!(req.response_schema.is_none());
    }
}
