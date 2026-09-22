//! Bridge that adapts a Greentic DW [`LlmProvider`] to Chronicle's [`LlmClient`].
//!
//! Chronicle drives entity / edge extraction through its own [`LlmClient`] trait,
//! whose `generate` returns a `serde_json::Value`. The Greentic DW LLM family exposes
//! a richer, provider-neutral contract ([`LlmProvider`]) whose `generate` is
//! synchronous and returns a normalized [`DwLlmResponse`]. This bridge maps between
//! the two so any DW LLM provider can power Chronicle's knowledge-graph extraction.
//!
//! Design notes:
//!
//! * **Retry.** Chronicle's convention is that retry lives *inside* the `LlmClient`
//!   implementation. DW providers already own their retry policy, so adding a retry
//!   wrapper here would double-retry. We deliberately do not retry in the bridge and
//!   surface the provider's error verbatim (mapped into [`LlmError`]).
//! * **Blocking.** [`LlmProvider::generate`] is synchronous, so we run it on a
//!   blocking thread via [`tokio::task::spawn_blocking`] to avoid stalling the async
//!   runtime that Chronicle's pipeline executes on.
//! * **Embeddings.** The bridge only covers the LLM surface; there is no DW
//!   embeddings family yet, so callers still pair this with an OpenAI embedder.

use std::sync::Arc;

use async_trait::async_trait;
use chronicle_core::llm::{LlmClient, LlmError, LlmRequest, ModelSize, Role};
use greentic_dw_llm::{
    LlmContentPart, LlmMessage, LlmMessageRole, LlmProvider, LlmRequest as DwLlmRequest,
    LlmStructuredOutputSpec,
};
use greentic_types::TenantCtx;
use serde_json::Value;

/// Adapts a DW [`LlmProvider`] to Chronicle's [`LlmClient`].
pub struct DwLlmBridge {
    provider: Arc<dyn LlmProvider>,
    tenant: TenantCtx,
}

impl DwLlmBridge {
    /// Builds a bridge over the given DW provider, scoped to a single tenant.
    pub fn new(provider: Arc<dyn LlmProvider>, tenant: TenantCtx) -> Self {
        Self { provider, tenant }
    }

    /// Translates a Chronicle [`LlmRequest`] into a normalized DW request.
    fn build_dw_request(&self, request: &LlmRequest) -> Result<DwLlmRequest, LlmError> {
        let mut messages = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            let role = map_role(message.role);
            let dw_message = LlmMessage::new(role, vec![LlmContentPart::text(&message.content)])
                .map_err(|e| LlmError::Transport(format!("invalid bridged message: {e}")))?;
            messages.push(dw_message);
        }

        let request_id = request
            .prompt_name
            .clone()
            .unwrap_or_else(|| "chronicle-bridge".to_string());

        let mut dw_request = DwLlmRequest::new(request_id, messages)
            .map_err(|e| LlmError::Transport(format!("invalid bridged request: {e}")))?;

        if let Some(schema) = &request.response_schema {
            let spec = LlmStructuredOutputSpec::new(schema.name.clone(), schema.schema.clone())
                .map_err(|e| {
                    LlmError::Transport(format!("invalid bridged structured output: {e}"))
                })?;
            dw_request = dw_request.with_structured_output(spec);
        }

        if let Some(max_tokens) = request.max_tokens {
            dw_request.max_output_tokens = Some(max_tokens);
        }

        if request.model_size == ModelSize::Small {
            // The DW contract has no model-size axis; small-model selection is left
            // to the provider's own configuration. We record the intent in metadata
            // so providers that honor it can route accordingly.
            dw_request.metadata = serde_json::json!({ "model_size": "small" });
        }

        Ok(dw_request)
    }
}

/// Maps a Chronicle message role onto the DW message-role enum.
fn map_role(role: Role) -> LlmMessageRole {
    match role {
        Role::System => LlmMessageRole::System,
        Role::User => LlmMessageRole::User,
        Role::Assistant => LlmMessageRole::Assistant,
    }
}

/// Extracts a JSON value from a DW response.
///
/// Prefers the provider's `structured_output` payload when present; otherwise parses
/// the concatenated text output as JSON. An empty text output maps to
/// [`LlmError::EmptyResponse`]; a non-JSON text output maps to
/// [`LlmError::InvalidJson`] via the `?` conversion.
fn extract_json(structured_output: Option<Value>, text: String) -> Result<Value, LlmError> {
    if let Some(value) = structured_output {
        return Ok(value);
    }
    if text.trim().is_empty() {
        return Err(LlmError::EmptyResponse);
    }
    let value = serde_json::from_str(&text)?;
    Ok(value)
}

#[async_trait]
impl LlmClient for DwLlmBridge {
    async fn generate(&self, request: LlmRequest) -> Result<Value, LlmError> {
        // Guard: a schema-bearing request against a provider without structured
        // output support cannot be satisfied reliably.
        if request.response_schema.is_some() && !self.provider.features().structured_outputs {
            return Err(LlmError::Transport(
                "dw llm provider lacks structured outputs".to_string(),
            ));
        }

        let dw_request = self.build_dw_request(&request)?;
        let provider = Arc::clone(&self.provider);
        let tenant = self.tenant.clone();

        let response = tokio::task::spawn_blocking(move || provider.generate(&tenant, dw_request))
            .await
            .map_err(|e| LlmError::Transport(format!("bridge join error: {e}")))?
            .map_err(|e| LlmError::Transport(format!("dw llm provider error: {e}")))?;

        let text = response.text_output();
        extract_json(response.structured_output, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use greentic_dw_llm::{
        LlmProviderFeatures, LlmRequest as DwLlmRequest, LlmResponse as DwLlmResponse, LlmResult,
    };

    fn tenant() -> TenantCtx {
        use greentic_types::{EnvId, TenantId};
        TenantCtx::new(
            EnvId::new("dev").expect("env id"),
            TenantId::new("tenant-a").expect("tenant id"),
        )
    }

    /// Provider returning a fixed structured-output payload.
    struct StructuredProvider {
        features: LlmProviderFeatures,
        payload: Value,
    }

    impl LlmProvider for StructuredProvider {
        fn features(&self) -> &LlmProviderFeatures {
            &self.features
        }

        fn generate(
            &self,
            _tenant: &TenantCtx,
            _request: DwLlmRequest,
        ) -> LlmResult<DwLlmResponse> {
            let mut response = DwLlmResponse::new(Vec::new());
            response.structured_output = Some(self.payload.clone());
            Ok(response)
        }
    }

    fn structured_features() -> LlmProviderFeatures {
        // chat + structured_outputs enabled.
        LlmProviderFeatures::new(true, true, false, false, false, false, false, false)
    }

    fn plain_features() -> LlmProviderFeatures {
        // chat only; no structured outputs.
        LlmProviderFeatures::new(true, false, false, false, false, false, false, false)
    }

    #[tokio::test]
    async fn structured_provider_yields_parsed_value() {
        let provider = Arc::new(StructuredProvider {
            features: structured_features(),
            payload: serde_json::json!({"answer": "42"}),
        });
        let bridge = DwLlmBridge::new(provider, tenant());

        let schema = chronicle_core::llm::ResponseSchema {
            name: "answer".to_string(),
            schema: serde_json::json!({"type": "object"}),
        };
        let request =
            LlmRequest::new(vec![chronicle_core::llm::Message::user("q")]).with_schema(schema);

        let value = bridge.generate(request).await.expect("bridge generate");
        assert_eq!(value["answer"], "42");
    }

    #[tokio::test]
    async fn plain_text_response_is_parsed_as_json() {
        struct TextProvider {
            features: LlmProviderFeatures,
        }
        impl LlmProvider for TextProvider {
            fn features(&self) -> &LlmProviderFeatures {
                &self.features
            }
            fn generate(
                &self,
                _tenant: &TenantCtx,
                _request: DwLlmRequest,
            ) -> LlmResult<DwLlmResponse> {
                let message = LlmMessage::new(
                    LlmMessageRole::Assistant,
                    vec![LlmContentPart::text("{\"k\":1}")],
                )
                .expect("message");
                Ok(DwLlmResponse::new(vec![message]))
            }
        }

        let bridge = DwLlmBridge::new(
            Arc::new(TextProvider {
                features: structured_features(),
            }),
            tenant(),
        );
        let request = LlmRequest::new(vec![chronicle_core::llm::Message::user("q")]);
        let value = bridge.generate(request).await.expect("bridge generate");
        assert_eq!(value["k"], 1);
    }

    #[tokio::test]
    async fn schema_request_without_structured_support_errors() {
        let provider = Arc::new(StructuredProvider {
            features: plain_features(),
            payload: Value::Null,
        });
        let bridge = DwLlmBridge::new(provider, tenant());

        let schema = chronicle_core::llm::ResponseSchema {
            name: "answer".to_string(),
            schema: serde_json::json!({"type": "object"}),
        };
        let request =
            LlmRequest::new(vec![chronicle_core::llm::Message::user("q")]).with_schema(schema);

        let err = bridge.generate(request).await.expect_err("expected error");
        match err {
            LlmError::Transport(msg) => {
                assert!(msg.contains("structured outputs"), "got: {msg}");
            }
            other => panic!("expected Transport error, got {other:?}"),
        }
    }
}
