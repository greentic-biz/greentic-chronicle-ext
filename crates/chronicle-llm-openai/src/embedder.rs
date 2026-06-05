// Ported from graphiti_core/embedder/openai.py @ 34f56e65 (v0.29.1).
//
// Upstream `DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"`. The embedding
// dimension default comes from the shared base config
// (`chronicle_core::embedder::DEFAULT_EMBEDDING_DIM`, upstream env-driven
// `EMBEDDING_DIM`, default 1024).
//
// Truncation semantics mirror upstream exactly: every returned vector is sliced
// to `embedding_dim` (`embedding[: embedding_dim]`). When the API returns a
// vector SHORTER than `embedding_dim`, the Python slice is a no-op — no error,
// no padding. We replicate that: truncate only, never pad, never error.

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::embeddings::CreateEmbeddingRequestArgs;
use async_trait::async_trait;
use chronicle_core::embedder::{DEFAULT_EMBEDDING_DIM, EmbedderClient, EmbedderError};

/// Upstream `DEFAULT_EMBEDDING_MODEL` (embedder/openai.py).
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Configuration for [`OpenAiEmbedder`].
///
/// Mirrors upstream `OpenAIEmbedderConfig` (model, api_key, base_url) plus the
/// inherited `embedding_dim` from the base `EmbedderConfig`.
#[derive(Debug, Clone)]
pub struct OpenAiEmbedderConfig {
    pub embedding_model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub embedding_dim: usize,
}

impl Default for OpenAiEmbedderConfig {
    fn default() -> Self {
        Self {
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            api_key: None,
            base_url: None,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
        }
    }
}

/// OpenAI-compatible [`EmbedderClient`] backed by `async-openai`.
pub struct OpenAiEmbedder {
    client: Client<OpenAIConfig>,
    config: OpenAiEmbedderConfig,
}

impl OpenAiEmbedder {
    /// Build an embedder from an [`OpenAiEmbedderConfig`].
    ///
    /// API-key handling mirrors the LLM client: when `api_key` is `None`, the
    /// async-openai default config reads `OPENAI_API_KEY` from the environment
    /// (matching upstream `AsyncOpenAI(api_key=None)` behavior).
    pub fn new(config: OpenAiEmbedderConfig) -> Result<Self, EmbedderError> {
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
}

/// Truncate a vector to at most `dim` elements.
///
/// Replicates upstream `embedding[: embedding_dim]`: shorter vectors are
/// returned unchanged (no padding, no error); longer vectors are clipped.
fn truncate(mut embedding: Vec<f32>, dim: usize) -> Vec<f32> {
    embedding.truncate(dim);
    embedding
}

/// Map an async-openai error into an [`EmbedderError`].
///
/// HTTP 429 → [`EmbedderError::RateLimit`]; other API errors →
/// [`EmbedderError::Provider`]; transport/SDK errors →
/// [`EmbedderError::Transport`].
fn map_embedder_error(err: OpenAIError) -> EmbedderError {
    match err {
        OpenAIError::ApiError(resp) => {
            if resp.status_code.as_u16() == 429 {
                EmbedderError::RateLimit
            } else {
                EmbedderError::Provider(resp.to_string())
            }
        }
        other => EmbedderError::Transport(other.to_string()),
    }
}

#[async_trait]
impl EmbedderClient for OpenAiEmbedder {
    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(self.config.embedding_model.clone())
            .input(input)
            .build()
            .map_err(|e| EmbedderError::Transport(format!("building request: {e}")))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(map_embedder_error)?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Provider("empty embedding response".to_string()))?
            .embedding;

        Ok(truncate(embedding, self.config.embedding_dim))
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Upstream issues a single embeddings call with all inputs.
        let request = CreateEmbeddingRequestArgs::default()
            .model(self.config.embedding_model.clone())
            .input(inputs.to_vec())
            .build()
            .map_err(|e| EmbedderError::Transport(format!("building request: {e}")))?;

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(map_embedder_error)?;

        Ok(response
            .data
            .into_iter()
            .map(|e| truncate(e.embedding, self.config.embedding_dim))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_upstream_model_and_dim() {
        let config = OpenAiEmbedderConfig::default();
        assert_eq!(config.embedding_model, "text-embedding-3-small");
        assert_eq!(config.embedding_dim, DEFAULT_EMBEDDING_DIM);
        assert!(config.api_key.is_none());
        assert!(config.base_url.is_none());
    }

    #[test]
    fn truncate_clips_longer_vectors() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(truncate(v, 3), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn truncate_leaves_shorter_vectors_unchanged() {
        // Upstream slice on a shorter vector is a no-op: no padding, no error.
        let v = vec![1.0, 2.0];
        assert_eq!(truncate(v, 5), vec![1.0, 2.0]);
    }

    #[test]
    fn truncate_exact_length_is_identity() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(truncate(v.clone(), 3), v);
    }

    #[test]
    fn embedding_dim_reflects_config() {
        let config = OpenAiEmbedderConfig {
            embedding_dim: 256,
            ..OpenAiEmbedderConfig::default()
        };
        let embedder = OpenAiEmbedder::new(config).unwrap();
        assert_eq!(embedder.embedding_dim(), 256);
    }

    #[test]
    fn map_embedder_error_maps_invalid_argument_to_transport() {
        let err = OpenAIError::InvalidArgument("nope".into());
        assert!(matches!(
            map_embedder_error(err),
            EmbedderError::Transport(_)
        ));
    }

    // Live API — requires a real OPENAI_API_KEY. Not run in CI.
    #[tokio::test]
    #[ignore = "live API; needs OPENAI_API_KEY"]
    async fn live_embedding_create() {
        let embedder = OpenAiEmbedder::new(OpenAiEmbedderConfig::default()).unwrap();
        let vector = embedder.create("hello world").await.unwrap();
        assert_eq!(vector.len(), DEFAULT_EMBEDDING_DIM);
    }
}
