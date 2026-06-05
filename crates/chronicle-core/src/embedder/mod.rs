// Ported from graphiti_core/embedder/client.py @ 34f56e65 (v0.29.1)
use async_trait::async_trait;
use thiserror::Error;

/// Upstream EMBEDDING_DIM default. NOTE: upstream reads this from env;
/// here it is a config default — override via EmbedderConfig.
pub const DEFAULT_EMBEDDING_DIM: usize = 1024;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("rate limit exceeded")]
    RateLimit,
}

impl EmbedderError {
    /// Mirrors `LlmError::is_retryable` semantics: rate limits retry;
    /// Transport is treated as non-retryable here for parity with LlmError
    /// (provider impls may retry transient transport errors internally).
    pub fn is_retryable(&self) -> bool {
        matches!(self, EmbedderError::RateLimit)
    }
}

/// Passed to [`EmbedderClient`] implementors at construction time.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub embedding_dim: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            embedding_dim: DEFAULT_EMBEDDING_DIM,
        }
    }
}

#[async_trait]
pub trait EmbedderClient: Send + Sync {
    fn embedding_dim(&self) -> usize;
    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError>;
    /// Inputs are owned Strings by design (pipeline callers own their name/fact buffers); see plan Task 5.
    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_via_boxed_dyn_returns_correct_dim() {
        struct Zero;
        #[async_trait::async_trait]
        impl EmbedderClient for Zero {
            fn embedding_dim(&self) -> usize {
                4
            }
            async fn create(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
                Ok(vec![0.0; 4])
            }
            async fn create_batch(
                &self,
                inputs: &[String],
            ) -> Result<Vec<Vec<f32>>, EmbedderError> {
                Ok(inputs.iter().map(|_| vec![0.0; 4]).collect())
            }
        }
        let c: Box<dyn EmbedderClient> = Box::new(Zero);
        assert_eq!(c.create("x").await.unwrap().len(), 4);
        assert_eq!(
            c.create_batch(&["a".to_string(), "b".to_string()])
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn rate_limit_is_retryable_transport_and_provider_are_not() {
        assert!(EmbedderError::RateLimit.is_retryable());
        assert!(!EmbedderError::Transport("conn reset".to_string()).is_retryable());
        assert!(!EmbedderError::Provider("bad key".to_string()).is_retryable());
    }
}
