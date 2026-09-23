//! Bridge that adapts a Greentic DW [`EmbeddingProvider`] to Chronicle's
//! [`EmbedderClient`].
//!
//! Chronicle drives vector recall through its own [`EmbedderClient`] trait
//! (async, returning `Vec<f32>`). The Greentic DW embedding family exposes a
//! provider-neutral contract ([`EmbeddingProvider`]) whose `embed` is
//! synchronous and batch-first. This bridge maps between the two so ANY DW
//! embedding backend (OpenAI-compatible, self-hosted, …) can power Chronicle's
//! embeddings — there is no hard dependency on a single provider.
//!
//! Design notes (mirroring [`crate::bridge::DwLlmBridge`]):
//!
//! * **Blocking.** [`EmbeddingProvider::embed`] is synchronous, so we run it on
//!   a blocking thread via [`tokio::task::spawn_blocking`] to avoid stalling the
//!   async runtime Chronicle executes on.
//! * **Tenant.** DW providers resolve credentials/metering per tenant, so the
//!   bridge is constructed with the tenant Chronicle's embedding calls run as.
//! * **Dimension.** Chronicle needs a fixed embedding dimension at construction
//!   (for the graph vector index); it is supplied to [`DwEmbedderBridge::new`].

use std::sync::Arc;

use async_trait::async_trait;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};
use greentic_dw_embedding::{EmbeddingProvider, EmbeddingRequest};
use greentic_types::TenantCtx;

/// Adapts a DW [`EmbeddingProvider`] to Chronicle's [`EmbedderClient`].
pub struct DwEmbedderBridge {
    provider: Arc<dyn EmbeddingProvider>,
    tenant: TenantCtx,
    embedding_dim: usize,
}

impl DwEmbedderBridge {
    /// Builds a bridge over the given DW embedding provider, scoped to a single
    /// tenant, advertising `embedding_dim` to Chronicle.
    #[must_use]
    pub fn new(
        provider: Arc<dyn EmbeddingProvider>,
        tenant: TenantCtx,
        embedding_dim: usize,
    ) -> Self {
        Self {
            provider,
            tenant,
            embedding_dim,
        }
    }

    /// Embed a batch of inputs through the DW provider on a blocking thread,
    /// returning one vector per input (order-preserving).
    async fn embed_inputs(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let provider = Arc::clone(&self.provider);
        let tenant = self.tenant.clone();
        let request = EmbeddingRequest::new("chronicle-embed", inputs);
        let response = tokio::task::spawn_blocking(move || provider.embed(&tenant, request))
            .await
            .map_err(|e| EmbedderError::Transport(format!("embedder bridge join: {e}")))?
            .map_err(|e| EmbedderError::Provider(e.to_string()))?;
        Ok(response.vectors)
    }
}

#[async_trait]
impl EmbedderClient for DwEmbedderBridge {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        let mut vectors = self.embed_inputs(vec![input.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| EmbedderError::Provider("embedding provider returned no vectors".into()))
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_inputs(inputs.to_vec()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use greentic_dw_embedding::{
        EmbeddingProviderFeatures, EmbeddingResponse, EmbeddingResult, EmbeddingUsage,
    };
    use greentic_types::{EnvId, TenantId};

    fn tenant() -> TenantCtx {
        TenantCtx::new(
            EnvId::try_from("dev").expect("env"),
            TenantId::try_from("acme").expect("tenant"),
        )
    }

    /// Provider that echoes a deterministic vector per input so the bridge's
    /// shape/ordering can be asserted without a real embedding backend.
    struct FakeEmbeddingProvider {
        features: EmbeddingProviderFeatures,
        dim: usize,
    }

    impl EmbeddingProvider for FakeEmbeddingProvider {
        fn features(&self) -> &EmbeddingProviderFeatures {
            &self.features
        }

        fn embed(
            &self,
            _tenant: &TenantCtx,
            request: EmbeddingRequest,
        ) -> EmbeddingResult<EmbeddingResponse> {
            let vectors: Vec<Vec<f32>> = request
                .inputs
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i as f32; self.dim])
                .collect();
            Ok(EmbeddingResponse {
                response_id: None,
                model: None,
                dim: self.dim,
                vectors,
                usage: Some(EmbeddingUsage {
                    prompt_tokens: None,
                    total_tokens: None,
                }),
                metadata: serde_json::Value::Null,
            })
        }
    }

    fn bridge(dim: usize) -> DwEmbedderBridge {
        DwEmbedderBridge::new(
            Arc::new(FakeEmbeddingProvider {
                features: EmbeddingProviderFeatures::new(true, true, true),
                dim,
            }),
            tenant(),
            dim,
        )
    }

    #[tokio::test]
    async fn embedding_dim_is_reported() {
        assert_eq!(bridge(8).embedding_dim(), 8);
    }

    #[tokio::test]
    async fn create_returns_single_vector_of_dim() {
        let v = bridge(4).create("hello").await.expect("embed");
        assert_eq!(v.len(), 4);
    }

    #[tokio::test]
    async fn create_batch_preserves_order_and_count() {
        let vectors = bridge(3)
            .create_batch(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .expect("embed batch");
        assert_eq!(vectors.len(), 3);
        // FakeEmbeddingProvider encodes the input index into the vector values.
        assert_eq!(vectors[0], vec![0.0; 3]);
        assert_eq!(vectors[1], vec![1.0; 3]);
        assert_eq!(vectors[2], vec![2.0; 3]);
    }
}
