//! Bridge that adapts a Greentic DW [`EmbeddingProvider`] to Chronicle's [`EmbedderClient`].
//!
//! Chronicle's doc-RAG pipeline drives embedding through its own async
//! [`EmbedderClient`] trait. The Greentic DW embedding family ([W1]) exposes a
//! synchronous [`EmbeddingProvider`] contract. This bridge maps between the two
//! so any DW embedding provider can power the Chronicle knowledge (document-RAG)
//! path without needing OpenAI directly.
//!
//! Design notes:
//!
//! * **Blocking.** [`EmbeddingProvider::embed`] is synchronous, so we run it on
//!   a blocking thread via [`tokio::task::spawn_blocking`] to avoid stalling the
//!   async runtime that Chronicle's pipeline executes on.
//! * **Dimension.** [`EmbeddingProviderFeatures`] carries no `dim` field (W1 design);
//!   the embedding dimension is supplied explicitly at bridge construction time
//!   (read from [`KnowledgeConfig::embedding_dim`](crate::KnowledgeConfig::embedding_dim)).
//! * **Batch.** [`EmbedderClient::create_batch`] maps to a single DW
//!   [`EmbeddingRequest`] whose `inputs` carries all texts; this keeps the bridge
//!   at one blocking call per batch.

use std::sync::Arc;

use async_trait::async_trait;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};
use greentic_dw_embedding::{EmbeddingProvider, EmbeddingRequest};
use greentic_types::TenantCtx;

/// Adapts a DW [`EmbeddingProvider`] (sync) to Chronicle's async [`EmbedderClient`].
pub struct DwEmbedderBridge {
    provider: Arc<dyn EmbeddingProvider>,
    tenant: TenantCtx,
    dim: usize,
}

impl DwEmbedderBridge {
    /// Builds a bridge over the given DW embedding provider, scoped to a single
    /// tenant, with the specified embedding dimension.
    ///
    /// `dim` must match what the underlying provider produces — it is returned
    /// verbatim from [`EmbedderClient::embedding_dim`] so Chronicle sizes its
    /// HNSW index correctly.
    pub fn new(provider: Arc<dyn EmbeddingProvider>, tenant: TenantCtx, dim: usize) -> Self {
        Self {
            provider,
            tenant,
            dim,
        }
    }

    /// Embeds a batch of inputs, running the sync provider on a blocking thread.
    async fn embed_inputs(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let provider = Arc::clone(&self.provider);
        let tenant = self.tenant.clone();
        let request = EmbeddingRequest::new("chronicle-doc-rag", inputs);
        let response = tokio::task::spawn_blocking(move || provider.embed(&tenant, request))
            .await
            .map_err(|e| EmbedderError::Transport(format!("embedder bridge join error: {e}")))?
            .map_err(|e| EmbedderError::Transport(format!("dw embedding provider error: {e}")))?;
        // Dimension is load-bearing: Chronicle sizes its HNSW index from
        // `embedding_dim()` (== `self.dim`). If the provider's actual output
        // dimension diverges (wrong model / misconfigured `embedding_dim`),
        // mismatched vectors would be indexed → backend insert error or silent
        // index corruption. Fail loud at the bridge instead.
        if response.dim != self.dim {
            return Err(EmbedderError::Transport(format!(
                "embedding dim mismatch: provider returned {}, expected {} (configured embedding_dim); \
                 corpus and query must use the same embedding model/dimension",
                response.dim, self.dim
            )));
        }
        Ok(response.vectors)
    }
}

#[async_trait]
impl EmbedderClient for DwEmbedderBridge {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        let mut out = self.embed_inputs(vec![input.to_string()]).await?;
        out.pop()
            .ok_or_else(|| EmbedderError::Transport("empty embedding response".to_string()))
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_inputs(inputs.to_vec()).await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use greentic_dw_embedding::{
        EmbeddingError, EmbeddingProviderFeatures, EmbeddingRequest, EmbeddingResponse,
        EmbeddingResult,
    };
    use greentic_types::{EnvId, TenantId};

    fn tenant() -> TenantCtx {
        TenantCtx::new(
            EnvId::try_from("dev").expect("env id"),
            TenantId::try_from("tenant-a").expect("tenant id"),
        )
    }

    const STUB_DIM: usize = 4;

    /// Minimal in-test embedding provider that returns fixed-length zero vectors.
    struct StubEmbeddingProvider;

    impl EmbeddingProvider for StubEmbeddingProvider {
        fn features(&self) -> &EmbeddingProviderFeatures {
            static FEATURES: EmbeddingProviderFeatures =
                EmbeddingProviderFeatures::new(true, false, false);
            &FEATURES
        }

        fn embed(
            &self,
            _tenant: &TenantCtx,
            request: EmbeddingRequest,
        ) -> EmbeddingResult<EmbeddingResponse> {
            let count = request.inputs.len();
            Ok(EmbeddingResponse {
                response_id: None,
                model: None,
                dim: STUB_DIM,
                vectors: vec![vec![0.1_f32; STUB_DIM]; count],
                usage: None,
                metadata: serde_json::Value::Null,
            })
        }
    }

    /// Provider that always returns an error (used to test error propagation).
    struct FailingEmbeddingProvider;

    impl EmbeddingProvider for FailingEmbeddingProvider {
        fn features(&self) -> &EmbeddingProviderFeatures {
            static FEATURES: EmbeddingProviderFeatures =
                EmbeddingProviderFeatures::new(true, false, false);
            &FEATURES
        }

        fn embed(
            &self,
            _tenant: &TenantCtx,
            _request: EmbeddingRequest,
        ) -> EmbeddingResult<EmbeddingResponse> {
            Err(EmbeddingError::provider("stub provider failure"))
        }
    }

    fn bridge() -> DwEmbedderBridge {
        DwEmbedderBridge::new(Arc::new(StubEmbeddingProvider), tenant(), STUB_DIM)
    }

    #[test]
    fn embedding_dim_matches_constructor() {
        let b = bridge();
        assert_eq!(b.embedding_dim(), STUB_DIM);
    }

    #[tokio::test]
    async fn create_returns_dim_length_vector() {
        let b = bridge();
        let v = b.create("hello world").await.expect("create succeeds");
        assert_eq!(v.len(), STUB_DIM);
    }

    #[tokio::test]
    async fn create_batch_returns_one_vector_per_input() {
        let b = bridge();
        let inputs = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let vecs = b.create_batch(&inputs).await.expect("batch succeeds");
        assert_eq!(vecs.len(), 3);
        for v in &vecs {
            assert_eq!(v.len(), STUB_DIM);
        }
    }

    #[tokio::test]
    async fn create_batch_empty_inputs_returns_empty() {
        let b = bridge();
        let vecs = b.create_batch(&[]).await.expect("empty batch ok");
        assert!(vecs.is_empty());
    }

    #[tokio::test]
    async fn provider_error_surfaces_as_embedder_transport_error() {
        let b = DwEmbedderBridge::new(Arc::new(FailingEmbeddingProvider), tenant(), STUB_DIM);
        let err = b.create("text").await.expect_err("expected error");
        match err {
            EmbedderError::Transport(msg) => {
                assert!(msg.contains("dw embedding provider error"), "got: {msg}");
            }
            other => panic!("expected Transport error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dim_mismatch_is_a_loud_error() {
        // Provider emits STUB_DIM (4) but the bridge is configured for 8 — the
        // guard must reject it rather than feed wrong-sized vectors to the index.
        let b = DwEmbedderBridge::new(Arc::new(StubEmbeddingProvider), tenant(), STUB_DIM + 4);
        let err = b.create("text").await.expect_err("dim mismatch must error");
        match err {
            EmbedderError::Transport(msg) => {
                assert!(msg.contains("dim mismatch"), "got: {msg}");
                assert!(
                    msg.contains("returned 4") && msg.contains("expected 8"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Transport error, got {other:?}"),
        }
    }
}
