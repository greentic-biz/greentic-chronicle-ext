// Ported from graphiti_core/cross_encoder/client.py @ 34f56e65 (v0.29.1)
//
// A cross-encoder ranks passages by relevance to a query. Unlike the
// algorithmic rerankers in `search/rerank.rs` (MMR, node-distance,
// episode-mentions) — which operate on graph structure and embeddings — a
// cross-encoder is a model-backed scorer that consumes the raw query/passage
// text. The trait is kept in its own top-level module (`cross_encoder`) so it
// is not confused with the structural rerankers under `search::rerank`.

use crate::llm::LlmError;

/// Interface for cross-encoder models that rank passages by relevance to a
/// query.
///
/// Mirrors upstream `CrossEncoderClient` (abstract base class in
/// `graphiti_core/cross_encoder/client.py`). Implementations may be model-
/// backed (e.g. the OpenAI logprob reranker) or scripted (the testkit mock).
#[async_trait::async_trait]
pub trait CrossEncoderClient: Send + Sync {
    /// Rank `passages` by relevance to `query`.
    ///
    /// Returns `(passage, score)` tuples sorted in **descending** order of
    /// relevance (highest score first), matching upstream `rank`.
    async fn rank(&self, query: &str, passages: &[String]) -> Result<Vec<(String, f64)>, LlmError>;
}
