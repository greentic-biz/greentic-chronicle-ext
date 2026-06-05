#![forbid(unsafe_code)]

//! OpenAI-compatible [`LlmClient`] and [`EmbedderClient`] implementations for
//! chronicle, backed by [`async-openai`].
//!
//! Ports `graphiti_core/llm_client/openai_generic_client.py` (+
//! `openai_base_client.py` for defaults) and `graphiti_core/embedder/openai.py`
//! @ 34f56e65 (v0.29.1).
//!
//! # Retry / typing layering
//!
//! The retry boundary lives **inside** [`OpenAiLlm::generate`]: only the
//! provider call is wrapped in [`chronicle_core::llm::retry::with_retry`].
//! Within that closure, provider-malformed JSON (the API returns text that is
//! not valid JSON) maps to [`chronicle_core::llm::LlmError::InvalidJson`],
//! which is retryable — the upstream `json.JSONDecodeError` analog, correctly
//! retried.
//!
//! Concrete-struct deserialization happens **outside** this crate, in
//! [`chronicle_core::llm::generate_typed`], which calls `generate` and then
//! deserializes the returned [`serde_json::Value`] into the caller's type. A
//! struct mismatch (upstream's non-retryable `ValidationError`) therefore never
//! enters the retry loop. This client never deserializes into a concrete
//! struct, so it cannot leak a struct-mismatch error into the retry path. The
//! two `InvalidJson` sub-cases noted in `chronicle_core::llm` are thus kept on
//! opposite sides of the retry boundary by construction.
//!
//! [`LlmClient`]: chronicle_core::llm::LlmClient
//! [`EmbedderClient`]: chronicle_core::embedder::EmbedderClient
//! [`async-openai`]: https://docs.rs/async-openai

mod embedder;
mod llm;

pub use embedder::{DEFAULT_EMBEDDING_MODEL, OpenAiEmbedder, OpenAiEmbedderConfig};
pub use llm::{DEFAULT_MODEL, DEFAULT_SMALL_MODEL, OpenAiLlm};
