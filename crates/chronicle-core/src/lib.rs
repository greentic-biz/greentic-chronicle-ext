#![forbid(unsafe_code)]

pub mod chronicle;
pub mod cross_encoder;
pub mod document_rag;
pub mod driver;
pub mod embedder;
pub mod errors;
pub mod helpers;
pub mod llm;
pub mod pipeline;
pub mod prompts;
pub mod search;
pub mod types;

pub use document_rag::{DOCUMENT_CHUNK_LABEL, DocumentChunk, DocumentChunkHit, chunk_text};
pub use errors::ChronicleError;
