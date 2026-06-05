use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChronicleError {
    // #[error("driver error: {0}")]
    // Driver(#[from] crate::driver::DriverError),      // enabled in task 8
    #[error("llm error: {0}")]
    Llm(#[from] crate::llm::LlmError),
    // #[error("embedder error: {0}")]
    // Embedder(#[from] crate::embedder::EmbedderError), // enabled in task 5
    #[error("node not found: {uuid}")]
    NodeNotFound { uuid: String },
    #[error("episode not found: {uuid}")]
    EpisodeNotFound { uuid: String },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
