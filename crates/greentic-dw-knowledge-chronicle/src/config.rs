//! Configuration for the Chronicle-backed knowledge (document-RAG) provider.

use std::fmt;

/// Default concurrency cap passed to the Chronicle engine.
pub const DEFAULT_MAX_CONCURRENCY: usize = 20;

/// Default number of chunks returned by [`search`](crate::KnowledgeChronicle::search)
/// when the caller does not specify a limit.
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Connection and embedding configuration for [`KnowledgeChronicle`](crate::KnowledgeChronicle).
///
/// Secrets ([`openai_api_key`](Self::openai_api_key),
/// [`llm_api_key`](Self::llm_api_key), and any neo4j password) are redacted by
/// the manual [`fmt::Debug`] implementation so they never appear in logs.
#[derive(Clone)]
pub struct KnowledgeConfig {
    /// Embedding model name. `None` uses the provider's configured default.
    pub embedding_model: Option<String>,
    /// Embedding dimension. Required: this feeds the bridge and the index.
    pub embedding_dim: usize,
    /// OpenAI (or compatible) API key for the embedder.
    /// When `None`, the underlying OpenAI client reads `OPENAI_API_KEY` from env.
    /// Redacted in [`fmt::Debug`].
    pub openai_api_key: Option<String>,
    /// Override base URL for OpenAI-compatible embedding endpoints.
    pub openai_base_url: Option<String>,
    /// Optional LLM API key (for future LLM-augmented RAG). Redacted in [`fmt::Debug`].
    pub llm_api_key: Option<String>,
    /// Optional LLM model name.
    pub llm_model: Option<String>,
    /// SurrealDB / local database path (used by the `surreal` feature).
    pub db_path: Option<String>,
    /// Neo4j Bolt URI (used by the `neo4j` feature).
    pub neo4j_uri: Option<String>,
    /// Neo4j username (used by the `neo4j` feature).
    pub neo4j_user: Option<String>,
    /// Neo4j password (used by the `neo4j` feature). Redacted in [`fmt::Debug`].
    pub neo4j_password: Option<String>,
    /// Neo4j database name (used by the `neo4j` feature).
    pub neo4j_database: Option<String>,
    /// Concurrency cap for the Chronicle engine. Defaults to [`DEFAULT_MAX_CONCURRENCY`].
    pub max_concurrency: usize,
    /// Default search result limit. Defaults to [`DEFAULT_SEARCH_LIMIT`].
    pub search_limit: usize,
}

impl KnowledgeConfig {
    /// Constructs a config with the required embedding dimension, applying
    /// defaults to all other fields.
    #[must_use]
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_model: None,
            embedding_dim,
            openai_api_key: None,
            openai_base_url: None,
            llm_api_key: None,
            llm_model: None,
            db_path: None,
            neo4j_uri: None,
            neo4j_user: None,
            neo4j_password: None,
            neo4j_database: None,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            search_limit: DEFAULT_SEARCH_LIMIT,
        }
    }

    /// Sets the embedding model name.
    #[must_use]
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    /// Sets the OpenAI API key (for the embedder).
    #[must_use]
    pub fn with_openai_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.openai_api_key = Some(api_key.into());
        self
    }

    /// Sets the OpenAI-compatible base URL.
    #[must_use]
    pub fn with_openai_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.openai_base_url = Some(base_url.into());
        self
    }

    /// Sets the LLM API key.
    #[must_use]
    pub fn with_llm_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.llm_api_key = Some(api_key.into());
        self
    }

    /// Sets the LLM model name.
    #[must_use]
    pub fn with_llm_model(mut self, model: impl Into<String>) -> Self {
        self.llm_model = Some(model.into());
        self
    }

    /// Sets the local database path (SurrealDB).
    #[must_use]
    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    /// Sets the engine concurrency cap.
    #[must_use]
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Sets the default search result limit.
    #[must_use]
    pub fn with_search_limit(mut self, search_limit: usize) -> Self {
        self.search_limit = search_limit;
        self
    }
}

/// Placeholder substituted for secrets in [`fmt::Debug`] output.
const REDACTED: &str = "<redacted>";

impl fmt::Debug for KnowledgeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnowledgeConfig")
            .field("embedding_model", &self.embedding_model)
            .field("embedding_dim", &self.embedding_dim)
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| REDACTED),
            )
            .field("openai_base_url", &self.openai_base_url)
            .field("llm_api_key", &self.llm_api_key.as_ref().map(|_| REDACTED))
            .field("llm_model", &self.llm_model)
            .field("db_path", &self.db_path)
            .field("neo4j_uri", &self.neo4j_uri)
            .field("neo4j_user", &self.neo4j_user)
            .field(
                "neo4j_password",
                &self.neo4j_password.as_ref().map(|_| REDACTED),
            )
            .field("neo4j_database", &self.neo4j_database)
            .field("max_concurrency", &self.max_concurrency)
            .field("search_limit", &self.search_limit)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_applies_defaults() {
        let cfg = KnowledgeConfig::new(1024);
        assert_eq!(cfg.embedding_dim, 1024);
        assert_eq!(cfg.max_concurrency, DEFAULT_MAX_CONCURRENCY);
        assert_eq!(cfg.search_limit, DEFAULT_SEARCH_LIMIT);
        assert!(cfg.openai_api_key.is_none());
        assert!(cfg.llm_api_key.is_none());
        assert!(cfg.embedding_model.is_none());
    }

    #[test]
    fn builder_methods_set_fields() {
        let cfg = KnowledgeConfig::new(512)
            .with_embedding_model("text-embedding-3-small")
            .with_openai_api_key("sk-test")
            .with_llm_api_key("lk-test")
            .with_max_concurrency(4)
            .with_search_limit(5);

        assert_eq!(cfg.embedding_dim, 512);
        assert_eq!(
            cfg.embedding_model.as_deref(),
            Some("text-embedding-3-small")
        );
        assert_eq!(cfg.openai_api_key.as_deref(), Some("sk-test"));
        assert_eq!(cfg.llm_api_key.as_deref(), Some("lk-test"));
        assert_eq!(cfg.max_concurrency, 4);
        assert_eq!(cfg.search_limit, 5);
    }

    #[test]
    fn debug_redacts_secrets() {
        let cfg = KnowledgeConfig::new(1024)
            .with_openai_api_key("sk-very-secret")
            .with_llm_api_key("lk-very-secret");

        let rendered = format!("{cfg:?}");
        assert!(rendered.contains(REDACTED), "expected redaction marker");
        assert!(
            !rendered.contains("sk-very-secret"),
            "openai api key must not appear in debug output"
        );
        assert!(
            !rendered.contains("lk-very-secret"),
            "llm api key must not appear in debug output"
        );
        assert!(rendered.contains("1024"), "embedding_dim should be visible");
    }

    #[test]
    fn debug_shows_none_for_absent_secrets() {
        let cfg = KnowledgeConfig::new(256);
        let rendered = format!("{cfg:?}");
        assert!(rendered.contains("openai_api_key: None"));
        assert!(rendered.contains("llm_api_key: None"));
        assert!(rendered.contains("neo4j_password: None"));
    }
}
