//! Configuration for the Chronicle-backed long-term memory provider.

use std::fmt;

/// Default Neo4j database name when none is supplied.
pub const DEFAULT_NEO4J_DATABASE: &str = "neo4j";

/// Default graph name for the FalkorDB backend.
pub const DEFAULT_FALKOR_GRAPH: &str = "chronicle";

/// Default concurrency cap passed to the Chronicle engine.
pub const DEFAULT_MAX_CONCURRENCY: usize = 20;

/// Default number of facts returned by [`recall`](crate::ChronicleLongTermMemory) when
/// the caller does not specify a limit.
pub const DEFAULT_RECALL_LIMIT: usize = 10;

/// Placeholder string substituted for secret fields in [`fmt::Debug`] output.
const REDACTED: &str = "<redacted>";

/// Graph-store backend that Chronicle persists episodes into.
///
/// Each variant carries exactly the connection parameters its driver needs.
/// Greentic is not locked to any single graph store: the operator chooses the
/// backend, and the long-term memory provider switches drivers accordingly.
///
/// Secrets ([`Neo4j::password`](Self::Neo4j) and the [`Falkor`](Self::Falkor)
/// connection string, which may embed credentials) are redacted by the manual
/// [`fmt::Debug`] implementation so they never leak into logs.
#[derive(Clone)]
pub enum ChronicleBackend {
    /// Neo4j over the Bolt protocol (a remote server).
    Neo4j {
        /// Bolt URI (for example `neo4j://localhost:7687`).
        uri: String,
        /// Neo4j username.
        user: String,
        /// Neo4j password. Redacted in [`fmt::Debug`].
        password: String,
        /// Neo4j database name.
        database: String,
    },
    /// FalkorDB (Redis-module openCypher) reached over a Redis connection string.
    Falkor {
        /// Redis connection string (for example `redis://localhost:6379`). May
        /// embed credentials; redacted wholesale in [`fmt::Debug`].
        connection: String,
        /// Graph key the episodes are stored under.
        graph: String,
    },
    /// Embedded SurrealDB persisted to a RocksDB directory on disk.
    SurrealEmbedded {
        /// Filesystem path of the RocksDB store.
        path: String,
    },
    /// Embedded in-memory SurrealDB. Ephemeral — every process starts empty.
    /// Intended for tests and disposable / development environments.
    SurrealMemory,
}

impl ChronicleBackend {
    /// Neo4j backend using the default database ([`DEFAULT_NEO4J_DATABASE`]).
    #[must_use]
    pub fn neo4j(
        uri: impl Into<String>,
        user: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self::Neo4j {
            uri: uri.into(),
            user: user.into(),
            password: password.into(),
            database: DEFAULT_NEO4J_DATABASE.to_string(),
        }
    }

    /// FalkorDB backend using the default graph name ([`DEFAULT_FALKOR_GRAPH`]).
    #[must_use]
    pub fn falkor(connection: impl Into<String>) -> Self {
        Self::Falkor {
            connection: connection.into(),
            graph: DEFAULT_FALKOR_GRAPH.to_string(),
        }
    }

    /// Embedded SurrealDB backend persisted to `path` (RocksDB on disk).
    #[must_use]
    pub fn surreal_embedded(path: impl Into<String>) -> Self {
        Self::SurrealEmbedded { path: path.into() }
    }

    /// Embedded in-memory SurrealDB backend (ephemeral).
    #[must_use]
    pub fn surreal_memory() -> Self {
        Self::SurrealMemory
    }

    /// Short, secret-free label of the backend kind, suitable for logs and
    /// telemetry.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Neo4j { .. } => "neo4j",
            Self::Falkor { .. } => "falkor",
            Self::SurrealEmbedded { .. } => "surreal-embedded",
            Self::SurrealMemory => "surreal-memory",
        }
    }
}

impl fmt::Debug for ChronicleBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neo4j {
                uri,
                user,
                database,
                ..
            } => f
                .debug_struct("Neo4j")
                .field("uri", uri)
                .field("user", user)
                .field("password", &REDACTED)
                .field("database", database)
                .finish(),
            Self::Falkor { graph, .. } => f
                .debug_struct("Falkor")
                // The connection string may embed credentials → redact wholesale.
                .field("connection", &REDACTED)
                .field("graph", graph)
                .finish(),
            Self::SurrealEmbedded { path } => f
                .debug_struct("SurrealEmbedded")
                .field("path", path)
                .finish(),
            Self::SurrealMemory => f.write_str("SurrealMemory"),
        }
    }
}

/// Connection and model configuration for [`ChronicleLongTermMemory`].
///
/// The graph store is selected by [`backend`](Self::backend); the remaining
/// fields configure the LLM and embedder Chronicle uses for extraction and
/// vector recall, plus engine tuning.
///
/// Secrets ([`openai_api_key`](Self::openai_api_key) and any inside
/// [`backend`](Self::backend)) are redacted by the manual [`fmt::Debug`]
/// implementation so they never leak into logs.
#[derive(Clone)]
pub struct ChronicleMemoryConfig {
    /// Graph-store backend to persist episodes into.
    pub backend: ChronicleBackend,
    /// OpenAI (or compatible) API key. When `None`, the underlying client reads
    /// `OPENAI_API_KEY` from the environment. Redacted in [`fmt::Debug`].
    pub openai_api_key: Option<String>,
    /// Override base URL for OpenAI-compatible endpoints.
    pub openai_base_url: Option<String>,
    /// Primary model name. When `None`, the Chronicle OpenAI client default applies.
    pub model: Option<String>,
    /// Small model name used for cheaper extraction calls. When `None`, the
    /// Chronicle OpenAI client default applies.
    pub small_model: Option<String>,
    /// Embedding model name. When `None`, the Chronicle embedder default applies.
    pub embedding_model: Option<String>,
    /// Embedding dimension. When `None`, the Chronicle embedder default applies.
    /// Also sizes the vector index for the Falkor / Surreal backends, so it MUST
    /// match the configured embedder's output dimension.
    pub embedding_dim: Option<usize>,
    /// Concurrency cap for the Chronicle engine. Defaults to [`DEFAULT_MAX_CONCURRENCY`].
    pub max_concurrency: usize,
    /// Default recall limit. Defaults to [`DEFAULT_RECALL_LIMIT`].
    pub recall_limit: usize,
}

impl ChronicleMemoryConfig {
    /// Builds a **Neo4j-backed** config from the required connection trio,
    /// applying defaults to every other field.
    ///
    /// Preserved for backward compatibility; use
    /// [`with_backend`](Self::with_backend) (or [`ChronicleBackend::falkor`] /
    /// [`ChronicleBackend::surreal_memory`] / …) to select a different backend.
    pub fn new(
        neo4j_uri: impl Into<String>,
        neo4j_user: impl Into<String>,
        neo4j_password: impl Into<String>,
    ) -> Self {
        Self::with_backend(ChronicleBackend::neo4j(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
        ))
    }

    /// Builds a config for an arbitrary [`ChronicleBackend`], applying defaults to
    /// the model and engine fields.
    #[must_use]
    pub fn with_backend(backend: ChronicleBackend) -> Self {
        Self {
            backend,
            openai_api_key: None,
            openai_base_url: None,
            model: None,
            small_model: None,
            embedding_model: None,
            embedding_dim: None,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            recall_limit: DEFAULT_RECALL_LIMIT,
        }
    }

    /// Sets the Neo4j database name. No-op when the backend is not Neo4j.
    #[must_use]
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        if let ChronicleBackend::Neo4j { database: db, .. } = &mut self.backend {
            *db = database.into();
        }
        self
    }

    /// Sets the OpenAI API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.openai_api_key = Some(api_key.into());
        self
    }

    /// Sets the OpenAI-compatible base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.openai_base_url = Some(base_url.into());
        self
    }

    /// Sets the primary and small model names.
    #[must_use]
    pub fn with_models(mut self, model: Option<String>, small_model: Option<String>) -> Self {
        self.model = model;
        self.small_model = small_model;
        self
    }

    /// Sets the embedding model name and dimension.
    #[must_use]
    pub fn with_embedding(
        mut self,
        embedding_model: Option<String>,
        embedding_dim: Option<usize>,
    ) -> Self {
        self.embedding_model = embedding_model;
        self.embedding_dim = embedding_dim;
        self
    }

    /// Sets the engine concurrency cap.
    #[must_use]
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Sets the default recall limit.
    #[must_use]
    pub fn with_recall_limit(mut self, recall_limit: usize) -> Self {
        self.recall_limit = recall_limit;
        self
    }
}

impl fmt::Debug for ChronicleMemoryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChronicleMemoryConfig")
            .field("backend", &self.backend)
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| REDACTED),
            )
            .field("openai_base_url", &self.openai_base_url)
            .field("model", &self.model)
            .field("small_model", &self.small_model)
            .field("embedding_model", &self.embedding_model)
            .field("embedding_dim", &self.embedding_dim)
            .field("max_concurrency", &self.max_concurrency)
            .field("recall_limit", &self.recall_limit)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults_to_neo4j_backend() {
        let cfg = ChronicleMemoryConfig::new("neo4j://localhost:7687", "neo4j", "secret-pw");
        assert_eq!(cfg.backend.kind(), "neo4j");
        match &cfg.backend {
            ChronicleBackend::Neo4j { database, .. } => {
                assert_eq!(database, DEFAULT_NEO4J_DATABASE);
            }
            other => panic!("expected Neo4j backend, got {other:?}"),
        }
        assert_eq!(cfg.max_concurrency, DEFAULT_MAX_CONCURRENCY);
        assert_eq!(cfg.recall_limit, DEFAULT_RECALL_LIMIT);
        assert!(cfg.openai_api_key.is_none());
    }

    #[test]
    fn with_database_overrides_neo4j_only() {
        let cfg = ChronicleMemoryConfig::new("uri", "user", "pw").with_database("graph-db");
        match &cfg.backend {
            ChronicleBackend::Neo4j { database, .. } => assert_eq!(database, "graph-db"),
            other => panic!("expected Neo4j, got {other:?}"),
        }
        // No-op on a non-Neo4j backend.
        let surreal = ChronicleMemoryConfig::with_backend(ChronicleBackend::surreal_memory())
            .with_database("ignored");
        assert_eq!(surreal.backend.kind(), "surreal-memory");
    }

    #[test]
    fn backend_constructors_carry_defaults() {
        assert_eq!(
            ChronicleBackend::falkor("redis://localhost:6379").kind(),
            "falkor"
        );
        match ChronicleBackend::falkor("redis://localhost:6379") {
            ChronicleBackend::Falkor { graph, .. } => assert_eq!(graph, DEFAULT_FALKOR_GRAPH),
            other => panic!("expected Falkor, got {other:?}"),
        }
        assert_eq!(
            ChronicleBackend::surreal_embedded("/tmp/store").kind(),
            "surreal-embedded"
        );
        assert_eq!(ChronicleBackend::surreal_memory().kind(), "surreal-memory");
    }

    #[test]
    fn debug_redacts_neo4j_password_and_api_key() {
        let cfg = ChronicleMemoryConfig::new("neo4j://localhost:7687", "neo4j", "super-secret-pw")
            .with_api_key("sk-very-secret-key");
        let rendered = format!("{cfg:?}");

        assert!(rendered.contains(REDACTED), "expected redaction marker");
        assert!(
            !rendered.contains("super-secret-pw"),
            "neo4j password must not appear in debug output"
        );
        assert!(
            !rendered.contains("sk-very-secret-key"),
            "openai api key must not appear in debug output"
        );
        // Non-secret fields remain visible.
        assert!(rendered.contains("neo4j://localhost:7687"));
    }

    #[test]
    fn debug_redacts_falkor_connection_string() {
        let cfg = ChronicleMemoryConfig::with_backend(ChronicleBackend::falkor(
            "redis://admin:hunter2@redis:6379",
        ));
        let rendered = format!("{cfg:?}");
        assert!(
            !rendered.contains("hunter2"),
            "falkor connection credentials must not appear in debug output"
        );
        assert!(rendered.contains(REDACTED));
    }

    #[test]
    fn debug_renders_absent_api_key_as_none() {
        let no_key = ChronicleMemoryConfig::new("uri", "user", "pw");
        let rendered = format!("{no_key:?}");
        assert!(rendered.contains("openai_api_key: None"));
    }
}
