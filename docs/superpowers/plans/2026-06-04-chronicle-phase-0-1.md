# Chronicle Phase 0+1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `greentic-chronicle-ext` workspace and deliver the graphiti core loop on Neo4j: ingest an episode end-to-end (extract → dedup → bi-temporal invalidation → persist) and recall facts via hybrid RRF search.

**Architecture:** Faithful Rust port of graphiti-core v0.29.1 (pinned upstream clone at `/home/bima-pangestu/Works/refs/graphiti`, commit `34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6`). Four crates: `chronicle-core` (types, traits, prompts, pipeline, search), `chronicle-driver-neo4j` (neo4rs), `chronicle-llm-openai` (async-openai), `chronicle-testkit` (FakeDriver + MockLlm for deterministic CI tests). Operation-level driver traits — no raw-Cypher passthrough.

**Tech Stack:** Rust 1.95 edition 2024, tokio, async-trait, serde/serde_json, schemars, thiserror, uuid, chrono, blake2, neo4rs, async-openai, tracing.

**Porting rules (apply to every task):**
- Upstream reference is ALWAYS `/home/bima-pangestu/Works/refs/graphiti/graphiti_core/<path>` at v0.29.1. Never consult a newer version.
- Prompt text and response-model field descriptions are ported **verbatim** (same wording, same casing). Each ported file starts with `// Ported from graphiti_core/<path> @ 34f56e65 (v0.29.1)`.
- No `unwrap()`/`panic!()`/`expect()` in non-test code. `#![forbid(unsafe_code)]` at every crate root.
- Conventional commits. NO Claude attribution trailers on any commit.
- After each task: `cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings` must pass before committing.

---

### Task 1: Workspace scaffold (Phase 0)

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `.gitignore`, `LICENSE`, `NOTICE`, `README.md`, `ci/local_check.sh`, `crates/.gitkeep`

- [ ] **Step 1: Root manifests**

`Cargo.toml`:
```toml
[workspace]
resolver = "3"
members = [
    "crates/chronicle-core",
    "crates/chronicle-testkit",
    "crates/chronicle-driver-neo4j",
    "crates/chronicle-llm-openai",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/greentic-biz/greentic-chronicle-ext"
rust-version = "1.95"

[workspace.dependencies]
chronicle-core = { path = "crates/chronicle-core" }
chronicle-testkit = { path = "crates/chronicle-testkit" }
tokio = { version = "1", features = ["macros", "rt-multi-thread", "sync", "time"] }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "1"
thiserror = "2"
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
blake2 = "0.10"
tracing = "0.1"
rand = "0.9"
neo4rs = "0.9"
async-openai = "0.32"
anyhow = "1"
```
(If a pinned minor doesn't exist when running `cargo update`, take the latest compatible and note it in the PR.)

`rust-toolchain.toml`:
```toml
[toolchain]
channel = "1.95.0"
components = ["rustfmt", "clippy"]
```

`.gitignore`:
```
/target
**/target
```

- [ ] **Step 2: LICENSE + NOTICE**

```bash
cp /home/bima-pangestu/Works/refs/graphiti/LICENSE LICENSE
```

`NOTICE`:
```
greentic-chronicle-ext

This product includes software ported from Graphiti
(https://github.com/getzep/graphiti), Copyright Zep Software, Inc.,
licensed under the Apache License, Version 2.0.

Port baseline: graphiti v0.29.1
(commit 34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6).
```

`README.md` (stub):
```markdown
# greentic-chronicle-ext

Bi-temporal knowledge-graph memory for Greentic digital workers.
Rust port of [Graphiti](https://github.com/getzep/graphiti) (Apache-2.0, see NOTICE).

Crates: `chronicle-core`, `chronicle-driver-neo4j`, `chronicle-llm-openai`, `chronicle-testkit`.

Design spec: `docs/superpowers/specs/2026-06-04-greentic-chronicle-ext-design.md`.
```

- [ ] **Step 3: CI script**

`ci/local_check.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```
`chmod +x ci/local_check.sh`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: scaffold workspace, toolchain pin, CI, Apache-2.0 NOTICE"
```
(Workspace won't build until Task 2 adds the first member — that's fine; don't run local_check yet.)

---

### Task 2: chronicle-core skeleton, errors, helpers

**Files:**
- Create: `crates/chronicle-core/Cargo.toml`, `src/lib.rs`, `src/errors.rs`, `src/helpers.rs`
- Test: inline `#[cfg(test)]` in `helpers.rs`

- [ ] **Step 1: Crate manifest**

`crates/chronicle-core/Cargo.toml`:
```toml
[package]
name = "chronicle-core"
description = "Bi-temporal knowledge-graph memory core (Rust port of graphiti-core)"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
tokio.workspace = true
async-trait.workspace = true
serde.workspace = true
serde_json.workspace = true
schemars.workspace = true
thiserror.workspace = true
uuid.workspace = true
chrono.workspace = true
blake2.workspace = true
tracing.workspace = true
rand.workspace = true
```

Temporarily reduce workspace `members` to `["crates/chronicle-core"]`; restore each member as its crate is created (Tasks 9, 15, 16).

- [ ] **Step 2: lib.rs + errors**

`src/lib.rs`:
```rust
#![forbid(unsafe_code)]

pub mod driver;
pub mod embedder;
pub mod errors;
pub mod helpers;
pub mod llm;
pub mod pipeline;
pub mod prompts;
pub mod search;
pub mod types;

pub use errors::ChronicleError;
```
(Declare modules as they land; start with `errors` + `helpers` only and grow the list per task.)

`src/errors.rs`:
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChronicleError {
    #[error("driver error: {0}")]
    Driver(#[from] crate::driver::DriverError),
    #[error("llm error: {0}")]
    Llm(#[from] crate::llm::LlmError),
    #[error("embedder error: {0}")]
    Embedder(#[from] crate::embedder::EmbedderError),
    #[error("node not found: {uuid}")]
    NodeNotFound { uuid: String },
    #[error("episode not found: {uuid}")]
    EpisodeNotFound { uuid: String },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
```
(The `Driver`/`Llm`/`Embedder` variants compile once Tasks 4 & 8 land; until then keep them commented with a `// enabled in task N` marker, or land errors.rs last in this task — implementer's choice, but the final shape is the above.)

- [ ] **Step 3: helpers with tests**

`src/helpers.rs` — port targets: `graphiti_core/helpers.py` (`utc_now`, `ensure_utc`) and `graphiti_core/utils/maintenance/dedup_helpers.py::_normalize_string_exact`:
```rust
use chrono::{DateTime, Utc};

/// Default concurrency for parallel LLM calls (upstream SEMAPHORE_LIMIT).
pub const SEMAPHORE_LIMIT: usize = 20;
/// Upstream RELEVANT_SCHEMA_LIMIT (search_utils.py): prior episodes pulled as context.
pub const RELEVANT_SCHEMA_LIMIT: usize = 10;
/// Upstream EPISODE_WINDOW_LEN (graph_data_operations.py).
pub const EPISODE_WINDOW_LEN: usize = 3;

pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Lowercase + collapse internal whitespace (upstream `_normalize_string_exact`).
pub fn normalize_string_exact(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_whitespace_and_lowercases() {
        assert_eq!(normalize_string_exact("  Alice   SMITH "), "alice smith");
        assert_eq!(normalize_string_exact(""), "");
    }
}
```

- [ ] **Step 4: Run and commit**

```bash
cargo test -p chronicle-core
```
Expected: PASS (1 test).
```bash
git add -A && git commit -m "feat(core): crate skeleton, error enum, normalization helpers"
```

---

### Task 3: Core domain types

**Files:**
- Create: `crates/chronicle-core/src/types/mod.rs`, `types/episode.rs`, `types/node.rs`, `types/edge.rs`
- Port from: `graphiti_core/nodes.py`, `graphiti_core/edges.py`

- [ ] **Step 1: Failing test (serde roundtrip + bi-temporal fields)**

`types/edge.rs` test section:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_edge_roundtrips_with_bitemporal_fields() {
        let edge = EntityEdge::new(
            "u1".into(), "u2".into(), "WORKS_AT".into(),
            "Alice works at Acme".into(), "g1".into(),
        );
        assert!(edge.valid_at.is_none() && edge.invalid_at.is_none()
            && edge.expired_at.is_none());
        let json = serde_json::to_string(&edge).unwrap();
        let back: EntityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fact, "Alice works at Acme");
    }
}
```

- [ ] **Step 2: Implement types**

`types/episode.rs`:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeType {
    Message,
    Text,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub source: EpisodeType,
    pub source_description: String,
    pub content: String,
    /// UUIDs of EntityEdges derived from this episode.
    pub entity_edges: Vec<String>,
    pub created_at: DateTime<Utc>,
    /// When the episode's content was true/occurred (reference_time).
    pub valid_at: DateTime<Utc>,
}
```

`types/node.rs`:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    /// Always contains "Entity"; specific entity-type labels appended.
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub summary: String,
    pub attributes: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_embedding: Option<Vec<f32>>,
}

impl EntityNode {
    pub fn new(name: String, group_id: String, created_at: DateTime<Utc>) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            name,
            group_id,
            labels: vec!["Entity".to_string()],
            created_at,
            summary: String::new(),
            attributes: Map::new(),
            name_embedding: None,
        }
    }
}
```

`types/edge.rs` — the bi-temporal heart:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Temporal entity-to-entity fact. Bi-temporal:
/// created_at/expired_at = transaction time, valid_at/invalid_at = valid time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    /// SCREAMING_SNAKE_CASE relation type (e.g. WORKS_AT).
    pub name: String,
    pub fact: String,
    pub group_id: String,
    /// Episode UUIDs in which this fact was mentioned.
    pub episodes: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expired_at: Option<DateTime<Utc>>,
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    pub attributes: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_embedding: Option<Vec<f32>>,
}

impl EntityEdge {
    pub fn new(
        source_node_uuid: String,
        target_node_uuid: String,
        name: String,
        fact: String,
        group_id: String,
    ) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4().to_string(),
            source_node_uuid,
            target_node_uuid,
            name,
            fact,
            group_id,
            episodes: Vec::new(),
            created_at: crate::helpers::utc_now(),
            expired_at: None,
            valid_at: None,
            invalid_at: None,
            attributes: Map::new(),
            fact_embedding: None,
        }
    }
}

/// MENTIONS edge: episode -> entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub uuid: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
}
```

`types/mod.rs` re-exports all of the above. Add `pub mod types;` to lib.rs.

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p chronicle-core
```
Expected: PASS.
```bash
git add -A && git commit -m "feat(core): episodic/entity node and bi-temporal edge types"
```

---

### Task 4: LLM abstraction

**Files:**
- Create: `crates/chronicle-core/src/llm/mod.rs`, `llm/message.rs`, `llm/config.rs`, `llm/retry.rs`
- Port from: `graphiti_core/llm_client/client.py`, `config.py`, `errors.py`

- [ ] **Step 1: Failing test (typed structured-output helper against a closure mock)**

In `llm/mod.rs` tests:
```rust
#[tokio::test]
async fn generate_typed_deserializes_structured_response() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct Out { answer: String }
    let mock = StaticLlm(serde_json::json!({"answer": "42"}));
    let out: Out = generate_typed(&mock, LlmRequest::new(vec![
        Message::system("s"), Message::user("u"),
    ])).await.unwrap();
    assert_eq!(out.answer, "42");
}
```
with `StaticLlm` defined in the test module implementing `LlmClient` by returning its `serde_json::Value`.

- [ ] **Step 2: Implement**

`llm/message.rs`:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
}
```

`llm/config.rs` — upstream defaults preserved:
```rust
pub const DEFAULT_MAX_TOKENS: u32 = 16384;
pub const DEFAULT_TEMPERATURE: f32 = 0.0;

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub small_model: Option<String>,
    pub base_url: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: None,
            small_model: None,
            base_url: None,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}
```

`llm/mod.rs`:
```rust
mod config;
mod message;
mod retry;

pub use config::{LlmConfig, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE};
pub use message::{Message, Role};
pub use retry::with_retry;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Small,
    Medium,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("rate limit exceeded")]
    RateLimit,
    #[error("llm refused: {0}")]
    Refusal(String),
    #[error("empty response")]
    EmptyResponse,
    #[error("server error ({status}): {message}")]
    Server { status: u16, message: String },
    #[error("invalid json from llm: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("transport error: {0}")]
    Transport(String),
}

impl LlmError {
    /// Upstream retries on RateLimitError, JSONDecodeError and 5xx.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            LlmError::RateLimit | LlmError::InvalidJson(_) | LlmError::Server { .. }
        )
    }
}

/// JSON-schema description of the expected response (Pydantic response_model analog).
#[derive(Debug, Clone)]
pub struct ResponseSchema {
    pub name: String,
    pub schema: serde_json::Value,
}

impl ResponseSchema {
    pub fn of<T: JsonSchema>() -> Self {
        Self {
            name: std::any::type_name::<T>()
                .rsplit("::")
                .next()
                .unwrap_or("structured_response")
                .to_string(),
            schema: serde_json::to_value(schemars::schema_for!(T))
                .unwrap_or(serde_json::Value::Null),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub messages: Vec<Message>,
    pub response_schema: Option<ResponseSchema>,
    pub max_tokens: Option<u32>,
    pub model_size: ModelSize,
    /// Telemetry label, e.g. "extract_nodes.extract_message".
    pub prompt_name: Option<String>,
}

impl LlmRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            response_schema: None,
            max_tokens: None,
            model_size: ModelSize::Medium,
            prompt_name: None,
        }
    }
    pub fn with_schema(mut self, schema: ResponseSchema) -> Self {
        self.response_schema = Some(schema);
        self
    }
    pub fn small(mut self) -> Self {
        self.model_size = ModelSize::Small;
        self
    }
    pub fn named(mut self, name: &str) -> Self {
        self.prompt_name = Some(name.to_string());
        self
    }
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Returns the parsed JSON object from the model.
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError>;
}

/// Typed wrapper: attaches T's schema and deserializes the result.
pub async fn generate_typed<T: DeserializeOwned + JsonSchema>(
    client: &dyn LlmClient,
    request: LlmRequest,
) -> Result<T, LlmError> {
    let request = request.with_schema(ResponseSchema::of::<T>());
    let value = client.generate(request).await?;
    Ok(serde_json::from_value(value)?)
}
```

`llm/retry.rs` — upstream: 4 attempts, `wait_random_exponential(multiplier=10, min=5, max=120)`:
```rust
use std::future::Future;
use std::time::Duration;

use super::LlmError;

pub const MAX_ATTEMPTS: u32 = 4;
const BACKOFF_MIN_SECS: f64 = 5.0;
const BACKOFF_MAX_SECS: f64 = 120.0;
const BACKOFF_MULTIPLIER: f64 = 10.0;

/// Retry an LLM call with randomized exponential backoff (upstream tenacity policy).
pub async fn with_retry<T, F, Fut>(mut call: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
{
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match call().await {
            Ok(value) => return Ok(value),
            Err(err) if err.is_retryable() && attempt < MAX_ATTEMPTS => {
                let cap = (BACKOFF_MULTIPLIER * 2f64.powi(attempt as i32 - 1))
                    .clamp(BACKOFF_MIN_SECS, BACKOFF_MAX_SECS);
                let wait = rand::random::<f64>() * cap;
                let wait = wait.max(BACKOFF_MIN_SECS).min(BACKOFF_MAX_SECS);
                tracing::warn!(attempt, wait_secs = wait, error = %err, "retrying LLM call");
                tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            }
            Err(err) => return Err(err),
        }
    }
}
```

Add `pub mod llm;` to lib.rs and enable the `Llm` variant in `ChronicleError`.

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p chronicle-core llm
```
Expected: PASS.
```bash
git add -A && git commit -m "feat(core): LlmClient trait with structured output and retry policy"
```

---

### Task 5: Embedder abstraction

**Files:**
- Create: `crates/chronicle-core/src/embedder/mod.rs`
- Port from: `graphiti_core/embedder/client.py`

- [ ] **Step 1: Implement (small enough to TDD in one pass — write the test below first)**

```rust
use async_trait::async_trait;
use thiserror::Error;

/// Upstream EMBEDDING_DIM default.
pub const DEFAULT_EMBEDDING_DIM: usize = 1024;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("provider error: {0}")]
    Provider(String),
}

#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub embedding_dim: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self { embedding_dim: DEFAULT_EMBEDDING_DIM }
    }
}

#[async_trait]
pub trait EmbedderClient: Send + Sync {
    fn embedding_dim(&self) -> usize;
    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError>;
    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}
```

Test: a unit-norm fake in `#[cfg(test)]` asserting `create_batch` default behavior if you provide one (or just trait-object compile check):
```rust
#[tokio::test]
async fn embedder_trait_is_object_safe() {
    struct Zero;
    #[async_trait::async_trait]
    impl EmbedderClient for Zero {
        fn embedding_dim(&self) -> usize { 4 }
        async fn create(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![0.0; 4])
        }
        async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            Ok(inputs.iter().map(|_| vec![0.0; 4]).collect())
        }
    }
    let c: Box<dyn EmbedderClient> = Box::new(Zero);
    assert_eq!(c.create("x").await.unwrap().len(), 4);
}
```

- [ ] **Step 2: Run, enable `Embedder` error variant, commit**

```bash
cargo test -p chronicle-core embedder
git add -A && git commit -m "feat(core): EmbedderClient trait"
```

---

### Task 6: Prompt response models (verbatim schemas)

**Files:**
- Create: `crates/chronicle-core/src/prompts/mod.rs`, `prompts/models.rs`, `prompts/helpers.rs`
- Port from: `graphiti_core/prompts/{models,prompt_helpers,extract_nodes,extract_edges,dedupe_nodes,dedupe_edges,summarize_nodes}.py` (response models only in this task)

- [ ] **Step 1: Failing test — schema field fidelity**

```rust
#[test]
fn extracted_entity_schema_has_upstream_descriptions() {
    let schema = serde_json::to_value(schemars::schema_for!(ExtractedEntity)).unwrap();
    let props = &schema["properties"];
    assert!(props["name"]["description"]
        .as_str().unwrap().contains("Name of the extracted entity"));
    assert!(props["entity_type_id"]["description"]
        .as_str().unwrap().contains("ID of the classified entity type"));
}
```

- [ ] **Step 2: Implement models**

`prompts/helpers.rs` (port of `prompt_helpers.py`):
```rust
/// Appended to every system prompt (upstream DO_NOT_ESCAPE_UNICODE).
pub const DO_NOT_ESCAPE_UNICODE: &str = "\nDo not escape unicode characters.\n";

/// Upstream to_prompt_json: minified JSON, unicode preserved.
pub fn to_prompt_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}
```

`prompts/models.rs` — every struct gets `#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]`; **field descriptions verbatim from upstream** via `#[schemars(description = "...")]`. Full inventory (descriptions abbreviated here with `…` ONLY in this plan — in code copy the full upstream text from the listed source file):

```rust
// Ported from graphiti_core/prompts/extract_nodes.py @ 34f56e65 (v0.29.1)
pub struct ExtractedEntity { pub name: String, pub entity_type_id: i64, pub episode_indices: Vec<i64> }
pub struct ExtractedEntities { pub extracted_entities: Vec<ExtractedEntity> }
pub struct EntitySummary { pub summary: String }
pub struct SummarizedEntity { pub name: String, pub summary: String }
pub struct SummarizedEntities { pub summaries: Vec<SummarizedEntity> }

// Ported from graphiti_core/prompts/extract_edges.py @ 34f56e65
pub struct ExtractedEdge {
    pub source_entity_name: String,
    pub target_entity_name: String,
    pub relation_type: String,
    pub fact: String,
    pub valid_at: Option<String>,   // ISO 8601 string per upstream contract
    pub invalid_at: Option<String>,
    pub episode_indices: Vec<i64>,
}
pub struct ExtractedEdges { pub edges: Vec<ExtractedEdge> }
pub struct EdgeTimestamps { pub valid_at: Option<String>, pub invalid_at: Option<String> }

// Ported from graphiti_core/prompts/dedupe_nodes.py @ 34f56e65
pub struct NodeDuplicate { pub id: i64, pub name: String, pub duplicate_candidate_id: i64 }
pub struct NodeResolutions { pub entity_resolutions: Vec<NodeDuplicate> }

// Ported from graphiti_core/prompts/dedupe_edges.py @ 34f56e65
pub struct EdgeDuplicate { pub duplicate_facts: Vec<i64>, pub contradicted_facts: Vec<i64> }

// Ported from graphiti_core/prompts/summarize_nodes.py @ 34f56e65
pub struct Summary { pub summary: String }
pub struct SummaryDescription { pub description: String }
```
Upstream `Edge` is renamed `ExtractedEdge` to avoid clashing with graph types — record this rename in the fidelity ledger (Task 17).

- [ ] **Step 3: Run, commit**

```bash
cargo test -p chronicle-core prompts
git add -A && git commit -m "feat(core): prompt response models with verbatim upstream schemas"
```

---

### Task 7: Prompt text port (verbatim)

**Files:**
- Create: `prompts/extract_nodes.rs`, `prompts/extract_edges.rs`, `prompts/dedupe_nodes.rs`, `prompts/dedupe_edges.rs`, `prompts/summarize_nodes.rs`, `prompts/snippets.rs`
- Port from: same-named `.py` files under `/home/bima-pangestu/Works/refs/graphiti/graphiti_core/prompts/`

Phase 1 needs these prompt functions (skip the rest until later phases):

| Rust fn | Upstream | Context keys |
|---|---|---|
| `extract_nodes::extract_message` | extract_nodes.py::extract_message | entity_types, previous_episodes, episode_content, custom_extraction_instructions |
| `extract_nodes::extract_text` | extract_nodes.py::extract_text | entity_types, episode_content, custom_extraction_instructions |
| `extract_nodes::extract_json` | extract_nodes.py::extract_json | entity_types, source_description, episode_content, custom_extraction_instructions |
| `extract_nodes::extract_attributes` | extract_nodes.py::extract_attributes | extracted_entities, entity_types, previous_episodes, episode_content, node, node_summary, attributes |
| `extract_edges::edge` | extract_edges.py::edge | previous_episodes, episode_content, nodes, reference_time, edge_types, custom_extraction_instructions |
| `extract_edges::extract_timestamps` | extract_edges.py::extract_timestamps | fact, reference_time |
| `dedupe_nodes::nodes` | dedupe_nodes.py::nodes | previous_episodes, episode_content, extracted_nodes, existing_nodes |
| `dedupe_edges::resolve_edge` | dedupe_edges.py::resolve_edge | existing_edges, edge_invalidation_candidates, new_edge |
| `summarize_nodes::summarize_context` | summarize_nodes.py::summarize_context | previous_episodes, episode_content, node_name, node_summary, attributes |

- [ ] **Step 1: Define the context pattern**

Each prompt fn is a plain function taking a typed context struct and returning `Vec<Message>`. Example shape (apply to all):

```rust
// Ported from graphiti_core/prompts/extract_nodes.py @ 34f56e65 (v0.29.1)
use crate::llm::Message;
use super::helpers::{to_prompt_json, DO_NOT_ESCAPE_UNICODE};

pub struct ExtractMessageContext<'a> {
    pub entity_types: &'a serde_json::Value,
    pub previous_episodes: &'a [String],
    pub episode_content: &'a str,
    pub custom_extraction_instructions: Option<&'a str>,
}

pub fn extract_message(ctx: &ExtractMessageContext<'_>) -> Vec<Message> {
    let system = format!(
        "{}{}",
        /* VERBATIM system prompt text from upstream extract_message() — copy
           the exact string from graphiti_core/prompts/extract_nodes.py */
        UPSTREAM_SYSTEM_TEXT,
        DO_NOT_ESCAPE_UNICODE,
    );
    let user = format!(
        /* VERBATIM user template with the same placeholder order as upstream */
        ...
    );
    vec![Message::system(system), Message::user(user)]
}
```

**Porting procedure per function (mandatory):**
1. Open the upstream `.py` file; locate the function.
2. Copy the system string and user f-string body **character-for-character** into Rust string literals (raw strings `r#"…"#` where quotes appear). Replace each `{context['x']}` interpolation with a `format!` argument fed by the context struct, preserving surrounding text exactly. Where upstream calls `to_prompt_json(...)`, call our `to_prompt_json`.
3. The `DO_NOT_ESCAPE_UNICODE` suffix on the system message reproduces upstream's `VersionWrapper` behavior.

- [ ] **Step 2: Tests — render snapshot per function**

For each ported fn, one test rendering with a small fixed context, asserting (a) message count and roles, (b) 2–3 distinctive verbatim sentences from the upstream prompt appear, e.g.:
```rust
#[test]
fn extract_message_renders_upstream_text() {
    let ctx = ExtractMessageContext {
        entity_types: &serde_json::json!([]),
        previous_episodes: &[],
        episode_content: "Alice joined Acme",
        custom_extraction_instructions: None,
    };
    let msgs = extract_message(&ctx);
    assert_eq!(msgs.len(), 2);
    assert!(msgs[0].content.contains("Do not escape unicode characters"));
    // + assertions on distinctive upstream sentences once copied
}
```

- [ ] **Step 3: Run, commit**

```bash
cargo test -p chronicle-core prompts
git add -A && git commit -m "feat(core): verbatim port of phase-1 prompt library"
```

---

### Task 8: Driver traits

**Files:**
- Create: `crates/chronicle-core/src/driver/mod.rs`
- Port shape from: `graphiti_core/driver/driver.py` (operation-property pattern → supertraits)

- [ ] **Step 1: Implement traits (compile-checked; behavior tested via FakeDriver in Task 9)**

```rust
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::types::{EntityEdge, EntityNode, EpisodicEdge, EpisodicNode, EpisodeType};

#[derive(Debug, Error)]
pub enum DriverError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("query error: {0}")]
    Query(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("decode error: {0}")]
    Decode(String),
}

#[async_trait]
pub trait EntityNodeOps: Send + Sync {
    async fn save_entity_nodes(&self, nodes: &[EntityNode]) -> Result<(), DriverError>;
    async fn get_entity_node(&self, uuid: &str) -> Result<Option<EntityNode>, DriverError>;
    async fn get_entity_nodes_by_uuids(&self, uuids: &[String]) -> Result<Vec<EntityNode>, DriverError>;
}

#[async_trait]
pub trait EntityEdgeOps: Send + Sync {
    async fn save_entity_edges(&self, edges: &[EntityEdge]) -> Result<(), DriverError>;
    async fn get_entity_edge(&self, uuid: &str) -> Result<Option<EntityEdge>, DriverError>;
    /// Edges from source_uuid -> target_uuid (single direction, mirrors upstream
    /// EntityEdge.get_between_nodes). Node-pair duplicate-candidate pool.
    /// AMENDED during Task 8 review: upstream v0.29.1 has NO topology-based
    /// "edges touching nodes" lookup — invalidation candidates come from hybrid
    /// fact-search in pipeline code (see Task 14 amendment).
    async fn get_edges_between_nodes(
        &self,
        source_uuid: &str,
        target_uuid: &str,
    ) -> Result<Vec<EntityEdge>, DriverError>;
}

#[async_trait]
pub trait EpisodeOps: Send + Sync {
    async fn save_episode(&self, episode: &EpisodicNode) -> Result<(), DriverError>;
    async fn get_episode(&self, uuid: &str) -> Result<Option<EpisodicNode>, DriverError>;
    async fn get_episodes_by_uuids(&self, uuids: &[String]) -> Result<Vec<EpisodicNode>, DriverError>;
    /// Last-n episodes with valid_at <= reference_time, chronological order.
    async fn retrieve_episodes(
        &self,
        reference_time: DateTime<Utc>,
        last_n: usize,
        group_ids: &[String],
        source: Option<EpisodeType>,
    ) -> Result<Vec<EpisodicNode>, DriverError>;
}

#[async_trait]
pub trait EpisodicEdgeOps: Send + Sync {
    async fn save_episodic_edges(&self, edges: &[EpisodicEdge]) -> Result<(), DriverError>;
}

/// Vector / fulltext / traversal search primitives the backend must provide.
#[async_trait]
pub trait SearchOps: Send + Sync {
    async fn edge_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn edge_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityEdge>, DriverError>;
    async fn node_fulltext_search(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<Vec<EntityNode>, DriverError>;
    async fn node_similarity_search(
        &self,
        search_vector: &[f32],
        group_ids: &[String],
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<EntityNode>, DriverError>;
}

#[async_trait]
pub trait SchemaOps: Send + Sync {
    async fn build_indices_and_constraints(&self, delete_existing: bool) -> Result<(), DriverError>;
}

/// Composite storage backend contract.
pub trait GraphDriver:
    EntityNodeOps + EntityEdgeOps + EpisodeOps + EpisodicEdgeOps + SearchOps + SchemaOps
{
    fn provider(&self) -> &'static str;
}
```
(SearchFilters/BFS params join in Phase 2 — keeping Phase 1 signatures lean is a deliberate scope cut; extend, don't redesign, in Phase 2.)

Enable `Driver` variant in `ChronicleError`. Add `pub mod driver;`.

- [ ] **Step 2: Build, commit**

```bash
cargo build -p chronicle-core && cargo clippy -p chronicle-core --all-targets -- -D warnings
git add -A && git commit -m "feat(core): operation-level GraphDriver trait family"
```

---### Task 9: chronicle-testkit — FakeDriver + MockLlm

**Files:**
- Create: `crates/chronicle-testkit/Cargo.toml`, `src/lib.rs`, `src/fake_driver.rs`, `src/mock_llm.rs`, `src/mock_embedder.rs`

- [ ] **Step 1: Manifest** (add member back to workspace `members`)

```toml
[package]
name = "chronicle-testkit"
description = "Deterministic test doubles for chronicle-core (in-memory driver, scripted LLM)"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
chronicle-core.workspace = true
tokio.workspace = true
async-trait.workspace = true
serde_json.workspace = true
chrono.workspace = true
```

- [ ] **Step 2: MockLlm — scripted, ordered responses**

`src/mock_llm.rs`:
```rust
use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use chronicle_core::llm::{LlmClient, LlmError, LlmRequest};

/// Replays queued JSON responses in order; records incoming requests.
pub struct MockLlm {
    responses: Mutex<VecDeque<serde_json::Value>>,
    pub requests: Mutex<Vec<LlmRequest>>,
}

impl MockLlm {
    pub fn new(responses: Vec<serde_json::Value>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        self.requests.lock().map_err(|_| LlmError::Transport("poisoned".into()))?
            .push(request);
        self.responses.lock().map_err(|_| LlmError::Transport("poisoned".into()))?
            .pop_front()
            .ok_or(LlmError::EmptyResponse)
    }
}
```

`src/mock_embedder.rs` — deterministic hash-based vectors so identical strings embed identically:
```rust
use async_trait::async_trait;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};

pub struct MockEmbedder {
    pub dim: usize,
}

impl MockEmbedder {
    fn embed(&self, input: &str) -> Vec<f32> {
        // Simple deterministic per-character rolling hash spread over dims, L2-normalized.
        let mut v = vec![0f32; self.dim];
        for (i, b) in input.bytes().enumerate() {
            v[i % self.dim] += f32::from(b) / 255.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl EmbedderClient for MockEmbedder {
    fn embedding_dim(&self) -> usize { self.dim }
    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embed(input))
    }
    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        Ok(inputs.iter().map(|s| self.embed(s)).collect())
    }
}
```

- [ ] **Step 3: FakeDriver — in-memory maps + brute-force search**

`src/fake_driver.rs`: `#[derive(Default)] pub struct FakeDriver { inner: Mutex<Inner> }` where `Inner` (also `Default`) holds `HashMap<String, EntityNode>`, `HashMap<String, EntityEdge>`, `HashMap<String, EpisodicNode>`, `Vec<EpisodicEdge>`. (Task 14's e2e test constructs it via `FakeDriver::default()`.) Implementations:
- `*_fulltext_search`: case-insensitive substring match on `name`/`fact`/`summary`/`content`, group-filtered, truncate to limit.
- `*_similarity_search`: brute-force cosine against stored embeddings, filter `> min_score`, sort desc, truncate.
- `retrieve_episodes`: filter `valid_at <= reference_time` + group + optional source, sort by valid_at desc, take n, reverse (chronological) — mirror upstream exactly.
- (AMENDED Task 8 review: no `get_edges_touching_nodes` — method removed from the trait; invalidation candidates are produced by hybrid fact-search in pipeline code.)

Write the cosine helper once:
```rust
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}
```

- [ ] **Step 4: Tests + commit**

Tests in `src/fake_driver.rs`: save/get roundtrip for node, edge, episode; `retrieve_episodes` ordering (insert 3 episodes out of order, expect chronological, reference_time cutoff respected); similarity search ranks identical-text embedding first.

```bash
cargo test -p chronicle-testkit
git add -A && git commit -m "feat(testkit): in-memory FakeDriver, scripted MockLlm, deterministic MockEmbedder"
```

---

### Task 10: Dedup helpers (exact port)

**Files:**
- Create: `crates/chronicle-core/src/pipeline/mod.rs`, `pipeline/dedup_helpers.rs`
- Port from: `graphiti_core/utils/maintenance/dedup_helpers.py`

- [ ] **Step 1: Failing tests — table-driven on upstream constants**

```rust
#[test]
fn entropy_gates_low_information_names() {
    // "aaaaaa" → entropy 0 → gated despite length 6
    assert!(!has_high_entropy("aaaaaa"));
    // "alice smith" → diverse chars + 2 tokens → passes
    assert!(has_high_entropy("alice smith"));
}

#[test]
fn jaccard_similarity_matches_definition() {
    let a = shingles("alice smith");
    let b = shingles("alice smith");
    assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-9);
    let c = shingles("bob jones");
    assert!(jaccard_similarity(&a, &c) < 0.2);
}

#[test]
fn minhash_lsh_buckets_identical_names_together() {
    let sig_a = minhash_signature(&shingles("acme corporation"));
    let sig_b = minhash_signature(&shingles("acme corporation"));
    assert_eq!(lsh_bands(&sig_a), lsh_bands(&sig_b));
}
```

- [ ] **Step 2: Implement — constants and algorithms exactly as upstream**

```rust
// Ported from graphiti_core/utils/maintenance/dedup_helpers.py @ 34f56e65 (v0.29.1)
use std::collections::BTreeSet;

use blake2::{digest::consts::U8, Blake2b, Digest};

pub const NAME_ENTROPY_THRESHOLD: f64 = 1.5;
pub const MIN_NAME_LENGTH: usize = 6;
pub const MIN_TOKEN_COUNT: usize = 2;
pub const FUZZY_JACCARD_THRESHOLD: f64 = 0.9;
pub const MINHASH_PERMUTATIONS: usize = 32;
pub const MINHASH_BAND_SIZE: usize = 4;

/// Keep alphanumerics + apostrophes, lowercase (upstream _normalize_name_for_fuzzy).
pub fn normalize_name_for_fuzzy(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Shannon entropy over characters of the space-stripped name.
pub fn name_entropy(normalized_name: &str) -> f64 {
    let chars: Vec<char> = normalized_name.chars().filter(|c| *c != ' ').collect();
    if chars.is_empty() {
        return 0.0;
    }
    let total = chars.len() as f64;
    let mut counts = std::collections::HashMap::new();
    for c in &chars {
        *counts.entry(*c).or_insert(0usize) += 1;
    }
    -counts
        .values()
        .map(|&n| {
            let p = n as f64 / total;
            p * p.log2()
        })
        .sum::<f64>()
}

pub fn has_high_entropy(normalized_name: &str) -> bool {
    let token_count = normalized_name.split_whitespace().count();
    if normalized_name.chars().filter(|c| *c != ' ').count() < MIN_NAME_LENGTH
        && token_count < MIN_TOKEN_COUNT
    {
        return false;
    }
    name_entropy(normalized_name) >= NAME_ENTROPY_THRESHOLD
}
```
**Fidelity check against upstream lines 79–85:** upstream gates with `if entropy < threshold AND (len < 6 OR tokens < 2)` — read `dedup_helpers.py` lines 79–85 in the clone and replicate the boolean structure exactly (the snippet above must be adjusted to whichever AND/OR composition upstream uses; the test in Step 1 plus a direct line-by-line read is the source of truth, not this plan).

```rust
/// 3-gram shingles (upstream _shingles).
pub fn shingles(normalized: &str) -> BTreeSet<String> {
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() < 3 {
        let mut s = BTreeSet::new();
        if !normalized.is_empty() {
            s.insert(normalized.to_string());
        }
        return s;
    }
    chars.windows(3).map(|w| w.iter().collect()).collect()
}
// NOTE: verify against upstream _shingles (lines 88–94) whether <3-char inputs
// produce the whole string or empty set, and replicate exactly.

/// blake2b(f"{seed}:{shingle}", digest_size=8) as big-endian u64 (upstream _hash_shingle).
pub fn hash_shingle(shingle: &str, seed: usize) -> u64 {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(format!("{seed}:{shingle}").as_bytes());
    let out = hasher.finalize();
    u64::from_be_bytes(out.into())
}

pub fn minhash_signature(shingles: &BTreeSet<String>) -> Vec<u64> {
    (0..MINHASH_PERMUTATIONS)
        .map(|seed| {
            shingles
                .iter()
                .map(|s| hash_shingle(s, seed))
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

pub fn lsh_bands(signature: &[u64]) -> Vec<Vec<u64>> {
    signature
        .chunks(MINHASH_BAND_SIZE)
        .map(|band| band.to_vec())
        .collect()
}

pub fn jaccard_similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0; // verify upstream's empty-set behavior (lines 131–140)
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}
```

- [ ] **Step 3: Run, commit**

```bash
cargo test -p chronicle-core dedup
git add -A && git commit -m "feat(core): exact port of minhash/lsh dedup helpers"
```

---

### Task 11: Temporal invalidation (exact port)

**Files:**
- Create: `crates/chronicle-core/src/pipeline/temporal.rs`
- Port from: `graphiti_core/utils/maintenance/edge_operations.py::resolve_edge_contradictions` (lines 538–573)

- [ ] **Step 1: Failing table-driven tests — every branch of the conditional**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn t(h: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, h, 0, 0).unwrap()
    }

    fn edge_with(valid: Option<u32>, invalid: Option<u32>) -> EntityEdge {
        let mut e = EntityEdge::new("a".into(), "b".into(), "R".into(), "f".into(), "g".into());
        e.valid_at = valid.map(t);
        e.invalid_at = invalid.map(t);
        e
    }

    #[test]
    fn skips_when_existing_already_invalid_before_new_valid() {
        // existing.invalid_at(2) <= new.valid_at(3) → skip
        let new_edge = edge_with(Some(3), None);
        let existing = edge_with(Some(1), Some(2));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn skips_when_new_already_invalid_before_existing_valid() {
        // new.invalid_at(2) <= existing.valid_at(3) → skip
        let new_edge = edge_with(Some(1), Some(2));
        let existing = edge_with(Some(3), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn invalidates_older_overlapping_edge() {
        // existing.valid_at(1) < new.valid_at(5), no disjoint windows → invalidate
        let new_edge = edge_with(Some(5), None);
        let existing = edge_with(Some(1), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].invalid_at, Some(t(5)));
        assert_eq!(out[0].expired_at, Some(t(12)));
    }

    #[test]
    fn preserves_preexisting_expired_at() {
        let new_edge = edge_with(Some(5), None);
        let mut existing = edge_with(Some(1), None);
        existing.expired_at = Some(t(2));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert_eq!(out[0].expired_at, Some(t(2)));
    }

    #[test]
    fn no_invalidation_without_valid_at_on_either_side() {
        let new_edge = edge_with(None, None);
        let existing = edge_with(Some(1), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }
}
```

- [ ] **Step 2: Implement — line-for-line**

```rust
// Ported from graphiti_core/utils/maintenance/edge_operations.py
// resolve_edge_contradictions (lines 538-573) @ 34f56e65 (v0.29.1)
use chrono::{DateTime, Utc};

use crate::types::EntityEdge;

/// Returns the subset of `invalidation_candidates` that the resolved edge
/// supersedes, with invalid_at/expired_at set. `now` is the transaction time.
pub fn resolve_edge_contradictions(
    resolved_edge: &EntityEdge,
    invalidation_candidates: Vec<EntityEdge>,
    now: DateTime<Utc>,
) -> Vec<EntityEdge> {
    let mut invalidated_edges = Vec::new();
    for mut edge in invalidation_candidates {
        // Skip if the windows are provably disjoint:
        // (1) existing expired before new became valid, OR
        // (2) new expired before existing became valid.
        let disjoint = matches!(
            (edge.invalid_at, resolved_edge.valid_at),
            (Some(ei), Some(rv)) if ei <= rv
        ) || matches!(
            (edge.valid_at, resolved_edge.invalid_at),
            (Some(ev), Some(ri)) if ri <= ev
        );
        if disjoint {
            continue;
        }
        // Invalidate when the existing fact started before the new fact.
        if let (Some(ev), Some(rv)) = (edge.valid_at, resolved_edge.valid_at) {
            if ev < rv {
                edge.invalid_at = resolved_edge.valid_at;
                edge.expired_at = Some(edge.expired_at.unwrap_or(now));
                invalidated_edges.push(edge);
            }
        }
    }
    invalidated_edges
}

/// Upstream post-step (edge_operations.py lines 822-823, 825-839): if the new
/// edge gained an invalid_at without expired_at, stamp expired_at = now; and if
/// an older-starting candidate begins after the new edge, close the new edge.
pub fn expire_new_edge_against_candidates(
    resolved_edge: &mut EntityEdge,
    invalidation_candidates: &[EntityEdge],
    now: DateTime<Utc>,
) {
    if resolved_edge.expired_at.is_none() {
        let mut candidates: Vec<&EntityEdge> = invalidation_candidates.iter().collect();
        candidates.sort_by_key(|e| e.valid_at);
        for candidate in candidates {
            if let (Some(cv), Some(rv)) = (candidate.valid_at, resolved_edge.valid_at) {
                if cv > rv {
                    resolved_edge.invalid_at = Some(cv);
                    resolved_edge.expired_at = Some(now);
                    break;
                }
            }
        }
    }
    if resolved_edge.invalid_at.is_some() && resolved_edge.expired_at.is_none() {
        resolved_edge.expired_at = Some(now);
    }
}
```

- [ ] **Step 3: Run, commit**

```bash
cargo test -p chronicle-core temporal
```
Expected: all 5 tests PASS.
```bash
git add -A && git commit -m "feat(core): line-for-line port of bi-temporal edge invalidation"
```

---

### Task 12: Search — RRF + config + hybrid edge search

**Files:**
- Create: `crates/chronicle-core/src/search/mod.rs`, `search/rrf.rs`, `search/config.rs`, `search/edge_search.rs`
- Port from: `graphiti_core/search/search_utils.py::rrf` (lines 1780–1795), `search_config.py`, `search.py::edge_search`

- [ ] **Step 1: Failing tests for rrf()**

```rust
#[test]
fn rrf_fuses_two_rankings() {
    let results = vec![
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec!["b".to_string(), "a".to_string()],
    ];
    let (uuids, scores) = rrf(&results, 1, 0.0);
    // a: 1/1 + 1/2 = 1.5 ; b: 1/2 + 1/1 = 1.5 ; c: 1/3
    assert_eq!(uuids.len(), 3);
    assert!(scores[0] >= scores[1] && scores[1] >= scores[2]);
    assert_eq!(uuids[2], "c");
}

#[test]
fn rrf_min_score_filters() {
    let results = vec![vec!["a".to_string(), "b".to_string()]];
    let (uuids, _) = rrf(&results, 1, 0.6);
    assert_eq!(uuids, vec!["a".to_string()]); // b scored 0.5
}
```

- [ ] **Step 2: Implement rrf (exact)**

```rust
// Ported from graphiti_core/search/search_utils.py::rrf (lines 1780-1795) @ 34f56e65
use std::collections::HashMap;

pub fn rrf(results: &[Vec<String>], rank_const: usize, min_score: f64) -> (Vec<String>, Vec<f64>) {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut first_seen: Vec<String> = Vec::new();
    for result in results {
        for (i, uuid) in result.iter().enumerate() {
            if !scores.contains_key(uuid) {
                first_seen.push(uuid.clone());
            }
            *scores.entry(uuid.clone()).or_insert(0.0) += 1.0 / (i + rank_const) as f64;
        }
    }
    // Stable sort preserving first-seen order on ties (Python dict iteration order).
    let mut scored: Vec<(String, f64)> = first_seen
        .into_iter()
        .map(|u| {
            let s = scores[&u];
            (u, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let filtered: Vec<(String, f64)> =
        scored.into_iter().filter(|(_, s)| *s >= min_score).collect();
    (
        filtered.iter().map(|(u, _)| u.clone()).collect(),
        filtered.iter().map(|(_, s)| *s).collect(),
    )
}
```

- [ ] **Step 3: Config types (full upstream shape, Phase-1 subset wired)**

`search/config.rs`:
```rust
pub const DEFAULT_SEARCH_LIMIT: usize = 10;
pub const DEFAULT_MIN_SCORE: f32 = 0.6;
pub const DEFAULT_MMR_LAMBDA: f64 = 0.5;
pub const MAX_SEARCH_DEPTH: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSearchMethod {
    CosineSimilarity,
    Bm25,
    BreadthFirstSearch, // implemented Phase 2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeReranker {
    Rrf,
    // Mmr, NodeDistance, EpisodeMentions, CrossEncoder — Phase 2
}

#[derive(Debug, Clone)]
pub struct EdgeSearchConfig {
    pub search_methods: Vec<EdgeSearchMethod>,
    pub reranker: EdgeReranker,
    pub sim_min_score: f32,
    pub mmr_lambda: f64,
    pub bfs_max_depth: usize,
}

impl Default for EdgeSearchConfig {
    fn default() -> Self {
        Self {
            search_methods: vec![EdgeSearchMethod::Bm25, EdgeSearchMethod::CosineSimilarity],
            reranker: EdgeReranker::Rrf,
            sim_min_score: DEFAULT_MIN_SCORE,
            mmr_lambda: DEFAULT_MMR_LAMBDA,
            bfs_max_depth: MAX_SEARCH_DEPTH,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub edge_config: Option<EdgeSearchConfig>,
    pub limit: usize,
    pub reranker_min_score: f64,
}
// Deliberately NO Default derive: a derived default would set limit = 0.
// Use edge_hybrid_search_rrf() as the default recipe.

/// Upstream EDGE_HYBRID_SEARCH_RRF recipe.
pub fn edge_hybrid_search_rrf() -> SearchConfig {
    SearchConfig {
        edge_config: Some(EdgeSearchConfig::default()),
        limit: DEFAULT_SEARCH_LIMIT,
        reranker_min_score: 0.0,
    }
}
```

- [ ] **Step 4: Hybrid edge search orchestration**

`search/edge_search.rs`:
```rust
use std::collections::HashMap;

use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::errors::ChronicleError;
use crate::types::EntityEdge;

use super::config::{EdgeSearchMethod, SearchConfig};
use super::rrf::rrf;

/// Upstream: each method fetches 2*limit candidates, RRF fuses, slice to limit.
pub async fn edge_search(
    driver: &dyn GraphDriver,
    embedder: &dyn EmbedderClient,
    query: &str,
    group_ids: &[String],
    config: &SearchConfig,
) -> Result<Vec<EntityEdge>, ChronicleError> {
    let Some(edge_config) = &config.edge_config else {
        return Ok(Vec::new());
    };
    let candidate_limit = 2 * config.limit;
    let mut result_lists: Vec<Vec<EntityEdge>> = Vec::new();

    if edge_config.search_methods.contains(&EdgeSearchMethod::Bm25) {
        result_lists.push(
            driver.edge_fulltext_search(query, group_ids, candidate_limit).await?,
        );
    }
    if edge_config.search_methods.contains(&EdgeSearchMethod::CosineSimilarity) {
        let vector = embedder.create(query).await?;
        result_lists.push(
            driver
                .edge_similarity_search(
                    &vector,
                    group_ids,
                    candidate_limit,
                    edge_config.sim_min_score,
                )
                .await?,
        );
    }

    let edge_uuid_map: HashMap<String, EntityEdge> = result_lists
        .iter()
        .flatten()
        .map(|e| (e.uuid.clone(), e.clone()))
        .collect();
    let uuid_lists: Vec<Vec<String>> = result_lists
        .iter()
        .map(|l| l.iter().map(|e| e.uuid.clone()).collect())
        .collect();
    let (ranked, _scores) = rrf(&uuid_lists, 1, config.reranker_min_score);

    Ok(ranked
        .into_iter()
        .filter_map(|u| edge_uuid_map.get(&u).cloned())
        .take(config.limit)
        .collect())
}
```

Test (in `edge_search.rs`, using chronicle-testkit as dev-dependency of chronicle-core): seed FakeDriver with 3 edges (one matching fulltext only, one matching embedding only, one both), assert the both-matcher ranks first and limit applies.

Add to `crates/chronicle-core/Cargo.toml`:
```toml
[dev-dependencies]
chronicle-testkit.workspace = true
```

- [ ] **Step 5: Run, commit**

```bash
cargo test -p chronicle-core search
git add -A && git commit -m "feat(core): RRF fusion and hybrid edge search"
```

---

### Task 13: Pipeline — node extraction & resolution

**Files:**
- Create: `crates/chronicle-core/src/pipeline/clients.rs`, `pipeline/node_ops.rs`
- Port from: `graphiti_core/utils/maintenance/node_operations.py` (`extract_nodes`, `resolve_extracted_nodes`, `_resolve_with_similarity`, `_resolve_with_llm`, `_promote_resolved_node`)

Upstream constants to carry over:
```rust
pub const MAX_NODES: usize = 30;
pub const NODE_DEDUP_CANDIDATE_LIMIT: usize = 15;
pub const NODE_DEDUP_COSINE_MIN_SCORE: f32 = 0.6;
```

- [ ] **Step 1: Clients bundle**

`pipeline/clients.rs`:
```rust
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::llm::LlmClient;

#[derive(Clone)]
pub struct Clients {
    pub driver: Arc<dyn GraphDriver>,
    pub llm: Arc<dyn LlmClient>,
    pub embedder: Arc<dyn EmbedderClient>,
    pub semaphore: Arc<Semaphore>,
}

impl Clients {
    pub fn new(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
    ) -> Self {
        Self {
            driver,
            llm,
            embedder,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }
}
```

- [ ] **Step 2: Failing test — extraction + dedup flow with scripted LLM**

In `pipeline/node_ops.rs` tests (uses MockLlm/MockEmbedder/FakeDriver):
```rust
#[tokio::test]
async fn resolve_reuses_exact_name_match_without_llm() {
    // Seed FakeDriver with existing node "Alice Smith".
    // Extract candidate also named "alice  SMITH" (different case/space).
    // Expect: resolved to existing uuid, no LLM dedupe call recorded.
}

#[tokio::test]
async fn resolve_escalates_ambiguous_to_llm() {
    // No exact match, low-entropy name → LLM consulted;
    // MockLlm scripted with NodeResolutions{duplicate_candidate_id: -1} → new node kept.
}
```
(Write these fully — seed data, scripted responses, assertions on `mock.requests` length.)

- [ ] **Step 3: Implement**

`pipeline/node_ops.rs` core functions (full bodies; signatures fixed here):
```rust
pub struct ExtractedNodeData {
    pub nodes: Vec<EntityNode>,
}

/// extract_nodes: prompt per episode source type, reflexion skipped in Phase 1
/// (upstream MAX_REFLEXION_ITERATIONS defaults to 0 — confirm in clone and mirror).
pub async fn extract_nodes(
    clients: &Clients,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
    entity_types: Option<&serde_json::Value>,
    custom_extraction_instructions: Option<&str>,
) -> Result<Vec<EntityNode>, ChronicleError>;

pub struct NodeResolutionOutcome {
    pub nodes: Vec<EntityNode>,
    /// extracted uuid -> canonical uuid
    pub uuid_map: std::collections::HashMap<String, String>,
    /// (extracted, canonical) pairs that were duplicates
    pub duplicates: Vec<(EntityNode, EntityNode)>,
}

pub async fn resolve_extracted_nodes(
    clients: &Clients,
    extracted_nodes: Vec<EntityNode>,
    episode: &EpisodicNode,
    previous_episodes: &[EpisodicNode],
) -> Result<NodeResolutionOutcome, ChronicleError>;
```

`resolve_extracted_nodes` implementation order (mirror upstream lines 627–708):
1. Embed each extracted name (`create_batch`), run `node_similarity_search` per node (limit 15, min_score 0.6) — gather candidates.
2. Deterministic pass: exact-normalized-name match (single hit → resolve; multi-hit → mark unresolved); entropy-gated MinHash/LSH fuzzy pass at jaccard ≥ 0.9 using Task 10 helpers.
3. Unresolved → ONE batched LLM call with `dedupe_nodes::nodes` prompt; parse `NodeResolutions`; `duplicate_candidate_id == -1` → keep extracted; else resolve to candidate and merge labels (`_promote_resolved_node`: union of labels, "Entity" always present).
4. Build `uuid_map` + `duplicates`.

- [ ] **Step 4: Run, commit**

```bash
cargo test -p chronicle-core node_ops
git add -A && git commit -m "feat(core): node extraction and three-stage dedup resolution"
```

---

### Task 14: Pipeline — edge ops, add_episode orchestration, Chronicle facade

**Files:**
- Create: `pipeline/edge_ops.rs`, `pipeline/add_episode.rs`, `crates/chronicle-core/src/chronicle.rs`
- Port from: `graphiti_core/utils/maintenance/edge_operations.py` (`extract_edges`, `resolve_extracted_edges`, `resolve_extracted_edge`), `graphiti_core/graphiti.py::add_episode` (lines 1067–1223)

- [ ] **Step 1: Failing end-to-end test (the Phase-1 acceptance test)**

`crates/chronicle-core/tests/add_episode_e2e.rs`:
```rust
//! End-to-end: two episodes about Alice's employer; the second contradicts the
//! first; expect the old fact invalidated (bi-temporal) and search to find the new.
use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::search::config::edge_hybrid_search_rrf;
use chronicle_core::types::EpisodeType;
use chronicle_testkit::{FakeDriver, MockEmbedder, MockLlm};

#[tokio::test]
async fn add_episode_invalidates_contradicted_fact_and_search_finds_new() {
    let driver = Arc::new(FakeDriver::default());
    // Script the exact LLM call sequence for episode 1 then episode 2:
    // ep1: extract_nodes → resolve(dedupe escalation may be skipped if exact-match)
    //      → extract_edges → resolve_edge → timestamps → summarize
    // ep2: same sequence, resolve_edge returns contradicted_facts=[idx of ep1 edge]
    let llm = Arc::new(MockLlm::new(vec![ /* scripted JSON values, written out fully here */ ]));
    let embedder = Arc::new(MockEmbedder { dim: 16 });
    let chronicle = Chronicle::new(driver.clone(), llm, embedder, 4);

    // AddEpisodeRequest has no Default (reference_time/group_id are mandatory);
    // use a small local helper to build both requests:
    fn req(name: &str, body: &str, at: chrono::DateTime<chrono::Utc>) -> AddEpisodeRequest {
        AddEpisodeRequest {
            name: name.into(),
            episode_body: body.into(),
            source: EpisodeType::Text,
            source_description: "test".into(),
            reference_time: at,
            group_id: "g1".into(),
            uuid: None,
            previous_episode_uuids: None,
            entity_types: None,
            custom_extraction_instructions: None,
        }
    }
    let t0 = chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let t1 = chrono::Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();

    chronicle.add_episode(req("ep1", "Alice works at Acme.", t0)).await.unwrap();
    let res2 = chronicle.add_episode(req("ep2", "Alice now works at Globex.", t1)).await.unwrap();

    // The Acme edge must be invalidated with invalid_at = t1.
    let invalidated: Vec<_> = res2.edges.iter().filter(|e| e.invalid_at.is_some()).collect();
    assert_eq!(invalidated.len(), 1);

    let hits = chronicle.search("where does Alice work", &["g1".into()],
        &edge_hybrid_search_rrf()).await.unwrap();
    assert!(hits.iter().any(|e| e.fact.contains("Globex")));
}
```
When implementing, write out the scripted MockLlm JSON values explicitly (ExtractedEntities, NodeResolutions, ExtractedEdges, EdgeDuplicate, EdgeTimestamps, Summary shapes from Task 6).

- [ ] **Step 2: Implement edge ops**

`pipeline/edge_ops.rs` signatures (bodies mirror upstream lines 623–847):
```rust
pub async fn extract_edges(
    clients: &Clients,
    episode: &EpisodicNode,
    nodes: &[EntityNode],
    previous_episodes: &[EpisodicNode],
    group_id: &str,
    custom_extraction_instructions: Option<&str>,
) -> Result<Vec<EntityEdge>, ChronicleError>;

pub struct EdgeResolutionOutcome {
    pub resolved_edges: Vec<EntityEdge>,
    pub invalidated_edges: Vec<EntityEdge>,
}

pub async fn resolve_extracted_edges(
    clients: &Clients,
    extracted_edges: Vec<EntityEdge>,
    episode: &EpisodicNode,
    nodes: &[EntityNode],
) -> Result<EdgeResolutionOutcome, ChronicleError>;
```
`resolve_extracted_edge` per-edge flow (parallel via semaphore, mirror upstream — AMENDED after verifying upstream v0.29.1 edge_operations.py lines 355-425):
1. Candidate retrieval (in `resolve_extracted_edges`, before the per-edge loop):
   a. Embed extracted edge facts FIRST (`create_entity_edge_embeddings` analog — cosine search needs fact embeddings).
   b. `valid_edges` per extracted edge = `driver.get_edges_between_nodes(source, target)` (node-pair pool).
   c. `related_edges` per edge = upstream re-ranks the pair pool via `EDGE_HYBRID_SEARCH_RRF` + `SearchFilters(edge_uuids=pair pool)`. Phase-1 approximation: use the pair pool directly, capped at RELEVANT_SCHEMA_LIMIT — candidate ORDERING may differ from upstream (the LLM sees the same candidate set; ordering deviation goes in the fidelity ledger).
   d. `existing_edges` (invalidation candidates) per edge = `edge_search(driver, embedder, &edge.fact, &[group_id], &edge_hybrid_search_rrf())` (Task 12 fn — faithful to upstream), minus any uuid already in `related_edges`.
2. Fast path both empty → extract timestamps (`extract_edges::extract_timestamps`, small model) → return as-is.
3. Fast path exact fact match (normalized) → append episode uuid, return existing edge.
4. Else LLM `dedupe_edges::resolve_edge` → `EdgeDuplicate`; first valid duplicate idx wins; map `contradicted_facts` indices to candidates (related first, then existing, continuous indexing).
5. New edges get timestamps; then `expire_new_edge_against_candidates` + `resolve_edge_contradictions` (Task 11).

- [ ] **Step 3: Implement add_episode orchestration + facade**

`pipeline/add_episode.rs` — follow the Task-list order from upstream (retrieve context → extract nodes → resolve nodes → extract edges → remap pointers via uuid_map → resolve edges → summarize/attributes via `summarize_nodes::summarize_context` per node → build MENTIONS episodic edges → embed (names + facts via `create_batch`) → persist all through driver). Persist order: episode, nodes, entity edges, episodic edges.

`src/chronicle.rs`:
```rust
pub struct AddEpisodeRequest {
    pub name: String,
    pub episode_body: String,
    pub source: EpisodeType,
    pub source_description: String,
    pub reference_time: chrono::DateTime<chrono::Utc>,
    pub group_id: String,
    pub uuid: Option<String>,
    pub previous_episode_uuids: Option<Vec<String>>,
    pub entity_types: Option<serde_json::Value>,
    pub custom_extraction_instructions: Option<String>,
}

pub struct AddEpisodeResults {
    pub episode: EpisodicNode,
    pub episodic_edges: Vec<EpisodicEdge>,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
}

pub struct Chronicle { clients: Clients }

impl Chronicle {
    pub fn new(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
    ) -> Self;

    pub async fn add_episode(&self, req: AddEpisodeRequest)
        -> Result<AddEpisodeResults, ChronicleError>;

    pub async fn retrieve_episodes(
        &self,
        reference_time: chrono::DateTime<chrono::Utc>,
        last_n: usize,
        group_ids: &[String],
    ) -> Result<Vec<EpisodicNode>, ChronicleError>;

    pub async fn search(
        &self,
        query: &str,
        group_ids: &[String],
        config: &SearchConfig,
    ) -> Result<Vec<EntityEdge>, ChronicleError>;

    pub async fn build_indices_and_constraints(&self, delete_existing: bool)
        -> Result<(), ChronicleError>;
}
```

- [ ] **Step 4: Run the e2e + full suite, commit**

```bash
cargo test -p chronicle-core
```
Expected: e2e passes; full suite green.
```bash
git add -A && git commit -m "feat(core): add_episode pipeline and Chronicle facade"
```

---

### Task 15: chronicle-driver-neo4j

**Files:**
- Create: `crates/chronicle-driver-neo4j/Cargo.toml`, `src/lib.rs`, `src/queries.rs`, `src/convert.rs`, `tests/neo4j_integration.rs`
- Port from: `graphiti_core/driver/neo4j/**`, `graphiti_core/models/{nodes,edges}/*_db_queries.py`, `graphiti_core/graph_queries.py`

- [ ] **Step 1: Manifest** (`neo4rs`, `chronicle-core`, tokio; dev-deps: chronicle-testkit not needed). Add to workspace members.

- [ ] **Step 2: Implement `Neo4jDriver`**

`src/lib.rs`: `pub struct Neo4jDriver { graph: neo4rs::Graph, database: String }` with `connect(uri, user, password, database) -> Result<Self, DriverError>`. Implement all six trait families. Key Cypher (verbatim targets in `queries.rs`):

```rust
pub const SAVE_ENTITY_NODES: &str = r#"
UNWIND $nodes AS node
MERGE (n:Entity {uuid: node.uuid})
SET n = node.props
WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
RETURN n.uuid AS uuid
"#;

pub const SAVE_ENTITY_EDGES: &str = r#"
UNWIND $entity_edges AS edge
MATCH (source:Entity {uuid: edge.source_node_uuid})
MATCH (target:Entity {uuid: edge.target_node_uuid})
MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
SET e = edge.props
WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)
RETURN edge.uuid AS uuid
"#;

pub const EDGE_SIMILARITY_SEARCH: &str = r#"
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
WHERE e.group_id IN $group_ids
WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $search_vector) AS score
WHERE score > $min_score
RETURN e.uuid AS uuid, n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
       e.group_id AS group_id, e.created_at AS created_at, e.name AS name,
       e.fact AS fact, e.episodes AS episodes, e.expired_at AS expired_at,
       e.valid_at AS valid_at, e.invalid_at AS invalid_at, properties(e) AS attributes
ORDER BY score DESC
LIMIT $limit
"#;

pub const EDGE_FULLTEXT_SEARCH: &str = r#"
CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $query, {limit: $limit})
YIELD relationship AS rel, score
MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
RETURN e.uuid AS uuid, n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
       e.group_id AS group_id, e.created_at AS created_at, e.name AS name,
       e.fact AS fact, e.episodes AS episodes, e.expired_at AS expired_at,
       e.valid_at AS valid_at, e.invalid_at AS invalid_at, properties(e) AS attributes
ORDER BY score DESC
LIMIT $limit
"#;
```
Plus episode save/get/retrieve, MENTIONS save, node fulltext (`node_name_and_summary` index) and node similarity (`vector.similarity.cosine(n.name_embedding, …)`), and the `get_edges_between_nodes` MATCH query (single direction `(n)-[e:RELATES_TO]->(m)`; `get_edges_touching_nodes` was removed from the trait in the Task 8 amendment) — copy the SELECT column lists from `graphiti_core/models/edges/edge_db_queries.py` exactly.

**Fulltext query sanitation** (upstream `build_fulltext_query`): escape Lucene specials `+ - & | ! ( ) { } [ ] ^ " ~ * ? : \ /`, prefix `group_id:"<gid>" AND (…)`; cap at 128 tokens → return empty results instead of erroring.

**Index DDL** (`build_indices_and_constraints`): port the full list from `graphiti_core/graph_queries.py` — Phase 1 needs at minimum: uuid + group_id range indices for Entity/Episodic/RELATES_TO/MENTIONS, `name_entity_index`, the two fulltext indices `node_name_and_summary` + `edge_name_and_fact`, and `valid_at`/`expired_at`/`invalid_at` edge indices.

`src/convert.rs`: record ↔ struct mapping. Strip core fields out of `properties(...)` maps into typed fields, keep the rest as `attributes` (mirror upstream record parsers; strip the `Entity_<group>` dynamic label on read). DateTime: neo4rs `DateTime` ↔ `chrono::DateTime<Utc>`.

- [ ] **Step 3: Integration tests (env-gated)**

`tests/neo4j_integration.rs`:
```rust
//! Requires a running Neo4j 5.x: NEO4J_TEST_URI=bolt://localhost:7687
//! NEO4J_TEST_USER/NEO4J_TEST_PASSWORD. Tests no-op (skip) when unset.
fn test_driver() -> Option<...> {
    let uri = std::env::var("NEO4J_TEST_URI").ok()?;
    ...
}

#[tokio::test]
async fn roundtrip_entity_node_and_edge() { /* save → get → field equality incl. temporal fields */ }

#[tokio::test]
async fn fulltext_and_similarity_search_return_saved_edge() { ... }

#[tokio::test]
async fn build_indices_is_idempotent() { /* run twice, no error */ }
```
Skip pattern: `let Some(d) = test_driver().await else { eprintln!("skipping: NEO4J_TEST_URI unset"); return; };`

- [ ] **Step 4: Run (unit compile + gated), commit**

```bash
cargo test -p chronicle-driver-neo4j            # skips integration without env
# optionally: docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
# NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_USER=neo4j NEO4J_TEST_PASSWORD=testpassword \
#   cargo test -p chronicle-driver-neo4j
git add -A && git commit -m "feat(neo4j): GraphDriver implementation with fulltext and vector search"
```

---

### Task 16: chronicle-llm-openai

**Files:**
- Create: `crates/chronicle-llm-openai/Cargo.toml`, `src/lib.rs`, `src/llm.rs`, `src/embedder.rs`
- Port from: `graphiti_core/llm_client/openai_generic_client.py`, `graphiti_core/embedder/openai.py`

- [ ] **Step 1: Manifest** (async-openai, chronicle-core, tokio, async-trait, serde_json, tracing). Add to workspace members.

- [ ] **Step 2: Implement LlmClient**

Upstream defaults: `DEFAULT_MODEL = "gpt-4.1-mini"`, `DEFAULT_SMALL_MODEL = "gpt-4.1-nano"`, temperature 0, max_tokens 16384. Use chat completions with `response_format = json_schema { name, schema, strict: true }` (the openai_generic_client path — works on OpenAI, Azure, Ollama, Groq, OpenRouter via `base_url`):

```rust
pub struct OpenAiLlm {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    config: chronicle_core::llm::LlmConfig,
}

#[async_trait::async_trait]
impl chronicle_core::llm::LlmClient for OpenAiLlm {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        chronicle_core::llm::with_retry(|| async {
            // build CreateChatCompletionRequest:
            //  - model: small_model if ModelSize::Small else model
            //  - messages: map Role::{System,User,Assistant}
            //  - response_format: json_schema when request.response_schema.is_some()
            //  - max_tokens / temperature from config
            // map errors: 429 → LlmError::RateLimit, 5xx → Server, others → Transport
            // parse choices[0].message.content as serde_json::Value
        })
        .await
    }
}
```
Map empty content → `LlmError::EmptyResponse`, refusal field → `LlmError::Refusal`.

- [ ] **Step 3: Implement Embedder**

`OpenAiEmbedder`: default model `text-embedding-3-small`, truncate every vector to `config.embedding_dim` (upstream truncates). `create_batch` = one embeddings call with all inputs.

- [ ] **Step 4: Tests**

Unit: request-mapping test via a `Fn`-injected transport is overkill — instead test the pure mapping helpers (role conversion, response_format construction, error mapping from status codes) as free functions. Live test `#[ignore]`d behind `OPENAI_API_KEY`:
```rust
#[tokio::test]
#[ignore = "live API; needs OPENAI_API_KEY"]
async fn live_structured_output_roundtrip() { ... }
```

- [ ] **Step 5: Run, commit**

```bash
cargo test -p chronicle-llm-openai
git add -A && git commit -m "feat(llm-openai): OpenAI-compatible LlmClient and embedder"
```

---

### Task 17: CI green, fidelity ledger, docs close-out

**Files:**
- Create: `docs/port-fidelity.md`
- Modify: `README.md`, `Cargo.toml` (confirm all 4 members restored)

- [ ] **Step 1: Fidelity ledger**

`docs/port-fidelity.md`: table of every ported module — columns: chronicle path | upstream path | upstream lines | status (verbatim / adapted / deferred) | notes (e.g. "Edge renamed ExtractedEdge"). One row per file created in Tasks 6–16. List Phase-2+ deferrals explicitly (reflexion loop, BFS search, MMR/cross-encoder, communities, saga, bulk, FalkorDB/Kuzu, SearchFilters wiring).

- [ ] **Step 2: Full local check**

```bash
cargo update --verbose
bash ci/local_check.sh
```
Expected: fmt clean, clippy clean (`-D warnings`), all workspace tests pass. Fix anything that fails before proceeding.

- [ ] **Step 3: README — usage example**

Add a minimal end-to-end snippet (Chronicle::new with Neo4jDriver + OpenAiLlm, add_episode, search) and the Phase table from the spec with Phase 1 marked done.

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "docs: port-fidelity ledger and usage docs; phase 0+1 complete"
```

---

## Execution notes

- Work on a `feat/phase-0-1-core-loop` branch off `research`; PR targets `research`.
- The upstream clone at `/home/bima-pangestu/Works/refs/graphiti` must stay checked out at v0.29.1 for the whole implementation — verify with `git -C /home/bima-pangestu/Works/refs/graphiti rev-parse HEAD` → `34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6`.
- Where this plan says "verbatim" and shows abbreviated text, the upstream file is the source of truth — copy from the clone, not from this plan.
- Where this plan's pseudocode and the upstream source disagree on logic details (boolean composition, edge-case behavior), **upstream wins**; note the discrepancy in `docs/port-fidelity.md`.
