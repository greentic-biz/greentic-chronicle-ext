# Phase 5: greentic-dw-providers Long-Term Memory Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** New `memory/long-term/` provider family in greentic-dw-providers whose `chronicle` implementation gives digital workers bi-temporal knowledge-graph memory via the chronicle crates.

**Architecture:** Family-level integration (mirrors `memory/short-term/{core,in-memory,redis}`): a new async `LongTermMemory` trait in a `core` crate + a `chronicle` impl crate wrapping the `Chronicle` facade, consumed as **git dependencies pinned to greentic-chronicle-ext tag v0.1.0 (rev `3faf8903bd2a1d3d9b3c71a31a3c2ed771f13642`)**. Capability helpers land in `greentic-dw-providers-common`; **NO ProviderCatalog wiring** (deferred until `feat/unified-catalog` lands — avoids conflicting with in-flight work). NO runtime KV adapter in this PR.

**Tech stack:** Rust 1.94 (this repo's pin — note: chronicle pins 1.95; git-dep consumers compile chronicle source with the consumer's toolchain, so verify chronicle builds on 1.94 in Task 2; if not, bump dw-providers toolchain is NOT ours to do — escalate), tokio, async-trait.

**Locked decisions:**
| Decision | Choice |
|---|---|
| Dependency mechanism | Cargo git deps on private `greentic-biz/greentic-chronicle-ext`, pinned `tag = "v0.1.0"`; CI auth via token (handoff note to devops for org secret) |
| Scope | Family crates + common helpers ONLY. Catalog wiring + runtime KV adapter = follow-up PRs |
| Crate names | core: `greentic-dw-memory-long-term` (folder `memory/long-term/core`), impl: `greentic-dw-memory-chronicle` (folder `memory/long-term/chronicle`) |
| Capability ids | uri `cap://dw.memory.long-term`, pack id `greentic.cap.memory.long-term`, provider_type `dw.memory.long-term.chronicle`, component_ref `component:memory.chronicle` |
| Tenancy | `TenantCtx` → chronicle `group_id` (validate against `^[a-zA-Z0-9_-]+$` — chronicle's neo4j driver enforces it) |
| Branch | git worktree off `origin/research` → `feat/memory-long-term-chronicle` → PR targets `research` |

**House rules (greentic-dw-providers):** follow its `CLAUDE.md` + `.codex/global_rules.md` — PRE-PR refresh of `.codex/repo_overview.md`, `bash ci/local_check.sh` before done, POST-PR sync, `#![forbid(unsafe_code)]`, no unwrap/panic in prod, Conventional Commits, NO attribution trailers, Cargo.lock committed.

---

### Task 1: Worktree + workspace wiring

- [ ] **Step 1:** `git -C /home/bima-pangestu/Works/greentic/greentic-dw-providers worktree add /tmp/dwp-longterm -b feat/memory-long-term-chronicle origin/research` (NEVER touch the user's `feat/unified-catalog` checkout). All subsequent work happens in `/tmp/dwp-longterm`.
- [ ] **Step 2:** Read `/tmp/dwp-longterm/CLAUDE.md`, `.codex/global_rules.md`, root `Cargo.toml`, and `memory/short-term/core/src/lib.rs` IN FULL — the short-term core is the style template (imports, `GResult`/error types, `TenantCtx` source crate, provider_decl/pack_manifest re-export pattern).
- [ ] **Step 3:** Root `Cargo.toml`: add members `"memory/long-term/core"`, `"memory/long-term/chronicle"`; add to `[workspace.dependencies]`:
```toml
greentic-dw-memory-long-term = { path = "memory/long-term/core" }
chronicle-core = { git = "ssh://git@github.com/greentic-biz/greentic-chronicle-ext.git", tag = "v0.1.0" }
chronicle-driver-neo4j = { git = "ssh://git@github.com/greentic-biz/greentic-chronicle-ext.git", tag = "v0.1.0" }
chronicle-llm-openai = { git = "ssh://git@github.com/greentic-biz/greentic-chronicle-ext.git", tag = "v0.1.0" }
chronicle-testkit = { git = "ssh://git@github.com/greentic-biz/greentic-chronicle-ext.git", tag = "v0.1.0" }
```
(If local SSH fetch of the private repo fails, try `https://github.com/...` + `gh auth token` via `CARGO_NET_GIT_FETCH_WITH_CLI=true`; record which form works — CI will need the same.)
- [ ] **Step 4:** `cargo metadata` resolves; chronicle compiles under THIS repo's toolchain (1.94). If chronicle's edition-2024/rust-version gate fails on 1.94: STOP, report BLOCKED (toolchain decision is repo-owner scope).
- [ ] **Step 5:** Commit `chore: add long-term memory family scaffolding and chronicle git deps`.

### Task 2: `memory/long-term/core` — the LongTermMemory contract

**Files:** `memory/long-term/core/{Cargo.toml,src/lib.rs}` (crate `greentic-dw-memory-long-term`).

- [ ] **Step 1 (failing test):** trait object-safety + DTO serde roundtrip tests (mirror short-term core's test style).
- [ ] **Step 2:** Implement, mirroring short-term core conventions (same error-handling crate, same TenantCtx import):
```rust
#[async_trait::async_trait]
pub trait LongTermMemory: Send + Sync {
    /// Ingest one episode (conversation turn, document, event) into long-term memory.
    async fn ingest_episode(&self, tenant: &TenantCtx, episode: EpisodeIngest)
        -> Result<IngestOutcome, LongTermMemoryError>;
    /// Semantic recall: natural-language query over remembered facts.
    async fn recall(&self, tenant: &TenantCtx, query: RecallQuery)
        -> Result<Vec<RecalledFact>, LongTermMemoryError>;
}

pub struct EpisodeIngest {
    pub name: String,
    pub body: String,
    pub source: EpisodeSource,          // Message | Text | Json
    pub source_description: String,
    pub reference_time: chrono::DateTime<chrono::Utc>,
}
pub struct IngestOutcome { pub episode_id: String, pub fact_count: usize, pub entity_count: usize }
pub struct RecallQuery { pub query: String, pub limit: Option<usize> }
pub struct RecalledFact {
    pub fact: String,
    pub relation: String,
    pub valid_at: Option<chrono::DateTime<chrono::Utc>>,
    pub invalid_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source_episode_ids: Vec<String>,
}
pub enum EpisodeSource { Message, Text, Json }
```
`LongTermMemoryError` thiserror enum: `Backend(String)`, `InvalidTenant(String)`, `NotConfigured(String)`. Keep the trait backend-agnostic — NO chronicle types leak here.
- [ ] **Step 3:** Tests pass; commit `feat(memory): long-term memory family contract`.

### Task 3: Capability helpers in `greentic-dw-providers-common`

**Files:** Modify `crates/greentic-dw-providers-common/src/memory.rs` ONLY (do NOT touch `catalog.rs` — conflicts with feat/unified-catalog).

- [ ] Mirror the short-term helper set 1:1 with long-term equivalents: `LongTermMemoryVariant` enum (single variant `Chronicle`), `provider_type()` → `dw.memory.long-term.chronicle`, `component_ref()` → `component:memory.chronicle`, `long_term_memory_capability_uri()` → `cap://dw.memory.long-term`, `long_term_memory_pack_capability_id()` → `greentic.cap.memory.long-term`, plus `provider_decl`/`pack_manifest` fixture helpers if short-term has them (read and mirror exactly). Unit tests asserting each string (mirror short-term's tests). Add a `// NOTE: catalog wiring intentionally deferred until unified-catalog lands` comment.
- [ ] Commit `feat(common): long-term memory capability helpers`.

### Task 4: `memory/long-term/chronicle` — the provider

**Files:** `memory/long-term/chronicle/{Cargo.toml,src/lib.rs,src/config.rs,src/bridge.rs}` (crate `greentic-dw-memory-chronicle`). Deps: greentic-dw-memory-long-term (path), chronicle-core/chronicle-driver-neo4j/chronicle-llm-openai (workspace git), greentic-dw-llm (path, for the bridge), tokio, async-trait, serde, thiserror, tracing. Dev-deps: chronicle-testkit.

- [ ] **Step 1 — config.rs:**
```rust
pub struct ChronicleMemoryConfig {
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,       // resolved by caller from secrets — never logged
    pub neo4j_database: String,       // default "neo4j"
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub model: Option<String>,
    pub small_model: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dim: Option<usize>,
    pub max_concurrency: usize,       // default chronicle SEMAPHORE_LIMIT analog (20)
    pub recall_limit: usize,          // default 10
}
```
Custom `Debug` impl redacting password/api_key.
- [ ] **Step 2 — lib.rs:** `pub struct ChronicleLongTermMemory { chronicle: chronicle_core::chronicle::Chronicle, recall_limit: usize }` with `pub async fn connect(config: ChronicleMemoryConfig) -> Result<Self, LongTermMemoryError>` (Neo4jDriver::connect + OpenAiLlm + OpenAiEmbedder from config, `Chronicle::new`, then `build_indices_and_constraints(false)`). Optional alternate constructor `from_parts(driver, llm, embedder, ...)` for tests (inject chronicle-testkit fakes).
- [ ] **Step 3 — impl LongTermMemory:**
  - `ingest_episode`: validate tenant id against `^[a-zA-Z0-9_-]+$` → `InvalidTenant`; map to `AddEpisodeRequest { group_id: tenant_id, source: map EpisodeSource→EpisodeType, reference_time, ... uuid/previous/entity_types/custom: None }`; call `chronicle.add_episode`; map result → `IngestOutcome { episode_id, fact_count: edges.len(), entity_count: nodes.len() }`. Errors → `Backend(err.to_string())` with `tracing::error!`.
  - `recall`: `chronicle.search(&query.query, &[tenant_id], &edge_hybrid_search_rrf())` with limit override (`SearchConfig { limit: query.limit.unwrap_or(self.recall_limit), ..recipe }`); map `EntityEdge` → `RecalledFact { fact, relation: name, valid_at, invalid_at, source_episode_ids: episodes }`.
- [ ] **Step 4 — bridge.rs (the day-1 LLM bridge):** `pub struct DwLlmBridge { provider: Arc<dyn greentic_dw_llm::LlmProvider>, tenant: TenantCtx }` implementing `chronicle_core::llm::LlmClient`:
  - READ `llm/core/src/lib.rs` first for exact `LlmRequest`/`LlmResponse`/message shapes.
  - Map chronicle `LlmRequest{messages, response_schema, max_tokens, model_size}` → dw `LlmRequest` (messages mapped; structured output: if `provider.features().structured_outputs` pass the JSON schema through the dw request's structured-output field — read how dw expresses it; if the provider lacks structured_outputs → return `LlmError::Transport("provider lacks structured outputs")`).
  - dw `generate` is SYNC → call via `tokio::task::spawn_blocking` (clone what's needed; map JoinError → Transport).
  - Parse dw response content as `serde_json::Value` (mirror chronicle-llm-openai: empty → EmptyResponse, malformed → InvalidJson).
  - NOTE in doc: embeddings have NO dw-providers abstraction yet, so embedder stays chronicle-llm-openai (OpenAI-compatible endpoints incl. self-hosted via base_url); revisit when dw grows an embeddings family.
  - Constructor on the provider: `ChronicleLongTermMemory::connect_with_dw_llm(config, provider: Arc<dyn LlmProvider>, tenant: TenantCtx)` wiring the bridge in place of OpenAiLlm.
- [ ] **Step 5 — tests:** (a) trait-level test with chronicle-testkit fakes via `from_parts`: ingest a scripted episode (MockLlm sequence mirroring chronicle's e2e) then recall finds the fact — THE acceptance test of this phase; (b) tenant validation rejects `bad tenant!`; (c) bridge unit test: scripted dw LlmProvider stub returns structured JSON, bridge yields parsed Value; provider without structured_outputs feature → error; (d) config Debug redaction test.
- [ ] **Step 6:** Commit `feat(memory): chronicle long-term memory provider with dw-llm bridge`.

### Task 5: CI auth for the private git dep

- [ ] Inspect `.github/workflows/*.yml`: every job running cargo needs access to the private repo. Add the minimal standard mechanism: a step before cargo commands —
```yaml
- name: Configure git auth for private chronicle dep
  run: git config --global url."https://x-access-token:${{ secrets.CHRONICLE_REPO_TOKEN }}@github.com/greentic-biz/".insteadOf "ssh://git@github.com/greentic-biz/"
```
plus `CARGO_NET_GIT_FETCH_WITH_CLI: "true"` env where needed. Adjust the insteadOf mapping to whatever URL form Task 1 recorded as working.
- [ ] **DO NOT create the org/repo secret** (devops-owned): add `docs/chronicle-dep.md` (or extend existing CI docs) describing: required secret `CHRONICLE_REPO_TOKEN` (fine-grained PAT or GitHub App token, read-only on greentic-biz/greentic-chronicle-ext), which workflows need it, and the local-dev requirement (SSH access to the repo). This is the explicit handoff note.
- [ ] Commit `ci: git auth wiring for private chronicle dependency`.

### Task 6: Local CI green + .codex sync + PR

- [ ] `bash ci/local_check.sh` from the worktree — ALL steps green (fmt, clippy -D warnings, lib+tests, provider_composition + provider_golden load-bearing tests, build --all-features, doc, gtpacks validate). The new family must not break provider_composition/provider_golden — if those tests enumerate families/providers, extend their fixtures per the established pattern (read the failures and mirror short-term's entries).
- [ ] PRE-PR sync: refresh `.codex/repo_overview.md` per `.codex/global_rules.md` (add the new family to the overview).
- [ ] Push branch, open PR → `research` titled `feat: memory/long-term family with chronicle provider`. Body: what/why, chronicle pin (v0.1.0 @ 3faf890), capability ids, deferred items (catalog wiring post-unified-catalog; runtime KV adapter; embeddings abstraction), CI secret handoff (CHRONICLE_REPO_TOKEN needed before merge CI can pass — note that CI WILL FAIL on the git fetch until devops adds the secret; state this explicitly in the PR body so red CI is understood). NO attribution trailers.
- [ ] POST-PR sync per .codex rules. Remove the worktree (`git worktree remove /tmp/dwp-longterm`) ONLY after PR is pushed.

---

## Risks
| Risk | Mitigation |
|---|---|
| chronicle (edition 2024, rust-version 1.95) fails on dw-providers' 1.94 toolchain | Task 1 Step 4 gate; BLOCKED → escalate toolchain decision |
| CI red until CHRONICLE_REPO_TOKEN secret exists | Explicit PR-body note + docs handoff; local_check proves green locally |
| provider_golden/composition fixtures unaware of new family | Extend fixtures mirroring short-term entries |
| Conflict with feat/unified-catalog | No catalog.rs changes; helpers-only in common/memory.rs |
