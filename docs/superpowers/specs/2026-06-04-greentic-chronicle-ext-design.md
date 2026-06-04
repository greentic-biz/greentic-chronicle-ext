# greentic-chronicle-ext — Design Spec

Date: 2026-06-04
Status: Approved (brainstorm phase complete)

## 1. Summary

`greentic-chronicle-ext` is a Rust port of [getzep/graphiti](https://github.com/getzep/graphiti)
(Apache-2.0, Python) — a temporal knowledge graph memory framework for AI agents —
packaged so Greentic digital workers gain long-term, bi-temporal, graph-structured memory.

Two integration halves:

1. **This repo**: a standalone, reusable Rust library workspace (the port itself).
2. **`greentic-dw-providers`** (separate repo, separate PRs): a new
   `memory/long-term/` provider family whose `chronicle` implementation wires the
   library into the DW runtime `MemoryProvider`/`MemoryPolicy` traits under
   capability id `greentic.cap.memory.long-term`.

## 2. Decisions (locked during brainstorm)

| Decision | Choice |
|---|---|
| Integration form | Native Rust provider + standalone library (NOT a WASM design extension — runtime memory needs DB sockets and full tokio) |
| Repo name | `greentic-chronicle-ext`, org `greentic-biz` (extension/product repo convention) |
| Graph backends v1 | Neo4j (`neo4rs`) + Kuzu (embedded), feature-gated; FalkorDB in v1.x; Neptune skipped |
| Driver abstraction | Operation-level traits (not raw-Cypher passthrough) to prevent Neo4j-ism lock-in |
| LLM wiring | Own `LlmClient`/`EmbedderClient` traits; v1 ships OpenAI-compatible impl (`async-openai`) AND a bridge adapter over the dw-providers LLM family (bridge lives in greentic-dw-providers) |
| Scope target | Full parity with graphiti-core, delivered in phases |
| Prompt fidelity | Line-for-line port of prompt text + response schemas from a pinned upstream commit; improvements deferred and tracked separately |
| License | Apache-2.0 with `NOTICE` attributing getzep/graphiti and recording the pinned upstream commit |
| Branching | Three-tier: `feat/* → research → develop → main`; research is canonical |
| Versioning | Extension line scheme `1.2.x-research` while publishing from research |

## 3. Workspace layout

```
greentic-chronicle-ext/
├── Cargo.toml                      # virtual workspace
├── rust-toolchain.toml             # 1.95.0 canonical pin
├── ci/local_check.sh               # fmt + clippy -D warnings + test
├── NOTICE                          # Apache-2.0 attribution to getzep/graphiti
├── docs/                           # specs, schema docs, port-fidelity ledger
└── crates/
    ├── chronicle-core/             # domain types, pipeline, search, prompts, traits
    ├── chronicle-driver-neo4j/     # GraphDriver impl via neo4rs
    ├── chronicle-driver-kuzu/      # GraphDriver impl, embedded
    ├── chronicle-llm-openai/       # LlmClient + EmbedderClient + cross-encoder via async-openai
    └── chronicle-testkit/          # driver-conformance suite + recorded-LLM fixtures
```

Backends are feature-gated from the consumer side (`features = ["neo4j", "kuzu"]`)
so Kuzu's static-linked C++ build cost is only paid by consumers that want it.

Workspace-wide guardrails: `#![forbid(unsafe_code)]`, no `unwrap()`/`panic!()` in
production paths (`anyhow`/`thiserror`), English-only source, committed `Cargo.lock`,
CI uses `--locked`, Conventional Commits.

## 4. chronicle-core architecture

Mirrors graphiti-core's module structure, Rust-idiomatic:

### types/
- `EpisodicNode` (episode types: message, text, json; toggleable raw-content storage),
  `EntityNode` (evolving summary, name embedding, typed attributes),
  `EntityEdge` (fact text + fact embedding), `CommunityNode`/`CommunityEdge`.
- **Bi-temporal model — four timestamps on edges**:
  `created_at`/`expired_at` (transaction time) and `valid_at`/`invalid_at` (valid time).
  Superseded facts are invalidated, never deleted.
- `group_id` namespacing on every node/edge/episode (multi-tenancy key).
- Typed ontologies: user-supplied entity types and edge types with an
  `edge_type_map` keyed on `(source_label, target_label)` — the Rust analog of
  graphiti's Pydantic `entity_types`/`edge_types`, expressed via `serde`+`schemars`.
- `uuid` + `chrono` throughout.

### driver/
Operation-level abstraction — backends implement operations, they do not receive
raw Cypher strings:

- `GraphDriver` exposes: `EntityNodeOps`, `EdgeOps`, `EpisodeOps`,
  `SearchOps` (vector similarity / fulltext / BFS traversal), `SchemaOps`
  (indices + constraints build/teardown).
- `Transaction` abstraction: real transactions on Neo4j; immediate-exec fallback on
  Kuzu (mirrors upstream behavior).
- Rationale: Kuzu is schema-full and serverless while Neo4j is schema-free over Bolt;
  an operation-level trait keeps the abstraction honest and prevents dialect lock-in.

### llm/, embedder/, rerank/
- `LlmClient`: structured output via `schemars`-generated JSON schemas (the
  `response_model` analog), `ModelSize` routing (small/large per pipeline task),
  retry with exponential backoff (4 attempts on 5xx/rate-limit), optional response
  cache, token usage surfaced through `tracing`.
- `EmbedderClient`: `create`, `create_batch`, fixed `embedding_dim` config.
- `CrossEncoderClient`: reranker trait; v1 impl is the OpenAI-reranker style in
  `chronicle-llm-openai`. Local BGE via `ort` is out of scope for v1.

### prompts/
All 13 upstream prompt modules ported verbatim from the pinned commit
(`extract_nodes`, `extract_edges`, `extract_nodes_and_edges`, `dedupe_nodes`,
`dedupe_edges`, `summarize_nodes`, `summarize_sagas`, `eval`, `models`, `lib`,
`prompt_helpers`, `snippets`): prompt text as consts, response schemas as
`serde`+`schemars` structs. Every file carries a
`// Ported from graphiti <path> @ <sha>` header; drift against upstream is tracked
in a port-fidelity ledger under `docs/`.

### pipeline/
`add_episode` orchestration, port of upstream sequence:

1. Retrieve prior-episode context for the group (`retrieve_episodes`).
2. Extract entity nodes + edges (including the combined-extraction path).
3. Entity dedup/resolution: embedding-similarity shortlist → LLM decision →
   `IS_DUPLICATE_OF` edges → summary/attribute merge.
4. Edge dedup against existing edges between the same node pair (parallel per-edge).
5. Temporal extraction: LLM fills `valid_at`/`invalid_at` from episode text +
   `reference_time`.
6. Contradiction resolution / invalidation: validity-window comparison rules ported
   line-for-line (**correctness risk #1** — covered by dedicated table-driven tests).
7. Embed + persist; optional community refresh.

Concurrency via `tokio::Semaphore` (the `max_coroutines` analog). Also:
`add_episode_bulk`, `add_triplet` (manual bypass), `remove_episode`, maintenance ops.

### search/
- Scopes: edges, nodes, episodes, communities — independently configurable.
- Methods per scope: cosine similarity (vector), BM25 (backend-native fulltext),
  breadth-first search (origin nodes / `center_node_uuid`).
- Rerankers: RRF, MMR (lambda-tunable), node-distance, episode-mentions,
  cross-encoder.
- `SearchConfig` composed of per-scope configs + `SearchFilters`
  (labels, edge types, time windows); full port of upstream
  `search_config_recipes` presets.
- `search()` returns edges (fact-centric, RRF default);
  `search_()` returns full multi-scope `SearchResults`.

### communities/
Label-propagation clustering + LLM-generated community summaries,
`build_communities()` on demand or `update_communities=true` on ingest; saga support.

### Errors & telemetry
`thiserror` per crate; `tracing` spans with configurable prefix (upstream tracer
analog).

### Public API
Method-for-method parity with the Python `Graphiti` class (`add_episode`,
`add_episode_bulk`, `add_triplet`, `retrieve_episodes`, `search`, `search_`,
`build_communities`, `get_nodes_and_edges_by_episode`, `remove_episode`,
`build_indices_and_constraints`, `close`) so upstream docs and evals stay relevant.

## 5. Greentic integration (lives in greentic-dw-providers)

- New provider family `memory/long-term/` following the existing
  `memory/short-term/{core,in-memory,redis}` pattern:
  - `memory/long-term/core` — contract crate (long-term memory semantics:
    ingest episode, recall query, temporal filters).
  - `memory/long-term/chronicle` — implementation depending on `chronicle-core`
    plus selected driver crates.
- Implements `MemoryProvider` + `MemoryPolicy` from `greentic-dw-runtime`;
  registers `greentic.cap.memory.long-term` via the established
  `capability_id(ProviderCategory::Memory, ...)` helper.
- **LLM bridge**: adapter implementing `chronicle_core::LlmClient` /
  `EmbedderClient` on top of the dw-providers LLM family, so a digital worker
  reuses its runtime-configured LLM provider — no separate API key for memory.
- Secrets (DB credentials) resolve through the runtime's existing secrets path,
  not bespoke config.

## 6. Delivery phases

Each phase gets its own implementation plan (separate spec→plan→impl cycle where
warranted). Phase 5 can start any time after Phase 1.

| Phase | Content | Deliverable |
|---|---|---|
| 0 | Repo scaffold: workspace, toolchain pin, CI `local_check.sh`, NOTICE, branch tiers | live repo, green CI |
| 1 | Types + all traits + Neo4j driver + OpenAI LLM/embedder + core `add_episode` loop end-to-end + basic RRF search | ingest & recall works on Neo4j |
| 2 | Full search parity: all recipes, MMR, cross-encoder, BFS, filters, multi-scope `search_` | search at upstream parity |
| 3 | Kuzu driver + conformance suite green on both backends | embedded serverless memory |
| 4 | Communities + saga + bulk ingestion + maintenance ops | full graphiti-core parity |
| 5 | Greentic integration in greentic-dw-providers (family, provider, LLM bridge, capability wiring) | digital workers get temporal KG memory |
| 6 (v1.x) | FalkorDB driver | third backend |

## 7. Testing & fidelity strategy

- **Pure unit tests**: temporal invalidation rules, dedup shortlisting, RRF/MMR
  math — table-driven, no I/O.
- **Prompt fidelity snapshots**: compare `schemars`-generated JSON schemas against
  upstream Pydantic schemas vendored as fixtures from the pinned commit.
- **Recorded-LLM tests**: mock `LlmClient` replaying responses recorded from
  Python graphiti runs — deterministic pipeline tests in CI without API keys.
- **Driver conformance**: one generic suite in `chronicle-testkit` run against
  Neo4j (testcontainers) and Kuzu (embedded temp dir).
- **Live-LLM E2E**: `#[ignore]` tests behind env vars, manual/nightly only.

## 8. Out of scope (v1)

- FalkorDB and Neptune backends (FalkorDB is v1.x; Neptune indefinitely deferred).
- Local BGE cross-encoder via ONNX (`ort`).
- graphiti's bundled FastAPI server and MCP server (Greentic has its own surfaces).
- Prompt redesign/improvement (fidelity first; changes tracked separately later).
- WASM design-time extension surface.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Temporal invalidation subtleties drift from upstream | Line-for-line port + dedicated table-driven tests + recorded-fixture comparison |
| Prompt drift degrades extraction quality | Verbatim port from pinned commit, fidelity ledger, schema snapshot tests |
| Driver trait ossifies around Neo4j | Two genuinely different backends (Bolt server vs embedded) in v1; conformance suite |
| Kuzu build cost slows CI | Feature gates; CI builds Kuzu jobs separately |
| Scope creep (full parity is multi-month) | Phased delivery; each phase independently shippable and planned |
