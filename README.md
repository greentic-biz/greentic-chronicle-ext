# greentic-chronicle-ext

Bi-temporal knowledge-graph memory for Greentic digital workers.

A faithful Rust port of [getzep/graphiti](https://github.com/getzep/graphiti) (Apache-2.0),
a temporal knowledge-graph memory framework for AI agents. Packaged as a standalone Rust library
workspace so Greentic digital workers gain long-term, bi-temporal, graph-structured memory.

**Attribution:** See [NOTICE](./NOTICE). Port baseline: graphiti v0.29.1
(commit `34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6`).

---

## Crates

| Crate | Description |
|---|---|
| `chronicle-core` | Domain types, pipeline (extract → dedup → invalidate → persist), search, prompts, traits |
| `chronicle-driver-neo4j` | `GraphDriver` implementation over Neo4j via `neo4rs` (Bolt) |
| `chronicle-driver-surreal` | `GraphDriver` implementation over **embedded** SurrealDB (`surrealdb` 3.1.3, `kv-rocksdb`/`kv-mem`) — server-less graph + HNSW vector + BM25 FTS in one engine |
| `chronicle-driver-falkor` | `GraphDriver` implementation over **FalkorDB** (`falkordb` 0.2.1) — openCypher on a Redis module; a Cypher-dialect adaptation of the Neo4j driver |
| `chronicle-llm-openai` | `LlmClient` + `EmbedderClient` over OpenAI-compatible endpoints via `async-openai` |
| `chronicle-testkit` | `FakeDriver`, `MockLlm`, `MockEmbedder` for deterministic unit tests; driver-conformance integration tests |

### Graph backends

| Backend | Crate | Status | Notes |
|---|---|---|---|
| Neo4j (server) | `chronicle-driver-neo4j` | Available | Bolt via `neo4rs`; the reference / conformance-baseline driver |
| SurrealDB (embedded) | `chronicle-driver-surreal` | Available | Pure-Rust, server-less; feature-gated. Behaviorally interchangeable with Neo4j behind the `GraphDriver` supertrait |
| FalkorDB (Redis server) | `chronicle-driver-falkor` | Available | openCypher on a Redis module via `falkordb` 0.2; a Cypher-dialect adaptation of the Neo4j driver. Behaviorally interchangeable behind the `GraphDriver` supertrait. Deviations: datetime stored as epoch-millis int, attrs as a JSON string, no atomic `save_all` (best-effort sequential), vector KNN distance→similarity post-filter, edge fulltext via a relationship-fulltext DDL index, requires a multi-threaded Tokio runtime — see `docs/port-fidelity.md` D-33–D-39 |
| Neptune | — | Skipped | Out of scope (no viable Rust crate) |
| ~~Kuzu~~ | — | Dropped | Upstream archived 2025-10-10 (Apple acquisition); superseded by SurrealDB embedded — see `docs/port-fidelity.md` |

---

## Usage

```rust,no_run
use std::sync::Arc;
use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::llm::LlmConfig;
use chronicle_core::search::config::edge_hybrid_search_rrf;
use chronicle_core::types::EpisodeType;
use chronicle_driver_neo4j::Neo4jDriver;
use chronicle_llm_openai::{OpenAiEmbedder, OpenAiEmbedderConfig, OpenAiLlm};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect driver (reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD or use defaults)
    let driver = Arc::new(
        Neo4jDriver::connect("bolt://localhost:7687", "neo4j", "password", "neo4j")
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    );

    // Build LLM client — reads OPENAI_API_KEY from env
    let llm = Arc::new(OpenAiLlm::new(LlmConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?);

    // Build embedder
    let embedder = Arc::new(
        OpenAiEmbedder::new(OpenAiEmbedderConfig::default())
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    );

    // Construct Chronicle (0 = use SEMAPHORE_LIMIT default of 20)
    let chronicle = Chronicle::new(driver, llm, embedder, 0);

    // Build indices (idempotent)
    chronicle.build_indices_and_constraints(false).await?;

    // Ingest an episode
    let result = chronicle
        .add_episode(AddEpisodeRequest {
            name: "episode-1".into(),
            episode_body: "Alice is an engineer at Acme Corp.".into(),
            source: EpisodeType::Text,
            source_description: "example".into(),
            reference_time: chrono::Utc::now(),
            group_id: "my-group".into(),
            uuid: None,
            previous_episode_uuids: None,
            entity_types: None,
            custom_extraction_instructions: None,
        })
        .await?;

    println!(
        "Ingested: {} nodes, {} edges",
        result.nodes.len(),
        result.edges.len()
    );

    // Hybrid BM25 + cosine RRF search
    let hits = chronicle
        .search("Alice engineer", &["my-group".to_string()], &edge_hybrid_search_rrf())
        .await?;

    for edge in &hits {
        println!("[{}] {}", edge.uuid, edge.fact);
    }

    Ok(())
}
```

A runnable version of this snippet lives at
[`crates/chronicle-llm-openai/examples/quickstart.rs`](crates/chronicle-llm-openai/examples/quickstart.rs).

### Embedded backend (SurrealDB)

For a server-less, node-local graph memory, swap the Neo4j driver for the
embedded SurrealDB driver — nothing else in the pipeline changes (it is
behaviorally interchangeable behind the `GraphDriver` supertrait):

```rust,ignore
use std::sync::Arc;
use chronicle_core::chronicle::Chronicle;
use chronicle_driver_surreal::SurrealDriver;

// Persistent embedded store backed by RocksDB on disk.
// `embedding_dim` must match your embedder's output dimension.
let driver = Arc::new(
    SurrealDriver::connect_embedded("/var/lib/chronicle/graph.db", 1536).await?,
);

// (`SurrealDriver::connect_memory(1536)` gives an ephemeral in-memory store,
// handy for tests.)

let chronicle = Chronicle::new(driver, llm, embedder, 0);
chronicle.build_indices_and_constraints(false).await?; // idempotent schema DDL
```

The embedded driver bundles graph traversal, HNSW vector similarity, and BM25
full-text search in one engine — no external service. It is native-only
(`kv-rocksdb` links RocksDB's C++); see the build note in the driver crate.

### Redis-server backend (FalkorDB)

For ops teams that already run Redis, FalkorDB (openCypher on a Redis module) is
a drop-in third backend — again, nothing else in the pipeline changes:

```rust,ignore
use std::sync::Arc;
use chronicle_core::chronicle::Chronicle;
use chronicle_driver_falkor::FalkorDriver;

// `graph_name` is the logical graph key in Redis; `embedding_dim` must match
// your embedder's output dimension. `connect()` builds the indices idempotently.
let driver = Arc::new(
    FalkorDriver::connect("falkor://127.0.0.1:6379", "chronicle", 1536).await?,
);

let chronicle = Chronicle::new(driver, llm, embedder, 0);
```

FalkorDB stores datetimes as epoch-millis integers and attributes as a JSON
string, has no atomic `save_all` (best-effort sequential), and **requires a
multi-threaded Tokio runtime** (the crate's schema refresh blocks). See
`docs/port-fidelity.md` D-33–D-39 for the full deviation list.

---

## Phase Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Workspace scaffold, crate skeletons, CI | Done |
| **Phase 1** | Core loop: add_episode (extract → dedup → bi-temporal invalidation → persist), hybrid RRF search, Neo4j driver, OpenAI LLM/embedder, 185 tests green | Done |
| **Phase 2** | Full search parity: BFS traversal, MMR / node-distance / episode-mentions / cross-encoder rerankers, `SearchFilters`, multi-scope `search_()` + `search_with_center()`, complete recipe set; D-3 edge-candidate re-ranking closed | Done |
| **Phase 4** | Communities (detection + summaries + community search scope), sagas (narrative threading + `summarize_saga`), bulk ingest (`add_episode_bulk` cross-episode dedup), `add_triplet`, `remove_episode`, `get_nodes_and_edges_by_episode`, transactional `save_all` (atomicity gap closed) — **full Graphiti-core parity** | Done |
| **Phase 3** | Embedded backend: `chronicle-driver-surreal` — full `GraphDriver` supertrait over embedded SurrealDB (graph + HNSW + BM25), e2e parity gate through the real driver. (Original spec named Kuzu; **superseded by SurrealDB** — Kuzu archived upstream, see spec amendment.) | Done |
| **Phase 6** | Third backend: `chronicle-driver-falkor` — full `GraphDriver` supertrait over FalkorDB (openCypher on a Redis module, `falkordb` 0.2), a Cypher-dialect adaptation of the Neo4j driver. e2e parity gate (bi-temporal invalidation + community/saga/bulk/triplet/remove) through the real driver against live `falkordb/falkordb:latest`. Completes the backend roadmap (Neo4j + SurrealDB + FalkorDB; Neptune skipped). | Done |
| **v1.x** | Entity/edge-type registries + attribute extraction, `extract_summaries_batch`, semaphore fan-out | Planned |

> **Graphiti-core parity:** with Phase 4 complete, chronicle ports the full Graphiti-core surface (ingest, bi-temporal invalidation, hybrid + community search, communities, sagas, bulk, triplets, maintenance). Phases 3 + 6 add the embedded (SurrealDB) and Redis-server (FalkorDB) backends alongside Neo4j. Remaining items (v1.x) are performance/registry enhancements, not core-semantic gaps.

Full fidelity notes and all ported-module statuses: [`docs/port-fidelity.md`](docs/port-fidelity.md).

Design spec: [`docs/superpowers/specs/2026-06-04-greentic-chronicle-ext-design.md`](docs/superpowers/specs/2026-06-04-greentic-chronicle-ext-design.md).

Implementation plan: [`docs/superpowers/plans/2026-06-04-chronicle-phase-0-1.md`](docs/superpowers/plans/2026-06-04-chronicle-phase-0-1.md).

---

## License

Apache-2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
