# Phase 3 Spec Amendment: Embedded Backend — Kuzu → SurrealDB

Date: 2026-06-06
Status: Approved direction (amends the original design spec's "Graph backends v1: Neo4j + Kuzu" decision)

## 1. Why this amendment

The original design picked **Kuzu** as the embedded backend (Phase 3). Verified primary-source spike (2026-06-06):

- **Kuzu is dead upstream**: `kuzudb/kuzu` was archived **2025-10-10** (Apple acquired Kùzu Inc.; surfaced Feb 2026 via DMA). Last release v0.11.3 / last `kuzu` crate publish ~Nov 2025. No future updates.
- Community forks (LadybugDB `lbug`, Vela, RyuGraph) are C++-FFI over Kuzu's ~163K-line core; only LadybugDB has an active crates.io crate, single-maintainer risk.
- The same long-term-maintenance lens that disqualified Kuzu in the spike points to **SurrealDB embedded** (`surrealdb` crate, `kv-rocksdb`): pure-Rust (no C++ FFI), corporate-backed ($44M funded), active release cadence (3.1.x Jun 2026), and ships graph + HNSW vector + BM25 FTS in one embedded engine.

**Decision: Phase 3 embedded backend = SurrealDB (`chronicle-driver-surreal`), `surrealdb` crate with `kv-rocksdb`. Kuzu dropped.** FalkorDB remains v1.x, Neptune skipped (both unchanged).

## 2. What stays identical

The operation-level `GraphDriver` trait family (Phase 1/2/4) is the abstraction the spike validated against. The Neo4j driver is the reference impl + conformance baseline. SurrealDB plugs in behind the SAME trait — no chronicle-core changes. The `chronicle-testkit` FakeDriver conformance suite is the parity oracle: the SurrealDB driver must pass the same behavioral tests the FakeDriver + Neo4j driver pass.

## 3. SurrealDB capability mapping (verified spike)

| GraphDriver op | SurrealQL approach | Confidence |
|---|---|---|
| Save/upsert Entity/Episodic/Community/Saga nodes (embedding `array<float>` TYPE F32, JSON `object` attrs, datetime) | `UPSERT entity:id SET ...` / `DEFINE TABLE ... SCHEMAFULL` | HIGH |
| RELATES_TO / MENTIONS / HAS_MEMBER / HAS_EPISODE / NEXT_EPISODE edges with props | `DEFINE TABLE rel TYPE RELATION IN .. OUT ..` + `RELATE a->rel->b SET ...` (edges are first-class tables with arbitrary fields) | HIGH |
| Vector similarity (cosine, top-k, min-score) | `DEFINE INDEX .. HNSW DIMENSION d DIST COSINE TYPE F32`; query `WHERE embedding <\|K,EF\|> $vec ORDER BY vector::distance::knn()`; **min-score = over-fetch + post-filter in driver** | HIGH (min-score MEDIUM) |
| Full-text BM25 (name/fact/content) | `DEFINE ANALYZER` + `DEFINE INDEX .. SEARCH ANALYZER .. BM25(..)`; `WHERE field @N@ $q` + `search::score(N)`; **one index per field** | HIGH |
| BFS depth-N from origins, edge-type filtered | recursive idiom `entity:o.{1..N}(->rel->entity).@`; edge filter `->(rel WHERE relation_type IN [..])->entity` | MEDIUM (perf uncharted) |
| 1-hop adjacency to center | `SELECT <->rel<->entity FROM entity:center` (undirected) / `->rel->` directed | HIGH |
| Edge/relationship-by-uuid | `SELECT * FROM rel WHERE id = rel:uuid` | HIGH |
| Transactions (multi-stmt atomic save_all) | `BEGIN; ..; COMMIT;` in one `db.query()` + `.check()`, or `db.begin()/tx.commit()/tx.cancel()` | HIGH |
| Filters (edge_types, edge_uuids, date OR-of-ANDs, labels) | `WHERE relation_type IN $t`, `WHERE id IN $uuids`, datetime comparisons, `WHERE labels CONTAINS $l` | HIGH |

Embedded init: `Surreal::new::<RocksDb>(path).await? ; db.use_ns(..).use_db(..)`. Typed deserialize via `response.take::<Vec<T>>(idx)`.

## 4. Locked implementation decisions (from spike risks)

1. **Datetime: use `surrealdb::sql::Datetime`, never bind `chrono::DateTime<Utc>` directly** (issues #2753/#2804 — chrono binds as string, breaks datetime comparisons). Driver converts chrono↔surreal Datetime at the boundary (convert.rs), exactly as the Neo4j driver does for BoltDateTime.
2. **HNSW `TYPE F32`** explicit on every vector index (default is F64; chronicle uses `Vec<f32>`).
3. **KNN min-score is post-filter**: trait `*_similarity_search(min_score)` impl over-fetches (K = limit×3 or a fixed pad) then filters `score >= min_score` in Rust. Document as a driver impl note; behavior identical to the trait contract.
4. **BFS depth hard cap = 5** (matches existing clamp pattern; SurrealDB recursive traversal perf at depth >5 is unbenchmarked). Inline sanitized integer like the Neo4j driver.
5. **FTS one-index-per-field**: separate `DEFINE INDEX ... SEARCH` for name / fact / content / summary; the driver's fulltext methods target the right index per node/edge type.
6. **HNSW lives in RAM, rebuilt on startup**: driver `connect()` runs schema DDL idempotently (`IF NOT EXISTS`/`OVERWRITE`); a readiness/warmup note in docs. Single-process embedded (correct for node-local memory; no clustering need).
7. **No WASM path**: `kv-rocksdb` is native-only (C++ RocksDB). chronicle driver is host-native — fine. Spec notes: a future wasm32 driver would need `kv-mem` or another store; out of scope.
8. **Feature-gated**: `chronicle-driver-surreal` is its own crate; consumers opt in (`features = ["surreal"]`), so SurrealDB's dep weight isn't paid by Neo4j-only users. Mirrors the original feature-gate intent.

## 5. Deliverable

New crate `chronicle-driver-surreal` implementing the full `GraphDriver` supertrait (EntityNodeOps, EntityEdgeOps, EpisodeOps, EpisodicEdgeOps, SearchOps, SchemaOps, CommunityOps, SagaOps, BulkSaveOps — everything Neo4j implements as of v0.3.0/1.2.0-research). Passes the chronicle-testkit conformance suite + a SurrealDB-specific embedded integration test set (no docker needed — embedded RocksDB in a tempdir). README + ledger updates. The spec's backend table becomes: **Neo4j (server) + SurrealDB (embedded), feature-gated; FalkorDB v1.x; Neptune skipped; Kuzu dropped (upstream archived).**

## 6. Risks carried into the plan

- BFS deep-traversal performance unbenchmarked → depth cap 5 + profile early.
- KNN min-score post-filter → over-fetch padding factor needs tuning (start ×3).
- Datetime binding gotcha → boundary conversion + a dedicated roundtrip test that asserts stored values are queryable as datetimes (not strings).
- Recursive-path edge-predicate expressiveness < Cypher → BFS filter applied per-hop; complex path predicates may need multi-step queries (acceptable for chronicle's BFS which only filters edge type + group).
- SurrealDB SurrealQL ≠ Cypher → this is a from-scratch query layer, not a Cypher port; the conformance suite (not Cypher-text fidelity) is the correctness oracle.
