# Phase 3: SurrealDB Embedded Driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** New crate `chronicle-driver-surreal` implementing the full `GraphDriver` supertrait (as of v0.3.0 / 1.2.0-research) against embedded SurrealDB (`surrealdb` crate, `kv-rocksdb`), giving digital workers a server-less graph-memory backend. Passes the same behavioral contract as the Neo4j driver.

**Spec amendment:** docs/superpowers/specs/2026-06-06-phase-3-embedded-backend-amendment.md (READ FIRST — capability mapping + 8 locked decisions + risks). This is NOT a Cypher port; SurrealQL is a from-scratch query layer. The correctness oracle is the **chronicle-testkit conformance suite + behavior parity with FakeDriver/Neo4j**, not query-text fidelity.

**Trait surface to implement (everything Neo4jDriver implements):** EntityNodeOps, EntityEdgeOps, EpisodeOps, EpisodicEdgeOps, SearchOps (incl. Phase-2: bfs x2, episode_fulltext, embeddings loaders, nodes_connected_to_center, episode_mention_counts, filters on the 4 base searches), SchemaOps, CommunityOps, SagaOps, BulkSaveOps (save_all transactional). READ crates/chronicle-core/src/driver/mod.rs IN FULL for the exact current signatures before starting.

**Branch:** `feat/phase-3-surreal-driver` off `research` (pull first). PR → research, tag after merge. House rules: no unwrap/panic in prod, fmt+clippy -D warnings, NO attribution trailers, conventional commits. ENV: export TMPDIR=$HOME/.cache/tmp (tmpfs near-full).

**Locked decisions (from spec §4):** surrealdb::sql::Datetime at boundary (never bind chrono directly); HNSW TYPE F32; KNN min-score = over-fetch ×3 + post-filter; BFS depth cap 5; one FTS index per field; idempotent schema DDL on connect; native-only (no wasm); feature-gated crate.

---

### Task 1: Crate scaffold + connect + schema + convert + node/episode/community/saga save+get

**Files:** crates/chronicle-driver-surreal/{Cargo.toml, src/lib.rs, src/schema.rs, src/convert.rs}. Root Cargo.toml: add member + `surrealdb = { version = "3", default-features = false, features = ["kv-rocksdb"] }` to workspace deps (verify the exact stable 3.x version + feature name on docs.rs; kv-mem also useful for tests — add if separate feature).

- [ ] **Step 1 — manifest + workspace member.** Crate `chronicle-driver-surreal`, deps: chronicle-core (workspace), surrealdb, tokio, async-trait, serde, serde_json, chrono, tracing, anyhow/thiserror. dev: chronicle-testkit, tempfile. Commit Cargo.lock.
- [ ] **Step 2 — schema.rs.** SurrealQL DDL string(s), idempotent (`DEFINE TABLE ... IF NOT EXISTS` / `OVERWRITE` — verify which 3.x supports): tables entity (label field distinguishes Entity/Episodic/Community/Saga — OR separate tables per node kind; DECIDE: separate tables `entity`, `episodic`, `community`, `saga` is cleaner for typed queries; edges `relates_to` (TYPE RELATION IN entity OUT entity), `mentions` (episodic→entity), `has_member` (community→entity), `has_episode` (saga→episodic), `next_episode` (episodic→episodic)). Fields per the chronicle types (uuid as the SurrealDB record id? — use `entity:⟨uuid⟩` record ids so uuid lookups are O(1); store the original uuid string field too for serde roundtrip). HNSW indexes (entity.name_embedding, relates_to.fact_embedding, community.name_embedding — DIMENSION configurable, default 1024, TYPE F32, DIST COSINE). BM25 FTS analyzer + per-field indexes (entity.name+summary, relates_to.fact, episodic.content, community.name). Range-equivalent: SurrealDB auto-indexes record ids; add explicit indexes on group_id fields for filter perf.
- [ ] **Step 3 — convert.rs.** chrono::DateTime<Utc> ↔ surrealdb::sql::Datetime helpers; Vec<f32> ↔ array<float> (native); serde_json::Value attrs ↔ object; chronicle type ↔ a serde struct mirroring the SurrealDB row shape (since uuid is the record id, map id↔uuid). Node/edge row structs with Serialize/Deserialize. A roundtrip unit test asserting datetime survives as a queryable datetime (not string) — the #2753/#2804 gotcha guard.
- [ ] **Step 4 — lib.rs SurrealDriver.** `pub struct SurrealDriver { db: Surreal<Db>, embedding_dim: usize }`; `pub async fn connect_embedded(path, embedding_dim) -> Result<Self, DriverError>` (Surreal::new::<RocksDb>(path), use_ns/use_db, run schema DDL via .query().check(); map errors → DriverError::Connection); `pub async fn connect_memory(embedding_dim)` for tests (kv-mem). Implement EntityNodeOps + EpisodeOps + EpisodicEdgeOps + CommunityOps(save/get) + SagaOps(save/get) node/edge save+get+get_by_uuids+by_group_ids+delete (UPSERT/SELECT/DELETE SurrealQL; RELATE for edges). SchemaOps::build_indices_and_constraints = re-run schema DDL idempotently (delete_existing → REMOVE then redefine).
- [ ] **Step 5 — tests** (in-crate, kv-mem): node/edge/episode/community/saga save→get roundtrip incl. embedding + attrs + temporal + datetime-is-datetime guard; get_by_uuids order; delete. Commit `feat(surreal): scaffold, schema, node/edge persistence`.

### Task 2: Search ops (vector, fulltext, BFS, embeddings, adjacency, mentions, filters)

- [ ] EntityEdgeOps::get_edges_between_nodes (directed RELATE query). SearchOps: edge/node fulltext (BM25 `@N@` + search::score), edge/node similarity (HNSW `<|K,EF|>` + vector::distance::knn, over-fetch ×3 + min-score post-filter in Rust), community fulltext+similarity, get_embeddings_for_{nodes,edges,communities}, node_bfs_search + edge_bfs_search (recursive idiom `.{1..N}`, depth cap 5, edge-type filter per-hop, group filter), nodes_connected_to_center (undirected 1-hop), episode_mention_counts (count mentions edges per entity), episode_fulltext_search.
- [ ] **Filters**: port the SearchFilters → SurrealQL WHERE builder (edge_types `relation_type IN $t`, edge_uuids `id IN $u`, date OR-of-ANDs groups, node_labels). Wire into the 4 base searches + BFS. Parameterize via .bind() (NO string interpolation of user values; label/field names sanitized if inlined).
- [ ] Tests (kv-mem): vector ranking + min-score cutoff, fulltext recall, BFS depth 1 vs 3 + edge-type filter, center adjacency, mention counts, filter combos (edge_types + date window + edge_uuids). Commit `feat(surreal): search, bfs, embeddings, filters`.

### Task 3: Bulk transactional save + remaining maintenance ops

- [ ] BulkSaveOps::save_all — single SurrealQL `BEGIN; ...; COMMIT;` (or db transaction API) saving episodes+episodic_edges+entity_nodes+entity_edges atomically; rollback on error (.check()/take_errors → CANCEL). EntityNodeOps::get_mentioned_nodes, delete ops, get_community_clusters projection (SurrealQL aggregation over relates_to grouped by group_id — may need a SELECT with count(); verify the GROUP BY syntax), community_of_member + neighbor_communities (graph queries), remove_communities, saga get_by_name/previous_episode/contents-since.
- [ ] Tests: transactional save_all all-present; rollback atomicity (mid-batch failure → nothing committed — feasible in kv-mem? if not, document + cover in a kv-rocksdb tempdir test); clusters projection; membership; saga queries. Commit `feat(surreal): transactional bulk save + maintenance/cluster/saga ops`.

### Task 4: Conformance + integration + ledger + PR

- [ ] **Conformance harness**: factor the FakeDriver behavioral tests into a reusable `chronicle-testkit` conformance fn set (if not already) OR write a SurrealDB integration test suite mirroring the Neo4j integration tests (tests/surreal_integration.rs, embedded RocksDb tempdir — NO docker needed). Cover: every trait family, the e2e add_episode flow (use Chronicle facade with SurrealDriver + MockLlm + MockEmbedder → ingest + recall + bi-temporal invalidation, mirroring the Phase-1 e2e gate), community build+search, saga, bulk, add_triplet, remove_episode.
- [ ] **Run the full suite** (tempdir embedded — runs in normal CI, no external service). Single-threaded if shared-state isolation needed.
- [ ] cargo update + bash ci/local_check.sh green (verify surrealdb's dep weight doesn't break the locked/--all-features build; if surrealdb pulls problematic transitive deps, pin minimally + note).
- [ ] docs/port-fidelity.md: SurrealDB driver row(s) + the 8 locked-decision deviations (post-filter min-score, depth cap, datetime boundary, etc.). README: backend table updated (Neo4j + SurrealDB embedded; Kuzu dropped); feature-gate usage. Spec's original "Kuzu" mention annotated as superseded.
- [ ] PR → research; verify headRefOid; after merge tag (next research line, e.g. 1.2.1-research or a v0.4.0 phase tag — match the established dual scheme: phase tag vX + the research line). Report.

## Risks
| Risk | Mitigation |
|---|---|
| surrealdb 3.x API differs from spike examples | Verify against docs.rs/surrealdb current at Task 1; adapt + report |
| chrono datetime binds as string | surreal Datetime at boundary + dedicated guard test (Task 1) |
| BFS recursive idiom syntax/perf | depth cap 5; test depth 1-3; profile; if syntax unavailable in pinned version, fall back to iterative multi-query BFS + document |
| KNN min-score post-filter recall loss | over-fetch ×3 padding; tune; document |
| Conformance gaps vs Neo4j | mirror Neo4j integration tests 1:1; the e2e add_episode gate is the headline parity proof |
| rollback test infeasible in kv-mem | use kv-rocksdb tempdir for the atomicity test |
