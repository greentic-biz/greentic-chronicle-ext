# Phase 6 (v1.x): FalkorDB Driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** New crate `chronicle-driver-falkor` implementing the full `GraphDriver` supertrait against FalkorDB (Redis-module openCypher graph), completing the original roadmap's third backend. FalkorDB is server-backed like Neo4j but Redis-native (ops-familiar — Greentic already runs Redis). This is largely a **Cypher dialect adaptation of the Neo4j driver**, not from-scratch like SurrealDB.

**Roadmap:** original design spec listed "FalkorDB in v1.x". Backends after this: Neo4j (server) + SurrealDB (embedded) + FalkorDB (Redis server). Neptune stays skipped (no viable Rust crate).

**Correctness oracle:** behavior parity with FakeDriver / Neo4j (the conformance + e2e gate), NOT Cypher-text fidelity. The Neo4j driver is the structural template; swap neo4rs→falkordb and apply the dialect deltas below.

**Crate (verified):** `falkordb = "0.2"` (0.2.1), async via `features = ["tokio"]`. `FalkorClientBuilder::new_async().with_connection_info("falkor://host:port".try_into()?).build().await`; `client.select_graph(name)`; `graph.query("CYPHER").with_timeout(ms).execute().await`; result types `FalkorValue` / `Node` / `Edge` / `LazyResultSet` (Iterator). Params: VERIFY the param-binding API in 0.2.x (query_with_params / a params map) at Task 1 — if absent, build queries with bound values via the crate's parameter support or careful typed construction (NO string-interpolating user values).

## Verified FalkorDB Cypher dialect deltas vs Neo4j (apply when adapting)

| Concern | Neo4j (current driver) | FalkorDB | Action |
|---|---|---|---|
| **Vector index** | `vector.similarity.cosine(n.emb,$v)` inline + `db.create.setNodeVectorProperty` | `CREATE VECTOR INDEX FOR (n:Label) ON (n.emb) OPTIONS {dimension:D, similarityFunction:'cosine', M:16, efConstruction:200}`; query `CALL db.idx.vector.queryNodes('Label','emb', K, vecf32([..])) YIELD node, score` | Procedure-based KNN; `vecf32([...])` wraps the query vector AND stored vectors on write (SET n.emb = vecf32($v)). score = distance → convert to similarity. |
| **Fulltext** | `db.index.fulltext.queryNodes("idx",$q)` / `queryRelationships` | `CALL db.idx.fulltext.createNodeIndex('Label','field1','field2')` + `CALL db.idx.fulltext.queryNodes('Label',$q) YIELD node, score` | **RISK: relationship fulltext may be unsupported** — chronicle indexes edge `fact`. VERIFY at Task 1; if unsupported, fall back to brute-force CONTAINS or store fact-search via a node. |
| **Datetime** | Bolt DateTime native | **NO native datetime type** | Store all 4 temporal fields as **epoch millis i64** (convert chrono↔i64 at boundary, convert.rs); bi-temporal comparisons become integer comparisons. This is THE key delta — a wrong mapping breaks invalidation + retrieve_episodes cutoff. Dedicated guard test. |
| **Attributes (JSON map)** | `properties(n)` map | **may not support nested map properties** | VERIFY; if unsupported, store attrs as a JSON **string** property (serialize on write, parse on read) — document as deviation. |
| **Vectors/lists as props** | native lists | openCypher lists supported; embeddings via vecf32 | episode-uuid lists = plain string arrays (verify); embeddings via vecf32. |
| **Variable-length BFS** | `-[:REL*1..N]->` | openCypher supports `-[*1..N]->` | Should map directly; verify depth perf, keep cap. |
| **Transactions** | neo4rs txn | Redis MULTI/EXEC; **crate txn support unclear** | VERIFY falkordb 0.2 transaction API; if none, save_all = best-effort sequential within one graph (document atomicity gap, or use a single multi-statement query if FalkorDB supports `;`-separated). |
| **Indices** | `CREATE INDEX ... FOR (n:L) ON (n.p)` | `CREATE INDEX FOR (n:L) ON (n.p)` (similar) + exact-match `CREATE INDEX ON :Label(prop)` legacy | Range/exact indices on uuid+group_id; verify syntax. |

## Tasks

### Task 1: scaffold + connect + schema/indices + convert (datetime-as-int) + node/edge persistence
- [ ] Crate `chronicle-driver-falkor`: falkordb (workspace dep `{ version = "0.2", features = ["tokio"] }`), chronicle-core, tokio, async-trait, serde, serde_json, chrono, tracing, thiserror. dev: chronicle-testkit. Member + Cargo.lock.
- [ ] **EMPIRICAL VERIFICATION FIRST** (Docker: `docker run -d --rm --name chronicle-falkor-test -p 6379:6379 falkordb/falkordb:latest`, wait, connect): confirm — param-binding API; vecf32 write+query; fulltext node + **whether relationship fulltext works**; whether nested map props work or must be JSON-string; transaction/multi-statement support; datetime storage (epoch int). RECORD findings; they drive the rest.
- [ ] Read crates/chronicle-core/src/driver/mod.rs (exact trait surface) + crates/chronicle-driver-neo4j/src/{lib,convert,queries}.rs (template). Read chronicle types.
- [ ] schema.rs: index DDL (range uuid+group_id per label; vector indexes entity.name_embedding/relates_to.fact_embedding/community.name_embedding; fulltext node indexes name+summary/fact/content/community-name — handle the relationship-fulltext finding). convert.rs: chrono↔epoch-millis i64, Vec<f32>↔vecf32, attrs↔(map or JSON string per finding), node/edge row extraction from FalkorValue/Node/Edge. lib.rs: FalkorDriver{client/graph, embedding_dim}, connect(conn_str, graph_name, embedding_dim). Implement node/episode/community/saga save+get+by_uuids+by_group_ids+delete, edge saves (CREATE/MERGE), retrieve_episodes, SchemaOps. Search/bulk/maintenance → loud `Err(DriverError::Query("phase-6 task-2/3 pending"))`.
- [ ] Tests (env-gated FALKOR_TEST_URI, Docker live): node/edge/episode/community/saga roundtrip incl embedding+attrs+temporal; **datetime-as-int guard** (store, query with int comparison, roundtrip to chrono); retrieve_episodes cutoff+order; delete. Commit `feat(falkor): scaffold, schema, persistence`.

### Task 2: search (vector procedure, fulltext, BFS, embeddings, adjacency, mentions, filters)
- [ ] get_edges_between_nodes; vector similarity (db.idx.vector.queryNodes over-fetch + min-score post-filter, distance→similarity); fulltext (db.idx.fulltext.queryNodes + score; relationship-fulltext per Task-1 finding); embeddings loaders; node_bfs/edge_bfs (`-[*1..N]->`, depth cap, edge-type filter, group filter); nodes_connected_to_center; episode_mention_counts; episode_fulltext; SearchFilters→Cypher WHERE (edge_types, edge_uuids, date-as-int OR-of-ANDs, labels). Filters wired into all base searches + BFS. Param-bound, no interpolation.
- [ ] Tests (Docker live): vector rank+min-score, fulltext recall, BFS depth, adjacency, mentions, filter combos. Commit `feat(falkor): search, bfs, embeddings, filters`.

### Task 3: bulk/maintenance/community/saga + conformance e2e + ledger + PR
- [ ] save_all (transactional per Task-1 finding — multi-statement or documented best-effort), get_mentioned_nodes, deletes (cascade per FalkorDB DETACH DELETE), get_community_clusters, community_of_member, neighbor_communities, remove_communities, saga previous/contents. No pending stubs after.
- [ ] **E2E parity gate** tests/falkor_e2e.rs (Docker live): Chronicle facade + real FalkorDriver + MockLlm/MockEmbedder — bi-temporal invalidation (Acme→Globex, old invalidated not deleted, search finds Globex) + community/saga/bulk/triplet/remove. Mirror the SurrealDB/Neo4j e2e.
- [ ] **DOCKER live run** (mandatory): falkordb/falkordb container, `FALKOR_TEST_URI=falkor://localhost:6379 cargo test -p chronicle-driver-falkor`, report pass count, stop container.
- [ ] cargo update + bash ci/local_check.sh. docs/port-fidelity.md FalkorDB rows + deviations (datetime-as-int, attrs-as-JSON-string if applicable, relationship-fulltext handling, vector procedure, transaction model). README backend table → +FalkorDB. PR → research; verify headRefOid; after merge tag (v0.5.0 + 1.2.2-research). Report.

## Risks
| Risk | Mitigation |
|---|---|
| falkordb 0.2 param API / result extraction differs from Neo4j pattern | Task-1 empirical verification before bulk impl |
| Relationship fulltext unsupported (chronicle indexes edge facts) | Task-1 verify; fall back to brute-force CONTAINS or node-side fact index; document |
| Datetime-as-int breaks bi-temporal silently | boundary convert + guard test + e2e invalidation proof |
| Nested-map attrs unsupported | JSON-string property fallback + deviation |
| Crate young (0.2.x) / API churn | pin exact 0.2.1; vendor-test against live FalkorDB |
| No real transactions | document atomicity gap (acceptable — Neo4j had the same until save_all; do best-effort or multi-statement) |
