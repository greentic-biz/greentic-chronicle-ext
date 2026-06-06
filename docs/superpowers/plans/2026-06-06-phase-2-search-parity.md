# Phase 2: Full Search Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Bring chronicle search to graphiti parity: BFS, MMR / node-distance / episode-mentions / cross-encoder rerankers, SearchFilters, multi-scope `search_()`, full recipe set — and close fidelity deviation D-3 (edge-candidate re-ranking) in the ingest pipeline.

**Upstream truth:** /home/bima-pangestu/Works/refs/graphiti @ 34f56e65 (verify before each task). All algorithm quotes below were extracted and verified from that commit — where this plan and upstream disagree, upstream wins; note divergences in docs/port-fidelity.md.

**Scope cuts (documented):** community scope DEFERRED to Phase 4 (`community_config: None` → upstream `community_search` returns empty — clean gate; combined recipes ship with `community_config: None` + ledger note). `property_filters` field ported but NOT applied in WHERE construction (dead field upstream at this commit — ledger note). BGE local reranker out of scope (v1 decision). `_search` deprecated upstream — not ported.

**Branch:** `feat/phase-2-search-parity` off `research` in /home/bima-pangestu/Works/greentic/greentic-chronicle-ext (repo checkout is on research, no foreign WIP — branch in place). PR → research. House rules: porting headers, no unwrap/panic in prod, fmt+clippy -D warnings, NO attribution trailers, conventional commits.

---

## Upstream behavioral reference (verified extracts — single source for all tasks)

### R1. Top-level search() (search.py)
- Empty query (`query.strip().is_empty()`) → empty SearchResults immediately.
- Query embedded ONCE iff any scope uses cosine_similarity OR mmr reranker; embed text = `query.replace('\n', " ")`; else vector unused.
- group_ids `[""]` or empty → treated as None (no filter).
- 4 scope searches run in parallel; `center_node_uuid`/`bfs_origin_node_uuids` go ONLY to edge_search + node_search.
- SearchResults{edges+scores, nodes+scores, episodes+scores, communities+scores}.

### R2. Per-scope candidate fetch + BFS self-seed
- Each enabled method fetches 2*limit candidates.
- BFS lazy-expand: if method list contains bfs AND bfs_origin_node_uuids is None → after the initial (non-BFS) results land, run BFS seeded from `source_node_uuids` (edges) / node uuids (nodes) collected across initial result sets, and add as another result list.

### R3. Edge reranker dispatch (exact)
- rrf | episode_mentions → `rrf(uuid_lists, min_score=reranker_min_score)`; episode_mentions then RE-SORTS reranked edges descending by `edge.episodes.len()` (in-memory, no DB).
- mmr → load embeddings fresh from DB (`get_embeddings_for_edges`) → `maximal_marginal_relevance(query_vector, uuid→embedding map, mmr_lambda, reranker_min_score)`.
- cross_encoder → take first `limit` of the uuid map values; map `edge.fact → uuid`; `cross_encoder.rank(query, facts)`; keep score >= reranker_min_score.
- node_distance → REQUIRES center_node_uuid (else SearchRerankerError); rrf-presort; group edge uuids by source_node_uuid; `node_distance_reranker(driver, source_uuids, center, min_score)`; expand back to edge uuids in returned node order.
- Slice `[:limit]` at the end (both uuids and scores).

### R4. Node reranker dispatch
Same shapes; cross_encoder ranks `node.name` (NO limit-truncate before ranking — unlike edges); episode_mentions calls `episode_mentions_reranker(driver, search_result_uuid_LISTS, min_score)` (DB-count based); node_distance presorts via rrf then `node_distance_reranker(driver, seeded_uuids, center, min_score)`.

### R5. Episode scope
Only method = bm25 (fulltext on `episode_content` index). Rerankers: rrf, cross_encoder (rrf-presort → take limit → rank `episode.content`). No similarity/MMR/distance/mentions.

### R6. maximal_marginal_relevance (port exactly; no ndarray needed — plain Vec math)
```python
candidate_arrays[uuid] = normalize_l2(embedding)   # candidates L2-normalized; query NOT normalized here
# pairwise similarity matrix (dot products), symmetric, diagonal 0
mmr = mmr_lambda * dot(query, candidate) + (mmr_lambda - 1) * max_sim_to_any_other_candidate
# sort desc by mmr; filter >= min_score (default -2.0)
```
NOTE: `max_sim = np.max(similarity_matrix[i, :])` includes the zero diagonal — for a single candidate max_sim=0. Replicate exactly (max over the row including zeros).

### R7. node_distance_reranker (port exactly — it is 1-HOP adjacency, NOT shortest path)
```python
filtered = uuids except center; scores = {center: 0.0}
# Cypher: UNWIND $node_uuids AS node_uuid
#   MATCH (center:Entity {uuid:$center_uuid})-[:RELATES_TO]-(n:Entity {uuid:node_uuid})
#   RETURN 1 AS score, node_uuid AS uuid          (UNDIRECTED single hop)
# missing → inf; sort ASC by score; if center was in input: scores[center]=0.1, prepend center
# return uuids where (1/score) >= min_score, scores = 1/score  (inf → 0.0 → filtered when min_score>0; center → 10.0)
```

### R8. episode_mentions_reranker (nodes; port exactly INCLUDING the counterintuitive ASC sort)
```python
sorted_uuids = rrf(uuid_lists)[0]
# Cypher: UNWIND ... MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid:node_uuid})
#         RETURN count(*) AS score, n.uuid AS uuid
# missing → inf; sort ASCENDING by count (yes — fewer mentions ranks higher; upstream quirk, port bug-for-bug + ledger note)
# filter score >= min_score (inf passes >=! but... port the exact code: `scores[uuid] >= min_score` — inf >= min_score is TRUE so unmentioned nodes are KEPT at the end. Re-read upstream lines and port the literal comparisons.)
```

### R9. BFS Cypher (Neo4j)
node_bfs: `UNWIND $bfs_origin_node_uuids AS origin_uuid MATCH (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..N]->(n:Entity) WHERE n.group_id = origin.group_id {filters} RETURN ... LIMIT $limit` (origin label-free — Entity or Episodic; group_ids additionally `n.group_id IN $group_ids AND origin.group_id IN $group_ids` when provided). Early-return empty when origins None/empty or depth < 1.
edge_bfs: path-expansion `MATCH path = (origin {uuid})-[:RELATES_TO|MENTIONS*1..N]->(:Entity) UNWIND relationships(path) AS rel MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity) {filters} RETURN DISTINCT ... LIMIT $limit`. Early-return when origins None/empty.

### R10. SearchFilters (full)
Fields: node_labels, edge_types, valid_at/invalid_at/created_at/expired_at as `Vec<Vec<DateFilter>>` (OUTER=OR, INNER=AND → `((c1 AND c2) OR (c3))`), edge_uuids, property_filters (ported, unused). DateFilter{date: Option<DateTime<Utc>>, comparison_operator}; ComparisonOperator{Eq "=", Neq "<>", Gt ">", Lt "<", Gte ">=", Lte "<=", IsNull "IS NULL", IsNotNull "IS NOT NULL"}.
Neo4j WHERE builders: node → `n:Label1|Label2`; edge → `e.name IN $edge_types`, `e.uuid IN $edge_uuids`, node_labels on edges → `n:L AND m:L`, date fragments per group; all fragments joined by AND.

### R11. Recipes (complete set — see extraction table). Key facts: COMBINED_* cover 4 scopes (ours: community_config None + note); *_CROSS_ENCODER recipes use methods bm25+sim+bfs; EDGE/NODE_CROSS_ENCODER set explicit limit 10; COMMUNITY_CROSS_ENCODER limit 3 (deferred with community scope); COMBINED_HYBRID_SEARCH_MMR uses mmr_lambda=1.0 for edge/node/community scopes, episodes rrf.

### R12. OpenAI cross-encoder (openai_reranker_client.py)
Boolean logprob classifier per passage, all passages concurrent: model gpt-4.1-nano; messages = system "You are an expert tasked with determining whether the passage is relevant to the query" + user `Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise. <PASSAGE>...</PASSAGE> <QUERY>...</QUERY>`; temperature 0, max_tokens 1, logit_bias {"6432":1,"7983":1} (token ids for " True"/" False"), logprobs true, top_logprobs 2. Score = exp(top_logprob) if top token =="true" (case-insensitive, stripped) else 1 - exp(top_logprob). Sort desc.

### R13. Public API (graphiti.py)
- `search()` simple: returns edges only; config = EDGE_HYBRID_SEARCH_RRF when center None, EDGE_HYBRID_SEARCH_NODE_DISTANCE when center given; limit = num_results.
- `search_()` advanced: default config COMBINED_HYBRID_SEARCH_CROSS_ENCODER; returns full SearchResults; search_filter None → default empty filters.

---

## Tasks

### Task 1: Branch + config/filters expansion (chronicle-core/src/search/)
- [ ] `git -C /home/bima-pangestu/Works/greentic/greentic-chronicle-ext checkout -b feat/phase-2-search-parity research` (pull first).
- [ ] `search/filters.rs`: SearchFilters/DateFilter/ComparisonOperator/PropertyFilter per R10 (serde where useful; property_filters with `// ported, not applied in queries — upstream dead field` note). Default = all None.
- [ ] `search/config.rs` expansion: NodeSearchConfig/EpisodeSearchConfig (+CommunitySearchConfig type present but undocumented-constructible? — include the struct for recipe-shape parity, always None in recipes, doc Phase 4), full reranker enums per scope (Edge: Rrf/Mmr/NodeDistance/EpisodeMentions/CrossEncoder; Node: same; Episode: Rrf/CrossEncoder), EdgeSearchMethod::BreadthFirstSearch now real, NodeSearchMethod{CosineSimilarity,Bm25,BreadthFirstSearch}, EpisodeSearchMethod{Bm25}. SearchConfig gains node_config/episode_config/community_config.
- [ ] `search/recipes.rs`: ALL recipes per R11 as fns (edge_hybrid_search_rrf stays; add the other 13; combined_* with community_config None + ledger-pointer comment).
- [ ] Unit tests: recipe shapes (spot 4), filter composition struct construction. Commit.

### Task 2: Driver trait extensions (chronicle-core/src/driver/mod.rs) + FakeDriver
- [ ] Extend SearchOps (additive — existing methods UNCHANGED except adding a `filters: &SearchFilters` param to the four existing search methods; ripple to Neo4j driver + FakeDriver + edge_search callers):
```rust
async fn node_bfs_search(&self, origins: &[String], filters: &SearchFilters, max_depth: usize, group_ids: &[String], limit: usize) -> Result<Vec<EntityNode>, DriverError>;
async fn edge_bfs_search(&self, origins: &[String], max_depth: usize, filters: &SearchFilters, group_ids: &[String], limit: usize) -> Result<Vec<EntityEdge>, DriverError>;
async fn episode_fulltext_search(&self, query: &str, group_ids: &[String], limit: usize) -> Result<Vec<EpisodicNode>, DriverError>;
async fn get_embeddings_for_nodes(&self, uuids: &[String]) -> Result<HashMap<String, Vec<f32>>, DriverError>;
async fn get_embeddings_for_edges(&self, uuids: &[String]) -> Result<HashMap<String, Vec<f32>>, DriverError>;
/// 1-hop UNDIRECTED adjacency to center (R7): returns uuids adjacent to center.
async fn nodes_connected_to_center(&self, node_uuids: &[String], center_uuid: &str) -> Result<Vec<String>, DriverError>;
/// MENTIONS in-degree per node uuid (R8).
async fn episode_mention_counts(&self, node_uuids: &[String]) -> Result<HashMap<String, u64>, DriverError>;
```
- [ ] FakeDriver: implement all (BFS = in-memory adjacency walk over RELATES_TO both?? — NO: directed per the Cypher `-[*1..N]->`; MENTIONS edges traverse episode→entity; adjacency check undirected for nodes_connected_to_center; filters honored: edge_types, edge_uuids, node_labels, date groups OR-of-ANDs). Tests for each (incl. filter semantics: OR-of-ANDs date windows; edge_uuids whitelist).
- [ ] Commit.

### Task 3: MMR + rerankers (chronicle-core/src/search/rerank.rs)
- [ ] `maximal_marginal_relevance(query: &[f32], candidates: &[(String, Vec<f32>)], mmr_lambda: f64, min_score: f64) -> (Vec<String>, Vec<f64>)` per R6 EXACTLY (L2-normalize candidates only; pairwise dot matrix with zero diagonal; max over full row; sort desc; >= min_score). Preserve candidate insertion order for ties (stable sort).
- [ ] `node_distance_rerank(driver, node_uuids, center_uuid, min_score)` per R7 (uses nodes_connected_to_center; inf handling; center prepend 0.1; 1/score semantics).
- [ ] `episode_mentions_rerank(driver, uuid_lists, min_score)` per R8 — READ upstream lines first and port the LITERAL comparisons (the ASC sort + inf >= min_score behavior); ledger note for the quirk.
- [ ] Table-driven tests: MMR diversity behavior (lambda 1.0 = pure relevance order; lambda 0.5 demotes near-duplicates), distance (adjacent/unreachable/center-in-list), mentions quirk pinned.
- [ ] Commit.

### Task 4: CrossEncoderClient (core trait + testkit mock + OpenAI impl)
- [ ] core `src/rerank/mod.rs` (or llm/cross_encoder.rs — follow spec layout `rerank/`): `#[async_trait] pub trait CrossEncoderClient: Send + Sync { async fn rank(&self, query: &str, passages: &[String]) -> Result<Vec<(String, f64)>, LlmError>; }` (desc-sorted).
- [ ] testkit `MockCrossEncoder`: scripted scores by substring match or FIFO; deterministic.
- [ ] chronicle-llm-openai `src/reranker.rs`: OpenAiReranker per R12 (concurrent per-passage via futures join_all; logit_bias/logprobs request fields — check async-openai 0.40 support for logprobs/top_logprobs/logit_bias on chat; if a field is unsupported by the typed builder, use the raw `serde_json` escape hatch or document + approximate with the Gemini-style numeric scoring as fallback — REPORT which path taken); unit tests for score math (exp/1-exp mapping) as pure fns; live test #[ignore].
- [ ] Commit.

### Task 5: Scope searches + top-level search (chronicle-core/src/search/)
- [ ] `edge_search` rewrite per R2+R3 (all 5 rerankers, BFS self-seed, 2*limit, SearchRerankerError → ChronicleError::InvalidInput for missing center).
- [ ] `node_search` per R4, `episode_search` per R5.
- [ ] `SearchResults` struct (edges/nodes/episodes/communities + 4 score Vecs; communities always empty + type placeholder `Vec<()>`? NO — use `Vec<EntityNode>`?? Communities have no type yet: use an empty `Vec<CommunityPlaceholder>`? Cleanest: omit community fields entirely in Phase 2 SearchResults and add in Phase 4 — note the struct-shape divergence in ledger).
- [ ] Top-level `search(driver, embedder, cross_encoder: Option<&dyn CrossEncoderClient>, query, group_ids, config, filters, center_node_uuid, bfs_origin_node_uuids) -> SearchResults` per R1 (empty-query guard, single embed decision, parallel scopes via tokio::join!). Cross-encoder absent but recipe demands it → ChronicleError::InvalidInput with clear message.
- [ ] Tests vs FakeDriver+MockEmbedder+MockCrossEncoder: per-reranker edge tests, node bfs self-seed test, multi-scope assembly test.
- [ ] Commit.

### Task 6: Neo4j driver parity
- [ ] queries.rs: BFS x2 (R9 — beware Cypher var-length param: depth must be inlined into query string, not a $param; sanitize as integer), episode fulltext (R5 Cypher), embeddings loaders (RETURN uuid + embedding WHERE uuid IN $uuids AND embedding IS NOT NULL), nodes_connected_to_center (R7 Cypher), episode_mention_counts (R8 Cypher), filter WHERE builders (R10: node label `n:L1|L2` — sanitize labels strictly `^[A-Za-z0-9_]+$` since labels can't be parameterized; edge_types/edge_uuids via params; date groups OR-of-ANDs with numbered params).
- [ ] Wire `filters` into the 4 existing search queries.
- [ ] Integration tests (docker, same env-gated pattern): BFS depth behavior, filtered search (edge_types + date window + edge_uuids), mention counts, embeddings loader.
- [ ] Commit.

### Task 7: D-3 closure + facade + close-out
- [ ] pipeline/edge_ops.rs `resolve_extracted_edges`: related_edges = full hybrid edge search on fact with `SearchFilters { edge_uuids: Some(pair-pool uuids), .. }` (faithful upstream path) — remove the Phase-1 approximation; invalidation candidates unchanged (already faithful) but now pass empty SearchFilters explicitly. Update e2e expectations if call sequence changes (it shouldn't — same LLM calls).
- [ ] chronicle.rs facade: `search()` center_node_uuid routing per R13 (None→RRF recipe, Some→NODE_DISTANCE recipe + center forwarded); new `search_(query, config, group_ids, center_node_uuid, bfs_origin_node_uuids, filters) -> SearchResults` default COMBINED_HYBRID_SEARCH_CROSS_ENCODER; Chronicle::new gains optional cross_encoder (builder method `with_cross_encoder(Arc<dyn CrossEncoderClient>)` to avoid breaking Phase-5 callers — dw-providers pins tag v0.1.0, unaffected until they bump).
- [ ] docs/port-fidelity.md: update D-3 (CLOSED), add new rows (filters, rerankers, recipes, cross-encoder) + new deviations (community scope/SearchResults shape Phase 4; property_filters dead field; episode_mentions quirk; logprob availability if fallback taken).
- [ ] README phase table: Phase 2 done. `cargo update` + `bash ci/local_check.sh` green. PR → research, then tag `v0.2.0` after merge.

## Risks
| Risk | Mitigation |
|---|---|
| async-openai may not expose logprobs/logit_bias typed fields | Task 4 explicit fallback path + report |
| Var-length Cypher depth not parameterizable | inline sanitized integer (Task 6) |
| Breaking Chronicle::new signature for dw-providers | builder method, not signature change; dw pin v0.1.0 unaffected |
| episode_mentions upstream quirk (ASC sort) surprising | port bug-for-bug + ledger + test pinning |
