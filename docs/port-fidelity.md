# Port-Fidelity Ledger

Upstream pinned: **getzep/graphiti v0.29.1 @ 34f56e65e0fe2096132c8d16f3a1a4ac9300a5f6**

Statuses:
- **verbatim** — prompt text, field descriptions, and algorithmic logic ported with byte-identical intent; any translation is purely mechanical (Python → Rust syntax).
- **adapted** — structure preserved, implementation adjusted for Rust idioms or type-system differences; logic is equivalent.
- **deviation** — documented behavioral difference from upstream; see notes.
- **deferred** — feature acknowledged but out of Phase-1 scope; see DEFERRED section.

---

## Module Table

| Chronicle path | Upstream path | Status | Notes |
|---|---|---|---|
| `crates/chronicle-core/src/types/node.rs` | `graphiti_core/nodes.py` | adapted | `EntityNode`/`EntityNode` fields verbatim; Python `Optional[str]` → `Option<String>`, `datetime` → `DateTime<Utc>`. UUIDs are `String` not `uuid::Uuid` to match upstream's str storage. |
| `crates/chronicle-core/src/types/edge.rs` | `graphiti_core/edges.py` | adapted | `EntityEdge` + `EpisodicEdge`; bi-temporal fields (`valid_at`, `invalid_at`, `created_at`, `expired_at`) verbatim. Field order preserved. `fact_triple` EpisodeType variant omitted (deviation #7). |
| `crates/chronicle-core/src/types/episode.rs` | `graphiti_core/nodes.py` | adapted | `EpisodicNode` + `EpisodeType` (Message/Text/Json). `EpisodeType::Json` is the Rust spelling of upstream `text_json`. `fact_triple` variant omitted (deviation #7). |
| `crates/chronicle-core/src/embedder/mod.rs` | `graphiti_core/embedder/client.py` | adapted | `EmbedderClient` trait; `embed` / `embed_many` surface verbatim; `EmbedderError` mirrors upstream error taxonomy. |
| `crates/chronicle-core/src/llm/mod.rs` | `graphiti_core/llm_client/client.py`, `config.py`, `errors.py` | adapted | `LlmClient` trait, `LlmError` variants, `ModelSize` enum, `DEFAULT_TEMPERATURE`/`DEFAULT_MAX_TOKENS` verbatim from upstream constants. |
| `crates/chronicle-core/src/llm/config.rs` | `graphiti_core/llm_client/config.py` | verbatim | `LlmConfig` fields (`model`, `small_model`, `temperature`, `max_tokens`) and defaults carry upstream prose. |
| `crates/chronicle-core/src/llm/message.rs` | `graphiti_core/prompts/models.py` | verbatim | `Message`/`Role` one-to-one; `LlmResponse` wraps the response JSON value. |
| `crates/chronicle-core/src/llm/retry.rs` | `graphiti_core/llm_client/client.py` | adapted | `with_retry` reproduces tenacity retry logic; deviation #10 (backoff floor) and deviation #9 (RateLimit handling, EmptyResponse path) documented inline. |
| `crates/chronicle-core/src/helpers.rs` | `graphiti_core/helpers.py`, `utils/datetime_utils.py` | adapted | `SEMAPHORE_LIMIT` (20), `RELEVANT_SCHEMA_LIMIT`, `utc_now`, `normalize_name`; compile-time const vs upstream env-var (deviation #11). |
| `crates/chronicle-core/src/errors.rs` | (synthesized from multiple upstream error classes) | adapted | Unified `ChronicleError` wrapping driver, LLM, embedder, and parse errors. |
| `crates/chronicle-core/src/prompts/models.rs` | Multiple upstream Pydantic models | verbatim | `ExtractedEntity`, `ExtractedEdge`, `NodeDuplicate`, `EdgeDuplicate` — all `description` strings copied verbatim from upstream Pydantic `Field(description=...)`. |
| `crates/chronicle-core/src/prompts/snippets.rs` | `graphiti_core/prompts/snippets.py` | verbatim | `DO_NOT_ESCAPE_UNICODE`, `CURRENT_DATE_PROMPT` constant strings byte-identical. |
| `crates/chronicle-core/src/prompts/helpers.rs` | `graphiti_core/prompts/prompt_helpers.py` | adapted | `to_prompt_json` mirrors `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))`. |
| `crates/chronicle-core/src/prompts/extract_nodes.rs` | `graphiti_core/prompts/extract_nodes.py` | deviation | System/user prompt text verbatim for message/text/json/attributes variants. `entity_types` interpolated via `to_prompt_json` (JSON double-quoted) vs upstream Python `str(list)` single-quoted repr — unavoidable Rust/Python divergence (deviation #1). |
| `crates/chronicle-core/src/prompts/extract_edges.rs` | `graphiti_core/prompts/extract_edges.py` | deviation | Prompt text verbatim. `fact_types` context rendered via `to_prompt_json` vs Python str(dict) repr (deviation #1). |
| `crates/chronicle-core/src/prompts/dedupe_nodes.rs` | `graphiti_core/prompts/dedupe_nodes.py` | verbatim | Prompt text and structure byte-identical. |
| `crates/chronicle-core/src/prompts/dedupe_edges.rs` | `graphiti_core/prompts/dedupe_edges.py` | deviation | Prompt text verbatim. `context` dict rendered via `python_repr_str` (deviation #2). |
| `crates/chronicle-core/src/prompts/summarize_nodes.rs` | `graphiti_core/prompts/summarize_nodes.py` | verbatim | `summarize_context` prompt text byte-identical. |
| `crates/chronicle-core/src/pipeline/clients.rs` | `graphiti_core/graphiti_types.py::GraphitiClients` | adapted | `Clients` struct bundles `Arc<dyn GraphDriver>`, `Arc<dyn LlmClient>`, `Arc<dyn EmbedderClient>`, `Arc<Semaphore>`. |
| `crates/chronicle-core/src/pipeline/dedup_helpers.rs` | `graphiti_core/utils/maintenance/dedup_helpers.py` | verbatim | `normalize_name`, `shingles`, `minhash`, `jaccard`, `lsh_candidate_indices`, `high_entropy`, `entropy` — all algorithmic logic ported verbatim including constants (`MINHASH_PERMUTATIONS=32`, `MINHASH_BAND_SIZE=4`, `FUZZY_JACCARD_THRESHOLD=0.9`, `NAME_ENTROPY_THRESHOLD=1.5`). |
| `crates/chronicle-core/src/pipeline/temporal.rs` | `graphiti_core/utils/maintenance/edge_operations.py` (temporal section) | verbatim | `invalidate_edges` bi-temporal invalidation logic ported verbatim; interval-overlap predicate identical. |
| `crates/chronicle-core/src/pipeline/node_ops.rs` | `graphiti_core/utils/maintenance/node_operations.py` | adapted | `extract_nodes`, `resolve_extracted_nodes` ported. Candidate-input gathering is sequential vs upstream `semaphore_gather` (deviation #4). Node summary hydration uses single-node summarize_context (deviation #8). Multi-episode extraction path, `_collapse_exact_duplicate_extracted_nodes`, `node_episode_index_map` are Phase 2 (deferred). |
| `crates/chronicle-core/src/pipeline/edge_ops.rs` | `graphiti_core/utils/maintenance/edge_operations.py` | adapted | `extract_edges`, `resolve_extracted_edges`, `resolve_extracted_edge`, `hydrate_node_summaries`. Edge candidate re-ranking via `EDGE_HYBRID_SEARCH_RRF + SearchFilters(edge_uuids)` now ported faithfully — **D-3 CLOSED** (upstream 392-405). |
| `crates/chronicle-core/src/pipeline/add_episode.rs` | `graphiti_core/graphiti.py::add_episode`, `_extract_and_resolve_edges`, `_process_episode_data`, `bulk_utils.py::resolve_edge_pointers` | adapted | Full single-episode core loop. Persist = 4 sequential driver calls (deviation #5). Community updates, sagas, excluded entity types, multi-episode bulk path are Phase 2+. |
| `crates/chronicle-core/src/pipeline/mod.rs` | `graphiti_core/utils/maintenance/` | adapted | Module re-exports; no direct behavior. |
| `crates/chronicle-core/src/search/rrf.rs` | `graphiti_core/search/search_utils.py::rrf` (lines ~1780-1795) | verbatim | RRF rank-fusion formula `1 / (rank + rank_const)` with upstream default `rank_const=1` (NOT the IR-conventional 60); inclusive `>=` min-score filter verbatim; first-seen tie order replicates Python dict insertion-order semantics. |
| `crates/chronicle-core/src/search/config.rs` | `graphiti_core/search/search_config.py` | adapted | `EdgeSearchConfig`, `NodeSearchConfig`, `EpisodeSearchConfig` + full reranker enums (Edge/Node: Rrf/Mmr/NodeDistance/EpisodeMentions/CrossEncoder; Episode: Rrf/CrossEncoder) and method enums (incl. `BreadthFirstSearch`). Constants (`DEFAULT_SEARCH_LIMIT=10`, `DEFAULT_MIN_SCORE=0.6`, `DEFAULT_MMR_LAMBDA=0.5`, `MAX_SEARCH_DEPTH=3`) verbatim. Phase 4: `CommunitySearchMethod` (cosine+bm25, NO bfs), `CommunityReranker` (Rrf/Mmr/CrossEncoder), `CommunitySearchConfig` (`bfs_max_depth` retained for parity but unused), `SearchConfig.community_config` (D-16 resolved). |
| `crates/chronicle-core/src/search/filters.rs` | `graphiti_core/search/search_filters.py` | adapted | `SearchFilters`/`DateFilter`/`ComparisonOperator`/`PropertyFilter`. Date filters are `Vec<Vec<DateFilter>>` (OUTER OR of INNER AND). `property_filters` ported but never applied in WHERE construction (D-17 — upstream dead field at this commit). |
| `crates/chronicle-core/src/search/recipes.rs` | `graphiti_core/search/search_config_recipes.py` | adapted | Full recipe set (edge/node/combined RRF/MMR/NodeDistance/EpisodeMentions/CrossEncoder). `*_CROSS_ENCODER` use bm25+sim+bfs methods; EDGE/NODE_CROSS_ENCODER limit 10; COMBINED_MMR uses mmr_lambda=1.0 for edge/node AND community (verified upstream). Phase 4: COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER(limit 3)} ported (bm25+cosine methods); the three COMBINED_* recipes now carry `community_config` (D-16 resolved). |
| `crates/chronicle-core/src/search/edge_search.rs` | `graphiti_core/search/search.py::edge_search` + `search_utils.py` | adapted | All 5 rerankers (rrf/mmr/node_distance/episode_mentions/cross_encoder), BFS self-seed, 2*limit candidate fetch, `SearchFilters` wiring complete. node_distance edge-scope expands edge-uuid groups in returned node order; score-list mismatch carried bug-for-bug (D-21). Missing center → `InvalidInput`. |
| `crates/chronicle-core/src/search/node_search.rs` | `graphiti_core/search/search.py::node_search` | adapted | Node rerankers per R4 (cross_encoder ranks `node.name`, no pre-truncate; episode_mentions DB-count via `episode_mention_counts`; node_distance rrf-presort). BFS self-seed from node uuids. |
| `crates/chronicle-core/src/search/episode_search.rs` | `graphiti_core/search/search.py::episode_search` | adapted | bm25-only fulltext; rrf + cross_encoder (rrf-presort → take limit → rank `episode.content`). |
| `crates/chronicle-core/src/search/rerank.rs` | `graphiti_core/search/search_utils.py` | verbatim | `maximal_marginal_relevance`, `node_distance_rerank`, `episode_mentions_rerank` (D-15 quirk). |
| `crates/chronicle-core/src/search/search.rs` | `graphiti_core/search/search.py::search` | adapted | Top-level multi-scope entrypoint; empty-query guard, single embed decision (now includes community cosine/mmr), 4 parallel scopes via `tokio::join!` (edge/node/episode/community). `SearchResults` carries community fields (D-16 resolved). |
| `crates/chronicle-core/src/search/community_search.rs` | `graphiti_core/search/search.py::community_search` | adapted | Community scope. UPSTREAM QUIRK: ALWAYS runs both fulltext + similarity regardless of `search_methods` list (replicated bug-for-bug). Rerankers rrf/mmr/cross_encoder (cross_encoder ranks `community.name`, no pre-truncate; missing encoder → `InvalidInput`). 2*limit candidates, parallel, slice `[:limit]`. |
| `crates/chronicle-core/src/cross_encoder/mod.rs` | `graphiti_core/cross_encoder/client.py` | adapted | `CrossEncoderClient` trait (`rank(query, passages) -> desc-sorted (passage, score)`). |
| `crates/chronicle-llm-openai/src/reranker.rs` | `graphiti_core/cross_encoder/openai_reranker_client.py` | adapted | OpenAI boolean-logprob reranker (gpt-4.1-nano, temperature 0, logit_bias True/False token ids, top_logprobs 2; score = exp(lp) if "true" else 1-exp(lp), desc sort). Uses deprecated `max_tokens` request field (D-20). |
| `crates/chronicle-core/src/search/edge_search.rs` (BFS) / `crates/chronicle-driver-neo4j/src/queries.rs` (BFS Cypher) | `graphiti_core/search/search_utils.py::*_bfs` | adapted | Edge-BFS undirected re-join (`(n)-[e]-(m)`) carries upstream duplicate-row behaviour bug-for-bug (D-19). Var-length depth inlined as sanitized integer (not a `$param`). |
| `crates/chronicle-core/src/chronicle.rs` | `graphiti_core/graphiti.py::Graphiti` | adapted | `Chronicle` facade: `add_episode`, `retrieve_episodes`, `search` (config-direct), `search_with_center` (R13 recipe-routing), `search_` (R13 advanced multi-scope, default COMBINED_HYBRID_SEARCH_CROSS_ENCODER), `with_cross_encoder` builder, `build_indices_and_constraints`. `build_communities`, batch operations deferred (Phase 4). |
| `crates/chronicle-driver-neo4j/src/lib.rs` | `graphiti_core/driver/neo4j_driver.py` + operations files | adapted | Full `GraphDriver` impl over `neo4rs`. Cross-call atomicity gap documented (deviation #5). |
| `crates/chronicle-driver-neo4j/src/queries.rs` | `graphiti_core/driver/neo4j/` query builders + `graph_data_operations.py::retrieve_episodes` + `search_utils.py::fulltext_query` | adapted | Cypher query builders. Lucene OR-precedence quirk reproduced bug-for-bug (deviation #6). `validate_group_id` pattern `^[a-zA-Z0-9_-]+$` verbatim from upstream. `MAX_QUERY_LENGTH=128` verbatim. Retrieve episodes tie order is backend-dependent (deviation #12). |
| `crates/chronicle-driver-neo4j/src/convert.rs` | (Neo4j ↔ domain type conversions, no direct upstream equivalent) | adapted | Bolt value ↔ Rust type bridge; no upstream analog. |
| `crates/chronicle-llm-openai/src/llm.rs` | `graphiti_core/llm_client/openai_generic_client.py`, `openai_base_client.py` | deviation | `DEFAULT_MODEL="gpt-4.1-mini"`, `DEFAULT_SMALL_MODEL="gpt-4.1-nano"`, temperature=0, max_tokens=16384 verbatim. `EmptyResponse` non-retryable (deviation #9). No error-context message appended on retry (deviation #9). RateLimit retried per base tenacity policy (deviation #9). |
| `crates/chronicle-llm-openai/src/embedder.rs` | `graphiti_core/embedder/openai.py` | adapted | `OpenAiEmbedder`; `DEFAULT_EMBEDDING_MODEL="text-embedding-3-small"`. `EMBEDDING_DIM` is compile-time const (deviation #11). |
| `crates/chronicle-testkit/src/fake_driver.rs` | (test fixture, no upstream equivalent) | adapted | In-memory `FakeDriver`; deterministic uuid tie-breaks in similarity sorts (deviation #13, test-only). |
| `crates/chronicle-testkit/src/mock_llm.rs` | (test fixture, no upstream equivalent) | adapted | `MockLlm` replay queue. |
| `crates/chronicle-testkit/src/mock_embedder.rs` | (test fixture, no upstream equivalent) | adapted | `MockEmbedder` keyed on string hash. |

---

## DEVIATIONS

These are all review-verified behavioral differences from upstream getzep/graphiti v0.29.1.

### D-1: Python str(dict/list) repr vs JSON rendering in four prompt functions

Affected: `extract_nodes.rs` (`extract_message`, `extract_text`, `extract_json`, `extract_attributes`) and `extract_edges.rs` (`extract_edges`).

Upstream interpolates `context['entity_types']` (a Python `list[dict]`) directly into an f-string, which invokes `str(list)` — Python's repr form with single-quoted keys, e.g. `[{'name': 'Person'}]`. True byte-identical reproduction is impossible in Rust without a complete Python repr emulator. These prompts render via `to_prompt_json` (JSON, double-quoted keys). The surrounding prompt text is byte-identical. The LLM sees semantically equivalent content; key quoting style differs. Documented inline in `extract_nodes.rs`.

### D-2: dedupe_edges context: Python repr reproduced including CPython quote-switch and control-char escapes

Affected: `edge_ops.rs::python_repr_str`.

Upstream `dedupe_edges.py` constructs the LLM context as `str({"edge": ...})` — Python's dict repr. This crate implements `python_repr_str` to reproduce CPython's repr behavior: default single-quoted strings, switch to double quotes when the string contains an apostrophe but no double quote, and escape control characters (`\n`, `\r`, `\t`, `\0`, `\x85` NEL, `\xa0` NBSP, `\x7f` DEL) in the same way CPython does. This is a deliberate bug-for-bug replication to keep prompt input identical for existing graphs. Tested in `edge_ops::tests::python_repr_str_*`.

### D-3: Edge candidate re-ranking — **CLOSED (Phase 2)**

Status: **CLOSED**. Upstream `resolve_extracted_edges` (edge_operations.py:392-405) re-ranks the node-pair candidate pool via `EDGE_HYBRID_SEARCH_RRF` filtered to the pool's UUIDs (`SearchFilters(edge_uuids=[...])`) before resolution. Phase-1 used the pool directly (same SET, possibly different order). Phase-2 ports the faithful path: `get_edges_between_nodes` is the UUID *source*; a group-scoped RRF hybrid edge search on `extracted_edge.fact` with `SearchFilters { edge_uuids: Some(pool_uuids), .. }` produces the ranked `related_edges`. An empty pool short-circuits to no candidates (no search issued) — observably identical to upstream's empty-`edge_uuids` search (both yield no related edges), recorded here as a micro-deviation. The invalidation-candidate search keeps explicit empty `SearchFilters` (upstream `SearchFilters()`), minus any UUID already in the related pool. The LLM call sequence is unchanged; only embedder/search call counts increase (one extra edge search per non-empty pool). e2e `add_episode_e2e` (both tests) green — they assert LLM call counts only.

### D-4: Candidate-input gathering: sequential vs upstream semaphore_gather (performance only)

Upstream fans out node/edge candidate queries concurrently under a semaphore. Phase-1 runs them sequentially. No behavioral difference on results; throughput is lower for large graphs. Phase-2: replace loops with `futures::future::join_all` under the shared semaphore.

### D-5: Persist = 4 sequential driver calls vs upstream single transaction (atomicity gap)

Upstream `add_episode` wraps `save_episode`, `save_entity_nodes`, `save_entity_edges`, and `save_episodic_edges` in a single Neo4j transaction. Chronicle executes these as four independent transactions. A crash between calls can leave a partially persisted episode. Phase-2 improvement: add a `save_all` transactional operation to `GraphDriver`.

### D-6: Lucene OR-precedence quirk in multi-group scoping reproduced bug-for-bug

Upstream `fulltext_query` builds multi-group Lucene queries as `(q) (group1 OR group2)` rather than `(q) AND (group1 OR group2)`. Because Lucene's default operator is OR, this widens the match beyond the intended groups. Chronicle reproduces this behavior identically so that query results are consistent with upstream-generated graphs. `group_id` validation pattern `^[a-zA-Z0-9_-]+$` ported verbatim from `helpers.py`.

### D-7: fact_triple EpisodeType variant omitted

Upstream `EpisodeType` includes a `fact_triple` variant used by older graph ingest pipelines. Phase-1 omits it. Graphs written with `fact_triple` episodes can still be read (the variant will deserialize as an unknown string), but it cannot be written via this crate. Read-compatibility gap with older upstream graphs. To be added in a future patch.

### D-8: Node summary hydration: single-node summarize_context vs upstream batched extract_summaries_batch

Upstream `extract_attributes_from_nodes` calls `extract_summaries_batch` (operating on `SummarizedEntities`, a batch of nodes). Phase-1 `hydrate_node_summaries` calls `summarize_context` once per node. Semantics are equivalent for single-node summaries; batch efficiency and any cross-node summary coherence in the batch prompt are lost. Phase-2: port `extract_summaries_batch`.

### D-9: OpenAI client retry policy diverges from upstream in three sub-cases

Three related sub-cases, all in `chronicle-llm-openai/src/llm.rs`:

1. **EmptyResponse non-retryable**: upstream `openai_generic_client.py` falls through to a retryable `json.JSONDecodeError` on an empty response string; this crate maps it to `LlmError::EmptyResponse` which is non-retryable. Downstream impact is minor (empty responses are usually terminal).
2. **No error-context message on retry**: upstream `openai_generic_client.py` appends a recovery hint (`"Previous response was: ..."`) to the user message on each retry attempt. This crate does not append recovery hints; the same prompt is retried verbatim.
3. **RateLimit retried**: upstream's concrete `openai_generic_client.py` does NOT retry `RateLimitError` (it is not in the concrete client's `retryable_exceptions`). Upstream's base `tenacity` policy in `LlmClient` does retry it. This crate follows the base-class policy and retries `RateLimit` — a deliberate decision to keep the retry logic uniform; callers with strict rate-budget requirements should handle `RateLimit` above this layer.

### D-10: Retry backoff distribution approximates tenacity wait_random_exponential

Upstream uses `tenacity.wait_random_exponential(multiplier=10, min=5, max=120)` with `stop_after_attempt(4)`. The `with_retry` implementation samples approximately the same distribution but draws below the 5 s floor collapse to exactly 5 s (a point-mass at the floor) rather than tenacity's continuous lower bound. Attempt count (4) and overall jitter character are equivalent; individual waits may differ.

### D-11: SEMAPHORE_LIMIT and EMBEDDING_DIM as compile-time consts vs upstream env-vars

Upstream reads `SEMAPHORE_LIMIT` from `SEMAPHORE_LIMIT` env var and embedding dimension from runtime model config. This crate defines both as compile-time constants (`SEMAPHORE_LIMIT=20` in `helpers.rs`; `DEFAULT_EMBEDDING_DIM=1024` in `chronicle-core::embedder`, matching upstream's env default). Callers can override the concurrency limit at `Chronicle::new` call time; embedding dimension override requires a config field on `OpenAiEmbedderConfig`.

### D-12: retrieve_episodes tie order is backend-dependent

Upstream `retrieve_episodes` returns episodes in `created_at DESC` order; ties within the same timestamp are not given a secondary sort key. The Rust port passes the same `ORDER BY created_at DESC` query to Neo4j; ties may be broken differently by different Neo4j versions or query planners. No secondary sort added, matching upstream behavior exactly.

### D-13: Deterministic uuid tie-breaks in FakeDriver similarity sorts (test-only)

`chronicle-testkit/src/fake_driver.rs` sorts similarity results by score DESC, breaking ties by `uuid` ASC for determinism in tests. Upstream has no in-memory driver equivalent; this divergence is test-only and does not affect production code paths.

### D-14: previous_episodes context: timestamp always String (valid_at non-optional)

Upstream `_build_episode_context` in `graphiti.py` formats `episode.valid_at` which is nullable (`Optional[datetime]`); when `None`, it formats as `"None"`. `EpisodicNode.valid_at` in this crate is non-optional (`DateTime<Utc>`), so the timestamp is always a formatted string. Episodes from older upstream graphs with `null` valid_at cannot be ingested via this crate without a pre-migration step.

### D-15: episode_mentions reranker — ASC sort + inf-retain quirk (bug-for-bug)

`episode_mentions_rerank` (`crates/chronicle-core/src/search/rerank.rs`) ports `episode_mentions_reranker` (`graphiti_core/search/search_utils.py` @ 34f56e65) exactly, including two counterintuitive upstream behaviours ("episode_mentions ASC quirk"):

1. **Ascending sort by mention count** — `sorted_uuids.sort(key=lambda u: scores[u])` orders nodes with *fewer* MENTIONS first. This is the opposite of the intuitive "most-mentioned is most relevant" ordering, but is exactly what upstream does, so it is reproduced.
2. **Unmentioned nodes retained at the end** — nodes with no MENTIONS get `float('inf')`, and the literal filter `scores[uuid] >= min_score` keeps them (`inf >= min_score` is always true). The returned scores are the raw counts / `inf`, not inverted. Reproduced literally.

Both `node_distance_rerank` and `maximal_marginal_relevance` in the same module are faithful ports with no behavioural deviation (the only adaptation is `f32` embeddings widened to `f64` before arithmetic, matching upstream numpy float64). `normalize_l2`'s zero-vector guard (`np.where(norm == 0, arr, arr / norm)` → zero vector returned unchanged) is reproduced.

### D-16: Community scope + SearchResults shape (RESOLVED in Phase 4)

RESOLVED. The community scope landed in Phase 4 Task 4. `SearchConfig.community_config`, `SearchResults.communities` + `community_reranker_scores`, `CommunitySearchConfig`/`CommunitySearchMethod`/`CommunityReranker`, `community_search`, the COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER(limit 3)} recipes, and `community_config` on the three COMBINED_* recipes are all present and match upstream. `SearchResults` is now shape-equivalent to upstream. Two upstream quirks are replicated bug-for-bug: (1) `community_search` ALWAYS runs both fulltext + similarity regardless of the `search_methods` list; (2) COMBINED_MMR sets community `mmr_lambda=1.0` (same as edge/node).

### D-17: `property_filters` ported but never applied (upstream dead field)

`SearchFilters::property_filters` exists for recipe/struct-shape parity but is NOT consumed in any WHERE-clause construction — because upstream at 34f56e65 also never reads it in its Neo4j WHERE builders (a dead field upstream at this commit). Carried as a field-shape match; flagged so a future upstream activation is noticed.

### D-18: Date-param collision — **upstream bug FIXED in this port**

This is the one place the port deliberately *diverges to be correct* rather than bug-for-bug. Upstream's edge/node date-filter WHERE builders name bound params by the INNER (per-AND-group) index only (`$valid_at_{j}`), which COLLIDES across OR-groups (group 2's `valid_at_0` overwrites group 1's) and across the four date fields in the same call. Upstream's fixtures never exercise multi-group OR-of-ANDs windows, so the bug is latent there. The R10 requirement (correct OR-of-ANDs date windows) forces a fix: every value-bearing date condition gets a GLOBALLY-unique `$p{n}` name from one monotonic counter shared across all four fields. The emitted Cypher SHAPE is identical (`((e.valid_at >= $p0 AND e.valid_at < $p1) OR (...))`); only the param identifiers are made unique. Documented inline in `queries.rs`.

### D-19: Edge-BFS undirected re-join — duplicate rows carried bug-for-bug

`edge_bfs` expands a directed var-length path, then re-matches each relationship UNDIRECTED (`(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)`). The undirected re-join can surface the same edge from both endpoint orientations, producing duplicate result rows for the same edge uuid — exactly as upstream does. Reproduced bug-for-bug so result multiplicity matches upstream-generated graphs (downstream RRF/dedup absorbs the duplicates).

### D-20: OpenAI reranker uses deprecated `max_tokens` request field

`chronicle-llm-openai/src/reranker.rs` sends `max_tokens: 1` (matching upstream `openai_reranker_client.py`). OpenAI has since deprecated `max_tokens` in favour of `max_completion_tokens` for chat completions, but upstream still emits `max_tokens` at 34f56e65, so the port mirrors it for byte-faithful request shape. If the OpenAI endpoint hard-rejects the field in future, switch to `max_completion_tokens` (semantics identical for the 1-token logprob classifier).

### D-21: node_distance edge-scope score-list mismatch carried bug-for-bug

In the edge-scope `node_distance` reranker, upstream groups edge uuids by `source_node_uuid`, reranks the *source nodes* via `node_distance_reranker`, then expands back to edge uuids in returned-node order — but the returned score list is the per-*node* distance scores, not per-edge, so the edge-score alignment after expansion does not 1:1 track the expanded edge uuids. The port reproduces this expansion + score-list behaviour exactly (bug-for-bug) rather than re-deriving per-edge scores. The final `[:limit]` slice and downstream consumers tolerate the mismatch identically to upstream.

---

## DEFERRED

Features acknowledged but out of Phase-1 scope. Listed with target phase.

| Feature | Target | Notes |
|---|---|---|
| Reflexion (self-critique loop) | absent in upstream v0.29.1 | Not present in pinned upstream; not applicable |
| Communities / saga / bulk ingest | Phase 4 | `build_communities`, `build_community_for_node`, community edge/node types, `community_config`, COMMUNITY_* recipes, `SearchResults` community fields (D-16) |
| BFS traversal in edge/node search | ✅ Done (Phase 2) | `EdgeSearchMethod::BreadthFirstSearch` + node BFS implemented (self-seed + Cypher); see D-19 |
| MMR / NodeDistance / EpisodeMentions rerankers | ✅ Done (Phase 2) | Ported in `search/rerank.rs` (D-15, D-21) + dispatched in edge/node scopes |
| CrossEncoder reranker | ✅ Done (Phase 2) | `CrossEncoderClient` trait + OpenAI logprob impl (D-20) + scope wiring |
| `SearchFilters` wiring in edge/node search | ✅ Done (Phase 2) | `filters.rs` (D-17) wired into all scopes + Neo4j WHERE builders (D-18) |
| `SearchFilters(edge_uuids)` in resolve_extracted_edges | ✅ Done (Phase 2) | Edge candidate re-ranking — D-3 CLOSED |
| `NodeSearchConfig`, `EpisodeSearchConfig` | ✅ Done (Phase 2) | Full config + reranker enums |
| Top-level `search_()` / multi-scope facade | ✅ Done (Phase 2) | `Chronicle::search_`, `search_with_center`, `with_cross_encoder` (R13) |
| `CommunitySearchConfig` + community search scope | ✅ Done (Phase 4) | `CommunitySearchConfig`/`CommunitySearchMethod`/`CommunityReranker`, `community_search`, COMMUNITY_* + COMBINED_* community recipes, `SearchResults` community fields — D-16 RESOLVED |
| Multi-episode extraction path | Phase 3 | `_process_episode_data` bulk path; `_collapse_exact_duplicate_extracted_nodes`; `node_episode_index_map` |
| Entity/edge-type registries + attribute extraction | Phase 3 | `entity_types` and `edge_types` Pydantic registry; `extract_attributes_from_nodes` batch path |
| `extract_summaries_batch` / `SummarizedEntities` | Phase 3 | Replace per-node hydration (D-8) |
| `save_all` transactional driver op | Phase 3 | Atomicity gap (D-5) |
| `semaphore_gather` equivalent fan-out | Phase 3 | Performance improvement for D-4 |
| `fact_triple` EpisodeType variant | Near-term patch | Read-compat gap (D-7) |
| Kuzu embedded driver | Phase 3 | `chronicle-driver-kuzu` crate not yet created |
| FalkorDB driver | v1.x | Planned post-v1 |
| Neptune driver | skipped | Out of scope |
| Upstream eval/ harness | not applicable | Python pytest-based evaluation suite |
| MCP / FastAPI servers | not applicable | Python server layer; Greentic integration via `greentic-dw-providers` |
