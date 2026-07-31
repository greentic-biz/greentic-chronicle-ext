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
| `crates/chronicle-core/src/types/community.rs` | `graphiti_core/nodes.py::CommunityNode` + `edges.py::CommunityEdge` | adapted | `CommunityNode` (uuid/name/group_id/labels `["Community"]`/created_at/name_embedding/summary default `""`) + `CommunityEdge` HAS_MEMBER (Community→Entity|Community). created_at-only (no bi-temporal). Field order + defaults verbatim (R1). |
| `crates/chronicle-core/src/types/saga.rs` | `graphiti_core/nodes.py::SagaNode` + saga edges | adapted | `SagaNode` (summary default `""`, first/last_episode_uuid, last_summarized_at, last_summarized_episode_valid_at) + `HasEpisodeEdge` (Saga→Episodic) + `NextEpisodeEdge` (Episodic→Episodic), each `{uuid,group_id,created_at}` (R9). |
| `crates/chronicle-core/src/pipeline/community_ops.rs` | `graphiti_core/utils/maintenance/community_operations.py` | deviation | `label_propagation` (R2, exact tuple-sort tie-break + `max(candidate,curr)` fallback) **plus a 1000-iteration safety cap (D-22, the one Phase-4 algorithm deviation)**; `get_community_clusters`, `build_community` pairwise-reduce (R3, `truncate_at_sentence` + `MAX_SUMMARY_CHARS=1000`), `build_community_edges`, `build_communities` (concurrency 10), `remove_communities`, `determine_entity_community` + `update_community` (R4, neighbor-vote only). `update_communities` is sequential per node (D-23). |
| `crates/chronicle-core/src/prompts/summarize_nodes.rs` (community) | `graphiti_core/prompts/summarize_nodes.py` | verbatim | `summarize_pair` + `summary_description` prompt fns + context structs VERBATIM (R3): summarize_pair system "You are a helpful assistant that combines summaries into a single dense factual summary" → `Summary{summary}`; summary_description system "...describes provided contents in a single sentence" → `SummaryDescription{description}`. |
| `crates/chronicle-core/src/pipeline/saga.rs` | `graphiti_core/graphiti.py` saga helpers + `summarize_sagas.py` | adapted | Saga threading (get-or-create by (name,group_id), prev-episode resolution, NEXT_EPISODE/HAS_EPISODE wiring, first/last pointer update — R9); `summarize_saga` two-watermark logic (`last_summarized_at` wall-clock filter + `last_summarized_episode_valid_at` episode-time). UTF-8 boundary truncation on summary input (D-24). |
| `crates/chronicle-core/src/prompts/summarize_sagas.rs` | `graphiti_core/prompts/summarize_sagas.py` | verbatim | `summarize_saga` prompt VERBATIM (R9): system "You extract durable knowledge from message threads..."; context `{saga_name, existing_summary, episodes}`; response `SagaSummary{summary}`. |
| `crates/chronicle-core/src/pipeline/bulk.rs` | `graphiti_core/utils/bulk_utils.py` + `graphiti.py::add_episode_bulk` | adapted | `RawEpisode`, `AddBulkEpisodeResults`, `_build_directed_uuid_map` (directed union-find, iterative path compression) + `compress_uuid_map` (undirected union-find, smallest-uuid-wins), `resolve_edge_pointers`, `dedupe_nodes_bulk`/`dedupe_edges_bulk`, `extract_nodes_and_edges_bulk`, `retrieve_previous_episodes_bulk`, `add_nodes_and_edges_bulk` (now transactional via `save_all` — D-5 CLOSED). `CHUNK_SIZE=10` const (caller-chunked). Communities never updated in bulk (upstream — `communities=[]`). |
| `crates/chronicle-core/src/pipeline/maintenance.rs` | `graphiti_core/graphiti.py` (`add_triplet`, `remove_episode`, `get_nodes_and_edges_by_episode`) | adapted | `add_triplet` (R7, synthetic-episode full dedup/invalidation, save via bulk `save_all`, no episodic/community), `remove_episode` (R5 cascade: primary-source edges + single-mention nodes only, DETACH episode), `get_nodes_and_edges_by_episode` (R10, trivial fan-out). |
| `crates/chronicle-core/src/driver/mod.rs` (Phase-4 ops) | `community_operations.py`, saga helpers, `bulk_utils.py` | adapted | New driver traits: `CommunityOps` (save/get community nodes+edges, fulltext+similarity+embeddings search, clusters projection, member/neighbor-vote lookups, remove_communities — R1/R2/R3/R4/R8), `SagaOps` (save saga node + HAS_EPISODE/NEXT_EPISODE, get-or-create, prev-episode, episode-contents — R9), `BulkSaveOps::save_all` (transactional group-save — D-5 CLOSED). |
| `crates/chronicle-core/src/cross_encoder/mod.rs` | `graphiti_core/cross_encoder/client.py` | adapted | `CrossEncoderClient` trait (`rank(query, passages) -> desc-sorted (passage, score)`). |
| `crates/chronicle-llm-openai/src/reranker.rs` | `graphiti_core/cross_encoder/openai_reranker_client.py` | adapted | OpenAI boolean-logprob reranker (gpt-4.1-nano, temperature 0, logit_bias True/False token ids, top_logprobs 2; score = exp(lp) if "true" else 1-exp(lp), desc sort). Uses deprecated `max_tokens` request field (D-20). |
| `crates/chronicle-core/src/search/edge_search.rs` (BFS) / `crates/chronicle-driver-neo4j/src/queries.rs` (BFS Cypher) | `graphiti_core/search/search_utils.py::*_bfs` | adapted | Edge-BFS undirected re-join (`(n)-[e]-(m)`) carries upstream duplicate-row behaviour bug-for-bug (D-19). Var-length depth inlined as sanitized integer (not a `$param`). |
| `crates/chronicle-core/src/chronicle.rs` | `graphiti_core/graphiti.py::Graphiti` | adapted | `Chronicle` facade: `add_episode` (+ optional saga / `update_communities` params, additive — see dw-providers note below), `retrieve_episodes`, `search` (config-direct), `search_with_center` (R13), `search_` (R13 advanced multi-scope), `with_cross_encoder`, `build_indices_and_constraints`. Phase 4 ADDS `add_episode_bulk`, `add_triplet`, `remove_episode`, `build_communities`, `get_nodes_and_edges_by_episode`, `summarize_saga` — full Graphiti-core parity. |
| `crates/chronicle-driver-neo4j/src/lib.rs` | `graphiti_core/driver/neo4j_driver.py` + operations files | adapted | Full `GraphDriver` impl over `neo4rs` incl. Phase-4 `CommunityOps`/`SagaOps`/`BulkSaveOps`. `save_all` runs episodes+nodes+entity-edges+episodic-edges in ONE `start_txn → run all → commit` (rollback on error) — **cross-call atomicity gap (D-5) CLOSED**. |
| `crates/chronicle-driver-neo4j/src/queries.rs` | `graphiti_core/driver/neo4j/` query builders + `graph_data_operations.py::retrieve_episodes` + `search_utils.py::fulltext_query` | adapted | Cypher query builders. Lucene OR-precedence quirk reproduced bug-for-bug (deviation #6). `validate_group_id` pattern `^[a-zA-Z0-9_-]+$` verbatim from upstream. `MAX_QUERY_LENGTH=128` verbatim. Retrieve episodes tie order is backend-dependent (deviation #12). |
| `crates/chronicle-driver-neo4j/src/convert.rs` | (Neo4j ↔ domain type conversions, no direct upstream equivalent) | adapted | Bolt value ↔ Rust type bridge; no upstream analog. |
| `crates/chronicle-driver-surreal/src/lib.rs` (node/episode/community/saga ops) | `graphiti_core/driver/` (SurrealQL is a from-scratch query layer, NOT a Cypher port) | adapted | Full `EntityNodeOps`/`EpisodeOps`/`CommunityOps`/`SagaOps` save/get/get_by_uuids/by_group_ids/delete over embedded SurrealDB (`UPSERT`/`SELECT`/`DELETE`/`RELATE`). Separate tables per node kind (`entity`/`episodic`/`community`/`saga`); uuid is the record id for O(1) lookup. Correctness oracle = behavior parity with FakeDriver/Neo4j (chronicle-testkit + `surreal_e2e` gate), not query-text fidelity. Deviations D-26/D-31/D-32. |
| `crates/chronicle-driver-surreal/src/lib.rs` (search ops) | `graphiti_core/search/search_utils.py::*` | adapted | `edge`/`node`/`community` fulltext (BM25 `@N@` + `search::score`, one FTS index per field), `*_similarity_search` (HNSW `<\|K,EF\|>` + `vector::distance::knn`, **over-fetch ×3 + min-score post-filter in Rust** — D-26), `node_bfs_search`/`edge_bfs_search` (**iterative multi-query BFS** + depth cap 5 — D-27/D-28), `nodes_connected_to_center` (two-query directed-union adjacency — D-32), `episode_mention_counts`, `get_embeddings_for_{nodes,edges,communities}`, `SearchFilters` WHERE builder (parameterized via `.bind`). |
| `crates/chronicle-driver-surreal/src/lib.rs` (bulk + maintenance) | `bulk_utils.py` + `community_operations.py` + saga helpers | adapted | `BulkSaveOps::save_all` atomic transaction via the `db.begin()`/`tx.commit()`/`tx.cancel()` handle API (rollback on first statement error — D-5 parity); `get_community_clusters` aggregation; `community_of_member`/`neighbor_communities`; `detach_entities`/`detach_episode` (explicit edge-detach delete, no graph cascade — D-31); saga `get_by_name`/`previous_episode`/`episode_contents`. |
| `crates/chronicle-driver-surreal/src/schema.rs` | `graphiti_core/driver/` index/constraint DDL | adapted | Idempotent SurrealQL DDL (`DEFINE … IF NOT EXISTS`/`OVERWRITE`): per-node-kind tables + `relates_to`/`mentions`/`has_member`/`has_episode`/`next_episode` RELATION tables; HNSW vector indexes `TYPE F32 DIST COSINE` (D-29); BM25 FTS analyzer + per-field SEARCH indexes; group_id indexes. `build_indices_and_constraints` re-runs DDL idempotently. |
| `crates/chronicle-driver-surreal/src/convert.rs` | (SurrealDB ↔ domain type conversions, no direct upstream equivalent) | adapted | `chrono::DateTime<Utc>` ↔ `surrealdb::types::Datetime` at the boundary (never bind chrono directly — D-30); `Vec<f32>` ↔ native `array<float>`; `serde_json::Value` attrs ↔ `object`. Row structs with Serialize/Deserialize; uuid↔record-id mapping. Datetime-is-datetime roundtrip guard test. |
| `crates/chronicle-driver-falkor/src/lib.rs` (persistence + search + bulk/maintenance) | `graphiti_core/driver/neo4j/` (Cypher-dialect adaptation of the Neo4j driver) | adapted | Full `GraphDriver` supertrait over FalkorDB (`falkordb` 0.2.1, openCypher on a Redis module). Persistence via per-item `MERGE … SET`; search via `db.idx.vector.query*` (distance→similarity post-filter, over-fetch ×3 — D-33) + `db.idx.fulltext.query{Nodes,Relationships}` (relationship fulltext via DDL index — D-37); BFS `*1..N` with inline-clamped depth; `get_community_clusters` `count(e)` aggregation; saga threading by epoch-int ordering. `save_all` best-effort sequential (no atomic batch — D-34). Correctness oracle = behaviour parity (live integration + search suites + `falkor_e2e` gate). Deviations D-33–D-39. |
| `crates/chronicle-driver-falkor/src/schema.rs` | `graphiti_core/driver/` index DDL | adapted | Idempotent FalkorDB DDL: range indices (uuid+group_id per label), HNSW vector indices (`CREATE VECTOR INDEX … OPTIONS {dimension, similarityFunction:'cosine'}`, node + relationship forms), node fulltext (`db.idx.fulltext.createNodeIndex`), relationship fulltext (`CREATE FULLTEXT INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact)` — D-37). `already_exists` swallows the no-`IF NOT EXISTS` re-create error class. |
| `crates/chronicle-driver-falkor/src/convert.rs` | (FalkorDB ↔ domain type conversions, no direct upstream equivalent) | adapted | Cypher-literal encoders (`lit_str`/`lit_int`/`lit_string_list`/`lit_vecf32`/`lit_attrs`) — every dynamic value escaped, no raw interpolation (D-38); datetime ↔ epoch-millis `i64` (D-35); attrs ↔ JSON-string `attrs_json` (D-36); `FalkorValue` (`Node`/`Edge`/`Vec32`) extraction. |
| `crates/chronicle-driver-falkor/src/filters.rs` | `graphiti_core/search/search_filters.py` | adapted | `SearchFilters` → Cypher WHERE fragments (edge_types / edge_uuids / node_labels / date OR-of-ANDs as epoch-int comparisons), rendered as escaped literals; threaded into all base searches + BFS. |
| `crates/chronicle-llm-openai/src/llm.rs` | `graphiti_core/llm_client/openai_generic_client.py`, `openai_base_client.py` | deviation | `DEFAULT_MODEL="gpt-4.1-mini"`, `DEFAULT_SMALL_MODEL="gpt-4.1-nano"`, temperature=0, max_tokens=16384 verbatim. `EmptyResponse` non-retryable (deviation #9). No error-context message appended on retry (deviation #9). RateLimit retried per base tenacity policy (deviation #9). |
| `crates/chronicle-llm-openai/src/embedder.rs` | `graphiti_core/embedder/openai.py` | adapted | `OpenAiEmbedder`; `DEFAULT_EMBEDDING_MODEL="text-embedding-3-small"`. `EMBEDDING_DIM` is compile-time const (deviation #11). |
| `crates/chronicle-testkit/src/fake_driver.rs` | (test fixture, no upstream equivalent) | adapted | In-memory `FakeDriver`; deterministic uuid tie-breaks in similarity sorts (deviation #13, test-only). Phase-4 `CommunityOps`/`SagaOps` in-memory; `BulkSaveOps` uses the default sequential `save_all` (no transaction needed — nothing can partially fail in memory). |
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

### D-5: Persist atomicity — **CLOSED (Phase 4)**

CLOSED. Phase 4 adds `BulkSaveOps::save_all(episodes, episodic_edges, entity_nodes, entity_edges)`, and both persist tails — `add_episode` (single-episode) and `add_nodes_and_edges_bulk` (batch/triplet) — now call it instead of four independent saves. The Neo4j backend implements `save_all` as ONE transaction (`start_txn → run each non-empty UNWIND statement → commit`, with rollback on any statement error), matching upstream `add_episode`/`add_nodes_and_edges_bulk`'s single-transaction persist: a mid-batch failure leaves the graph unchanged (verified by `save_all_rolls_back_on_mid_batch_failure_live`). The in-memory `FakeDriver` inherits the trait's default sequential `save_all` (nothing can partially fail in memory). The granular four save ops remain on the trait for callers that need per-collection control.

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

### D-22: `label_propagation` iteration cap — **safety deviation (the one Phase-4 algorithm divergence)**

Upstream `label_propagation` (`community_operations.py`) loops `while True` with NO iteration bound. Synchronous label propagation can oscillate forever on pathological non-cluster inputs (e.g. a bare 2-node path `a—b` with equal edge weights, where each pass swaps the two labels), which would hang `build_communities`. This port adds `MAX_LABEL_PROPAGATION_ITERATIONS = 1000`: on overflow it emits a `tracing::warn!` and returns the last assignment instead of spinning. This is a pure safety fix and **cannot change behaviour on correct input** — real `get_community_clusters` projections converge in a handful of passes (verified: all four convergence/tie-break unit tests and the live `community_clusters_and_neighbor_vote_live` test stay well under the cap; `label_propagation_caps_on_oscillating_input` confirms termination on the degenerate case). The cap is the only deliberate algorithm-level divergence introduced in Phase 4.

### D-23: `update_communities` runs sequentially per node (vs upstream parallel)

On ingest with `update_communities=True`, upstream fans out `update_community` per affected node under a semaphore. This port iterates affected nodes sequentially (each `determine_entity_community` → `update_community` → save). Output is identical (neighbor-vote membership is independent per node); only the wall-clock fan-out differs. Performance-only deviation, consistent with D-4.

### D-24: `summarize_saga` truncates summary input on a UTF-8 char boundary

`summarize_saga` (`pipeline/saga.rs`) caps the rolling saga summary / concatenated-episode context at the upstream char budget. Python slices on code-point indices; Rust strings are byte-indexed, so the port truncates at the nearest **char boundary at or below** the byte budget (never mid-multibyte-sequence) to avoid panicking on non-ASCII content. For ASCII content the cut point is identical to upstream; for multibyte content the Rust cut may fall a few bytes earlier (never later). Behaviour-equivalent for the summarization prompt.

### D-25: Community / saga save uses UNWIND-batch Cypher vs upstream per-node single-MERGE

Upstream's `get_community_node_save_query` / saga save queries emit a single-node `MERGE` per call (the Python caller loops). The Neo4j port batches them as one `UNWIND $rows AS row MERGE (...)` statement per collection (identical shape to the existing entity-node/edge save queries), so a list of communities/sagas/edges persists in one round-trip. The resulting graph state is identical (MERGE-by-uuid is idempotent and order-independent); only the statement count per save call differs. Consistent with how Phase-1 already batches entity saves.

---

## SurrealDB driver deviations (Phase 3)

These are the locked implementation decisions for `chronicle-driver-surreal` (embedded SurrealDB, `surrealdb` 3.1.3, `kv-rocksdb`/`kv-mem`). All are documented in the Phase-3 spec amendment (`docs/superpowers/specs/2026-06-06-phase-3-embedded-backend-amendment.md` §4). The SurrealDB driver is NOT a query-text port; the correctness oracle is **behavior parity** with the FakeDriver/Neo4j reference, proven by the chronicle-testkit conformance suite + the `surreal_e2e` gate (bi-temporal invalidation + search recall through the real driver).

### D-26: KNN min-score is a post-filter (over-fetch ×3)

SurrealDB's HNSW `<|K,EF|>` operator returns the K nearest neighbours but does not natively filter by a similarity threshold. The trait `*_similarity_search(min_score)` contract requires dropping results below `min_score`. The driver over-fetches `K = limit × 3` candidates, computes `vector::distance::knn()`, then filters `score >= min_score` and truncates to `limit` in Rust. The ×3 padding factor compensates for candidates lost to the post-filter; behaviour is identical to the trait contract (verified by `node_similarity_ranks_exact_match_first_and_cuts_min_score`). Tunable if recall loss is observed on large indexes.

### D-27: BFS is iterative multi-query, not the recursive path idiom

SurrealDB 3.1.3's recursive-path idiom (`origin.{1..N}(->rel->node)`) has limited per-hop edge-predicate expressiveness for chronicle's BFS (which filters edge type + group per hop). Rather than fight the recursive syntax, the driver implements BFS **iteratively**: one adjacency query per hop, accumulating the visited frontier in Rust, applying the edge-type / group filter on each step. Result set is identical to a faithful breadth-first traversal; only the query count scales with depth (bounded by D-28). Verified by `node_bfs_depth_1_vs_3`, `node_bfs_edge_type_filter`, `edge_bfs_returns_only_relates_to`.

### D-28: BFS depth hard cap = 5

Matching the existing clamp pattern and because SurrealDB recursive/iterative traversal performance at depth > 5 is unbenchmarked, the driver clamps the requested BFS depth to a maximum of 5 (sanitized inline integer, never a user-bound parameter). chronicle's search recipes use `MAX_SEARCH_DEPTH = 3`, so the cap is never hit in normal operation; it is a defensive bound on pathological inputs.

### D-29: HNSW `TYPE F32` explicit on every vector index

SurrealDB's default vector index element type is F64. chronicle stores `Vec<f32>` embeddings, so every `DEFINE INDEX … HNSW` carries explicit `TYPE F32 DIST COSINE` to avoid an F64 widening mismatch. The HNSW graph lives in RAM and is rebuilt on `connect()` (idempotent schema DDL); single-process embedded, correct for node-local memory.

### D-30: chrono datetime converted at the boundary (never bound directly)

Binding `chrono::DateTime<Utc>` to a SurrealQL parameter stores it as a **string**, breaking datetime range/comparison queries (surrealdb issues #2753/#2804). The driver converts `chrono::DateTime<Utc>` ↔ `surrealdb::types::Datetime` in `convert.rs` at the persistence boundary — exactly as the Neo4j driver does for BoltDateTime — so stored temporal fields remain queryable as native datetimes. Guarded by a roundtrip test asserting stored values deserialize back as datetimes (not strings).

### D-31: DELETE performs explicit edge-detach, no native graph cascade

SurrealDB's `DELETE` removes a record but does not cascade-delete its incident RELATION edges the way Neo4j's `DETACH DELETE` does. The driver therefore explicitly deletes the incident edges (`relates_to`/`mentions`/`has_member`/`has_episode`/`next_episode`) before/with the node delete (`detach_entities`/`detach_episode`). End-state graph is identical to a Neo4j `DETACH DELETE`; the cascade is just performed in driver code rather than by the engine. Verified by `remove_communities_clears_nodes_and_membership` + the `remove_episode_cascade_surreal` e2e gate.

### D-32: Separate-tables-per-node-kind model + two-query undirected adjacency (no UNION ALL)

Node kinds are modelled as separate SurrealDB tables (`entity`/`episodic`/`community`/`saga`) rather than a single `node` table with a `label` discriminator — cleaner typed queries and per-kind indexes. Consequently, undirected adjacency (`nodes_connected_to_center`) that Cypher expresses with a single `(c)-[]-(n)` pattern is built from **two directed SurrealQL queries** (outgoing `->rel->` and incoming `<-rel<-`) unioned in Rust, because SurrealQL's `UNION ALL` of graph-traversal projections is not used here. Result is the deduplicated 1-hop neighbourhood, identical to the Cypher undirected match.

---

## FalkorDB driver deviations (Phase 6)

These are the locked implementation decisions for `chronicle-driver-falkor` (FalkorDB, the openCypher graph on a Redis module, `falkordb` 0.2.1, `features = ["tokio"]`). FalkorDB is a **Cypher-dialect adaptation of the Neo4j driver**, not a from-scratch query layer; the structural template is the Neo4j driver and the correctness oracle is **behaviour parity** with the FakeDriver/Neo4j/SurrealDB reference, proven by the live-server integration + search suites and the `falkor_e2e` gate (bi-temporal invalidation + community/saga/bulk/triplet/remove through the real driver). All deviations below are verified against live `falkordb/falkordb:latest`.

The op groups and their FalkorDB realisation:

| Op group | FalkorDB realisation |
|---|---|
| node/episode/community/saga persistence | `MERGE (n:Label {uuid}) SET …`, one statement per item (no nested-map params, no multi-statement batch) |
| edge persistence (RELATES_TO/MENTIONS/HAS_MEMBER/HAS_EPISODE/NEXT_EPISODE) | `MATCH endpoints … MERGE (a)-[e:TYPE {uuid}]->(b) SET …` |
| getters / by_uuids / by_group_ids | `MATCH … RETURN n/e`; `UNWIND <list> AS wanted MATCH {uuid: wanted}` for order-preserving by-uuid |
| `retrieve_episodes` | epoch-int `valid_at <=` filter, `ORDER BY valid_at DESC LIMIT`, reversed to chronological |
| vector search (node/edge/community) | `CALL db.idx.vector.query{Nodes,Relationships}(label,prop,K,vecf32([…]))`, over-fetch ×3, distance→similarity post-filter (D-33/D-36) |
| fulltext search (node/episode/community) | `CALL db.idx.fulltext.queryNodes(label,$q) YIELD node, score` |
| edge fulltext (`fact`) | `CALL db.idx.fulltext.queryRelationships('RELATES_TO',$q)` over a relationship-fulltext DDL index (D-37) |
| BFS (node/edge) | `-[:RELATES_TO\|MENTIONS*1..N]->` with inline-clamped depth, edge re-MATCH by uuid for filters |
| `get_community_clusters` | per-group, per-node `(n)-[e:RELATES_TO]-(m)` with `count(e)` aggregation → `GroupClusterProjection` |
| `community_of_member` / `neighbor_communities` | `(c:Community)-[:HAS_MEMBER]->…`; neighbour query returns one row per membership (NOT deduped) |
| deletes / `remove_communities` | `DETACH DELETE` for nodes (D-35), `DELETE e` for edges |
| saga threading | `saga_previous_episode_uuid` / `saga_episode_contents` order by epoch-int `valid_at`/`created_at` |
| `save_all` | best-effort sequential — no atomic batch (D-34) |

### D-33: KNN distance score → similarity, min-score is a post-filter (over-fetch ×3)

FalkorDB's `db.idx.vector.queryNodes`/`queryRelationships` procedures yield `(node/relationship, score)` where `score` is the cosine **distance** (smaller = closer), and they cannot post-filter inline on `min_score`, group, or node labels. The driver over-fetches `K = max(limit × 3, 30)` candidates ordered by distance ASC, converts each to a similarity (`1 - distance`), applies the group/label/filter WHERE in the query and the `similarity >= min_score` cut in Rust, then truncates to `limit`. Identical contract to the Neo4j inline `vector.similarity.cosine` path; verified by `node_similarity_ranks_and_applies_min_score` / `edge_similarity_ranks_and_cuts`.

### D-34: `save_all` is best-effort sequential, NOT atomic

FalkorDB's `GRAPH.QUERY` rejects `;`-separated multi-statement bodies and the `falkordb` 0.2 crate exposes no MULTI/EXEC transaction handle, so there is no way to wrap the four collection writes (episodes, entity nodes, entity edges, episodic edges) in a single atomic batch. Unlike the Neo4j backend — which OVERRIDES `save_all` with a real `start_txn → run all → commit` and rollback (D-5) — this backend performs the four writes **sequentially in the same order** with no cross-call rollback: a mid-batch failure leaves earlier collections persisted. This is an accepted FalkorDB limitation (the in-memory FakeDriver has the same non-atomic default, where nothing can partially fail). The ordering (nodes before edges) ensures a later failure cannot succeed-then-orphan. Exercised by the `add_episode_bulk_cross_dedup_falkor` e2e gate.

### D-35: datetime stored as epoch-millis `i64` (no native datetime type)

FalkorDB has no native datetime type. All temporal fields (`created_at`, `valid_at`, `expired_at`, `invalid_at`, the saga watermarks) are stored as **epoch-millisecond integers**; every bi-temporal comparison (`valid_at <=`, `created_at >`, ordering) becomes an integer comparison. `convert::datetime_to_millis`/`millis_to_datetime` are the only boundary. This is THE key delta a wrong mapping would silently break invalidation + retrieve-episodes cutoff; guarded by the `datetime_stored_as_int_and_filters_correctly` integration test and proven end-to-end by the `add_episode_two_episodes_invalidates_old_edge_and_keeps_it_falkor` gate (invalid_at round-trips to t1, edge stays retrievable).

### D-36: attributes stored as a single JSON-string property (`attrs_json`)

FalkorDB rejects nested-map properties ("Property values can only be of primitive types or arrays of primitive types"). The flat-or-nested `attributes` map is serialised to a single JSON **string** property `attrs_json` on write and parsed back on read, rather than stored as real node/edge properties (as the Neo4j backend does for flat attributes). Round-trip is lossless for any JSON-object attribute payload; the trade-off is that attribute values are not independently indexable/queryable on the server. Verified by the entity-node/edge attrs round-trip integration tests.

### D-37: relationship fulltext via the DDL index form (not `createNodeIndex`)

chronicle indexes the edge `fact` for fulltext recall. FalkorDB's `db.idx.fulltext.createNodeIndex` silently indexes nothing for a relationship type (verified empirically). The working path is the DDL form `CREATE FULLTEXT INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact)`, queried with `db.idx.fulltext.queryRelationships('RELATES_TO', $q) YIELD relationship, score`. The driver re-MATCHes each yielded relationship by uuid so the endpoint aliases exist for the node-label / group / date filters. Verified by `edge_fulltext_recall_on_fact`.

### D-38: parameters are textual Cypher literals (no typed wire params), encoded via an escaping layer

The `falkordb` 0.2 crate binds parameters as `CYPHER key=<literal> <query>` — every parameter *value* is a textual **Cypher literal expression**, not a typed Bolt wire value (unlike neo4rs). The driver therefore renders every dynamic value (strings, int/epoch, string lists, embeddings via `vecf32([…])`) through the escaping encoders in `convert.rs` so user data can NEVER break out of a literal — there is no raw string interpolation of user input. Embeddings additionally require the `vecf32([…])` constructor inline on both write and query, and BFS depth is an inline-clamped `usize` (Cypher disallows a parameter in `*1..N`), so the `[1,5]` clamp is the injection guard. Verified by `entity_node_handles_quote_injection_safely`.

### D-39: driver requires a multi-threaded Tokio runtime

The `falkordb` 0.2 crate refreshes the graph schema (label / property-key id → name maps, needed to decode `--compact` result rows) via an internal **blocking** Redis round-trip. Under a single-threaded Tokio runtime that blocking call aborts and rows decode as `Unparseable`. The driver MUST therefore run on a multi-threaded runtime (`#[tokio::test(flavor = "multi_thread")]` / `Runtime::new()` with `rt-multi-thread`). Greentic's production runtime is multi-threaded, so this is only a constraint for tests; all live tests carry the `multi_thread` flavour and serialise the setup+query phase through a shared `tokio::sync::Mutex` (the shared connection can otherwise race a just-built index).

---

## Consumer migration: `AddEpisodeRequest` → dw-providers v0.3.0

Phase 4 adds **three new fields** to `AddEpisodeRequest` (all additive, all defaulting to "off"):

- `update_communities: bool` (default `false`) — run neighbour-vote community membership update after ingest.
- `saga: Option<String>` (default `None`) — associate the episode with a named saga thread (HAS_EPISODE / NEXT_EPISODE wiring).
- `saga_previous_episode_uuid: Option<String>` (default `None`) — explicit previous-episode override for the NEXT_EPISODE chain (falls back to the saga's latest-by-valid_at episode when `None`).

`AddEpisodeRequest` derives `Default`, so the breaking change is source-level only. The `greentic-dw-providers` crate currently pins chronicle **v0.2.0** and constructs the struct by named fields, so it is **unaffected until it bumps to v0.3.0**. When it does:

- Switch field-by-field construction to spread the defaults: `AddEpisodeRequest { name, episode_body, source, source_description, reference_time, group_id, ..Default::default() }`. This keeps the call site compiling across any future additive field.
- Default behaviour is byte-identical to v0.2.0 (no community updates, no saga threading) — opt into the new behaviour only where the provider wants it.

No other public-API breaks in v0.3.0; the new facade methods (`add_episode_bulk`, `add_triplet`, `remove_episode`, `build_communities`, `get_nodes_and_edges_by_episode`, `summarize_saga`) are pure additions.

---

## DEFERRED

Features acknowledged but out of Phase-1 scope. Listed with target phase.

| Feature | Target | Notes |
|---|---|---|
| Reflexion (self-critique loop) | absent in upstream v0.29.1 | Not present in pinned upstream; not applicable |
| Communities / saga / bulk ingest | ✅ Done (Phase 4) | `build_communities`, `determine_entity_community`/`update_community`, community + saga node/edge types, `add_episode_bulk`, `add_triplet`, `remove_episode`, `summarize_saga`, `community_config`, COMMUNITY_* recipes, `SearchResults` community fields (D-16/D-22/D-23/D-24/D-25) |
| BFS traversal in edge/node search | ✅ Done (Phase 2) | `EdgeSearchMethod::BreadthFirstSearch` + node BFS implemented (self-seed + Cypher); see D-19 |
| MMR / NodeDistance / EpisodeMentions rerankers | ✅ Done (Phase 2) | Ported in `search/rerank.rs` (D-15, D-21) + dispatched in edge/node scopes |
| CrossEncoder reranker | ✅ Done (Phase 2) | `CrossEncoderClient` trait + OpenAI logprob impl (D-20) + scope wiring |
| `SearchFilters` wiring in edge/node search | ✅ Done (Phase 2) | `filters.rs` (D-17) wired into all scopes + Neo4j WHERE builders (D-18) |
| `SearchFilters(edge_uuids)` in resolve_extracted_edges | ✅ Done (Phase 2) | Edge candidate re-ranking — D-3 CLOSED |
| `NodeSearchConfig`, `EpisodeSearchConfig` | ✅ Done (Phase 2) | Full config + reranker enums |
| Top-level `search_()` / multi-scope facade | ✅ Done (Phase 2) | `Chronicle::search_`, `search_with_center`, `with_cross_encoder` (R13) |
| `CommunitySearchConfig` + community search scope | ✅ Done (Phase 4) | `CommunitySearchConfig`/`CommunitySearchMethod`/`CommunityReranker`, `community_search`, COMMUNITY_* + COMBINED_* community recipes, `SearchResults` community fields — D-16 RESOLVED |
| Multi-episode extraction path | ✅ Done (Phase 4) | `add_episode_bulk` cross-episode dedup (`dedupe_nodes_bulk`/`dedupe_edges_bulk`, directed + undirected union-find) in `pipeline/bulk.rs` |
| Entity/edge-type registries + attribute extraction | Phase 3 | `entity_types` and `edge_types` Pydantic registry; `extract_attributes_from_nodes` batch path |
| `extract_summaries_batch` / `SummarizedEntities` | Phase 3 | Replace per-node hydration (D-8) |
| `save_all` transactional driver op | ✅ Done (Phase 4) | Atomicity gap CLOSED — `BulkSaveOps::save_all`, Neo4j single-tx + rollback (D-5) |
| `semaphore_gather` equivalent fan-out | Phase 3 | Performance improvement for D-4 |
| `fact_triple` EpisodeType variant | Near-term patch | Read-compat gap (D-7) |
| ~~Kuzu embedded driver~~ | ❌ Superseded | **Dropped.** Kuzu archived upstream 2025-10-10 (Apple acquisition); see Phase-3 spec amendment. Embedded backend is now SurrealDB. |
| SurrealDB embedded driver | ✅ Done (Phase 3) | `chronicle-driver-surreal` — full `GraphDriver` supertrait over embedded SurrealDB (`surrealdb` 3.1.3, `kv-rocksdb`/`kv-mem`), feature-gated. Behavior parity with FakeDriver/Neo4j proven by chronicle-testkit + `surreal_e2e` gate. Deviations D-26–D-32. |
| FalkorDB driver | ✅ Done (Phase 6) | `chronicle-driver-falkor` — full `GraphDriver` supertrait over FalkorDB (`falkordb` 0.2.1, openCypher on a Redis module), a Cypher-dialect adaptation of the Neo4j driver. Was "Planned post-v1" in the original spec; now shipped. Behaviour parity proven by the live integration + search suites and the `falkor_e2e` gate (bi-temporal invalidation + community/saga/bulk/triplet/remove through the real driver). Deviations D-33–D-39. Completes the backend roadmap: Neo4j (server) + SurrealDB (embedded) + FalkorDB (Redis server); Neptune stays skipped. |
| Neptune driver | skipped | Out of scope |
| Upstream eval/ harness | not applicable | Python pytest-based evaluation suite |
| MCP / FastAPI servers | not applicable | Python server layer; Greentic integration via `greentic-dw-providers` |
