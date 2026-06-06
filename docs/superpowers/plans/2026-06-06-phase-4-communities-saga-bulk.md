# Phase 4: Communities + Saga + Bulk + Maintenance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Complete graphiti-core parity: community detection + summaries + community search scope, saga narrative grouping, bulk ingestion, `add_triplet`, `remove_episode`, `get_nodes_and_edges_by_episode`, plus the new node/edge types and indices they need.

**Upstream truth:** /home/bima-pangestu/Works/refs/graphiti @ 34f56e65 (verify each task). All algorithm/Cypher quotes below were extracted+verified from that commit; upstream wins on any disagreement — note divergences in docs/port-fidelity.md.

**Branch:** `feat/phase-4-communities-saga-bulk` off `research` (pull first; checkout is on research, no foreign WIP). PR → research, tag v0.3.0 after merge. House rules: porting headers, no unwrap/panic in prod, fmt+clippy -D warnings, NO attribution trailers, conventional commits, docker neo4j integration where flagged.

**Reference (R-sections) — verified upstream extracts the tasks rely on:**

- **R1 CommunityNode** (nodes.py): fields uuid/name/group_id/labels(["Community"])/created_at/name_embedding/summary(default ""). **CommunityEdge** HAS_MEMBER (Community→Entity): uuid/group_id/source_node_uuid/target_node_uuid/created_at. Save Cypher: `MERGE (n:Community {uuid})... SET n = {...} ... db.create.setNodeVectorProperty(n,"name_embedding",...)`; edge `MATCH (community:Community)... MATCH (node:Entity|Community)... MERGE (community)-[e:HAS_MEMBER {uuid}]->(node) SET e = {...}`. RETURN projections + `community_name` fulltext index `FOR (n:Community) ON EACH [n.name, n.group_id]` + range indices community_uuid/community_group_id/has_member_uuid.
- **R2 label_propagation** (community_operations.py): projection = per-node neighbor list with RELATES_TO edge_count (`MATCH (n:Entity{group_id,uuid})-[e:RELATES_TO]-(m:Entity{group_id}) WITH count(e) AS count, m.uuid AS uuid RETURN uuid, count`). Algorithm: each node starts own community (int); iterate — for each node sum neighbor edge_counts per community, sort desc, if top count>1 take it else `max(candidate, current)`; converge when no change. Clusters = grouped by final community int.
- **R3 build_community** (per cluster): PAIRWISE-REDUCE over member `entity.summary` strings — binary-tree merge via `summarize_pair` (odd one carried), log₂(N) rounds, `truncate_at_sentence(result, MAX_SUMMARY_CHARS=1000)` → summary; `name = generate_summary_description(summary)` (summary_description prompt). build_community_edges = HAS_MEMBER per member. summarize_pair + summary_description prompts: PORT VERBATIM (system+user text quoted in extraction; summarize_pair system "You are a helpful assistant that combines summaries into a single dense factual summary", response Summary{summary}; summary_description system "...describes provided contents in a single sentence", response SummaryDescription{description}). MAX_COMMUNITY_BUILD_CONCURRENCY=10 semaphore. graphiti.build_communities(): remove_communities (DETACH DELETE all :Community) → build → parallel name embeddings → parallel save nodes+edges.
- **R4 update_community / determine_entity_community** (update_communities=True on ingest): determine = (a) already a member? `MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity{uuid})` → return (community, is_new=False); (b) else neighbor-vote `MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity{uuid})` → mode community → return (community, is_new=True); (c) none → (None,False). update_community: if community None→noop; new_summary=summarize_pair(entity.summary, community.summary); new_name=generate_summary_description; if is_new save HAS_MEMBER edge; regen name embedding; save community. NEIGHBOR-VOTE only, no embedding sim at ingest.
- **R5 remove_episode** (graphiti.py): get episode; edges = get_by_uuids(episode.entity_edges); delete edges where `edge.episodes[0] == episode.uuid` (primary-source only); nodes = get_mentioned_nodes([episode]); delete nodes where MENTIONS episode_count==1 (single-episode only); delete episode (DETACH). No saga/community cascade.
- **R6 add_episode_bulk** (graphiti.py + bulk_utils.py): RawEpisode{name, uuid?, content, source_description, source, reference_time}; AddBulkEpisodeResults{episodes, episodic_edges, nodes, edges, communities=[], community_edges=[]}. Pipeline: create+save episodes → retrieve_previous_episodes_bulk (EPISODE_WINDOW_LEN per ep) → extract_nodes_and_edges_bulk (parallel LLM) → dedupe_nodes_bulk → build episodic edges → resolve_edge_pointers → dedupe_edges_bulk → resolve_extracted_nodes/edges per ep → add_nodes_and_edges_bulk (one save). dedupe_nodes_bulk: pass1 resolve_extracted_nodes per ep, pass2 intra-batch exact-name + MinHash candidate indexes; union all uuid_maps + intra pairs → `_build_directed_uuid_map` (union-find iterative path compression, direction-preserving). dedupe_edges_bulk: embed all, compare same src+tgt + word-overlap/cosine≥0.6, resolve_extracted_edge parallel, compress_uuid_map (undirected union-find, smallest uuid wins). resolve_edge_pointers remaps src/tgt via map. CHUNK_SIZE=10 defined but caller-chunked (not internal). Saga association if saga param: sort by valid_at, chain NEXT_EPISODE + HAS_EPISODE.
- **R7 add_triplet** (graphiti.py): embed nodes+edge if missing; get_by_uuid source/target else resolve_extracted_nodes; merge caller attributes/summary/labels into resolved; set edge endpoints to resolved uuids; uuid-conflict regen; valid_edges=get_between_nodes + related_edges=hybrid fact search; resolve_extracted_edge (synthetic episode, full dedup/invalidation); save via bulk. NO episode/episodic edges/community. AddTripletResults{nodes, edges(resolved+invalidated)}.
- **R8 community search** (search.py + search_utils): CommunitySearchMethod{cosine_similarity, bm25} (NO bfs), CommunityReranker{rrf, mmr, cross_encoder}, CommunitySearchConfig{search_methods, reranker=rrf, sim_min_score=0.6, mmr_lambda=0.5, bfs_max_depth=3 unused}. community_search ALWAYS runs both fulltext+similarity regardless of methods list. community_fulltext_search (index community_name, group filter, COMMUNITY_NODE_RETURN, ORDER score DESC); community_similarity_search (`MATCH (c:Community) WHERE group... WITH c, vector.similarity.cosine(c.name_embedding,$vec) AS score WHERE score > $min RETURN... ORDER DESC`). Rerankers: rrf / mmr (get_embeddings_for_communities) / cross_encoder (rank node.name). Recipes COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER(limit 3)} + add community_config to the 3 COMBINED_* recipes.
- **R9 Saga** (nodes.py + summarize_sagas.py + graphiti.py): SagaNode{uuid/name/group_id/labels/created_at, summary="", first_episode_uuid?, last_episode_uuid?, last_summarized_at?, last_summarized_episode_valid_at?}. Edges HAS_EPISODE (Saga→Episodic) + NEXT_EPISODE (Episodic→Episodic), each {uuid,group_id,created_at}. Save Cypher quoted in extraction. add_episode saga threading: `saga: str|SagaNode`, `saga_previous_episode_uuid`; get_or_create_saga by (name,group_id); prev = param or `_saga_get_previous_episode_uuid` (latest HAS_EPISODE episode by valid_at DESC); save NEXT_EPISODE (prev→current) if prev; save HAS_EPISODE; update saga first/last_episode_uuid. summarize_saga: two watermarks (last_summarized_at wall-clock filter `e.created_at > $since`; last_summarized_episode_valid_at episode-time); summarize_saga prompt VERBATIM (system "You extract durable knowledge from message threads..."; context {saga_name, existing_summary, episodes: list[str]}; response SagaSummary{summary}).
- **R10 get_nodes_and_edges_by_episode** (trivial): EpisodicNode.get_by_uuids + get_mentioned_nodes + EntityEdge.get_by_uuids(episode.entity_edges) → SearchResults-like.

**Scope note:** Phase 4 completes Graphiti-class parity. `_search` deprecated alias not ported (Phase 2 decision). Bulk never updates communities (upstream — communities=[]).

---

## Tasks

### Task 1: Community + Saga types + indices
- [ ] Branch off research. types/community.rs: CommunityNode (R1), CommunityEdge (HAS_MEMBER). types/saga.rs: SagaNode (R9), HasEpisodeEdge, NextEpisodeEdge. serde + Default + new() ctors mirroring existing types; porting headers; bi-temporal N/A (these carry created_at only). Register in types/mod.rs.
- [ ] Driver SchemaOps: extend build_indices_and_constraints index set with community_name fulltext + Community/Saga/HAS_MEMBER/HAS_EPISODE/NEXT_EPISODE range indices (Neo4j queries.rs + FakeDriver no-op). Verify upstream graph_queries.py list.
- [ ] Tests: serde roundtrips; index DDL idempotency (neo4j integration, docker). Commit.

### Task 2: Driver ops for communities + saga (trait + FakeDriver + Neo4j)
- [ ] SearchOps/new CommunityOps+SagaOps traits (or extend GraphDriver): save/get community nodes+edges, community_fulltext_search + community_similarity_search + get_embeddings_for_communities (R8 Cypher), get/save saga node + HAS_EPISODE/NEXT_EPISODE edges, get_or_create_saga lookup, saga_previous_episode_uuid query, the community-membership queries (already-member + neighbor-vote, R4), get_community_clusters projection query (R2), remove_communities, get_mentioned_nodes, episode mention-count-for-node (R5), get_entity_edges_by_uuids / get_entity_nodes_by_uuids (if not present), delete_edges_by_uuids / delete_nodes_by_uuids + episode delete (R5). Audit existing driver first — add only what's missing.
- [ ] FakeDriver impls (in-memory) + Neo4j impls (verbatim Cypher). Tests both. Commit.

### Task 3: Community prompts + build_communities
- [ ] prompts/summarize_nodes.rs: add summarize_pair + summary_description prompt fns VERBATIM (R3) + their context structs. (Summary/SummaryDescription models already exist from Phase 1.)
- [ ] pipeline/community_ops.rs: label_propagation (R2 exact), get_community_clusters (driver), build_community pairwise-reduce (R3 — truncate_at_sentence helper: port upstream's sentence-boundary truncation), build_community_edges, build_communities orchestration (concurrency 10), remove_communities. determine_entity_community + update_community (R4).
- [ ] Tests: label_propagation convergence (small graphs, tie-break max), pairwise-reduce odd/even member counts (MockLlm scripted), neighbor-vote membership. Commit.

### Task 4: Community search scope
- [ ] search/config.rs: CommunitySearchMethod, CommunityReranker, CommunitySearchConfig (R8) — RESTORE the types removed in Phase 2 Task 1 (now with correct CommunitySearchMethod enum). SearchConfig gains community_config. recipes.rs: COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER limit 3} + add community_config to combined recipes.
- [ ] search/community_search.rs: per R8 (always both methods; rrf/mmr/cross_encoder dispatch; ranks community.name). results.rs SearchResults gains communities + community_reranker_scores. Top-level search() adds 4th scope (community_search) to the parallel join; embed-decision includes community cosine/mmr.
- [ ] Tests: community rrf/mmr/cross_encoder vs FakeDriver; top-level 4-scope assembly. Commit.

### Task 5: Bulk ingestion
- [ ] pipeline/bulk.rs: RawEpisode, AddBulkEpisodeResults, union-find (_build_directed_uuid_map directed + compress_uuid_map undirected, R6 exact), resolve_edge_pointers (exists? reuse), dedupe_nodes_bulk + dedupe_edges_bulk + extract_nodes_and_edges_bulk + retrieve_previous_episodes_bulk + add_nodes_and_edges_bulk. add_episode_bulk orchestration (R6). CHUNK_SIZE const (documented caller-chunked).
- [ ] Tests: union-find path compression + direction; cross-episode dedup (two episodes same entity → merged, MockLlm scripted); bulk e2e (2 episodes → AddBulkEpisodeResults). Commit.

### Task 6: add_triplet + remove_episode + get_nodes_and_edges_by_episode + facade
- [ ] pipeline: add_triplet (R7), remove_episode (R5), get_nodes_and_edges_by_episode (R10).
- [ ] chronicle.rs facade methods: add_episode_bulk, add_triplet, remove_episode, build_communities, get_nodes_and_edges_by_episode; add_episode gains optional saga params (update_communities flag + saga/saga_previous_episode_uuid) — additive to AddEpisodeRequest (Option fields, default None; dw-providers v0.2.0 unaffected). Wire saga threading (R9) + update_communities path into add_episode.
- [ ] summarize_saga method + summarize_sagas.rs prompt VERBATIM (R9). SagaSummary model (add to prompts/models.rs).
- [ ] Tests: add_triplet (testkit), remove_episode cascade (primary-edge + single-mention-node only), saga chaining (NEXT_EPISODE/HAS_EPISODE), summarize_saga watermarks. Commit.

### Task 7: Neo4j parity for new ops + integration + close-out
- [ ] Ensure every new driver method has a real Neo4j impl (Task 2 covers most — verify community search, embeddings loader, saga queries, bulk save path all Cypher-complete; bulk save = add_nodes_and_edges_bulk single-tx — implement as a multi-statement tx per the Phase-1 atomicity note, now is the time to add a transactional save).
- [ ] Docker neo4j integration tests: community build+search end-to-end, saga chain + summarize, add_triplet, remove_episode cascade, bulk ingest. Run live (--test-threads=1).
- [ ] docs/port-fidelity.md: community/saga/bulk rows; mark Phase 4 deferred items CLOSED; new deviations. README phase table Phase 4 done. cargo update + bash ci/local_check.sh green.
- [ ] PR → research; verify headRefOid; after merge tag v0.3.0.

## Risks
| Risk | Mitigation |
|---|---|
| Phase 2 removed CommunitySearchConfig — restore cleanly | Task 4 reintroduces with correct enum |
| Bulk union-find direction subtle | R6 exact port + path-compression tests |
| add_nodes_and_edges_bulk single-tx vs our 4-call persist | Task 7 adds transactional save (closes Phase-1 atomicity note) |
| truncate_at_sentence boundary logic | port upstream helper exactly + test |
| AddEpisodeRequest growth breaks Phase-5 provider | Option fields default None; provider pins v0.2.0, opts in later |
