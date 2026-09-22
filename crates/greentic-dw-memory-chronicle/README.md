# greentic-dw-memory-chronicle

Chronicle-backed implementation of the Greentic DW long-term memory contract
(`greentic-dw-memory-long-term`).

It adapts the `LongTermMemory` trait onto the [Chronicle](https://github.com/greentic-biz/greentic-chronicle-ext)
bi-temporal knowledge-graph engine: episodes are ingested into a Neo4j-backed graph
and recalled through Chronicle's hybrid (BM25 + cosine, RRF-fused) edge search.

Tenant isolation maps every `TenantCtx` to a Chronicle `group_id`; all reads and
writes are scoped to that group.

## Backends

- **Graph:** Neo4j (via `chronicle-driver-neo4j`).
- **Extraction LLM:** OpenAI-compatible (via `chronicle-llm-openai`), or any
  Greentic DW `LlmProvider` through the `DwLlmBridge`.
- **Embeddings:** OpenAI-compatible. There is no DW embeddings family yet, so the
  vector side of recall always uses an OpenAI embedder, even when extraction is
  driven by a DW provider.

## Construction

- `ChronicleLongTermMemory::connect(config)` — OpenAI LLM + embedder; provisions
  indices and constraints.
- `ChronicleLongTermMemory::connect_with_dw_llm(config, provider, tenant)` — DW
  provider for extraction, OpenAI embedder for vectors.
- `ChronicleLongTermMemory::from_parts(driver, llm, embedder, max_concurrency,
  recall_limit)` — direct dependency injection for tests.
