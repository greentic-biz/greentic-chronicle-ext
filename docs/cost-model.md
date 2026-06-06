# Chronicle cost model & metering harness

This document explains the cost-measurement harness for the chronicle memory
pipeline: what it measures, what it estimates, how to run it, and how to validate
the estimate against reality.

## TL;DR

- The harness turns the per-operation LLM/embedding cost from a hand-wave
  estimate into **MEASURED token counts × a configurable pricing table**.
- **Token counts are MEASURED** — input tokens are an exact `tiktoken` count over
  the verbatim prompts the pipeline actually sends; output tokens are an exact
  `tiktoken` count over the model's JSON response.
- **USD is ESTIMATED** — `tokens × pricing-table rate`. The pricing table carries
  APPROXIMATE current OpenAI prices and must be verified before quoting.
- Run it: `cargo run -p chronicle-testkit --example cost_report`

## Architecture

The metering layer lives in `crates/chronicle-testkit/src/metering.rs`:

- `MeteringLlm` wraps any `Arc<dyn LlmClient>`. On every `generate`, it counts
  input tokens over the request `messages` (tiktoken `o200k_base`), calls the
  inner client, counts output tokens over the returned `serde_json::Value`
  serialized compactly, computes cost via the `PricingTable` keyed on the
  request's `ModelSize`, and pushes a `CallRecord` onto a shared `Meter`.
- `MeteringEmbedder` wraps any `Arc<dyn EmbedderClient>`. For each input string
  in `create` / `create_batch` it counts input tokens (tiktoken `cl100k_base`)
  and records cost via the embedding pricing (`output_tokens = 0`).
- `Meter` is a cloneable `Arc<Mutex<Vec<CallRecord>>>` + the `PricingTable`. The
  harness calls `meter.reset()` before each operation and `meter.drain()` after
  it, so cost is attributed to a single operation ("add_episode (simple)",
  "search (RRF)", …).
- `CostReport` aggregates the per-operation snapshots: per-call ledger,
  per-operation subtotals, by-`prompt_name` rollup, grand total, and a monthly
  ingest projection at 100 / 1,000 / 10,000 episodes/day.

### Tokenizer / encoding choice

`tiktoken-rs` bundles its BPE ranks, so token counting is offline and
deterministic (no network). Encodings:

- **Chat (gpt-4.1-mini / gpt-4.1-nano): `o200k_base`** — the gpt-4.1 family uses
  this encoding.
- **Embeddings (text-embedding-3-small): `cl100k_base`** — text-embedding-3-\*
  uses this encoding.

Chat input counting adds a small documented per-message overhead (~4 tokens/msg
+ 3 to prime the reply, from OpenAI's chat token-counting cookbook). Content
tokens — the dominant term — are counted exactly; the per-message constant is a
minor correction, not the headline number.

## What is measured vs. estimated

| Quantity | Mock mode | Live mode |
| --- | --- | --- |
| **Input tokens** | EXACT (verbatim pipeline prompts) | EXACT (verbatim pipeline prompts) |
| **Output tokens** | REPRESENTATIVE (scripted responses, realistic size) | EXACT-ish (tiktoken over real model JSON) |
| **USD** | ESTIMATED (pricing table) | ESTIMATED (pricing table) |

The scripted mock responses (in `examples/cost_report.rs`) reuse the realistic
JSON shapes from the e2e test (`ExtractedEntities`, `ExtractedEdges`, `Summary`,
…) so output-token counts are representative of a real run. A real model would
vary, so treat the mock-mode USD as an order-of-magnitude estimate, not a bill.

## Pricing table (APPROXIMATE — VERIFY)

Defaults in `PricingTable::default()` (USD per 1,000,000 tokens):

| Model | Role | Input | Output |
| --- | --- | --- | --- |
| gpt-4.1-mini | chat medium | $0.40 | $1.60 |
| gpt-4.1-nano | chat small | $0.10 | $0.40 |
| text-embedding-3-small | embedding | $0.02 | — |

These are **APPROXIMATE** and are flagged as such in the code and in the rendered
report. **Verify against current OpenAI pricing before quoting any number.** The
table is trivially overridable — construct `PricingTable` directly, or take the
default and mutate the public fields.

## How to run

### Mock mode (free, deterministic — the default)

```bash
cargo run -p chronicle-testkit --example cost_report
```

This exercises:

1. `add_episode (simple)` — "Alice works at Acme." (fresh store)
2. `add_episode (rich)` — a 4-entity / 3-relation paragraph (fresh store)
3. `search (RRF)` — `edge_hybrid_search_rrf()`
4. `search (cross-encoder)` — `edge_hybrid_search_cross_encoder()` + `MockCrossEncoder`

### Live ground-truth mode (gated; spends real cents)

```bash
export OPENAI_API_KEY=sk-...
export CHRONICLE_COST_LIVE=1
cargo run -p chronicle-testkit --example cost_report
```

This wraps the **real** `OpenAiLlm` + `OpenAiEmbedder` (from
`chronicle-llm-openai`) in the same metering wrappers, runs one simple
`add_episode` + one search, and prints the report.

> **Ground-truth caveat.** The cleanest ground truth would be the OpenAI API
> `usage` field (exact billed prompt/completion tokens). However, the current
> `OpenAiLlm` (`chronicle-llm-openai/src/llm.rs`) **discards `usage`** — its
> trait returns only `serde_json::Value` (the parsed content), with no slot for
> token metadata. So even in live mode the metering `tiktoken` count is the
> available proxy, NOT the billed truth. Plumbing `usage` out of `OpenAiLlm` is
> a separate, non-trivial change and is intentionally not done here. This is the
> known fidelity gap between this harness and an exact bill.

## Snapshot of the example output

> The numbers below are from a real `cargo run` of the mock-mode harness. They
> are **mock-response-based estimates**: input tokens are exact (real pipeline
> prompts), output tokens are from representative scripted responses, and USD is
> from the APPROXIMATE pricing table above.

```
================ chronicle cost report ================
NOTE: token counts are MEASURED (tiktoken on real pipeline prompts);
      USD is ESTIMATED from the pricing table below.
      In mock mode, OUTPUT tokens come from representative scripted
      responses; INPUT tokens are exact (the real prompts the pipeline sends).

-- pricing assumptions (APPROXIMATE — VERIFY against current OpenAI pricing) --
  chat medium [gpt-4.1-mini]: $0.40/1M in, $1.60/1M out
  chat small  [gpt-4.1-nano]: $0.10/1M in, $0.40/1M out
  embedding   [text-embedding-3-small]: $0.02/1M tok
  chat encoding: o200k_base   embedding encoding: cl100k_base

-- per-operation subtotals --
  add_episode (simple)         4 llm +  6 emb calls |   4181 in /    131 out tok | $0.001519
  add_episode (rich)           6 llm + 14 emb calls |   5425 in /    317 out tok | $0.001907
  search (RRF)                 0 llm +  1 emb calls |      4 in /      0 out tok | $0.000000
  search (cross-encoder)       0 llm +  1 emb calls |      4 in /      0 out tok | $0.000000

-- cost by prompt_name (descending) --
  extract_nodes.extract_message   3729 in /    106 out tok | $0.001661
  extract_edges.edge           2555 in /    233 out tok | $0.001395
  summarize_nodes.summarize_context   3248 in /    109 out tok | $0.000368
  embedding                      82 in /      0 out tok | $0.000002

-- grand total --
  10 llm calls + 22 embedding calls
  9614 input tokens + 448 output tokens
  ESTIMATED $0.003426 total

-- monthly projection (ingest only, at avg add_episode cost) --
     100 episodes/day -> $      5.14/month
    1000 episodes/day -> $     51.39/month
   10000 episodes/day -> $    513.88/month

=======================================================
```

(The full per-call ledger is also printed by the example; it is elided here for
brevity.)

## Reading the numbers

- **A simple `add_episode` costs ~$0.0015; a rich one ~$0.0019** (mock-estimate).
- **The dominant cost is the extraction prompts** — `extract_nodes.extract_message`
  and `extract_edges.edge` together account for the large majority of the spend.
  Their input-token counts (~1,850 tokens per call) dwarf the responses: the
  **verbatim system+user prompts** are the headline cost driver, not the model
  output. This is the lever to pull for cost reduction (prompt trimming,
  few-shot pruning, caching).
- **Search is effectively free** — RRF and cross-encoder searches issue no LLM
  calls; only the query embedding hits the embedder, which is a rounding error.
  (Note: the `MockCrossEncoder` is scored locally and does not bill; a real
  OpenAI logprob cross-encoder would add per-passage LLM cost — out of scope for
  this mock harness.)
- **Embedding cost is negligible** at these volumes.

## Caveats on fidelity

- Output tokens in mock mode are from scripted responses; a real model varies.
- The pricing table is APPROXIMATE — verify against current OpenAI pricing.
- The per-message chat overhead constant (~4 + 3) is an approximation of the
  chat framing; content tokens are exact.
- Live mode cannot yet read the billed `usage` (see ground-truth caveat above),
  so the tiktoken count remains the proxy even there.
- The monthly projection assumes a fixed per-episode cost equal to the mean of
  the measured `add_episode` operations; real workloads mix simple/rich episodes
  and incur extra dedupe/invalidation LLM calls on a populated store (not
  modeled in the two fresh-store sample episodes).
