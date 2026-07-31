//! Runnable cost-measurement harness for the chronicle memory pipeline.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p chronicle-testkit --example cost_report
//! ```
//!
//! ## What this does
//!
//! It wires the real `Chronicle` engine over the in-memory `FakeDriver`, a
//! mock embedder, and a *scripted* mock LLM, but routes every LLM and embedder
//! call through the metering wrappers ([`MeteringLlm`] / [`MeteringEmbedder`]).
//! The wrappers count the REAL tokens of the actual prompts the pipeline builds
//! (via tiktoken) and the scripted responses, apply a pricing table, and
//! accumulate a per-call ledger. The harness resets the meter between
//! operations so each `add_episode` / `search` gets its own subtotal, then
//! prints a full cost report.
//!
//! ## Fidelity
//!
//! - INPUT tokens are EXACT — they are the verbatim prompts the pipeline sends.
//! - OUTPUT tokens are REPRESENTATIVE — the scripted mock responses are sized
//!   to look like real extraction/summary JSON, but a real model would vary.
//! - USD is ESTIMATED from the pricing table (verify against current pricing).
//!
//! ## Live ground-truth mode (gated, off by default; spends real cents)
//!
//! Set `CHRONICLE_COST_LIVE=1` and `OPENAI_API_KEY=...` to run ONE simple
//! `add_episode` + one search against the real OpenAI clients and report cost
//! from the same metering wrappers. See [`run_live`] for caveats — notably that
//! the current `OpenAiLlm` discards the API `usage` field, so the tiktoken count
//! is the available proxy even in live mode.

use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::driver::GraphDriver;
use chronicle_core::embedder::EmbedderClient;
use chronicle_core::llm::LlmClient;
use chronicle_core::search::recipes::{edge_hybrid_search_cross_encoder, edge_hybrid_search_rrf};
use chronicle_core::types::EpisodeType;

use chronicle_testkit::{
    CostReport, FakeDriver, Meter, MeteringEmbedder, MeteringLlm, MockCrossEncoder, MockEmbedder,
    MockLlm, PricingTable,
};

use chrono::{TimeZone, Utc};

const EMB_DIM: usize = 1536; // text-embedding-3-small dimensionality.

fn req(name: &str, body: &str, ts: chrono::DateTime<Utc>) -> AddEpisodeRequest {
    AddEpisodeRequest {
        name: name.to_string(),
        episode_body: body.to_string(),
        source: EpisodeType::Message,
        source_description: "cost-harness".to_string(),
        reference_time: ts,
        group_id: "cost".to_string(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Scripted responses (realistic in size → representative output-token counts)
// ---------------------------------------------------------------------------

/// Fresh-store simple episode: "Alice works at Acme."
///
/// Call sequence on a fresh store (concurrency 1):
/// 1. extract_message → ExtractedEntities[Alice, Acme]
///    (no dedupe — fresh store, no candidates)
/// 2. extract_edges.edge → ExtractedEdges[Alice WORKS_AT Acme]
/// 3. summarize_context × 2 (Alice, Acme)
fn simple_responses() -> Vec<serde_json::Value> {
    let entities = serde_json::json!({
        "extracted_entities": [
            {"name": "Alice", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Acme", "entity_type_id": 0, "episode_indices": [0]}
        ]
    });
    let edges = serde_json::json!({
        "edges": [{
            "source_entity_name": "Alice",
            "target_entity_name": "Acme",
            "relation_type": "WORKS_AT",
            "fact": "Alice works at Acme.",
            "valid_at": "2025-01-01T00:00:00Z",
            "invalid_at": null,
            "episode_indices": [0]
        }]
    });
    let summary_alice = serde_json::json!({
        "summary": "Alice is an employee at Acme; this captures her current employment relationship."
    });
    let summary_acme = serde_json::json!({
        "summary": "Acme is an organization that currently employs Alice."
    });
    vec![entities, edges, summary_alice, summary_acme]
}

/// Fresh-store rich episode: a 4-entity, 3-relation paragraph.
///
/// Body: "Bob is a senior engineer at Globex. He reports to Carol, who leads
/// the platform team. Globex was founded in Berlin."
///
/// Call sequence on a fresh store (concurrency 1):
/// 1. extract_message → ExtractedEntities[Bob, Globex, Carol, Berlin]
/// 2. extract_edges.edge → ExtractedEdges[3 edges]
/// 3. summarize_context × 4 (Bob, Globex, Carol, Berlin)
fn rich_responses() -> Vec<serde_json::Value> {
    let entities = serde_json::json!({
        "extracted_entities": [
            {"name": "Bob", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Globex", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Carol", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Berlin", "entity_type_id": 0, "episode_indices": [0]}
        ]
    });
    let edges = serde_json::json!({
        "edges": [
            {
                "source_entity_name": "Bob",
                "target_entity_name": "Globex",
                "relation_type": "WORKS_AT",
                "fact": "Bob is a senior engineer at Globex.",
                "valid_at": "2025-03-01T00:00:00Z",
                "invalid_at": null,
                "episode_indices": [0]
            },
            {
                "source_entity_name": "Bob",
                "target_entity_name": "Carol",
                "relation_type": "REPORTS_TO",
                "fact": "Bob reports to Carol.",
                "valid_at": "2025-03-01T00:00:00Z",
                "invalid_at": null,
                "episode_indices": [0]
            },
            {
                "source_entity_name": "Globex",
                "target_entity_name": "Berlin",
                "relation_type": "FOUNDED_IN",
                "fact": "Globex was founded in Berlin.",
                "valid_at": "2025-03-01T00:00:00Z",
                "invalid_at": null,
                "episode_indices": [0]
            }
        ]
    });
    let s_bob = serde_json::json!({
        "summary": "Bob is a senior engineer at Globex who reports to Carol on the platform team."
    });
    let s_globex = serde_json::json!({
        "summary": "Globex is an organization founded in Berlin that employs Bob and is led in part by Carol."
    });
    let s_carol = serde_json::json!({
        "summary": "Carol leads the platform team at Globex and is Bob's manager."
    });
    let s_berlin = serde_json::json!({
        "summary": "Berlin is the city where Globex was founded."
    });
    vec![entities, edges, s_bob, s_globex, s_carol, s_berlin]
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    if std::env::var("CHRONICLE_COST_LIVE").as_deref() == Ok("1") {
        run_live().await;
        return;
    }

    let meter = Meter::new(PricingTable::default());
    let mut report = CostReport::new(meter.pricing().clone());

    let t0 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
    let t1 = Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap();

    // ----- Operation 1: add_episode (simple), fresh store --------------------
    {
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MeteringEmbedder::new(
            Arc::new(MockEmbedder::new(EMB_DIM)),
            meter.clone(),
        ));
        let llm = Arc::new(MeteringLlm::new(
            Arc::new(MockLlm::new(simple_responses())),
            meter.clone(),
        ));
        let chronicle = Chronicle::new(
            driver as Arc<dyn GraphDriver>,
            llm as Arc<dyn LlmClient>,
            embedder as Arc<dyn EmbedderClient>,
            1,
        );

        meter.reset();
        chronicle
            .add_episode(req("simple", "Alice works at Acme.", t0))
            .await
            .expect("add_episode (simple)");
        report.add_operation("add_episode (simple)", meter.drain());
    }

    // ----- Operation 2: add_episode (rich), fresh store ----------------------
    {
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MeteringEmbedder::new(
            Arc::new(MockEmbedder::new(EMB_DIM)),
            meter.clone(),
        ));
        let llm = Arc::new(MeteringLlm::new(
            Arc::new(MockLlm::new(rich_responses())),
            meter.clone(),
        ));
        let chronicle = Chronicle::new(
            driver as Arc<dyn GraphDriver>,
            llm as Arc<dyn LlmClient>,
            embedder as Arc<dyn EmbedderClient>,
            1,
        );

        meter.reset();
        chronicle
            .add_episode(req(
                "rich",
                "Bob is a senior engineer at Globex. He reports to Carol, who leads \
                 the platform team. Globex was founded in Berlin.",
                t1,
            ))
            .await
            .expect("add_episode (rich)");
        report.add_operation("add_episode (rich)", meter.drain());
    }

    // ----- Operations 3 & 4: search (RRF) and search (cross-encoder) ---------
    // Build a populated store first (the simple episode), then run both
    // searches against it. Searches issue NO LLM calls; only the query
    // embedding hits the embedder, so these subtotals are near-zero.
    {
        let driver = Arc::new(FakeDriver::new());
        let embedder = Arc::new(MeteringEmbedder::new(
            Arc::new(MockEmbedder::new(EMB_DIM)),
            meter.clone(),
        ));
        // Ingest needs its own responses; searches need none.
        let llm = Arc::new(MeteringLlm::new(
            Arc::new(MockLlm::new(simple_responses())),
            meter.clone(),
        ));
        let cross_encoder = Arc::new(MockCrossEncoder::from_pairs([(
            "Alice works at Acme.",
            0.9,
        )]));
        let chronicle = Chronicle::new(
            driver as Arc<dyn GraphDriver>,
            llm as Arc<dyn LlmClient>,
            embedder as Arc<dyn EmbedderClient>,
            1,
        )
        .with_cross_encoder(cross_encoder);

        // Populate (not metered into a reported op).
        chronicle
            .add_episode(req("seed", "Alice works at Acme.", t0))
            .await
            .expect("seed add_episode");

        // search (RRF) — issues no LLM calls; only the query embedding hits the
        // embedder, so this subtotal is near-zero.
        meter.reset();
        let _rrf = chronicle
            .search(
                "where does Alice work",
                &["cost".to_string()],
                &edge_hybrid_search_rrf(),
            )
            .await
            .expect("search RRF");
        report.add_operation("search (RRF)", meter.drain());

        // search (cross-encoder)
        meter.reset();
        let _ce = chronicle
            .search(
                "where does Alice work",
                &["cost".to_string()],
                &edge_hybrid_search_cross_encoder(),
            )
            .await
            .expect("search cross-encoder");
        report.add_operation("search (cross-encoder)", meter.drain());
    }

    println!("{}", report.render());
}

// ---------------------------------------------------------------------------
// Live ground-truth mode (gated; not run in CI/tests)
// ---------------------------------------------------------------------------

/// Run one simple `add_episode` + one search against the REAL OpenAI clients,
/// reporting cost from the metering wrappers.
///
/// GROUND-TRUTH CAVEAT: the goal of this mode is to validate the mock estimate
/// against reality. The cleanest ground truth is the OpenAI API `usage` field
/// (exact prompt/completion tokens billed). However, the current `OpenAiLlm`
/// (chronicle-llm-openai/src/llm.rs) discards `usage` — it returns only the
/// parsed JSON content. So even here the metering tiktoken count is the
/// available proxy, NOT the billed truth. Plumbing `usage` out of `OpenAiLlm`
/// is a separate, non-trivial change (the trait returns `serde_json::Value`,
/// with no slot for token metadata) and is intentionally NOT done here. This
/// gap is documented in docs/cost-model.md.
async fn run_live() {
    use chronicle_core::llm::LlmConfig;
    use chronicle_llm_openai::{OpenAiEmbedder, OpenAiEmbedderConfig, OpenAiLlm};

    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("CHRONICLE_COST_LIVE=1 but OPENAI_API_KEY is unset; aborting.");
        return;
    }

    let meter = Meter::new(PricingTable::default());
    let mut report = CostReport::new(meter.pricing().clone());

    let driver = Arc::new(FakeDriver::new());
    let real_llm = OpenAiLlm::new(LlmConfig::default()).expect("OpenAiLlm");
    let real_emb = OpenAiEmbedder::new(OpenAiEmbedderConfig {
        embedding_dim: EMB_DIM,
        ..Default::default()
    })
    .expect("OpenAiEmbedder");

    let llm = Arc::new(MeteringLlm::new(
        Arc::new(real_llm) as Arc<dyn LlmClient>,
        meter.clone(),
    ));
    let embedder = Arc::new(MeteringEmbedder::new(
        Arc::new(real_emb) as Arc<dyn EmbedderClient>,
        meter.clone(),
    ));

    let chronicle = Chronicle::new(
        driver as Arc<dyn GraphDriver>,
        llm as Arc<dyn LlmClient>,
        embedder as Arc<dyn EmbedderClient>,
        1,
    );

    let t0 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();

    meter.reset();
    chronicle
        .add_episode(req("live-simple", "Alice works at Acme.", t0))
        .await
        .expect("live add_episode");
    report.add_operation("LIVE add_episode (simple)", meter.drain());

    meter.reset();
    let _ = chronicle
        .search(
            "where does Alice work",
            &["cost".to_string()],
            &edge_hybrid_search_rrf(),
        )
        .await
        .expect("live search");
    report.add_operation("LIVE search (RRF)", meter.drain());

    println!("{}", report.render());
    println!(
        "NOTE (live): OUTPUT tokens are tiktoken over the REAL model JSON, but the \n\
         OpenAI `usage` field (the billed ground truth) is discarded by OpenAiLlm. \n\
         The tiktoken count is the available proxy. See docs/cost-model.md."
    );
}
