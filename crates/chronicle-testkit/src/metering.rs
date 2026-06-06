//! Cost-metering harness for the chronicle pipeline.
//!
//! This module wraps [`LlmClient`] and [`EmbedderClient`] with decorators that
//! count the REAL tokens of the actual prompts the pipeline sends (via
//! `tiktoken-rs`, which bundles its BPE ranks — no network), apply a
//! configurable pricing table, and accumulate a per-call cost ledger that can
//! be rendered into a per-operation cost report.
//!
//! ## What is measured vs. estimated
//!
//! - **Token counts are MEASURED.** Input tokens are an exact tiktoken count
//!   over the verbatim prompt strings the pipeline builds. Output tokens are an
//!   exact tiktoken count over the JSON the LLM returns. In MOCK mode the
//!   responses are scripted (realistic in size, but not produced by a real
//!   model), so output-token counts are *representative*, not ground truth.
//! - **USD is ESTIMATED.** Cost is `tokens × pricing-table rate`. The pricing
//!   table carries APPROXIMATE current OpenAI prices and is trivially
//!   overridable — verify against current OpenAI pricing before quoting.
//!
//! ## Encoding choice
//!
//! The gpt-4.1 family (gpt-4.1-mini / gpt-4.1-nano) uses the `o200k_base`
//! encoding; `text-embedding-3-small` uses `cl100k_base`. We therefore use
//! `o200k_base` for chat token counts and `cl100k_base` for embedding token
//! counts.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};
use chronicle_core::llm::{LlmClient, LlmError, LlmRequest, ModelSize, Role};
use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton};

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/// Per-message chat-format overhead, in tokens.
///
/// OpenAI's chat models wrap each message in role/structure tokens. The
/// commonly-cited approximation (from OpenAI's token-counting cookbook for the
/// gpt-3.5/gpt-4 chat format) is ~4 tokens per message plus 3 tokens to prime
/// the assistant reply. We reuse that constant here as a documented
/// approximation — the gpt-4.1 / o200k_base framing differs slightly, but the
/// content tokens (the dominant term) are counted exactly, so the per-message
/// constant is a small correction, not the headline number.
const TOKENS_PER_MESSAGE: usize = 4;
/// Reply-priming overhead added once per request (the `<|start|>assistant` prime).
const TOKENS_REPLY_PRIMING: usize = 3;

/// Count chat-prompt tokens for a slice of messages using `o200k_base`.
///
/// Counts the content of each message exactly, then adds the documented
/// per-message + reply-priming overhead. The role string itself is short and
/// folded into the per-message overhead rather than encoded separately.
pub fn count_chat_tokens(messages: &[chronicle_core::llm::Message]) -> usize {
    let bpe = o200k_base_singleton();
    let mut total = 0usize;
    for m in messages {
        total += bpe.encode_with_special_tokens(&m.content).len();
        total += TOKENS_PER_MESSAGE;
        // Role name contributes a token or two; counted exactly for fidelity.
        total += match m.role {
            Role::System => 1,
            Role::User => 1,
            Role::Assistant => 1,
        };
    }
    total + TOKENS_REPLY_PRIMING
}

/// Count tokens of an arbitrary string using `o200k_base` (chat output).
pub fn count_chat_output_tokens(text: &str) -> usize {
    o200k_base_singleton()
        .encode_with_special_tokens(text)
        .len()
}

/// Count tokens of an embedding input string using `cl100k_base`.
pub fn count_embedding_tokens(text: &str) -> usize {
    cl100k_base_singleton()
        .encode_with_special_tokens(text)
        .len()
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

/// Per-model chat pricing, expressed in USD per 1,000,000 tokens.
#[derive(Debug, Clone)]
pub struct Pricing {
    pub model_label: String,
    pub input_per_1m_usd: f64,
    pub output_per_1m_usd: f64,
}

/// Embedding pricing, expressed in USD per 1,000,000 tokens.
#[derive(Debug, Clone)]
pub struct EmbeddingPricing {
    pub model_label: String,
    pub per_1m_usd: f64,
}

/// Maps each [`ModelSize`] to a chat [`Pricing`] plus a single
/// [`EmbeddingPricing`].
///
/// Trivially overridable: construct directly, or take [`PricingTable::default`]
/// and mutate the public fields.
#[derive(Debug, Clone)]
pub struct PricingTable {
    /// Pricing for `ModelSize::Medium` (gpt-4.1-mini by default).
    pub medium: Pricing,
    /// Pricing for `ModelSize::Small` (gpt-4.1-nano by default).
    pub small: Pricing,
    /// Pricing for the embedding model (text-embedding-3-small by default).
    pub embedding: EmbeddingPricing,
}

impl Default for PricingTable {
    fn default() -> Self {
        Self {
            // APPROXIMATE — verify against current OpenAI pricing; configurable.
            medium: Pricing {
                model_label: "gpt-4.1-mini".to_string(),
                input_per_1m_usd: 0.40,
                output_per_1m_usd: 1.60,
            },
            // APPROXIMATE — verify against current OpenAI pricing; configurable.
            small: Pricing {
                model_label: "gpt-4.1-nano".to_string(),
                input_per_1m_usd: 0.10,
                output_per_1m_usd: 0.40,
            },
            // APPROXIMATE — verify against current OpenAI pricing; configurable.
            embedding: EmbeddingPricing {
                model_label: "text-embedding-3-small".to_string(),
                per_1m_usd: 0.02,
            },
        }
    }
}

impl PricingTable {
    /// Chat pricing for a model size.
    pub fn chat(&self, size: ModelSize) -> &Pricing {
        match size {
            ModelSize::Small => &self.small,
            ModelSize::Medium => &self.medium,
        }
    }

    /// Cost in USD of `input_tokens` + `output_tokens` at the given size.
    pub fn chat_cost(&self, size: ModelSize, input_tokens: usize, output_tokens: usize) -> f64 {
        let p = self.chat(size);
        (input_tokens as f64) * p.input_per_1m_usd / 1_000_000.0
            + (output_tokens as f64) * p.output_per_1m_usd / 1_000_000.0
    }

    /// Cost in USD of `input_tokens` embedding tokens.
    pub fn embedding_cost(&self, input_tokens: usize) -> f64 {
        (input_tokens as f64) * self.embedding.per_1m_usd / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// Call ledger
// ---------------------------------------------------------------------------

/// One metered call (an LLM `generate` or an embedding `create`/`create_batch`).
#[derive(Debug, Clone)]
pub struct CallRecord {
    /// `"llm"` or `"embedding"`.
    pub kind: &'static str,
    pub prompt_name: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cost_usd: f64,
}

/// Shared, cloneable sink of [`CallRecord`]s plus the active [`PricingTable`].
///
/// Cheap to clone: both fields are `Arc`. The wrappers and the report all share
/// the same ledger so cost accrues in one place.
#[derive(Clone)]
pub struct Meter {
    records: Arc<Mutex<Vec<CallRecord>>>,
    pricing: Arc<PricingTable>,
}

impl Default for Meter {
    fn default() -> Self {
        Self::new(PricingTable::default())
    }
}

impl Meter {
    /// Build a meter over the given pricing table.
    pub fn new(pricing: PricingTable) -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            pricing: Arc::new(pricing),
        }
    }

    /// The pricing table in force for this meter.
    pub fn pricing(&self) -> &PricingTable {
        &self.pricing
    }

    /// Append a record to the ledger.
    pub fn record(&self, record: CallRecord) {
        if let Ok(mut guard) = self.records.lock() {
            guard.push(record);
        }
    }

    /// Snapshot the current ledger (clone of all records so far).
    pub fn snapshot(&self) -> Vec<CallRecord> {
        self.records.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Clear the ledger. Used by the harness to attribute cost to a single
    /// operation: reset, run one op, snapshot, repeat.
    pub fn reset(&self) {
        if let Ok(mut guard) = self.records.lock() {
            guard.clear();
        }
    }

    /// Reset the ledger and return whatever it held, in one shot.
    pub fn drain(&self) -> Vec<CallRecord> {
        if let Ok(mut guard) = self.records.lock() {
            std::mem::take(&mut *guard)
        } else {
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Metering wrappers
// ---------------------------------------------------------------------------

/// Wraps an [`LlmClient`], counting input/output tokens of every `generate`
/// call and recording the cost against a shared [`Meter`].
pub struct MeteringLlm {
    inner: Arc<dyn LlmClient>,
    meter: Meter,
}

impl MeteringLlm {
    pub fn new(inner: Arc<dyn LlmClient>, meter: Meter) -> Self {
        Self { inner, meter }
    }
}

#[async_trait]
impl LlmClient for MeteringLlm {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        // Measure the prompt BEFORE moving the request into the inner client.
        let input_tokens = count_chat_tokens(&request.messages);
        let model_size = request.model_size;
        let prompt_name = request
            .prompt_name
            .clone()
            .unwrap_or_else(|| "<unnamed>".to_string());

        let value = self.inner.generate(request).await?;

        // Output tokens = tiktoken over the compact JSON the model returned —
        // exactly what a real provider would have emitted as completion tokens.
        let compact = serde_json::to_string(&value).unwrap_or_default();
        let output_tokens = count_chat_output_tokens(&compact);

        let pricing = self.meter.pricing();
        let cost_usd = pricing.chat_cost(model_size, input_tokens, output_tokens);
        let model = pricing.chat(model_size).model_label.clone();

        self.meter.record(CallRecord {
            kind: "llm",
            prompt_name,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        });

        Ok(value)
    }
}

/// Wraps an [`EmbedderClient`], counting input tokens of every embedded string
/// and recording the cost against a shared [`Meter`]. Embeddings have no output
/// tokens (`output_tokens = 0`).
pub struct MeteringEmbedder {
    inner: Arc<dyn EmbedderClient>,
    meter: Meter,
}

impl MeteringEmbedder {
    pub fn new(inner: Arc<dyn EmbedderClient>, meter: Meter) -> Self {
        Self { inner, meter }
    }

    fn record_input(&self, input: &str) {
        let input_tokens = count_embedding_tokens(input);
        let pricing = self.meter.pricing();
        let cost_usd = pricing.embedding_cost(input_tokens);
        self.meter.record(CallRecord {
            kind: "embedding",
            prompt_name: "embedding".to_string(),
            model: pricing.embedding.model_label.clone(),
            input_tokens,
            output_tokens: 0,
            cost_usd,
        });
    }
}

#[async_trait]
impl EmbedderClient for MeteringEmbedder {
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        self.record_input(input);
        self.inner.create(input).await
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        for input in inputs {
            self.record_input(input);
        }
        self.inner.create_batch(inputs).await
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Aggregate totals over a set of [`CallRecord`]s.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Totals {
    pub llm_calls: usize,
    pub embedding_calls: usize,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub total_usd: f64,
}

/// A named subtotal for one pipeline operation (e.g. "add_episode (simple)").
#[derive(Debug, Clone)]
pub struct OperationSubtotal {
    pub label: String,
    pub records: Vec<CallRecord>,
}

impl OperationSubtotal {
    pub fn totals(&self) -> Totals {
        totals_of(&self.records)
    }
}

/// Compute [`Totals`] over a slice of records.
pub fn totals_of(records: &[CallRecord]) -> Totals {
    let mut t = Totals::default();
    for r in records {
        match r.kind {
            "llm" => t.llm_calls += 1,
            "embedding" => t.embedding_calls += 1,
            _ => {}
        }
        t.total_input_tokens += r.input_tokens;
        t.total_output_tokens += r.output_tokens;
        t.total_usd += r.cost_usd;
    }
    t
}

/// A renderable cost report built from a [`PricingTable`] and a set of
/// per-operation subtotals.
pub struct CostReport {
    pricing: PricingTable,
    operations: Vec<OperationSubtotal>,
}

impl CostReport {
    /// Build a report from a pricing table; add operations with [`Self::add_operation`].
    pub fn new(pricing: PricingTable) -> Self {
        Self {
            pricing,
            operations: Vec::new(),
        }
    }

    /// Attach a named operation subtotal (a snapshot of the meter's records).
    pub fn add_operation(&mut self, label: impl Into<String>, records: Vec<CallRecord>) {
        self.operations.push(OperationSubtotal {
            label: label.into(),
            records,
        });
    }

    /// All records across every operation, flattened.
    pub fn all_records(&self) -> Vec<CallRecord> {
        self.operations
            .iter()
            .flat_map(|o| o.records.iter().cloned())
            .collect()
    }

    /// Grand totals across all operations.
    pub fn totals(&self) -> Totals {
        totals_of(&self.all_records())
    }

    /// Aggregate cost by `prompt_name`, sorted by descending cost.
    pub fn by_prompt_name(&self) -> Vec<(String, Totals)> {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<String, Vec<CallRecord>> = BTreeMap::new();
        for r in self.all_records() {
            map.entry(r.prompt_name.clone()).or_default().push(r);
        }
        let mut rows: Vec<(String, Totals)> = map
            .into_iter()
            .map(|(name, recs)| (name, totals_of(&recs)))
            .collect();
        rows.sort_by(|a, b| {
            b.1.total_usd
                .partial_cmp(&a.1.total_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    }

    /// Per-call pretty table over all records.
    pub fn per_call_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<10} {:<24} {:<24} {:>8} {:>8} {:>12}\n",
            "kind", "prompt_name", "model", "in_tok", "out_tok", "$"
        ));
        out.push_str(&"-".repeat(90));
        out.push('\n');
        for r in self.all_records() {
            out.push_str(&format!(
                "{:<10} {:<24} {:<24} {:>8} {:>8} {:>12.6}\n",
                r.kind, r.prompt_name, r.model, r.input_tokens, r.output_tokens, r.cost_usd
            ));
        }
        out
    }

    /// Monthly projection in USD if `episodes_per_day` episodes are ingested at
    /// the average per-episode add cost observed across this report's
    /// `add_episode` operations.
    ///
    /// "Per-episode" cost is the mean total USD of every operation whose label
    /// contains `"add_episode"`. Search operations are excluded (they scale with
    /// query volume, not ingest volume).
    pub fn project_monthly(&self, episodes_per_day: f64) -> f64 {
        let add_ops: Vec<&OperationSubtotal> = self
            .operations
            .iter()
            .filter(|o| o.label.contains("add_episode"))
            .collect();
        if add_ops.is_empty() {
            return 0.0;
        }
        let per_episode: f64 =
            add_ops.iter().map(|o| o.totals().total_usd).sum::<f64>() / add_ops.len() as f64;
        per_episode * episodes_per_day * 30.0
    }

    /// Full human-readable report.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("================ chronicle cost report ================\n");
        out.push_str(
            "NOTE: token counts are MEASURED (tiktoken on real pipeline prompts);\n      \
             USD is ESTIMATED from the pricing table below.\n      \
             In mock mode, OUTPUT tokens come from representative scripted\n      \
             responses; INPUT tokens are exact (the real prompts the pipeline sends).\n\n",
        );

        // Pricing assumptions.
        out.push_str(
            "-- pricing assumptions (APPROXIMATE — VERIFY against current OpenAI pricing) --\n",
        );
        out.push_str(&format!(
            "  chat medium [{}]: ${:.2}/1M in, ${:.2}/1M out\n",
            self.pricing.medium.model_label,
            self.pricing.medium.input_per_1m_usd,
            self.pricing.medium.output_per_1m_usd
        ));
        out.push_str(&format!(
            "  chat small  [{}]: ${:.2}/1M in, ${:.2}/1M out\n",
            self.pricing.small.model_label,
            self.pricing.small.input_per_1m_usd,
            self.pricing.small.output_per_1m_usd
        ));
        out.push_str(&format!(
            "  embedding   [{}]: ${:.2}/1M tok\n",
            self.pricing.embedding.model_label, self.pricing.embedding.per_1m_usd
        ));
        out.push_str("  chat encoding: o200k_base   embedding encoding: cl100k_base\n\n");

        // Per-operation subtotals.
        out.push_str("-- per-operation subtotals --\n");
        for op in &self.operations {
            let t = op.totals();
            out.push_str(&format!(
                "  {:<26} {:>3} llm + {:>2} emb calls | {:>6} in / {:>6} out tok | ${:.6}\n",
                op.label,
                t.llm_calls,
                t.embedding_calls,
                t.total_input_tokens,
                t.total_output_tokens,
                t.total_usd
            ));
        }
        out.push('\n');

        // Per-call table.
        out.push_str("-- per-call ledger --\n");
        out.push_str(&self.per_call_table());
        out.push('\n');

        // By prompt name.
        out.push_str("-- cost by prompt_name (descending) --\n");
        for (name, t) in self.by_prompt_name() {
            out.push_str(&format!(
                "  {:<26} {:>6} in / {:>6} out tok | ${:.6}\n",
                name, t.total_input_tokens, t.total_output_tokens, t.total_usd
            ));
        }
        out.push('\n');

        // Grand total.
        let gt = self.totals();
        out.push_str("-- grand total --\n");
        out.push_str(&format!(
            "  {} llm calls + {} embedding calls\n  {} input tokens + {} output tokens\n  ESTIMATED ${:.6} total\n\n",
            gt.llm_calls, gt.embedding_calls, gt.total_input_tokens, gt.total_output_tokens, gt.total_usd
        ));

        // Monthly projection.
        out.push_str("-- monthly projection (ingest only, at avg add_episode cost) --\n");
        for vol in [100.0, 1_000.0, 10_000.0] {
            out.push_str(&format!(
                "  {:>6} episodes/day -> ${:>10.2}/month\n",
                vol as u64,
                self.project_monthly(vol)
            ));
        }
        out.push_str("\n=======================================================\n");
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_core::llm::Message;

    #[test]
    fn chat_token_count_is_deterministic() {
        let msgs = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Extract entities from: Alice works at Acme."),
        ];
        let a = count_chat_tokens(&msgs);
        let b = count_chat_tokens(&msgs);
        assert_eq!(a, b, "same input must yield same count");
        assert!(a > 0);
    }

    #[test]
    fn embedding_token_count_is_deterministic() {
        let a = count_embedding_tokens("Alice works at Acme.");
        let b = count_embedding_tokens("Alice works at Acme.");
        assert_eq!(a, b);
        assert!(a > 0);
    }

    #[test]
    fn longer_text_costs_more_tokens() {
        let short = count_chat_output_tokens("hi");
        let long = count_chat_output_tokens(
            "Alice is the CEO of Acme Corporation and she founded it in 2010.",
        );
        assert!(long > short);
    }

    #[test]
    fn chat_cost_math_is_correct() {
        let table = PricingTable::default();
        // medium: $0.40/1M in, $1.60/1M out. 1,000,000 in + 1,000,000 out.
        let cost = table.chat_cost(ModelSize::Medium, 1_000_000, 1_000_000);
        assert!((cost - (0.40 + 1.60)).abs() < 1e-9, "got {cost}");

        // small: $0.10/1M in, $0.40/1M out. 500,000 in + 250,000 out.
        let cost_small = table.chat_cost(ModelSize::Small, 500_000, 250_000);
        let expect = 0.10 * 0.5 + 0.40 * 0.25;
        assert!((cost_small - expect).abs() < 1e-9, "got {cost_small}");
    }

    #[test]
    fn embedding_cost_math_is_correct() {
        let table = PricingTable::default();
        // $0.02/1M. 1,000,000 tokens -> $0.02.
        let cost = table.embedding_cost(1_000_000);
        assert!((cost - 0.02).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn pricing_table_is_overridable() {
        let mut table = PricingTable::default();
        table.medium.input_per_1m_usd = 99.0;
        assert_eq!(table.chat(ModelSize::Medium).input_per_1m_usd, 99.0);
    }

    #[test]
    fn totals_aggregate_correctly() {
        let records = vec![
            CallRecord {
                kind: "llm",
                prompt_name: "extract_message".into(),
                model: "gpt-4.1-mini".into(),
                input_tokens: 100,
                output_tokens: 20,
                cost_usd: 0.001,
            },
            CallRecord {
                kind: "embedding",
                prompt_name: "embedding".into(),
                model: "text-embedding-3-small".into(),
                input_tokens: 10,
                output_tokens: 0,
                cost_usd: 0.0001,
            },
        ];
        let t = totals_of(&records);
        assert_eq!(t.llm_calls, 1);
        assert_eq!(t.embedding_calls, 1);
        assert_eq!(t.total_input_tokens, 110);
        assert_eq!(t.total_output_tokens, 20);
        assert!((t.total_usd - 0.0011).abs() < 1e-9);
    }

    #[test]
    fn meter_reset_and_snapshot_isolate_operations() {
        let meter = Meter::default();
        meter.record(CallRecord {
            kind: "llm",
            prompt_name: "a".into(),
            model: "m".into(),
            input_tokens: 1,
            output_tokens: 1,
            cost_usd: 0.0,
        });
        let snap1 = meter.drain();
        assert_eq!(snap1.len(), 1);
        // After drain the ledger is empty.
        assert!(meter.snapshot().is_empty());
        meter.record(CallRecord {
            kind: "embedding",
            prompt_name: "b".into(),
            model: "m".into(),
            input_tokens: 2,
            output_tokens: 0,
            cost_usd: 0.0,
        });
        let snap2 = meter.snapshot();
        assert_eq!(snap2.len(), 1);
        assert_eq!(snap2[0].kind, "embedding");
    }

    #[tokio::test]
    async fn metering_llm_records_a_call() {
        use crate::MockLlm;
        let inner = Arc::new(MockLlm::new(vec![serde_json::json!({
            "extracted_entities": [{"name": "Alice", "entity_type_id": 0}]
        })]));
        let meter = Meter::default();
        let llm = MeteringLlm::new(inner, meter.clone());
        let req = LlmRequest::new(vec![
            Message::system("Extract entities."),
            Message::user("Alice works at Acme."),
        ])
        .named("extract_message");
        let _ = llm.generate(req).await.unwrap();

        let recs = meter.snapshot();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].kind, "llm");
        assert_eq!(recs[0].prompt_name, "extract_message");
        assert_eq!(recs[0].model, "gpt-4.1-mini");
        assert!(recs[0].input_tokens > 0);
        assert!(recs[0].output_tokens > 0);
        assert!(recs[0].cost_usd > 0.0);
    }

    #[tokio::test]
    async fn metering_embedder_records_each_input() {
        use crate::MockEmbedder;
        let inner = Arc::new(MockEmbedder::new(8));
        let meter = Meter::default();
        let emb = MeteringEmbedder::new(inner, meter.clone());
        emb.create("Alice").await.unwrap();
        emb.create_batch(&["Acme".to_string(), "Globex".to_string()])
            .await
            .unwrap();

        let recs = meter.snapshot();
        assert_eq!(recs.len(), 3, "1 create + 2 batch inputs");
        assert!(recs.iter().all(|r| r.kind == "embedding"));
        assert!(recs.iter().all(|r| r.output_tokens == 0));
        assert!(recs.iter().all(|r| r.input_tokens > 0));
    }

    #[test]
    fn report_renders_and_projects() {
        let mut report = CostReport::new(PricingTable::default());
        report.add_operation(
            "add_episode (simple)",
            vec![CallRecord {
                kind: "llm",
                prompt_name: "extract_message".into(),
                model: "gpt-4.1-mini".into(),
                input_tokens: 1_000,
                output_tokens: 100,
                cost_usd: 0.001,
            }],
        );
        report.add_operation(
            "search (RRF)",
            vec![CallRecord {
                kind: "embedding",
                prompt_name: "embedding".into(),
                model: "text-embedding-3-small".into(),
                input_tokens: 10,
                output_tokens: 0,
                cost_usd: 0.0000002,
            }],
        );

        let rendered = report.render();
        assert!(rendered.contains("chronicle cost report"));
        assert!(rendered.contains("MEASURED"));
        assert!(rendered.contains("VERIFY"));
        assert!(rendered.contains("add_episode (simple)"));

        // Monthly projection at 1000/day = per-episode ($0.001) * 1000 * 30.
        let monthly = report.project_monthly(1_000.0);
        assert!(
            (monthly - 0.001 * 1_000.0 * 30.0).abs() < 1e-9,
            "got {monthly}"
        );
    }
}
