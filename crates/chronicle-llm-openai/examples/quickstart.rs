//! Minimal end-to-end usage example.
//!
//! Requires:
//!   - A running Neo4j instance (default: `bolt://localhost:7687`)
//!   - `OPENAI_API_KEY` env var (or set via `LlmConfig::api_key`)
//!
//! Run with:
//!   ```sh
//!   NEO4J_URI=bolt://localhost:7687 \
//!   NEO4J_USER=neo4j \
//!   NEO4J_PASSWORD=password \
//!   OPENAI_API_KEY=sk-... \
//!   cargo run -p chronicle-llm-openai --example quickstart
//!   ```

use std::sync::Arc;

use chronicle_core::chronicle::{AddEpisodeRequest, Chronicle};
use chronicle_core::llm::LlmConfig;
use chronicle_core::search::config::edge_hybrid_search_rrf;
use chronicle_core::types::EpisodeType;
use chronicle_driver_neo4j::Neo4jDriver;
use chronicle_llm_openai::{OpenAiEmbedder, OpenAiEmbedderConfig, OpenAiLlm};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // --- Connect driver ---
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".into());

    let driver = Arc::new(
        Neo4jDriver::connect(&uri, &user, &password, "neo4j")
            .await
            .map_err(|e| anyhow::anyhow!("Neo4j connect: {e}"))?,
    );

    // --- Build LLM client (reads OPENAI_API_KEY from env) ---
    let llm = Arc::new(OpenAiLlm::new(LlmConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?);

    // --- Build embedder ---
    let embedder = Arc::new(
        OpenAiEmbedder::new(OpenAiEmbedderConfig::default())
            .map_err(|e| anyhow::anyhow!("embedder: {e}"))?,
    );

    // --- Construct Chronicle ---
    let chronicle = Chronicle::new(
        driver, llm, embedder, 0, /* use default SEMAPHORE_LIMIT */
    );

    // --- Build indices (idempotent) ---
    chronicle.build_indices_and_constraints(false).await?;

    // --- Add an episode ---
    let result = chronicle
        .add_episode(AddEpisodeRequest {
            name: "quickstart-episode-1".into(),
            episode_body:
                "Alice is an engineer at Acme Corp. She works with Bob on the Graphiti project."
                    .into(),
            source: EpisodeType::Text,
            source_description: "quickstart example".into(),
            reference_time: chrono::Utc::now(),
            group_id: "quickstart".into(),
            uuid: None,
            previous_episode_uuids: None,
            entity_types: None,
            custom_extraction_instructions: None,
        })
        .await?;

    println!(
        "Ingested episode {} — {} nodes, {} edges",
        result.episode.uuid,
        result.nodes.len(),
        result.edges.len(),
    );

    // --- Hybrid search ---
    let group_ids = vec!["quickstart".to_string()];
    let search_cfg = edge_hybrid_search_rrf();
    let hits = chronicle
        .search("Alice engineer", &group_ids, &search_cfg)
        .await?;

    println!("Search returned {} edge(s):", hits.len());
    for edge in &hits {
        println!("  [{}] {}", edge.uuid, edge.fact);
    }

    Ok(())
}
