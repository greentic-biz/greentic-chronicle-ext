// Ported from graphiti_core/graphiti_types.py::GraphitiClients @ 34f56e65 (v0.29.1)
//
// Upstream `GraphitiClients` bundles the driver, LLM client, embedder, and the
// cross-call concurrency semaphore (`SEMAPHORE_LIMIT`). Pipeline functions take
// this bundle so they can fan out parallel work (candidate search, batch
// embedding) under a shared concurrency cap, exactly as upstream does via
// `semaphore_gather`.

use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::driver::GraphDriver;
use crate::embedder::EmbedderClient;
use crate::llm::LlmClient;

/// Shared client bundle passed to every pipeline operation.
#[derive(Clone)]
pub struct Clients {
    pub driver: Arc<dyn GraphDriver>,
    pub llm: Arc<dyn LlmClient>,
    pub embedder: Arc<dyn EmbedderClient>,
    /// Caps concurrent fan-out (upstream `SEMAPHORE_LIMIT`-bounded
    /// `semaphore_gather`).
    pub semaphore: Arc<Semaphore>,
}

impl Clients {
    /// Build a bundle with a fresh semaphore permitting `max_concurrency`
    /// simultaneous in-flight operations.
    pub fn new(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        max_concurrency: usize,
    ) -> Self {
        // Semaphore::new(0) would deadlock every fan-out; clamp to at least 1.
        let permits = max_concurrency.max(1);
        Self {
            driver,
            llm,
            embedder,
            semaphore: Arc::new(Semaphore::new(permits)),
        }
    }
}
