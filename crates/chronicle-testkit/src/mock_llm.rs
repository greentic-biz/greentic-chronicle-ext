use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use async_trait::async_trait;
use chronicle_core::llm::{LlmClient, LlmError, LlmRequest};

/// Routing strategy for [`MockLlm`] responses.
enum Mode {
    /// FIFO queue: each call pops the next response regardless of prompt.
    Ordered(VecDeque<serde_json::Value>),
    /// Keyed by `prompt_name`: each call returns the registered response for its
    /// prompt name. This decouples scripting from call ORDER, which is essential
    /// when the pipeline fans out parallel tasks (bulk ingest). A prompt with no
    /// registered key yields [`LlmError::EmptyResponse`].
    Keyed(HashMap<String, serde_json::Value>),
}

/// Replays queued JSON responses; records incoming requests.
///
/// Two modes:
/// - [`MockLlm::new`] — ordered FIFO queue (deterministic only when call order is
///   deterministic, e.g. concurrency pinned to 1 along a single linear path).
/// - [`MockLlm::keyed`] — route by `prompt_name`, order-independent. Use this for
///   the parallel bulk path.
pub struct MockLlm {
    mode: Mutex<Mode>,
    pub requests: Mutex<Vec<LlmRequest>>,
}

impl MockLlm {
    pub fn new(responses: Vec<serde_json::Value>) -> Self {
        Self {
            mode: Mutex::new(Mode::Ordered(responses.into())),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Route responses by `prompt_name`. Each entry `(prompt_name, response)` is
    /// returned for every call whose request carries that `prompt_name`.
    pub fn keyed(responses: Vec<(&str, serde_json::Value)>) -> Self {
        let map = responses
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Self {
            mode: Mutex::new(Mode::Keyed(map)),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Number of LLM calls made so far.
    pub fn call_count(&self) -> usize {
        self.requests.lock().map(|r| r.len()).unwrap_or(0)
    }
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        let prompt_name = request.prompt_name.clone();
        self.requests
            .lock()
            .map_err(|_| LlmError::Transport("poisoned".into()))?
            .push(request);
        let mut mode = self
            .mode
            .lock()
            .map_err(|_| LlmError::Transport("poisoned".into()))?;
        match &mut *mode {
            Mode::Ordered(queue) => queue.pop_front().ok_or(LlmError::EmptyResponse),
            Mode::Keyed(map) => prompt_name
                .as_deref()
                .and_then(|name| map.get(name).cloned())
                .ok_or(LlmError::EmptyResponse),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronicle_core::llm::message::Message;

    #[tokio::test]
    async fn replays_responses_in_order() {
        let mock = MockLlm::new(vec![
            serde_json::json!({"step": 1}),
            serde_json::json!({"step": 2}),
        ]);
        let req1 = LlmRequest::new(vec![Message::user("first")]);
        let req2 = LlmRequest::new(vec![Message::user("second")]);

        let r1 = mock.generate(req1).await.unwrap();
        let r2 = mock.generate(req2).await.unwrap();

        assert_eq!(r1["step"], 1);
        assert_eq!(r2["step"], 2);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn records_requests() {
        let mock = MockLlm::new(vec![serde_json::json!({})]);
        let req = LlmRequest::new(vec![Message::user("hello")]).named("test-prompt");
        mock.generate(req).await.unwrap();

        let requests = mock.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].prompt_name.as_deref(), Some("test-prompt"));
    }

    #[tokio::test]
    async fn empty_response_when_exhausted() {
        let mock = MockLlm::new(vec![]);
        let req = LlmRequest::new(vec![Message::user("too many")]);
        let err = mock.generate(req).await.unwrap_err();
        assert!(matches!(err, LlmError::EmptyResponse));
    }
}
