use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use chronicle_core::llm::{LlmClient, LlmError, LlmRequest};

/// Replays queued JSON responses in order; records incoming requests.
pub struct MockLlm {
    responses: Mutex<VecDeque<serde_json::Value>>,
    pub requests: Mutex<Vec<LlmRequest>>,
}

impl MockLlm {
    pub fn new(responses: Vec<serde_json::Value>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
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
        self.requests
            .lock()
            .map_err(|_| LlmError::Transport("poisoned".into()))?
            .push(request);
        self.responses
            .lock()
            .map_err(|_| LlmError::Transport("poisoned".into()))?
            .pop_front()
            .ok_or(LlmError::EmptyResponse)
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
