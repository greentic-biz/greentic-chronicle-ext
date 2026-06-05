// Ported from graphiti_core/llm_client/client.py @ 34f56e65 (v0.29.1)
//
// Upstream tenacity policy on `_generate_response_with_retry`:
//   stop_after_attempt(4)
//   wait_random_exponential(multiplier=10, min=5, max=120)
//   retry_if_exception(is_server_or_retry_error)
//     where is_server_or_retry_error returns True for:
//       RateLimitError, json.JSONDecodeError, and httpx.HTTPStatusError with 5xx status

use std::future::Future;
use std::time::Duration;

use super::LlmError;

pub const MAX_ATTEMPTS: u32 = 4;
const BACKOFF_MIN_SECS: f64 = 5.0;
const BACKOFF_MAX_SECS: f64 = 120.0;
const BACKOFF_MULTIPLIER: f64 = 10.0;

/// Retry an LLM call with randomized exponential backoff (upstream tenacity policy:
/// up to 4 attempts, wait_random_exponential(multiplier=10, min=5, max=120)).
pub async fn with_retry<T, F, Fut>(mut call: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
{
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match call().await {
            Ok(value) => return Ok(value),
            Err(err) if err.is_retryable() && attempt < MAX_ATTEMPTS => {
                let cap = (BACKOFF_MULTIPLIER * 2f64.powi(attempt as i32 - 1))
                    .clamp(BACKOFF_MIN_SECS, BACKOFF_MAX_SECS);
                let wait = (rand::random::<f64>() * cap).clamp(BACKOFF_MIN_SECS, BACKOFF_MAX_SECS);
                tracing::warn!(attempt, wait_secs = wait, error = %err, "retrying LLM call");
                tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            }
            Err(err) => return Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    #[tokio::test(start_paused = true)]
    async fn retryable_error_retries_up_to_max_attempts() {
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = Arc::clone(&count);

        let result = with_retry(|| {
            let c = Arc::clone(&count_clone);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(LlmError::RateLimit)
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), MAX_ATTEMPTS);
    }

    #[tokio::test(start_paused = true)]
    async fn non_retryable_error_returns_after_one_attempt() {
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = Arc::clone(&count);

        let result = with_retry(|| {
            let c = Arc::clone(&count_clone);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(LlmError::Refusal("I refuse".into()))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
