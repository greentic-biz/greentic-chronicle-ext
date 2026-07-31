use std::collections::HashMap;

use async_trait::async_trait;
use chronicle_core::cross_encoder::CrossEncoderClient;
use chronicle_core::llm::LlmError;

/// Deterministic [`CrossEncoderClient`] for tests.
///
/// Scores are scripted by exact passage string via a `HashMap`. Any passage
/// not present in the map scores `0.0` (mirroring the "unmentioned" default of
/// the real reranker). `rank` returns the passages sorted in descending score
/// order using a **stable** sort, so passages with equal scores preserve their
/// input order — keeping test assertions deterministic.
#[derive(Debug, Default, Clone)]
pub struct MockCrossEncoder {
    scores: HashMap<String, f64>,
}

impl MockCrossEncoder {
    /// Build a mock from an explicit passage → score map.
    pub fn new(scores: HashMap<String, f64>) -> Self {
        Self { scores }
    }

    /// Convenience constructor from `(passage, score)` pairs.
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, f64)>,
        S: Into<String>,
    {
        Self {
            scores: pairs.into_iter().map(|(p, s)| (p.into(), s)).collect(),
        }
    }

    /// Score for a passage, defaulting to `0.0` when not scripted.
    fn score_for(&self, passage: &str) -> f64 {
        self.scores.get(passage).copied().unwrap_or(0.0)
    }
}

#[async_trait]
impl CrossEncoderClient for MockCrossEncoder {
    async fn rank(
        &self,
        _query: &str,
        passages: &[String],
    ) -> Result<Vec<(String, f64)>, LlmError> {
        let mut ranked: Vec<(String, f64)> = passages
            .iter()
            .map(|p| (p.clone(), self.score_for(p)))
            .collect();
        // Stable, descending by score. `total_cmp` keeps NaN-free f64 ordering
        // well-defined; `reverse()` of `a.cmp(b)` yields descending order while
        // a stable sort preserves input order for ties.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ranks_descending_by_score() {
        let mock = MockCrossEncoder::from_pairs([("low", 0.1), ("high", 0.9), ("mid", 0.5)]);
        let passages = vec!["low".to_string(), "high".to_string(), "mid".to_string()];
        let ranked = mock.rank("q", &passages).await.unwrap();
        let order: Vec<&str> = ranked.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(order, vec!["high", "mid", "low"]);
        assert_eq!(ranked[0].1, 0.9);
    }

    #[tokio::test]
    async fn missing_passages_score_zero() {
        let mock = MockCrossEncoder::from_pairs([("known", 0.7)]);
        let passages = vec!["known".to_string(), "unknown".to_string()];
        let ranked = mock.rank("q", &passages).await.unwrap();
        assert_eq!(ranked[0], ("known".to_string(), 0.7));
        assert_eq!(ranked[1], ("unknown".to_string(), 0.0));
    }

    #[tokio::test]
    async fn ties_preserve_input_order_stable() {
        let mock = MockCrossEncoder::from_pairs([("a", 0.5), ("b", 0.5), ("c", 0.5)]);
        let passages = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        let ranked = mock.rank("q", &passages).await.unwrap();
        let order: Vec<&str> = ranked.iter().map(|(p, _)| p.as_str()).collect();
        // Equal scores keep the original input ordering.
        assert_eq!(order, vec!["b", "a", "c"]);
    }

    #[tokio::test]
    async fn deterministic_across_runs() {
        let mock = MockCrossEncoder::from_pairs([("x", 0.3), ("y", 0.8)]);
        let passages = vec!["x".to_string(), "y".to_string()];
        let first = mock.rank("q", &passages).await.unwrap();
        let second = mock.rank("q", &passages).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn empty_passages_yields_empty() {
        let mock = MockCrossEncoder::default();
        let ranked = mock.rank("q", &[]).await.unwrap();
        assert!(ranked.is_empty());
    }
}
