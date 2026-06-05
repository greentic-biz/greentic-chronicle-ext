use async_trait::async_trait;
use chronicle_core::embedder::{EmbedderClient, EmbedderError};

/// Deterministic embedder using a rolling-byte-hash accumulation, L2-normalised.
/// Identical strings always produce identical vectors; distinctly spelled strings
/// produce meaningfully different vectors. Not semantically meaningful — for
/// deterministic test assertions only.
pub struct MockEmbedder {
    pub dim: usize,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self { dim: 4 }
    }
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn embed_internal(&self, input: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for (i, b) in input.bytes().enumerate() {
            v[i % self.dim] += f32::from(b) / 255.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl EmbedderClient for MockEmbedder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    async fn create(&self, input: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embed_internal(input))
    }

    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        Ok(inputs.iter().map(|s| self.embed_internal(s)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_string_identical_embedding() {
        let emb = MockEmbedder::new(8);
        let a = emb.create("hello").await.unwrap();
        let b = emb.create("hello").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn different_strings_different_embeddings() {
        let emb = MockEmbedder::new(8);
        let a = emb.create("alpha").await.unwrap();
        let b = emb.create("beta").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn embedding_is_l2_normalised() {
        let emb = MockEmbedder::new(4);
        let v = emb.create("test").await.unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[tokio::test]
    async fn batch_matches_individual() {
        let emb = MockEmbedder::new(4);
        let batch = emb
            .create_batch(&["foo".to_string(), "bar".to_string()])
            .await
            .unwrap();
        let single_foo = emb.create("foo").await.unwrap();
        let single_bar = emb.create("bar").await.unwrap();
        assert_eq!(batch[0], single_foo);
        assert_eq!(batch[1], single_bar);
    }

    #[test]
    fn dim_reported_correctly() {
        let emb = MockEmbedder::new(16);
        assert_eq!(emb.embedding_dim(), 16);
    }
}
