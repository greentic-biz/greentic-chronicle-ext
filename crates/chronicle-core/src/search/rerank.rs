// Ported from graphiti_core/search/search_utils.py @ 34f56e65:
//   - maximal_marginal_relevance (lines ~1901-1939)
//   - node_distance_reranker     (lines ~1798-1857)
//   - episode_mentions_reranker  (lines ~1860-1898)
// normalize_l2 ported from graphiti_core/helpers.py::normalize_l2 (lines ~116-119).
//
// Upstream is numpy float64. Embeddings arrive as f32; we convert to f64 *before*
// any arithmetic so the math matches upstream precision as closely as possible.
//
// Fidelity notes (see docs/port-fidelity.md):
//   - normalize_l2 zero-vector guard: `np.where(norm == 0, arr, arr / norm)` →
//     a zero vector is returned unchanged (NOT divided by zero). Replicated.
//   - MMR `max_sim = np.max(similarity_matrix[i, :])` includes the zero diagonal,
//     so for a single candidate max_sim = 0, and in general max_sim >= 0 because the
//     diagonal 0 participates in the max. Replicated (max over the full row incl. 0).
//   - node_distance returns `1 / score` and filters on `(1 / score) >= min_score`
//     (1/inf = 0.0). Replicated literally.
//   - episode_mentions UPSTREAM QUIRK: ASC sort by mention count (fewer mentions
//     rank higher) and the literal filter `score >= min_score` retains unmentioned
//     nodes (score = inf) at the END. Replicated bug-for-bug.

use crate::driver::GraphDriver;
use crate::errors::ChronicleError;
use crate::search::rrf::rrf;
use std::collections::HashMap;

/// L2-normalize an embedding, replicating upstream `normalize_l2`.
///
/// Upstream (`helpers.py`):
/// ```python
/// norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
/// return np.where(norm == 0, embedding_array, embedding_array / norm)
/// ```
/// A zero vector (`norm == 0`) is returned unchanged rather than producing NaN.
fn normalize_l2(embedding: &[f32]) -> Vec<f64> {
    let v: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        // Zero-vector guard: return the (zero) vector unchanged.
        v
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Maximal Marginal Relevance reranking.
///
/// Ported from `maximal_marginal_relevance` (search_utils.py @ 34f56e65).
///
/// - `candidates` are insertion-ordered `(uuid, raw embedding)`; the returned order
///   for tied MMR scores preserves this insertion order (stable sort).
/// - Candidate embeddings are L2-normalized; the query vector is NOT normalized
///   (matches upstream — only `candidate_arrays` go through `normalize_l2`).
/// - Pairwise dot-product similarity matrix with a ZERO diagonal.
/// - `mmr = mmr_lambda * dot(query, cand) + (mmr_lambda - 1) * max_sim`, where
///   `max_sim` is the max over the FULL similarity-matrix row (including the zero
///   diagonal) → single candidate ⇒ max_sim = 0.
/// - Sort descending by MMR, then filter `>= min_score` (upstream default -2.0).
pub fn maximal_marginal_relevance(
    query_vector: &[f32],
    candidates: &[(String, Vec<f32>)],
    mmr_lambda: f64,
    min_score: f64,
) -> (Vec<String>, Vec<f64>) {
    // Query vector to f64 (NOT normalized — mirrors upstream `np.array(query_vector)`).
    let query: Vec<f64> = query_vector.iter().map(|&x| x as f64).collect();

    // Preserve insertion order of uuids; normalize each candidate embedding.
    let uuids: Vec<String> = candidates.iter().map(|(u, _)| u.clone()).collect();
    let arrays: Vec<Vec<f64>> = candidates.iter().map(|(_, e)| normalize_l2(e)).collect();

    let n = uuids.len();

    // Symmetric pairwise dot-product matrix, diagonal left at 0.0.
    let mut similarity = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..i {
            let s = dot(&arrays[i], &arrays[j]);
            similarity[i][j] = s;
            similarity[j][i] = s;
        }
    }

    // MMR score per candidate.
    let mut mmr_scores: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        // max over the FULL row including the zero diagonal (upstream
        // `np.max(similarity_matrix[i, :])`). Seed with 0.0 so an empty/single
        // row yields 0.0 and the diagonal zero always participates.
        let mut max_sim = 0.0_f64;
        for &s in &similarity[i] {
            if s > max_sim {
                max_sim = s;
            }
        }
        let mmr = mmr_lambda * dot(&query, &arrays[i]) + (mmr_lambda - 1.0) * max_sim;
        mmr_scores.push(mmr);
    }

    // Stable sort descending by MMR score (ties keep insertion order).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        mmr_scores[b]
            .partial_cmp(&mmr_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out_uuids = Vec::new();
    let mut out_scores = Vec::new();
    for idx in order {
        if mmr_scores[idx] >= min_score {
            out_uuids.push(uuids[idx].clone());
            out_scores.push(mmr_scores[idx]);
        }
    }
    (out_uuids, out_scores)
}

/// Node-distance reranking (1-HOP undirected adjacency to a center node).
///
/// Ported from `node_distance_reranker` (search_utils.py @ 34f56e65).
///
/// - `filtered = node_uuids` minus `center` (order preserved).
/// - Adjacency comes from `driver.nodes_connected_to_center` → score 1.0; missing
///   uuids get `f64::INFINITY`.
/// - Sort ASC by score (stable).
/// - If `center` was present in the input, set its score to 0.1 and PREPEND it.
/// - Return `(uuids, 1/score)` keeping only entries where `(1 / score) >= min_score`.
///   `1 / inf = 0.0`, so unreachable nodes survive iff `min_score <= 0`.
pub async fn node_distance_rerank(
    driver: &dyn GraphDriver,
    node_uuids: &[String],
    center_node_uuid: &str,
    min_score: f64,
) -> Result<(Vec<String>, Vec<f64>), ChronicleError> {
    // filter out the center node (order preserved).
    let mut filtered: Vec<String> = node_uuids
        .iter()
        .filter(|u| u.as_str() != center_node_uuid)
        .cloned()
        .collect();

    // scores starts with {center: 0.0} (matches upstream seed; only consulted if
    // the center is later prepended).
    let mut scores: HashMap<String, f64> = HashMap::new();
    scores.insert(center_node_uuid.to_string(), 0.0);

    // Adjacency: connected uuids → score 1.0.
    let connected = driver
        .nodes_connected_to_center(&filtered, center_node_uuid)
        .await?;
    for uuid in &connected {
        scores.insert(uuid.clone(), 1.0);
    }

    // Missing → inf.
    for uuid in &filtered {
        scores.entry(uuid.clone()).or_insert(f64::INFINITY);
    }

    // Sort ascending by score (stable — ties keep prior order).
    filtered.sort_by(|a, b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Add back the center if it was in the input: score 0.1, prepended.
    if node_uuids.iter().any(|u| u.as_str() == center_node_uuid) {
        scores.insert(center_node_uuid.to_string(), 0.1);
        filtered.insert(0, center_node_uuid.to_string());
    }

    // Return uuids where (1 / score) >= min_score, with scores = 1 / score.
    let mut out_uuids = Vec::new();
    let mut out_scores = Vec::new();
    for uuid in &filtered {
        let inv = 1.0 / scores[uuid];
        if inv >= min_score {
            out_uuids.push(uuid.clone());
            out_scores.push(inv);
        }
    }
    Ok((out_uuids, out_scores))
}

/// Episode-mentions reranking.
///
/// Ported from `episode_mentions_reranker` (search_utils.py @ 34f56e65).
///
/// - RRF presort over the input uuid lists using upstream defaults
///   (`rank_const = 1`, `min_score = 0`).
/// - Mention counts come from `driver.episode_mention_counts`; missing uuids get
///   `f64::INFINITY`.
///
/// UPSTREAM QUIRK (bug-for-bug):
///   1. Sort is ASCENDING by mention count — **fewer mentions rank higher**. This
///      is counterintuitive but is exactly what upstream does
///      (`sorted_uuids.sort(key=lambda u: scores[u])`).
///   2. The filter is the literal `scores[uuid] >= min_score`, and unmentioned
///      uuids carry `inf`. Since `inf >= min_score` is always true, **unmentioned
///      nodes are RETAINED at the END** with score `inf`. The returned scores are
///      the raw counts / `inf` (NOT inverted).
///
/// See docs/port-fidelity.md ("episode_mentions ASC quirk").
pub async fn episode_mentions_rerank(
    driver: &dyn GraphDriver,
    uuid_result_lists: &[Vec<String>],
    min_score: f64,
) -> Result<(Vec<String>, Vec<f64>), ChronicleError> {
    // RRF presort with upstream defaults (rank_const=1, min_score=0).
    let (mut sorted_uuids, _) = rrf(uuid_result_lists, 1, 0.0);

    // Mention counts; missing → inf.
    let counts = driver.episode_mention_counts(&sorted_uuids).await?;
    let mut scores: HashMap<String, f64> = HashMap::new();
    for uuid in &sorted_uuids {
        let score = counts.get(uuid).map(|&c| c as f64).unwrap_or(f64::INFINITY);
        scores.insert(uuid.clone(), score);
    }

    // UPSTREAM QUIRK (bug-for-bug): ascending sort by count (fewer mentions first).
    sorted_uuids.sort_by(|a, b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // UPSTREAM QUIRK (bug-for-bug): literal `score >= min_score` filter. `inf` passes,
    // so unmentioned nodes are kept at the end. Returned scores are raw counts / inf.
    let mut out_uuids = Vec::new();
    let mut out_scores = Vec::new();
    for uuid in &sorted_uuids {
        let s = scores[uuid];
        if s >= min_score {
            out_uuids.push(uuid.clone());
            out_scores.push(s);
        }
    }
    Ok((out_uuids, out_scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalize_l2 ──────────────────────────────────────────────────────────

    #[test]
    fn normalize_l2_zero_vector_returned_unchanged() {
        // Upstream guard: norm == 0 → return input unchanged (no division by zero).
        let out = normalize_l2(&[0.0, 0.0, 0.0]);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
        assert!(out.iter().all(|x| x.is_finite()), "no NaN from zero vector");
    }

    #[test]
    fn normalize_l2_unit_length() {
        let out = normalize_l2(&[3.0, 4.0]);
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    // ── maximal_marginal_relevance ────────────────────────────────────────────

    fn cand(uuid: &str, e: &[f32]) -> (String, Vec<f32>) {
        (uuid.to_string(), e.to_vec())
    }

    #[test]
    fn mmr_lambda_one_is_pure_query_relevance_order() {
        // lambda=1.0 → mmr = dot(query, normalized_cand); diversity term drops out.
        // Query [1,0]; candidates ordered by descending alignment with query.
        let query = vec![1.0_f32, 0.0];
        let candidates = vec![
            cand("c_far", &[0.0, 1.0]),  // orthogonal → score 0
            cand("c_near", &[1.0, 0.0]), // aligned → score 1
            cand("c_mid", &[1.0, 1.0]),  // 45° → ~0.707
        ];
        let (uuids, scores) = maximal_marginal_relevance(&query, &candidates, 1.0, -2.0);
        assert_eq!(uuids, vec!["c_near", "c_mid", "c_far"]);
        // descending
        assert!(scores[0] >= scores[1] && scores[1] >= scores[2]);
        assert!((scores[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mmr_lambda_half_diverse_beats_near_duplicate() {
        // Two near-identical candidates + one diverse, all with *comparable* query
        // relevance (query bisects them). With lambda=0.5 the diversity penalty
        // demotes the second near-duplicate below the diverse candidate.
        //
        // query=[1,1]; dup_a/dup_b ≈ [1,0]; diverse = [0,1]. Each candidate has the
        // same query relevance (dot with normalized query ≈ 0.707), but dup_b is
        // highly similar to dup_a (max_sim ≈ 1) while diverse is orthogonal to the
        // dups (max_sim ≈ 0), so:
        //   dup_a:   0.5*0.707 + (-0.5)*~1.0    ≈ -0.146
        //   dup_b:   0.5*0.707 + (-0.5)*~1.0    ≈ -0.146  (penalized by dup_a)
        //   diverse: 0.5*0.707 + (-0.5)*~0.0    ≈  0.354  (no near-duplicate)
        let query = vec![1.0_f32, 1.0];
        let candidates = vec![
            cand("dup_a", &[1.0, 0.0]),
            cand("dup_b", &[1.0, 0.0]),
            cand("diverse", &[0.0, 1.0]),
        ];
        let (uuids, _scores) = maximal_marginal_relevance(&query, &candidates, 0.5, -2.0);
        let pos_diverse = uuids.iter().position(|u| u == "diverse").unwrap();
        let pos_dup_a = uuids.iter().position(|u| u == "dup_a").unwrap();
        let pos_dup_b = uuids.iter().position(|u| u == "dup_b").unwrap();
        assert_eq!(pos_diverse, 0, "diverse leads (order {uuids:?})");
        assert!(
            pos_diverse < pos_dup_a && pos_diverse < pos_dup_b,
            "diverse beats BOTH near-duplicates (order {uuids:?})"
        );
    }

    #[test]
    fn mmr_single_candidate_max_sim_is_zero() {
        // Single candidate → similarity row is just the zero diagonal → max_sim 0.
        // mmr = lambda*dot(query,cand) + (lambda-1)*0 = lambda*dot.
        let query = vec![1.0_f32, 0.0];
        let candidates = vec![cand("only", &[1.0, 0.0])];
        let (uuids, scores) = maximal_marginal_relevance(&query, &candidates, 0.5, -2.0);
        assert_eq!(uuids, vec!["only"]);
        assert!((scores[0] - 0.5).abs() < 1e-9, "0.5*1.0 + (-0.5)*0 = 0.5");
    }

    #[test]
    fn mmr_empty_candidates_returns_empty() {
        let query = vec![1.0_f32, 0.0];
        let (uuids, scores) = maximal_marginal_relevance(&query, &[], 0.5, -2.0);
        assert!(uuids.is_empty());
        assert!(scores.is_empty());
    }

    #[test]
    fn mmr_min_score_filters() {
        // High min_score drops everything; the orthogonal candidate scores 0 < 0.5.
        let query = vec![1.0_f32, 0.0];
        let candidates = vec![cand("ortho", &[0.0, 1.0])];
        let (uuids, _) = maximal_marginal_relevance(&query, &candidates, 1.0, 0.5);
        assert!(uuids.is_empty(), "score 0.0 < min_score 0.5 → filtered");
    }

    #[test]
    fn mmr_zero_vector_candidate_does_not_nan() {
        // A zero-vector candidate is normalized to zeros (guard) → dot with query 0.
        let query = vec![1.0_f32, 0.0];
        let candidates = vec![cand("aligned", &[1.0, 0.0]), cand("zero", &[0.0, 0.0])];
        let (uuids, scores) = maximal_marginal_relevance(&query, &candidates, 1.0, -2.0);
        assert!(scores.iter().all(|s| s.is_finite()), "no NaN");
        // aligned (score 1) ranks above zero (score 0).
        assert_eq!(uuids, vec!["aligned", "zero"]);
    }

    #[test]
    fn mmr_tie_preserves_insertion_order() {
        // Two candidates with identical embeddings → identical MMR → stable order.
        let query = vec![1.0_f32, 0.0];
        let candidates = vec![cand("first", &[1.0, 0.0]), cand("second", &[1.0, 0.0])];
        let (uuids, _) = maximal_marginal_relevance(&query, &candidates, 1.0, -2.0);
        assert_eq!(uuids, vec!["first", "second"]);
    }
}
