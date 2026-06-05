// Ported from graphiti_core/search/search_utils.py::rrf (lines ~1780-1795) @ 34f56e65.
//
// Upstream semantics verified:
//   - rank_const default = 1 (not the more common 60 used in IR literature; upstream explicitly uses 1)
//   - min_score filter uses `>= min_score` (inclusive). Confirmed from upstream:
//       `[uuid for uuid in sorted_uuids if scores[uuid] >= min_score]`
//   - defaultdict(float) means first-seen insertion order is preserved by Python dict (3.7+)
//   - Tie ordering: Python dict preserves insertion order, so equal-score items sort by first-seen.
//     We replicate this with an explicit `first_seen` Vec.

use std::collections::HashMap;

/// Reciprocal Rank Fusion over multiple ranked result lists.
///
/// `results`    — ordered lists of UUIDs (position 0 = rank 1).
/// `rank_const` — upstream default 1 (not 60); controls score magnitude.
/// `min_score`  — inclusive lower bound (`>= min_score`); upstream default 0.
///
/// Returns `(uuids, scores)` sorted descending by score.
/// Ties keep first-seen order (mirrors Python dict insertion-order iteration).
pub fn rrf(results: &[Vec<String>], rank_const: usize, min_score: f64) -> (Vec<String>, Vec<f64>) {
    let mut scores: HashMap<String, f64> = HashMap::new();
    // Preserves first-seen order for stable tie-breaking (upstream dict iteration order).
    let mut first_seen: Vec<String> = Vec::new();

    for result in results {
        for (i, uuid) in result.iter().enumerate() {
            if !scores.contains_key(uuid) {
                first_seen.push(uuid.clone());
            }
            *scores.entry(uuid.clone()).or_insert(0.0) += 1.0 / (i + rank_const) as f64;
        }
    }

    // Build ordered pairs from first_seen (HashMap iteration is non-deterministic for ties).
    let mut scored: Vec<(String, f64)> = first_seen
        .into_iter()
        .map(|u| {
            let s = scores[&u];
            (u, s)
        })
        .collect();

    // Sort descending; ties keep first_seen order because sort_by is stable.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Filter: >= min_score (upstream inclusive bound).
    let filtered: Vec<(String, f64)> = scored
        .into_iter()
        .filter(|(_, s)| *s >= min_score)
        .collect();

    (
        filtered.iter().map(|(u, _)| u.clone()).collect(),
        filtered.iter().map(|(_, s)| *s).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuses_two_rankings() {
        let results = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["b".to_string(), "a".to_string()],
        ];
        let (uuids, scores) = rrf(&results, 1, 0.0);
        // a: 1/1 + 1/2 = 1.5 ; b: 1/2 + 1/1 = 1.5 ; c: 1/3
        assert_eq!(uuids.len(), 3);
        assert!(scores[0] >= scores[1] && scores[1] >= scores[2]);
        assert_eq!(uuids[2], "c");
    }

    #[test]
    fn rrf_min_score_filters() {
        let results = vec![vec!["a".to_string(), "b".to_string()]];
        // a: 1/1 = 1.0, b: 1/2 = 0.5; min_score=0.6 so b is excluded
        let (uuids, _) = rrf(&results, 1, 0.6);
        assert_eq!(uuids, vec!["a".to_string()]); // b scored 0.5
    }

    #[test]
    fn rrf_min_score_inclusive_boundary() {
        // Verify >= (not >): score exactly at min_score must be included.
        let results = vec![vec!["a".to_string()]];
        // a: 1/1 = 1.0 with rank_const=1
        let (uuids, scores) = rrf(&results, 1, 1.0);
        assert_eq!(
            uuids,
            vec!["a".to_string()],
            "score == min_score must be included"
        );
        assert!((scores[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rrf_tie_order_is_first_seen_stable() {
        // x and y both appear at rank 0 in separate single-element lists — equal score 1.0.
        // x is seen first; after stable sort x should appear before y.
        let results = vec![vec!["x".to_string()], vec!["y".to_string()]];
        let (uuids, scores) = rrf(&results, 1, 0.0);
        assert_eq!(uuids.len(), 2);
        assert_eq!(scores[0], scores[1], "x and y must have equal scores");
        assert_eq!(uuids[0], "x", "first-seen item must win on tie");
        assert_eq!(uuids[1], "y");
    }

    #[test]
    fn rrf_single_list_preserves_rank_order() {
        let results = vec![vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ]];
        let (uuids, scores) = rrf(&results, 1, 0.0);
        assert_eq!(uuids.len(), 3);
        // scores should be strictly decreasing
        assert!(scores[0] > scores[1] && scores[1] > scores[2]);
        assert_eq!(uuids[0], "first");
    }

    #[test]
    fn rrf_empty_input() {
        let (uuids, scores) = rrf(&[], 1, 0.0);
        assert!(uuids.is_empty());
        assert!(scores.is_empty());
    }

    #[test]
    fn rrf_all_filtered_by_min_score() {
        let results = vec![vec!["a".to_string()]];
        // a scores 1.0; min_score=2.0 means nothing passes
        let (uuids, _) = rrf(&results, 1, 2.0);
        assert!(uuids.is_empty());
    }
}
