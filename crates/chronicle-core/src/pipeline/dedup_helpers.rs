// Ported from graphiti_core/utils/maintenance/dedup_helpers.py @ 34f56e65 (v0.29.1)
//
// Line-for-line port of the deterministic deduplication heuristics: name
// normalization, Shannon-entropy gating, 3-gram shingles, blake2b-based MinHash
// signatures, LSH banding, Jaccard similarity, and the candidate-index /
// resolution-state structures consumed by node resolution (Task 13).
//
// Exact constants and boolean structure are load-bearing — divergences from the
// task sketch are noted inline where upstream wins.

use std::collections::{BTreeSet, HashMap};

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

use crate::helpers::normalize_string_exact;
use crate::types::EntityNode;

/// Blake2b with an 8-byte (64-bit) digest, matching upstream
/// `blake2b(..., digest_size=8)`.
type Blake2b64 = Blake2b<U8>;

// Upstream lines ~31-36.
pub const NAME_ENTROPY_THRESHOLD: f64 = 1.5;
pub const MIN_NAME_LENGTH: usize = 6;
pub const MIN_TOKEN_COUNT: usize = 2;
pub const FUZZY_JACCARD_THRESHOLD: f64 = 0.9;
pub const MINHASH_PERMUTATIONS: usize = 32;
pub const MINHASH_BAND_SIZE: usize = 4;

/// Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram
/// shingles. Upstream `_normalize_name_for_fuzzy`:
///
/// ```python
/// normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
/// normalized = normalized.strip()
/// return re.sub(r'[\s]+', ' ', normalized)
/// ```
///
/// `_normalize_string_exact` already lowercases, so the char class is purely
/// lowercase `a-z`, digits, apostrophe, and space; everything else becomes a
/// space, then whitespace is collapsed and trimmed.
pub fn normalize_name_for_fuzzy(name: &str) -> String {
    let exact = normalize_string_exact(name);
    // Replace any char that is not [a-z0-9' ] with a space.
    let replaced: String = exact
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    // strip() then collapse runs of whitespace to a single space.
    // `split_whitespace().join(" ")` performs both trim and collapse.
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Approximate text specificity using Shannon entropy over characters.
/// Upstream `_name_entropy`: empty string → 0.0, spaces are stripped before
/// counting, log base 2.
pub fn name_entropy(normalized_name: &str) -> f64 {
    if normalized_name.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<char, usize> = HashMap::new();
    for ch in normalized_name.chars().filter(|c| *c != ' ') {
        *counts.entry(ch).or_insert(0) += 1;
    }

    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let mut entropy = 0.0_f64;
    for count in counts.values() {
        let probability = *count as f64 / total;
        entropy -= probability * probability.log2();
    }

    entropy
}

/// Filter out very short or low-entropy names that are unreliable for fuzzy
/// matching. Upstream `_has_high_entropy` (lines ~79-85):
///
/// ```python
/// token_count = len(normalized_name.split())
/// if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
///     return False
/// return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD
/// ```
///
/// Note the gate is an AND: a name is only rejected outright when it is BOTH
/// shorter than `MIN_NAME_LENGTH` AND has fewer than `MIN_TOKEN_COUNT` tokens.
/// `len(normalized_name)` is the full string length INCLUDING spaces (Python
/// counts the raw string here, not the space-stripped form).
pub fn has_high_entropy(normalized_name: &str) -> bool {
    // Python str.split() with no args splits on runs of whitespace and drops
    // empties; split_whitespace() matches that exactly.
    let token_count = normalized_name.split_whitespace().count();
    // len(normalized_name) in Python counts Unicode code points; chars().count()
    // matches that (str::len would count UTF-8 bytes — wrong for non-ASCII).
    let char_len = normalized_name.chars().count();
    if char_len < MIN_NAME_LENGTH && token_count < MIN_TOKEN_COUNT {
        return false;
    }

    name_entropy(normalized_name) >= NAME_ENTROPY_THRESHOLD
}

/// Create 3-gram shingles from the normalized name for MinHash calculations.
/// Upstream `_shingles`:
///
/// ```python
/// cleaned = normalized_name.replace(' ', '')
/// if len(cleaned) < 2:
///     return {cleaned} if cleaned else set()
/// return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}
/// ```
///
/// DIVERGENCE FROM TASK SKETCH: the short-string guard is `len(cleaned) < 2`,
/// not `< 3`. So a 2-char string like "ab" falls through to the comprehension
/// `range(len - 2) = range(0)` → empty set (NOT `{"ab"}`). Only strings of
/// length 0 (empty set) or 1 (`{c}`) hit the early return. A 3-char string
/// yields `{whole}`; longer strings yield the sliding 3-gram set.
pub fn shingles(normalized_name: &str) -> BTreeSet<String> {
    let cleaned: Vec<char> = normalized_name.chars().filter(|c| *c != ' ').collect();
    let mut set = BTreeSet::new();

    if cleaned.len() < 2 {
        if !cleaned.is_empty() {
            set.insert(cleaned.iter().collect::<String>());
        }
        return set;
    }

    // range(len(cleaned) - 2): i in 0..=len-3, slice cleaned[i..i+3].
    for i in 0..(cleaned.len() - 2) {
        let gram: String = cleaned[i..i + 3].iter().collect();
        set.insert(gram);
    }
    set
}

/// Generate a deterministic 64-bit hash for a shingle given the permutation
/// seed. Upstream `_hash_shingle`:
///
/// ```python
/// digest = blake2b(f'{seed}:{shingle}'.encode(), digest_size=8)
/// return int.from_bytes(digest.digest(), 'big')
/// ```
///
/// 8-byte digest, big-endian → u64.
pub fn hash_shingle(shingle: &str, seed: u64) -> u64 {
    let mut hasher = Blake2b64::new();
    hasher.update(format!("{seed}:{shingle}").as_bytes());
    let digest = hasher.finalize(); // 8 bytes
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest);
    u64::from_be_bytes(bytes)
}

/// Compute the MinHash signature across the predefined permutations. Upstream
/// `_minhash_signature`: empty shingles → empty tuple; otherwise for each seed
/// in `0..MINHASH_PERMUTATIONS` take the minimum shingle hash.
pub fn minhash_signature(shingles: &BTreeSet<String>) -> Vec<u64> {
    if shingles.is_empty() {
        return Vec::new();
    }

    let mut signature = Vec::with_capacity(MINHASH_PERMUTATIONS);
    for seed in 0..MINHASH_PERMUTATIONS as u64 {
        let min_hash = shingles
            .iter()
            .map(|s| hash_shingle(s, seed))
            .min()
            .expect("shingles non-empty checked above");
        signature.push(min_hash);
    }
    signature
}

/// Split the MinHash signature into fixed-size bands for LSH. Upstream
/// `_lsh_bands`: empty signature → empty; otherwise non-overlapping chunks of
/// `MINHASH_BAND_SIZE`, DROPPING any trailing partial band (only full bands are
/// retained).
pub fn lsh_bands(signature: &[u64]) -> Vec<Vec<u64>> {
    if signature.is_empty() {
        return Vec::new();
    }

    let mut bands = Vec::new();
    let mut start = 0;
    while start < signature.len() {
        let end = (start + MINHASH_BAND_SIZE).min(signature.len());
        let band = &signature[start..end];
        if band.len() == MINHASH_BAND_SIZE {
            bands.push(band.to_vec());
        }
        start += MINHASH_BAND_SIZE;
    }
    bands
}

/// Return the Jaccard similarity between two shingle sets. Upstream
/// `_jaccard_similarity`:
///
/// ```python
/// if not a and not b:
///     return 1.0
/// if not a or not b:
///     return 0.0
/// intersection = len(a.intersection(b))
/// union = len(a.union(b))
/// return intersection / union if union else 0.0
/// ```
///
/// Both-empty → 1.0; exactly-one-empty → 0.0.
pub fn jaccard_similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Precomputed lookup structures that drive entity deduplication heuristics.
/// Upstream `DedupCandidateIndexes`.
///
/// `lsh_buckets` is keyed by `(band_index, band)` exactly as upstream builds and
/// queries it. The band tuple becomes a `Vec<u64>` here (hashable map key).
#[derive(Debug, Default, Clone)]
pub struct DedupCandidateIndexes {
    pub existing_nodes: Vec<EntityNode>,
    pub nodes_by_uuid: HashMap<String, EntityNode>,
    pub normalized_existing: HashMap<String, Vec<EntityNode>>,
    pub shingles_by_candidate: HashMap<String, BTreeSet<String>>,
    pub lsh_buckets: HashMap<(usize, Vec<u64>), Vec<String>>,
}

/// Mutable resolution bookkeeping shared across deterministic and LLM passes.
/// Upstream `DedupResolutionState`.
#[derive(Debug, Default)]
pub struct DedupResolutionState {
    pub resolved_nodes: Vec<Option<EntityNode>>,
    pub uuid_map: HashMap<String, String>,
    pub unresolved_indices: Vec<usize>,
    pub duplicate_pairs: Vec<(EntityNode, EntityNode)>,
}

/// Upgrade a generic canonical node when a duplicate carries a specific type.
/// Upstream `_promote_resolved_node`.
pub fn promote_resolved_node(
    extracted_node: &EntityNode,
    resolved_node: &EntityNode,
) -> EntityNode {
    let resolved_specific: Vec<&String> = resolved_node
        .labels
        .iter()
        .filter(|l| *l != "Entity")
        .collect();
    if !resolved_specific.is_empty() {
        return resolved_node.clone();
    }

    let extracted_specific: Vec<String> = extracted_node
        .labels
        .iter()
        .filter(|l| *l != "Entity")
        .cloned()
        .collect();
    if extracted_specific.is_empty() {
        return resolved_node.clone();
    }

    // ['Entity', *resolved_node.labels, *extracted_specific_labels], dedup,
    // preserving first-seen order.
    let mut promoted_labels: Vec<String> = Vec::new();
    let chain = std::iter::once("Entity".to_string())
        .chain(resolved_node.labels.iter().cloned())
        .chain(extracted_specific);
    for label in chain {
        if !promoted_labels.contains(&label) {
            promoted_labels.push(label);
        }
    }

    let mut promoted = resolved_node.clone();
    promoted.labels = promoted_labels;
    promoted
}

/// Precompute exact and fuzzy lookup structures once per dedupe run. Upstream
/// `_build_candidate_indexes`. (`_cached_shingles` is an `lru_cache` wrapper
/// around `_shingles`; we call `shingles` directly — the cache is a perf detail
/// with no behavioral effect.)
pub fn build_candidate_indexes(existing_nodes: Vec<EntityNode>) -> DedupCandidateIndexes {
    let mut normalized_existing: HashMap<String, Vec<EntityNode>> = HashMap::new();
    let mut nodes_by_uuid: HashMap<String, EntityNode> = HashMap::new();
    let mut shingles_by_candidate: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut lsh_buckets: HashMap<(usize, Vec<u64>), Vec<String>> = HashMap::new();

    for candidate in &existing_nodes {
        let normalized = normalize_string_exact(&candidate.name);
        normalized_existing
            .entry(normalized)
            .or_default()
            .push(candidate.clone());
        nodes_by_uuid.insert(candidate.uuid.clone(), candidate.clone());

        let cand_shingles = shingles(&normalize_name_for_fuzzy(&candidate.name));
        shingles_by_candidate.insert(candidate.uuid.clone(), cand_shingles.clone());

        let signature = minhash_signature(&cand_shingles);
        for (band_index, band) in lsh_bands(&signature).into_iter().enumerate() {
            lsh_buckets
                .entry((band_index, band))
                .or_default()
                .push(candidate.uuid.clone());
        }
    }

    DedupCandidateIndexes {
        existing_nodes,
        nodes_by_uuid,
        normalized_existing,
        shingles_by_candidate,
        lsh_buckets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn node(name: &str) -> EntityNode {
        EntityNode::new(name.to_string(), "g".to_string(), Utc::now())
    }

    // --- normalize_name_for_fuzzy ---

    #[test]
    fn fuzzy_keeps_alnum_and_apostrophe_collapses_ws() {
        assert_eq!(normalize_name_for_fuzzy("O'Brien, Inc."), "o'brien inc");
        assert_eq!(normalize_name_for_fuzzy("  Alice   Smith! "), "alice smith");
        assert_eq!(normalize_name_for_fuzzy("a@#b"), "a b");
        assert_eq!(normalize_name_for_fuzzy("café"), "caf"); // 'é' -> space, trimmed
    }

    // --- name_entropy ---

    #[test]
    fn entropy_known_values() {
        assert_eq!(name_entropy(""), 0.0);
        // "ab": two distinct chars, each p=0.5 -> -2*(0.5*log2 0.5) = 1.0
        assert!((name_entropy("ab") - 1.0).abs() < 1e-12);
        // all-same char -> 0 entropy
        assert!((name_entropy("aaaaaa") - 0.0).abs() < 1e-12);
        // spaces stripped: "a a" == "aa" -> 0.0
        assert!((name_entropy("a a") - 0.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_low_vs_high() {
        let low = name_entropy("aaaaaa");
        let high = name_entropy("alice smith");
        assert!(low < high);
        assert!(high >= NAME_ENTROPY_THRESHOLD);
    }

    // --- has_high_entropy (truth table) ---

    #[test]
    fn high_entropy_short_low_single_token_gated() {
        // len < 6 AND token_count < 2 -> false regardless of entropy.
        assert!(!has_high_entropy("aaa")); // short, 1 token, low entropy
        // "abc": len 3 < 6, 1 token -> gated false even though entropy high.
        assert!(!has_high_entropy("abc"));
    }

    #[test]
    fn high_entropy_multi_token_escapes_length_gate() {
        // "a b": len 3 < 6 BUT token_count 2 >= MIN_TOKEN_COUNT -> not gated,
        // falls through to entropy check. "ab" entropy = 1.0 < 1.5 -> false.
        assert!(!has_high_entropy("a b"));
    }

    #[test]
    fn high_entropy_long_low_entropy_false() {
        // len 7 >= 6 -> not gated, but entropy 0 < 1.5 -> false.
        assert!(!has_high_entropy("aaaaaaa"));
    }

    #[test]
    fn high_entropy_long_high_entropy_true() {
        assert!(has_high_entropy("alice smith"));
        assert!(has_high_entropy("abcdef")); // len 6, 6 distinct -> entropy log2(6)~2.58
    }

    // --- shingles ---

    #[test]
    fn shingles_three_gram_and_edge_cases() {
        assert_eq!(shingles("abc"), BTreeSet::from(["abc".to_string()]));
        assert_eq!(
            shingles("abcd"),
            BTreeSet::from(["abc".to_string(), "bcd".to_string()])
        );
        // len(cleaned) < 2 guard: "ab" -> range(0) -> EMPTY (upstream divergence
        // from sketch's "< 3"); only length 0/1 early-return.
        assert!(shingles("ab").is_empty());
        assert_eq!(shingles("a"), BTreeSet::from(["a".to_string()]));
        assert!(shingles("").is_empty());
        // spaces removed before shingling.
        assert_eq!(
            shingles("a b c d"),
            BTreeSet::from(["abc".to_string(), "bcd".to_string()])
        );
    }

    // --- hash_shingle (cross-language constant) ---

    #[test]
    fn hash_shingle_cross_language_constant() {
        // python3 -c "from hashlib import blake2b; \
        //   print(int.from_bytes(blake2b(b'0:abc',digest_size=8).digest(),'big'))"
        // => 7705568351334315143
        assert_eq!(hash_shingle("abc", 0), 7705568351334315143);
        // python3 -c "...b'1:abc'... => 17109218681013626300"
        assert_eq!(hash_shingle("abc", 1), 17109218681013626300);
    }

    #[test]
    fn hash_shingle_deterministic_and_seed_sensitive() {
        assert_eq!(hash_shingle("abc", 0), hash_shingle("abc", 0));
        assert_ne!(hash_shingle("abc", 0), hash_shingle("abc", 1));
    }

    // --- minhash_signature ---

    #[test]
    fn minhash_identical_sets_identical_signatures() {
        let s1 = shingles(&normalize_name_for_fuzzy("Alice Smith"));
        let s2 = shingles(&normalize_name_for_fuzzy("alice   smith"));
        let sig1 = minhash_signature(&s1);
        let sig2 = minhash_signature(&s2);
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), MINHASH_PERMUTATIONS);
    }

    #[test]
    fn minhash_empty_shingles_empty_signature() {
        assert!(minhash_signature(&BTreeSet::new()).is_empty());
    }

    // --- lsh_bands ---

    #[test]
    fn lsh_drops_partial_band() {
        // length 32 -> 8 full bands.
        let sig: Vec<u64> = (0..32).collect();
        assert_eq!(lsh_bands(&sig).len(), 8);
        // length 6 -> one full band of 4, trailing 2 dropped.
        let sig: Vec<u64> = (0..6).collect();
        let bands = lsh_bands(&sig);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], vec![0, 1, 2, 3]);
        assert!(lsh_bands(&[]).is_empty());
    }

    #[test]
    fn lsh_identical_names_share_band() {
        let s1 = shingles(&normalize_name_for_fuzzy("Alice Smith"));
        let s2 = shingles(&normalize_name_for_fuzzy("Alice Smith"));
        let b1 = lsh_bands(&minhash_signature(&s1));
        let b2 = lsh_bands(&minhash_signature(&s2));
        assert_eq!(b1, b2);
        assert!(!b1.is_empty());
    }

    // --- jaccard_similarity ---

    #[test]
    fn jaccard_edge_cases() {
        let empty: BTreeSet<String> = BTreeSet::new();
        let a = BTreeSet::from(["abc".to_string(), "bcd".to_string()]);
        assert_eq!(jaccard_similarity(&empty, &empty), 1.0); // both empty -> 1.0
        assert_eq!(jaccard_similarity(&a, &empty), 0.0); // one empty -> 0.0
        assert_eq!(jaccard_similarity(&empty, &a), 0.0);
        assert_eq!(jaccard_similarity(&a, &a), 1.0); // identical -> 1.0
        let b = BTreeSet::from(["xyz".to_string()]);
        assert_eq!(jaccard_similarity(&a, &b), 0.0); // disjoint -> 0.0
        // partial overlap: {abc,bcd} vs {bcd,cde} -> inter 1, union 3 -> 1/3.
        let c = BTreeSet::from(["bcd".to_string(), "cde".to_string()]);
        assert!((jaccard_similarity(&a, &c) - 1.0 / 3.0).abs() < 1e-12);
    }

    // --- candidate indexes ---

    #[test]
    fn candidate_indexes_exact_map_groups_same_normalized_name() {
        let n1 = node("Alice Smith");
        let n2 = node("  alice   smith ");
        let n3 = node("Bob Jones");
        let idx = build_candidate_indexes(vec![n1.clone(), n2.clone(), n3.clone()]);

        let group = idx.normalized_existing.get("alice smith").unwrap();
        assert_eq!(group.len(), 2);
        assert!(idx.nodes_by_uuid.contains_key(&n1.uuid));
        assert!(idx.nodes_by_uuid.contains_key(&n3.uuid));
        assert!(idx.shingles_by_candidate.contains_key(&n1.uuid));
        // identical fuzzy names land in the same LSH buckets.
        assert!(!idx.lsh_buckets.is_empty());
        let shared = idx
            .lsh_buckets
            .values()
            .any(|ids| ids.contains(&n1.uuid) && ids.contains(&n2.uuid));
        assert!(shared);
    }

    // --- promote_resolved_node ---

    #[test]
    fn promote_keeps_resolved_when_specific() {
        let extracted = node("x");
        let mut resolved = node("y");
        resolved.labels = vec!["Entity".to_string(), "Person".to_string()];
        let out = promote_resolved_node(&extracted, &resolved);
        assert_eq!(out.labels, vec!["Entity", "Person"]);
    }

    #[test]
    fn promote_upgrades_generic_resolved_from_extracted() {
        let mut extracted = node("x");
        extracted.labels = vec!["Entity".to_string(), "Person".to_string()];
        let resolved = node("y"); // generic: ["Entity"]
        let out = promote_resolved_node(&extracted, &resolved);
        assert_eq!(out.labels, vec!["Entity", "Person"]);
        assert_eq!(out.uuid, resolved.uuid); // identity stays resolved's
    }

    #[test]
    fn promote_noop_when_extracted_also_generic() {
        let extracted = node("x");
        let resolved = node("y");
        let out = promote_resolved_node(&extracted, &resolved);
        assert_eq!(out.labels, vec!["Entity"]);
    }
}
