// Ported from graphiti_core/utils/maintenance/edge_operations.py
// resolve_edge_contradictions (lines ~538-573) and the new-edge expiration
// block of resolve_extracted_edge (lines ~820-839) @ 34f56e65 (v0.29.1)
//
// This is the bi-temporal edge invalidation core. Two operations:
//
//  1. `resolve_edge_contradictions` — given a newly resolved edge and a set of
//     existing candidate edges that contradict it, decide which existing edges
//     must be expired (their valid-time interval closed at the new edge's
//     `valid_at`).
//
//  2. `expire_new_edge_against_candidates` — decide whether the *new* edge is
//     already stale because a more-recent contradicting fact exists.
//
// `now` (transaction time) is taken as a parameter rather than read inside via
// `utc_now()` as upstream does, so the logic is deterministic and testable.
// Production callers pass `crate::helpers::utc_now()`.

use crate::helpers::ensure_utc;
use crate::types::EntityEdge;
use chrono::{DateTime, Utc};

/// Determine which contradictory candidate edges must be expired by the newly
/// resolved edge.
///
/// Returns the candidates that were invalidated, each mutated so that its
/// valid-time interval closes at the new edge's `valid_at` and its
/// transaction-time `expired_at` is stamped (preserving any pre-existing
/// `expired_at`).
///
/// Port of upstream `resolve_edge_contradictions` (edge_operations.py
/// ~538-573). `now` replaces the upstream inline `utc_now()` call (line 570).
pub fn resolve_edge_contradictions(
    resolved_edge: &EntityEdge,
    invalidation_candidates: Vec<EntityEdge>,
    now: DateTime<Utc>,
) -> Vec<EntityEdge> {
    // if len(invalidation_candidates) == 0: return []
    if invalidation_candidates.is_empty() {
        return Vec::new();
    }

    // Determine which contradictory edges need to be expired
    let mut invalidated_edges: Vec<EntityEdge> = Vec::new();
    for mut edge in invalidation_candidates {
        // edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        // resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        // edge_valid_at_utc = ensure_utc(edge.valid_at)
        // resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)
        let edge_invalid_at_utc = ensure_utc(edge.invalid_at);
        let resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at);
        let edge_valid_at_utc = ensure_utc(edge.valid_at);
        let resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at);

        // (Edge invalid before new edge becomes valid)
        //   or (new edge invalid before edge becomes valid)
        //
        // if (
        //     edge_invalid_at_utc is not None
        //     and resolved_edge_valid_at_utc is not None
        //     and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        // ) or (
        //     edge_valid_at_utc is not None
        //     and resolved_edge_invalid_at_utc is not None
        //     and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        // ):
        //     continue
        let existing_invalid_before_new_valid = matches!(
            (edge_invalid_at_utc, resolved_edge_valid_at_utc),
            (Some(ei), Some(rv)) if ei <= rv
        );
        let new_invalid_before_existing_valid = matches!(
            (resolved_edge_invalid_at_utc, edge_valid_at_utc),
            (Some(ri), Some(ev)) if ri <= ev
        );
        if existing_invalid_before_new_valid || new_invalid_before_existing_valid {
            continue;
        }
        // New edge invalidates edge
        // elif (
        //     edge_valid_at_utc is not None
        //     and resolved_edge_valid_at_utc is not None
        //     and edge_valid_at_utc < resolved_edge_valid_at_utc
        // ):
        //     edge.invalid_at = resolved_edge.valid_at
        //     edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
        //     invalidated_edges.append(edge)
        else if matches!(
            (edge_valid_at_utc, resolved_edge_valid_at_utc),
            // strict `<`: an existing edge with the SAME valid_at is NOT invalidated.
            (Some(ev), Some(rv)) if ev < rv
        ) {
            edge.invalid_at = resolved_edge.valid_at;
            edge.expired_at = match edge.expired_at {
                Some(existing) => Some(existing),
                None => Some(now),
            };
            invalidated_edges.push(edge);
        }
    }

    invalidated_edges
}

/// Decide whether the newly resolved edge is itself already stale, expiring it
/// in place if a more-recent contradicting candidate exists.
///
/// Port of the new-edge expiration block in upstream `resolve_extracted_edge`
/// (edge_operations.py ~820-839). `now` replaces the upstream inline
/// `utc_now()` call (line 820).
///
/// Upstream ordering (replicated exactly):
///   1. (~822-823) If `invalid_at` is already set but `expired_at` is not,
///      stamp `expired_at = now` FIRST.
///   2. (~826-839) Only if `expired_at` is still None, sort candidates by
///      `valid_at` and expire the new edge against the earliest candidate whose
///      `valid_at` is strictly after the new edge's `valid_at`.
pub fn expire_new_edge_against_candidates(
    resolved_edge: &mut EntityEdge,
    invalidation_candidates: &[EntityEdge],
    now: DateTime<Utc>,
) {
    // if resolved_edge.invalid_at and not resolved_edge.expired_at:
    //     resolved_edge.expired_at = now
    if resolved_edge.invalid_at.is_some() && resolved_edge.expired_at.is_none() {
        resolved_edge.expired_at = Some(now);
    }

    // Determine if the new_edge needs to be expired
    // if resolved_edge.expired_at is None:
    if resolved_edge.expired_at.is_none() {
        // invalidation_candidates.sort(
        //     key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at))
        // )
        //
        // Sort by `valid_at` ascending with None last. Upstream's tuple key puts
        // `valid_at is None` (True) after Some (False); within the Some group it
        // orders by the timestamp. We work on a local copy of references to avoid
        // mutating the caller's slice (upstream sorts the list in place, but the
        // list is local to resolve_extracted_edge and not observed afterwards).
        let mut sorted: Vec<&EntityEdge> = invalidation_candidates.iter().collect();
        sorted.sort_by(|a, b| {
            let a_none = a.valid_at.is_none();
            let b_none = b.valid_at.is_none();
            // (c.valid_at is None) primary key: false (Some) sorts before true (None).
            a_none.cmp(&b_none).then_with(|| {
                // Secondary key only discriminates within the Some group; for the
                // None group both sides are None and compare equal (stable sort
                // preserves input order, matching upstream where the None tail is
                // never reached by the expiry check below).
                match (ensure_utc(a.valid_at), ensure_utc(b.valid_at)) {
                    (Some(av), Some(bv)) => av.cmp(&bv),
                    _ => std::cmp::Ordering::Equal,
                }
            })
        });

        // for candidate in invalidation_candidates:
        for candidate in sorted {
            let candidate_valid_at_utc = ensure_utc(candidate.valid_at);
            let resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at);
            // if (
            //     candidate_valid_at_utc is not None
            //     and resolved_edge_valid_at_utc is not None
            //     and candidate_valid_at_utc > resolved_edge_valid_at_utc
            // ):
            //     resolved_edge.invalid_at = candidate.valid_at
            //     resolved_edge.expired_at = now
            //     break
            if matches!(
                (candidate_valid_at_utc, resolved_edge_valid_at_utc),
                (Some(cv), Some(rv)) if cv > rv
            ) {
                // Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at;
                resolved_edge.expired_at = Some(now);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn t(h: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, h, 0, 0).unwrap()
    }

    fn edge_with(valid: Option<u32>, invalid: Option<u32>) -> EntityEdge {
        let mut e = EntityEdge::new("a".into(), "b".into(), "R".into(), "f".into(), "g".into());
        e.valid_at = valid.map(t);
        e.invalid_at = invalid.map(t);
        e
    }

    // ---- resolve_edge_contradictions ----

    #[test]
    fn skips_when_existing_already_invalid_before_new_valid() {
        // existing.invalid_at(2) <= new.valid_at(3) → disjoint → skip
        let new_edge = edge_with(Some(3), None);
        let existing = edge_with(Some(1), Some(2));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn skips_when_new_already_invalid_before_existing_valid() {
        let new_edge = edge_with(Some(1), Some(2));
        let existing = edge_with(Some(3), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn invalidates_older_overlapping_edge() {
        let new_edge = edge_with(Some(5), None);
        let existing = edge_with(Some(1), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].invalid_at, Some(t(5)));
        assert_eq!(out[0].expired_at, Some(t(12)));
    }

    #[test]
    fn preserves_preexisting_expired_at() {
        let new_edge = edge_with(Some(5), None);
        let mut existing = edge_with(Some(1), None);
        existing.expired_at = Some(t(2));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert_eq!(out[0].expired_at, Some(t(2)));
    }

    #[test]
    fn no_invalidation_without_valid_at_on_either_side() {
        let new_edge = edge_with(None, None);
        let existing = edge_with(Some(1), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn empty_candidates_returns_empty() {
        let new_edge = edge_with(Some(5), None);
        let out = resolve_edge_contradictions(&new_edge, vec![], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn equal_valid_at_does_not_invalidate() {
        // ev == rv: upstream uses strict `<`, so equal valid_at is NOT invalidated.
        let new_edge = edge_with(Some(5), None);
        let existing = edge_with(Some(5), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn newer_existing_not_invalidated() {
        // existing is newer than new edge → ev(7) < rv(5) is false → not invalidated.
        let new_edge = edge_with(Some(5), None);
        let existing = edge_with(Some(7), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn boundary_existing_invalid_equals_new_valid_skips() {
        // existing.invalid_at(3) == new.valid_at(3): `<=` is true → disjoint → skip.
        // (Even though existing.valid_at(1) < new.valid_at(3) would otherwise invalidate.)
        let new_edge = edge_with(Some(3), None);
        let existing = edge_with(Some(1), Some(3));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn boundary_new_invalid_equals_existing_valid_skips() {
        // new.invalid_at(3) == existing.valid_at(3): `<=` is true → disjoint → skip.
        let new_edge = edge_with(Some(1), Some(3));
        let existing = edge_with(Some(3), None);
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert!(out.is_empty());
    }

    #[test]
    fn invalidates_when_intervals_overlap_with_invalid_bound() {
        // existing.invalid_at(10) > new.valid_at(5) → not disjoint on first clause;
        // new has no invalid_at → second clause inapplicable;
        // existing.valid_at(1) < new.valid_at(5) → invalidate.
        let new_edge = edge_with(Some(5), None);
        let existing = edge_with(Some(1), Some(10));
        let out = resolve_edge_contradictions(&new_edge, vec![existing], t(12));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].invalid_at, Some(t(5)));
        assert_eq!(out[0].expired_at, Some(t(12)));
    }

    #[test]
    fn mixed_candidates_only_qualifying_invalidated() {
        let new_edge = edge_with(Some(5), None);
        let older = edge_with(Some(1), None); // invalidated
        let equal = edge_with(Some(5), None); // not (strict <)
        let newer = edge_with(Some(9), None); // not
        let disjoint = edge_with(Some(2), Some(4)); // invalid_at(4) <= valid(5) → skip
        let out =
            resolve_edge_contradictions(&new_edge, vec![older, equal, newer, disjoint], t(12));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].valid_at, Some(t(1)));
        assert_eq!(out[0].invalid_at, Some(t(5)));
    }

    // ---- expire_new_edge_against_candidates ----

    #[test]
    fn new_edge_expired_by_later_candidate() {
        // (a) candidate starting AFTER new edge → new gets invalid_at=candidate.valid_at + expired_at=now
        let mut new_edge = edge_with(Some(3), None);
        let candidate = edge_with(Some(7), None);
        expire_new_edge_against_candidates(&mut new_edge, &[candidate], t(12));
        assert_eq!(new_edge.invalid_at, Some(t(7)));
        assert_eq!(new_edge.expired_at, Some(t(12)));
    }

    #[test]
    fn earliest_qualifying_candidate_wins() {
        // (b) candidates sorted by valid_at — earliest qualifying (>3) wins: t(5), not t(9).
        let mut new_edge = edge_with(Some(3), None);
        let c_late = edge_with(Some(9), None);
        let c_early = edge_with(Some(5), None);
        // Pass out of order to prove sorting selects the earliest.
        expire_new_edge_against_candidates(&mut new_edge, &[c_late, c_early], t(12));
        assert_eq!(new_edge.invalid_at, Some(t(5)));
        assert_eq!(new_edge.expired_at, Some(t(12)));
    }

    #[test]
    fn candidate_not_after_new_edge_does_not_expire() {
        // candidate.valid_at(3) == new.valid_at(3): strict `>` false → no expiry.
        let mut new_edge = edge_with(Some(3), None);
        let candidate = edge_with(Some(3), None);
        expire_new_edge_against_candidates(&mut new_edge, &[candidate], t(12));
        assert_eq!(new_edge.invalid_at, None);
        assert_eq!(new_edge.expired_at, None);
    }

    #[test]
    fn new_edge_with_invalid_at_but_no_expired_at_gets_stamped() {
        // (c) new edge with invalid_at but no expired_at → expired_at stamped FIRST.
        // Because expired_at becomes Some, the candidate-expiry block is skipped,
        // so invalid_at is preserved (not overwritten by the later candidate).
        let mut new_edge = edge_with(Some(3), Some(4));
        let candidate = edge_with(Some(7), None);
        expire_new_edge_against_candidates(&mut new_edge, &[candidate], t(12));
        assert_eq!(new_edge.expired_at, Some(t(12)));
        assert_eq!(new_edge.invalid_at, Some(t(4)));
    }

    #[test]
    fn already_expired_new_edge_untouched() {
        // (d) new edge already expired → untouched by both blocks.
        let mut new_edge = edge_with(Some(3), None);
        new_edge.expired_at = Some(t(2));
        let candidate = edge_with(Some(7), None);
        expire_new_edge_against_candidates(&mut new_edge, &[candidate], t(12));
        assert_eq!(new_edge.expired_at, Some(t(2)));
        assert_eq!(new_edge.invalid_at, None);
    }

    #[test]
    fn none_valid_at_candidates_do_not_expire_new_edge() {
        // Candidates with no valid_at sort last and never satisfy the strict `>` check.
        let mut new_edge = edge_with(Some(3), None);
        let c_none = edge_with(None, None);
        expire_new_edge_against_candidates(&mut new_edge, &[c_none], t(12));
        assert_eq!(new_edge.invalid_at, None);
        assert_eq!(new_edge.expired_at, None);
    }

    #[test]
    fn new_edge_without_valid_at_not_expired() {
        // resolved.valid_at is None → strict `>` clause inapplicable → no expiry.
        let mut new_edge = edge_with(None, None);
        let candidate = edge_with(Some(7), None);
        expire_new_edge_against_candidates(&mut new_edge, &[candidate], t(12));
        assert_eq!(new_edge.invalid_at, None);
        assert_eq!(new_edge.expired_at, None);
    }

    #[test]
    fn mixed_none_and_some_candidates_pick_earliest_some() {
        // None-valued candidates interleaved must not disturb earliest-Some selection.
        let mut new_edge = edge_with(Some(3), None);
        let c_none = edge_with(None, None);
        let c_late = edge_with(Some(9), None);
        let c_early = edge_with(Some(5), None);
        expire_new_edge_against_candidates(&mut new_edge, &[c_none, c_late, c_early], t(12));
        assert_eq!(new_edge.invalid_at, Some(t(5)));
        assert_eq!(new_edge.expired_at, Some(t(12)));
    }
}
