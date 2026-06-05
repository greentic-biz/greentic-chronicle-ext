// Ported from graphiti_core/helpers.py, utils/datetime_utils.py,
// utils/maintenance/dedup_helpers.py @ 34f56e65 (v0.29.1)

use chrono::{DateTime, SecondsFormat, Utc};

/// Default concurrency for parallel LLM calls (upstream `SEMAPHORE_LIMIT`,
/// `graphiti_core/helpers.py`). NOTE: upstream reads this from env; this crate
/// uses a compile-time const — callers can override at call site.
pub const SEMAPHORE_LIMIT: usize = 20;

/// Max results returned per graph-schema search call (upstream `RELEVANT_SCHEMA_LIMIT`,
/// `graphiti_core/search/search_utils.py`). Also used as the prior-episode window
/// when `previous_episode_uuids` is None.
pub const RELEVANT_SCHEMA_LIMIT: usize = 10;

/// Upstream EPISODE_WINDOW_LEN (graph_data_operations.py).
pub const EPISODE_WINDOW_LEN: usize = 3;

pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Ensures a datetime is UTC; tz-naive values are treated as UTC.
/// Port of `graphiti_core/utils/datetime_utils.py::ensure_utc`. Since
/// `chrono::DateTime<Utc>` is UTC by construction this is an identity,
/// kept so temporal code reads symmetrically with upstream.
#[inline]
pub fn ensure_utc(dt: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
    dt
}

/// Format a UTC datetime exactly as Python's `datetime.isoformat()` would for a
/// tz-aware UTC value, for byte-identical prompt fidelity with upstream graphiti
/// (which interpolates `ep.valid_at.isoformat()` into the `previous_episodes`
/// context — see `pipeline::node_ops::previous_episodes_context`).
///
/// Python's default `isoformat()` emits fractional seconds ONLY when the
/// microsecond component is nonzero, and when it does it emits EXACTLY six
/// digits (never trimming trailing zeros). The UTC offset renders as `+00:00`.
///
/// Mapping to `chrono`:
/// - zero sub-second → `SecondsFormat::Secs` → `...T00:00:00+00:00` (matches).
/// - nonzero sub-second → `SecondsFormat::Micros` → six fixed digits
///   `...T00:00:00.123000+00:00` (matches; `chrono`'s `to_rfc3339` / `AutoSi`
///   would WRONGLY trim trailing zeros to `.123`, so it is not used here).
///
/// Residual divergence: Python `datetime` resolution caps at microseconds, and
/// `SecondsFormat::Micros` truncates any nanosecond tail to microseconds, so the
/// two agree across the entire representable range of a `chrono` UTC instant.
pub fn isoformat(dt: DateTime<Utc>) -> String {
    if dt.timestamp_subsec_micros() == 0 {
        dt.to_rfc3339_opts(SecondsFormat::Secs, false)
    } else {
        dt.to_rfc3339_opts(SecondsFormat::Micros, false)
    }
}

/// Lowercase + collapse internal whitespace (upstream `_normalize_string_exact`).
pub fn normalize_string_exact(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isoformat_matches_python_datetime_isoformat() {
        use chrono::TimeZone;
        // Zero sub-second: Python emits no fractional part.
        let zero = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(isoformat(zero), "2026-01-01T00:00:00+00:00");
        // Full microseconds: Python emits all six digits.
        let micro = zero + chrono::Duration::microseconds(123_456);
        assert_eq!(isoformat(micro), "2026-01-01T00:00:00.123456+00:00");
        // Trailing-zero microseconds: Python keeps six digits (.123000), it does
        // NOT trim to .123 — this is exactly the case `to_rfc3339`/AutoSi gets wrong.
        let milli = zero + chrono::Duration::microseconds(123_000);
        assert_eq!(isoformat(milli), "2026-01-01T00:00:00.123000+00:00");
        // Sub-microsecond nanoseconds truncate to microseconds (Python caps there too).
        let nano = zero + chrono::Duration::nanoseconds(123_456_789);
        assert_eq!(isoformat(nano), "2026-01-01T00:00:00.123456+00:00");
    }

    #[test]
    fn normalize_collapses_whitespace_and_lowercases() {
        assert_eq!(normalize_string_exact("  Alice   SMITH "), "alice smith");
        assert_eq!(normalize_string_exact(""), "");
        assert_eq!(normalize_string_exact("\t Alice \n SMITH\t"), "alice smith");
        assert_eq!(normalize_string_exact("  "), "");
        assert_eq!(normalize_string_exact("ok"), "ok");
    }
}
