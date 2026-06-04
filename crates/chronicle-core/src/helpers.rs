// Ported from graphiti_core/helpers.py, utils/datetime_utils.py,
// utils/maintenance/dedup_helpers.py @ 34f56e65 (v0.29.1)

use chrono::{DateTime, Utc};

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
    fn normalize_collapses_whitespace_and_lowercases() {
        assert_eq!(normalize_string_exact("  Alice   SMITH "), "alice smith");
        assert_eq!(normalize_string_exact(""), "");
        assert_eq!(normalize_string_exact("\t Alice \n SMITH\t"), "alice smith");
        assert_eq!(normalize_string_exact("  "), "");
        assert_eq!(normalize_string_exact("ok"), "ok");
    }
}
