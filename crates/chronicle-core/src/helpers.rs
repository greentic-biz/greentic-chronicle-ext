use chrono::{DateTime, Utc};

/// Default concurrency for parallel LLM calls (upstream SEMAPHORE_LIMIT).
pub const SEMAPHORE_LIMIT: usize = 20;
/// Upstream RELEVANT_SCHEMA_LIMIT (search_utils.py): prior episodes pulled as context.
pub const RELEVANT_SCHEMA_LIMIT: usize = 10;
/// Upstream EPISODE_WINDOW_LEN (graph_data_operations.py).
pub const EPISODE_WINDOW_LEN: usize = 3;

pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
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
    }
}
