use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

use crate::wire::MAX_DIMS;

pub const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Embedding dimensions an index may be created with unless
/// `CHRONICLE_INDEX_ALLOWED_DIMS` says otherwise. Each distinct dimension
/// opens its own graph store for the life of the process, so the set is
/// bounded by the operator rather than by whatever a tenant asks for.
pub const DEFAULT_ALLOWED_DIMS: [usize; 5] = [384, 768, 1024, 1536, 3072];
const MIN_BOOTSTRAP_KEY_LEN: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("CHRONICLE_INDEX_BOOTSTRAP_KEY is not set")]
    MissingBootstrapKey,
    #[error("CHRONICLE_INDEX_BOOTSTRAP_KEY must be at least 32 characters")]
    WeakBootstrapKey,
    #[error("CHRONICLE_INDEX_DATA_DIR is not set")]
    MissingDataDir,
    #[error("CHRONICLE_INDEX_BIND is not a socket address: {0}")]
    InvalidBind(String),
    #[error("CHRONICLE_INDEX_MAX_BODY_BYTES must be a positive integer: {0}")]
    InvalidMaxBody(String),
    #[error(
        "CHRONICLE_INDEX_ALLOWED_DIMS must be a comma-separated list of integers between 1 and {MAX_DIMS}: {0:?}"
    )]
    InvalidAllowedDims(String),
}

pub struct Config {
    pub bind: SocketAddr,
    pub data_dir: PathBuf,
    pub bootstrap_key: String,
    pub max_body_bytes: usize,
    /// Sorted and deduplicated.
    pub allowed_dims: Vec<usize>,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("bind", &self.bind)
            .field("data_dir", &self.data_dir)
            .field("bootstrap_key", &"<redacted>")
            .field("max_body_bytes", &self.max_body_bytes)
            .field("allowed_dims", &self.allowed_dims)
            .finish()
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let bootstrap_key = get("CHRONICLE_INDEX_BOOTSTRAP_KEY")
            .filter(|k| !k.trim().is_empty())
            .ok_or(ConfigError::MissingBootstrapKey)?;
        if bootstrap_key.len() < MIN_BOOTSTRAP_KEY_LEN {
            return Err(ConfigError::WeakBootstrapKey);
        }
        let data_dir = get("CHRONICLE_INDEX_DATA_DIR")
            .filter(|d| !d.trim().is_empty())
            .map(PathBuf::from)
            .ok_or(ConfigError::MissingDataDir)?;
        let bind = match get("CHRONICLE_INDEX_BIND") {
            Some(raw) => raw.parse().map_err(|_| ConfigError::InvalidBind(raw))?,
            None => SocketAddr::from(([0, 0, 0, 0], 8088)),
        };
        let max_body_bytes = match get("CHRONICLE_INDEX_MAX_BODY_BYTES") {
            Some(raw) => raw
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or(ConfigError::InvalidMaxBody(raw))?,
            None => DEFAULT_MAX_BODY_BYTES,
        };
        let allowed_dims = match get("CHRONICLE_INDEX_ALLOWED_DIMS") {
            Some(raw) => parse_allowed_dims(&raw).ok_or(ConfigError::InvalidAllowedDims(raw))?,
            None => DEFAULT_ALLOWED_DIMS.to_vec(),
        };
        Ok(Self {
            bind,
            data_dir,
            bootstrap_key,
            max_body_bytes,
            allowed_dims,
        })
    }
}

/// `None` for an empty list or any entry that is not an integer in
/// `1..=MAX_DIMS` — a typo must stop the server, not silently narrow or
/// widen what it accepts.
fn parse_allowed_dims(raw: &str) -> Option<Vec<usize>> {
    let max = usize::try_from(MAX_DIMS).ok()?;
    let mut dims = raw
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=max).contains(n))
        })
        .collect::<Option<Vec<_>>>()?;
    dims.sort_unstable();
    dims.dedup();
    Some(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn defaults_apply_when_only_required_values_are_set() {
        let cfg = Config::from_lookup(lookup(&[
            ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
            ("CHRONICLE_INDEX_DATA_DIR", "/data"),
        ]))
        .expect("config");
        assert_eq!(cfg.bind.to_string(), "0.0.0.0:8088");
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }

    #[test]
    fn a_missing_or_short_bootstrap_key_refuses_to_start() {
        assert!(matches!(
            Config::from_lookup(lookup(&[("CHRONICLE_INDEX_DATA_DIR", "/data")])),
            Err(ConfigError::MissingBootstrapKey)
        ));
        assert!(matches!(
            Config::from_lookup(lookup(&[
                ("CHRONICLE_INDEX_BOOTSTRAP_KEY", "short"),
                ("CHRONICLE_INDEX_DATA_DIR", "/data"),
            ])),
            Err(ConfigError::WeakBootstrapKey)
        ));
    }

    #[test]
    fn an_unparseable_bind_or_body_limit_is_an_error_not_a_default() {
        assert!(
            Config::from_lookup(lookup(&[
                ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
                ("CHRONICLE_INDEX_DATA_DIR", "/data"),
                ("CHRONICLE_INDEX_BIND", "not-an-addr"),
            ]))
            .is_err()
        );
        assert!(
            Config::from_lookup(lookup(&[
                ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
                ("CHRONICLE_INDEX_DATA_DIR", "/data"),
                ("CHRONICLE_INDEX_MAX_BODY_BYTES", "0"),
            ]))
            .is_err()
        );
    }

    #[test]
    fn allowed_dims_default_to_the_common_embedding_sizes() {
        let cfg = Config::from_lookup(lookup(&[
            ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
            ("CHRONICLE_INDEX_DATA_DIR", "/data"),
        ]))
        .expect("config");
        assert_eq!(cfg.allowed_dims, DEFAULT_ALLOWED_DIMS);
    }

    #[test]
    fn allowed_dims_can_be_set_and_are_sorted_and_deduplicated() {
        let cfg = Config::from_lookup(lookup(&[
            ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
            ("CHRONICLE_INDEX_DATA_DIR", "/data"),
            ("CHRONICLE_INDEX_ALLOWED_DIMS", " 768, 4,768 "),
        ]))
        .expect("config");
        assert_eq!(cfg.allowed_dims, vec![4, 768]);
    }

    #[test]
    fn unusable_allowed_dims_are_an_error_not_a_default() {
        for raw in ["", " ", "0", "384,x", "384,,768", "-4", "9000"] {
            assert!(
                matches!(
                    Config::from_lookup(lookup(&[
                        ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
                        ("CHRONICLE_INDEX_DATA_DIR", "/data"),
                        ("CHRONICLE_INDEX_ALLOWED_DIMS", raw),
                    ])),
                    Err(ConfigError::InvalidAllowedDims(_))
                ),
                "{raw:?} must be refused"
            );
        }
    }

    #[test]
    fn debug_never_prints_the_bootstrap_key() {
        let cfg = Config::from_lookup(lookup(&[
            ("CHRONICLE_INDEX_BOOTSTRAP_KEY", KEY),
            ("CHRONICLE_INDEX_DATA_DIR", "/data"),
        ]))
        .expect("config");
        assert!(!format!("{cfg:?}").contains(KEY));
    }
}
