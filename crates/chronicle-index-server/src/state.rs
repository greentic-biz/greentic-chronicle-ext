use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chronicle_driver_surreal::SurrealDriver;
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::error::ApiError;
use crate::meta::{MetaError, MetaStore};

#[derive(Debug, thiserror::Error)]
pub enum StartupError {
    #[error(transparent)]
    Meta(#[from] MetaError),
    #[error("create data dir: {0}")]
    Io(#[from] std::io::Error),
}

enum GraphBackend {
    Disk(PathBuf),
    Memory,
}

/// One chronicle graph store per embedding dimension: SurrealDB's HNSW
/// index fixes its dimension when the store is opened.
pub struct GraphPool {
    backend: GraphBackend,
    drivers: tokio::sync::Mutex<HashMap<usize, Arc<SurrealDriver>>>,
}

impl GraphPool {
    fn new(backend: GraphBackend) -> Self {
        Self {
            backend,
            drivers: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    pub async fn for_dims(&self, dims: usize) -> Result<Arc<SurrealDriver>, ApiError> {
        let mut drivers = self.drivers.lock().await;
        if let Some(driver) = drivers.get(&dims) {
            return Ok(Arc::clone(driver));
        }
        let driver = match &self.backend {
            GraphBackend::Disk(root) => {
                let dir = root.join(format!("dim-{dims}"));
                std::fs::create_dir_all(&dir).map_err(ApiError::internal)?;
                let path = dir.to_str().ok_or_else(|| {
                    ApiError::internal(format!("non-UTF-8 path {}", dir.display()))
                })?;
                SurrealDriver::connect_embedded(path, dims).await
            }
            GraphBackend::Memory => SurrealDriver::connect_memory(dims).await,
        }
        .map_err(ApiError::internal)?;
        let driver = Arc::new(driver);
        drivers.insert(dims, Arc::clone(&driver));
        Ok(driver)
    }
}

/// Serialises writes per index so the graph and the meta records move
/// together. Reads (search, stats) do not take it.
#[derive(Default)]
pub struct IndexLocks {
    inner: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl IndexLocks {
    pub async fn lock(&self, group_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut map = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Arc::clone(map.entry(group_id.to_string()).or_default())
        };
        lock.lock_owned().await
    }
}

#[derive(Clone)]
pub struct AppState {
    pub meta: MetaStore,
    pub graphs: Arc<GraphPool>,
    pub locks: Arc<IndexLocks>,
    allowed_dims: Arc<[usize]>,
    bootstrap_hash: Arc<[u8; 32]>,
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

impl AppState {
    pub async fn open(cfg: &Config) -> Result<Self, StartupError> {
        std::fs::create_dir_all(&cfg.data_dir)?;
        let meta_dir = cfg.data_dir.join("meta");
        std::fs::create_dir_all(&meta_dir)?;
        Ok(Self {
            meta: MetaStore::open(&meta_dir).await?,
            graphs: Arc::new(GraphPool::new(GraphBackend::Disk(cfg.data_dir.clone()))),
            locks: Arc::new(IndexLocks::default()),
            allowed_dims: Arc::from(cfg.allowed_dims.as_slice()),
            bootstrap_hash: Arc::new(sha256(cfg.bootstrap_key.as_bytes())),
        })
    }

    /// Everything in memory — for tests and local experiments.
    pub async fn in_memory(
        bootstrap_key: &str,
        allowed_dims: &[usize],
    ) -> Result<Self, StartupError> {
        Ok(Self {
            meta: MetaStore::memory().await?,
            graphs: Arc::new(GraphPool::new(GraphBackend::Memory)),
            locks: Arc::new(IndexLocks::default()),
            allowed_dims: Arc::from(allowed_dims),
            bootstrap_hash: Arc::new(sha256(bootstrap_key.as_bytes())),
        })
    }

    /// Whether an index may be created with `dims` on this server.
    pub fn allows_dims(&self, dims: i64) -> bool {
        usize::try_from(dims).is_ok_and(|d| self.allowed_dims.contains(&d))
    }

    pub fn bootstrap_hash(&self) -> &[u8; 32] {
        &self.bootstrap_hash
    }
}
