use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::sync::OnceCell;

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

#[derive(Clone)]
enum GraphBackend {
    Disk(PathBuf),
    Memory,
}

/// One chronicle graph store per embedding dimension: SurrealDB's HNSW
/// index fixes its dimension when the store is opened.
///
/// Opening a store is slow on a cold, populated volume and holds an exclusive
/// RocksDB `LOCK`. Two rules follow, and both exist because the opposite was
/// shipped and looped forever on the demo lane:
///
/// - the open runs in its OWN task, so a caller that goes away (a client
///   timeout drops the request future) does not abort it half way. An aborted
///   open leaves the lock held while the next request opens the same path
///   again, which then fails with "lock hold by current process";
/// - the opened driver is kept in a per-dimension [`OnceCell`], so every
///   waiter shares the one open and it is never repeated.
pub struct GraphPool {
    backend: GraphBackend,
    cells: Mutex<HashMap<usize, Arc<OnceCell<Arc<SurrealDriver>>>>>,
}

impl GraphPool {
    fn new(backend: GraphBackend) -> Self {
        Self {
            backend,
            cells: Mutex::new(HashMap::new()),
        }
    }

    pub async fn for_dims(&self, dims: usize) -> Result<Arc<SurrealDriver>, ApiError> {
        let cell = {
            let mut cells = self
                .cells
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Arc::clone(cells.entry(dims).or_default())
        };
        if let Some(driver) = cell.get() {
            return Ok(Arc::clone(driver));
        }
        let backend = self.backend.clone();
        let opening = tokio::spawn(async move {
            cell.get_or_try_init(|| open_driver(backend, dims))
                .await
                .map(Arc::clone)
        });
        match opening.await {
            Ok(result) => result,
            Err(join) => Err(ApiError::internal(format!("graph store open task: {join}"))),
        }
    }

    /// Open, in the background, every dimension that already has a store on
    /// disk, so the first request after a restart finds it ready instead of
    /// paying for the open. A dimension with no directory is left alone: it is
    /// created on first use, and creating it here would open stores nobody has
    /// asked for.
    pub fn warm_existing(self: &Arc<Self>) {
        let GraphBackend::Disk(root) = &self.backend else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(dims) = name
                .to_str()
                .and_then(|n| n.strip_prefix("dim-"))
                .and_then(|n| n.parse::<usize>().ok())
            else {
                continue;
            };
            let pool = Arc::clone(self);
            tokio::spawn(async move {
                let started = std::time::Instant::now();
                match pool.for_dims(dims).await {
                    Ok(_) => tracing::info!(
                        dims,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "graph store warmed"
                    ),
                    Err(_) => tracing::warn!(dims, "graph store warm-up failed"),
                }
            });
        }
    }
}

async fn open_driver(backend: GraphBackend, dims: usize) -> Result<Arc<SurrealDriver>, ApiError> {
    let driver = match backend {
        GraphBackend::Disk(root) => {
            let dir = root.join(format!("dim-{dims}"));
            std::fs::create_dir_all(&dir).map_err(ApiError::internal)?;
            let path = dir
                .to_str()
                .ok_or_else(|| ApiError::internal(format!("non-UTF-8 path {}", dir.display())))?;
            SurrealDriver::connect_embedded(path, dims).await
        }
        GraphBackend::Memory => SurrealDriver::connect_memory(dims).await,
    }
    .map_err(ApiError::internal)?;
    Ok(Arc::new(driver))
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
        let graphs = Arc::new(GraphPool::new(GraphBackend::Disk(cfg.data_dir.clone())));
        graphs.warm_existing();
        Ok(Self {
            meta: MetaStore::open(&meta_dir).await?,
            graphs,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrent_callers_share_one_open() {
        let pool = Arc::new(GraphPool::new(GraphBackend::Memory));
        let (a, b, c) = tokio::join!(pool.for_dims(4), pool.for_dims(4), pool.for_dims(4));
        let (a, b, c) = (a.unwrap(), b.unwrap(), c.unwrap());
        assert!(Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&b, &c));
    }

    #[tokio::test]
    async fn a_caller_that_goes_away_does_not_cancel_the_open() {
        let pool = Arc::new(GraphPool::new(GraphBackend::Memory));
        let caller = {
            let pool = Arc::clone(&pool);
            tokio::spawn(async move { pool.for_dims(4).await.map(|_| ()) })
        };
        caller.abort(); // a client timeout drops the request future like this
        let _ = caller.await;
        let first = pool.for_dims(4).await.unwrap();
        let second = pool.for_dims(4).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn warm_existing_opens_only_dimensions_already_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        // Create a real store for dim 8, then let it close.
        {
            let pool = GraphPool::new(GraphBackend::Disk(dir.path().to_path_buf()));
            pool.for_dims(8).await.unwrap();
        }
        let pool = Arc::new(GraphPool::new(GraphBackend::Disk(dir.path().to_path_buf())));
        // The previous pool's store releases its RocksDB lock asynchronously.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        pool.warm_existing();
        let warmed = pool.for_dims(8).await.unwrap();
        assert!(Arc::ptr_eq(&warmed, &pool.for_dims(8).await.unwrap()));
        assert!(
            !dir.path().join("dim-16").exists(),
            "warm-up must not create stores nobody asked for"
        );
    }
}
