//! Server-side bookkeeping kept beside the graph: API keys (hash only),
//! index metadata, and per-document `content_hash` + chunk indexes. The
//! chunk indexes are what make deletion exact — chunk UUIDs are
//! deterministic (`chronicle_core::chunk_uuid`), so a document's nodes are
//! deleted by UUID without scanning the graph.

use std::path::Path;

use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem, RocksDb};
use surrealdb::types::{RecordId, SurrealValue};

#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    #[error("meta store: {0}")]
    Store(String),
}

fn store(e: impl std::fmt::Display) -> MetaError {
    MetaError::Store(e.to_string())
}

#[derive(Debug, Clone, PartialEq, SurrealValue)]
pub struct KeyRecord {
    pub key_id: String,
    pub hash: String,
    pub tenant_slug: String,
    pub teams: Vec<String>,
    pub label: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, SurrealValue)]
pub struct IndexRecord {
    pub group_id: String,
    pub index_id: String,
    pub name: String,
    pub embedding_model: String,
    pub dims: i64,
    pub chunk_size: i64,
    pub chunk_overlap: i64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, SurrealValue)]
pub struct DocRecord {
    pub group_id: String,
    pub document_id: String,
    pub content_hash: String,
    pub chunk_indexes: Vec<i64>,
}

const DDL: &str = "
DEFINE TABLE IF NOT EXISTS cix_key SCHEMALESS;
DEFINE INDEX IF NOT EXISTS cix_key_hash ON cix_key FIELDS hash UNIQUE;
DEFINE TABLE IF NOT EXISTS cix_index SCHEMALESS;
DEFINE TABLE IF NOT EXISTS cix_doc SCHEMALESS;
DEFINE INDEX IF NOT EXISTS cix_doc_group ON cix_doc FIELDS group_id;
";

fn key_rid(key_id: &str) -> RecordId {
    RecordId::new("cix_key", key_id)
}

fn index_rid(group_id: &str) -> RecordId {
    RecordId::new("cix_index", group_id)
}

fn doc_rid(group_id: &str, document_id: &str) -> RecordId {
    RecordId::new("cix_doc", format!("{group_id}\u{1f}{document_id}"))
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn rfc3339(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .to_rfc3339()
}

#[derive(Clone)]
pub struct MetaStore {
    db: Surreal<Db>,
}

impl MetaStore {
    pub async fn open(path: &Path) -> Result<Self, MetaError> {
        let path = path
            .to_str()
            .ok_or_else(|| MetaError::Store(format!("non-UTF-8 path {}", path.display())))?;
        Self::init(Surreal::new::<RocksDb>(path).await.map_err(store)?).await
    }

    pub async fn memory() -> Result<Self, MetaError> {
        Self::init(Surreal::new::<Mem>(()).await.map_err(store)?).await
    }

    async fn init(db: Surreal<Db>) -> Result<Self, MetaError> {
        db.use_ns("chronicle_index")
            .use_db("meta")
            .await
            .map_err(store)?;
        db.query(DDL).await.map_err(store)?.check().map_err(store)?;
        Ok(Self { db })
    }

    async fn upsert<R: SurrealValue>(&self, id: RecordId, row: R) -> Result<(), MetaError> {
        self.db
            .query("UPSERT $id CONTENT $row")
            .bind(("id", id))
            .bind(("row", row))
            .await
            .map_err(store)?
            .check()
            .map_err(store)?;
        Ok(())
    }

    pub async fn insert_key(&self, rec: &KeyRecord) -> Result<(), MetaError> {
        self.upsert(key_rid(&rec.key_id), rec.clone()).await
    }

    pub async fn key_by_hash(&self, hash: &str) -> Result<Option<KeyRecord>, MetaError> {
        let mut res = self
            .db
            .query("SELECT * FROM cix_key WHERE hash = $hash LIMIT 1")
            .bind(("hash", hash.to_string()))
            .await
            .map_err(store)?;
        Ok(res
            .take::<Vec<KeyRecord>>(0)
            .map_err(store)?
            .into_iter()
            .next())
    }

    pub async fn list_keys(&self) -> Result<Vec<KeyRecord>, MetaError> {
        let mut res = self
            .db
            .query("SELECT * FROM cix_key ORDER BY created_at_ms")
            .await
            .map_err(store)?;
        res.take::<Vec<KeyRecord>>(0).map_err(store)
    }

    pub async fn delete_key(&self, key_id: &str) -> Result<bool, MetaError> {
        let mut res = self
            .db
            .query("DELETE $id RETURN BEFORE")
            .bind(("id", key_rid(key_id)))
            .await
            .map_err(store)?;
        Ok(!res.take::<Vec<KeyRecord>>(0).map_err(store)?.is_empty())
    }

    pub async fn get_index(&self, group_id: &str) -> Result<Option<IndexRecord>, MetaError> {
        let mut res = self
            .db
            .query("SELECT * FROM $id")
            .bind(("id", index_rid(group_id)))
            .await
            .map_err(store)?;
        Ok(res
            .take::<Vec<IndexRecord>>(0)
            .map_err(store)?
            .into_iter()
            .next())
    }

    pub async fn put_index(&self, rec: &IndexRecord) -> Result<(), MetaError> {
        self.upsert(index_rid(&rec.group_id), rec.clone()).await
    }

    pub async fn delete_index(&self, group_id: &str) -> Result<(), MetaError> {
        self.db
            .query("DELETE cix_doc WHERE group_id = $group; DELETE $id")
            .bind(("group", group_id.to_string()))
            .bind(("id", index_rid(group_id)))
            .await
            .map_err(store)?
            .check()
            .map_err(store)?;
        Ok(())
    }

    pub async fn get_doc(
        &self,
        group_id: &str,
        document_id: &str,
    ) -> Result<Option<DocRecord>, MetaError> {
        let mut res = self
            .db
            .query("SELECT * FROM $id")
            .bind(("id", doc_rid(group_id, document_id)))
            .await
            .map_err(store)?;
        Ok(res
            .take::<Vec<DocRecord>>(0)
            .map_err(store)?
            .into_iter()
            .next())
    }

    pub async fn put_doc(&self, rec: &DocRecord) -> Result<(), MetaError> {
        self.upsert(doc_rid(&rec.group_id, &rec.document_id), rec.clone())
            .await
    }

    pub async fn delete_doc(
        &self,
        group_id: &str,
        document_id: &str,
    ) -> Result<Option<DocRecord>, MetaError> {
        let mut res = self
            .db
            .query("DELETE $id RETURN BEFORE")
            .bind(("id", doc_rid(group_id, document_id)))
            .await
            .map_err(store)?;
        Ok(res
            .take::<Vec<DocRecord>>(0)
            .map_err(store)?
            .into_iter()
            .next())
    }

    pub async fn list_docs(&self, group_id: &str) -> Result<Vec<DocRecord>, MetaError> {
        let mut res = self
            .db
            .query("SELECT * FROM cix_doc WHERE group_id = $group")
            .bind(("group", group_id.to_string()))
            .await
            .map_err(store)?;
        res.take::<Vec<DocRecord>>(0).map_err(store)
    }
}

impl From<MetaError> for crate::error::ApiError {
    fn from(err: MetaError) -> Self {
        Self::internal(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: &str, hash: &str) -> KeyRecord {
        KeyRecord {
            key_id: id.into(),
            hash: hash.into(),
            tenant_slug: "acme".into(),
            teams: vec!["*".into()],
            label: "test".into(),
            created_at_ms: 1,
        }
    }

    fn index(group: &str) -> IndexRecord {
        IndexRecord {
            group_id: group.into(),
            index_id: "kb1".into(),
            name: "Policies".into(),
            embedding_model: "text-embedding-3-small".into(),
            dims: 4,
            chunk_size: 1000,
            chunk_overlap: 100,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    fn doc(group: &str, id: &str, hash: &str, indexes: &[i64]) -> DocRecord {
        DocRecord {
            group_id: group.into(),
            document_id: id.into(),
            content_hash: hash.into(),
            chunk_indexes: indexes.to_vec(),
        }
    }

    #[tokio::test]
    async fn keys_are_found_by_hash_and_revocable() {
        let meta = MetaStore::memory().await.expect("meta");
        meta.insert_key(&key("k1", "h1")).await.expect("insert");
        assert_eq!(
            meta.key_by_hash("h1").await.expect("get").map(|k| k.key_id),
            Some("k1".into())
        );
        assert!(meta.key_by_hash("nope").await.expect("get").is_none());
        assert_eq!(meta.list_keys().await.expect("list").len(), 1);
        assert!(meta.delete_key("k1").await.expect("delete"));
        assert!(!meta.delete_key("k1").await.expect("delete again"));
        assert!(meta.key_by_hash("h1").await.expect("get").is_none());
    }

    #[tokio::test]
    async fn documents_are_scoped_to_their_group() {
        let meta = MetaStore::memory().await.expect("meta");
        meta.put_doc(&doc("g1", "d1", "h", &[0, 1]))
            .await
            .expect("put");
        meta.put_doc(&doc("g2", "d1", "other", &[0]))
            .await
            .expect("put");
        assert_eq!(
            meta.get_doc("g1", "d1")
                .await
                .expect("get")
                .map(|d| d.content_hash),
            Some("h".into())
        );
        assert_eq!(meta.list_docs("g1").await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn deleting_an_index_removes_its_documents_only() {
        let meta = MetaStore::memory().await.expect("meta");
        meta.put_index(&index("g1")).await.expect("put");
        meta.put_doc(&doc("g1", "d1", "h", &[0]))
            .await
            .expect("put");
        meta.put_doc(&doc("g2", "d1", "h", &[0]))
            .await
            .expect("put");
        meta.delete_index("g1").await.expect("delete");
        assert!(meta.get_index("g1").await.expect("get").is_none());
        assert!(meta.list_docs("g1").await.expect("list").is_empty());
        assert_eq!(meta.list_docs("g2").await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn delete_doc_returns_what_it_removed() {
        let meta = MetaStore::memory().await.expect("meta");
        meta.put_doc(&doc("g1", "d1", "h", &[0, 3]))
            .await
            .expect("put");
        let removed = meta.delete_doc("g1", "d1").await.expect("delete");
        assert_eq!(removed.map(|d| d.chunk_indexes), Some(vec![0, 3]));
        assert!(meta.delete_doc("g1", "d1").await.expect("delete").is_none());
    }

    #[test]
    fn rfc3339_formats_epoch_millis() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00+00:00");
    }
}
