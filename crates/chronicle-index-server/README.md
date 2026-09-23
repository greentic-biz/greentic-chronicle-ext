# chronicle-index-server

## What it is

`chronicle-index-server` is the `/v1/indexes` server the Greentic designer
syncs a tenant's knowledge bases into, and the server a deployed runtime
searches at retrieval time. It never embeds: every document chunk arrives
with its own vector already computed by the caller, and every search request
carries its own query vector alongside the query text. The server stores
chunks, serves vector + text hybrid search over them, and answers a stats
view the designer uses to confirm a sync landed.

## Running it

Four environment variables configure the process:

| Variable | Required | Default |
|---|---|---|
| `CHRONICLE_INDEX_BOOTSTRAP_KEY` | yes | none — the process refuses to start without one at least 32 characters long |
| `CHRONICLE_INDEX_DATA_DIR` | yes | none — the process refuses to start without one |
| `CHRONICLE_INDEX_BIND` | no | `0.0.0.0:8088` |
| `CHRONICLE_INDEX_MAX_BODY_BYTES` | no | `16777216` (16 MiB) |

Build and run the Docker image:

```bash
docker build -f crates/chronicle-index-server/Dockerfile -t chronicle-index-server .
docker run -p 8088:8088 -v chronicle-index:/data \
  -e CHRONICLE_INDEX_BOOTSTRAP_KEY=… \
  chronicle-index-server
```

**Single replica: one process owns the data directory.** The meta store and
every per-dimension graph store are embedded RocksDB, opened exclusively —
a second process pointed at the same `CHRONICLE_INDEX_DATA_DIR` fails to
start rather than sharing it.

## Minting a key

The bootstrap key authenticates only `/admin/v1/keys`. Mint a tenant-scoped
API key with it:

```bash
curl -X POST -H "Authorization: Bearer $BOOTSTRAP" \
  -d '{"tenant_slug":"acme","teams":["*"],"label":"designer"}' \
  http://localhost:8088/admin/v1/keys
```

The plaintext key appears only in that response — the server stores a
SHA-256 hash and never the key itself. `DELETE /admin/v1/keys/{key_id}`
revokes the key at once; the next request bearing it answers `401
unauthorized`.

## Routes

`tenant key` means `Authorization: Bearer <key>` plus `X-Greentic-Tenant`
(required) and `X-Greentic-Team` (optional, defaults to `general`). Every
error leaves as `{"error":{"code","message"}}`.

| Method + path | Auth | Success | Error codes |
|---|---|---|---|
| `GET /healthz` | none | `200 ok` | — |
| `POST /admin/v1/keys` | bootstrap | `201` | `unauthorized`, `bad_request` |
| `GET /admin/v1/keys` | bootstrap | `200` | `unauthorized` |
| `DELETE /admin/v1/keys/{key_id}` | bootstrap | `204` | `unauthorized`, `key_not_found` |
| `PUT /v1/indexes/{index_id}` | tenant key | `200` (existing) / `201` (created) | `unauthorized`, `bad_request`, `model_mismatch`, `dim_mismatch`, 413 |
| `DELETE /v1/indexes/{index_id}` | tenant key | `204` | `unauthorized`, `bad_request`, `index_not_found` |
| `GET /v1/indexes/{index_id}/stats` | tenant key | `200` | `unauthorized`, `bad_request`, `index_not_found` |
| `POST /v1/indexes/{index_id}/documents` | tenant key | `200` | `unauthorized`, `bad_request`, `index_not_found`, `dim_mismatch`, 413 |
| `DELETE /v1/indexes/{index_id}/documents/{document_id}` | tenant key | `204` | `unauthorized`, `bad_request`, `index_not_found`, `document_not_found` |
| `POST /v1/indexes/{index_id}/search` | tenant key | `200` | `unauthorized`, `bad_request`, `index_not_found`, `dim_mismatch`, 413 |

A document id or index id carrying a control character, or that is empty or
blank, answers `bad_request` — never reaching the store, since the meta
store's own record keys use a control character as a separator internally.

## Storage layout

```
<data>/meta        # keys, index records, and per-document content-hash + chunk-index records
<data>/dim-<n>      # one embedded chronicle graph store per embedding dimension in use
```

Back up by stopping the process and copying the whole `CHRONICLE_INDEX_DATA_DIR`
directory; there is no live-backup path.

## Known limits

- Single replica only — the storage layer takes an exclusive lock on the
  data directory.
- HNSW post-filtering by tenant/team/index can lower recall in a store
  holding very many indexes, since the graph's approximate search runs
  before the tenant filter narrows the result set.
- `GET /v1/indexes/{index_id}/stats` reads the meta records (document and
  chunk-index counts kept beside each document), not a scan of the graph
  store.
