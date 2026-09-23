# greentic-runner-chronicle

Chronicle long-term memory and knowledge (document-RAG) for the Greentic runner,
as `AgentRuntimeExtension`s, plus **`greentic-runner-full`** — the stock runner
CLI with both registered.

They used to be `greentic-runner-host`'s `long-term-chronicle` and
`knowledge-chronicle` cargo features. `cargo publish` refuses a crate whose
dependency has a git source and no registry version, even an optional one, and
the Chronicle crates are private — so those two features made the whole runner
workspace unpublishable on the `1.2.0-dev` crates.io lane. The host kept the
seam it always had (both mounts installed a trait object), and the concrete
backends live here. See greentic-runner#787.

## Build

```bash
cargo build -p greentic-runner-chronicle --bin greentic-runner-full --release
```

Needs `clang` (RocksDB, via the embedded SurrealDB driver).

## Run

`greentic-runner-full` takes exactly the arguments `greentic-runner` takes. The
backends stay disabled until the operator configures them, as before:

- long-term memory: `GREENTIC_CHRONICLE_*`
- knowledge: `GREENTIC_KNOWLEDGE_EMBED_*`, graph store via
  `GREENTIC_KNOWLEDGE_BACKEND` (default: embedded SurrealDB, no server needed)

A stock `greentic-runner` given that environment now logs one warning saying
nothing is registered to serve it, instead of running with no memory and no
retrieval in silence.
