#![forbid(unsafe_code)]
//! HTTP knowledge-index server: the `/v1/indexes` contract the Greentic
//! designer syncs into, a vector search route for runtimes, and a
//! tenant-bound API-key admin API. It never embeds: every document chunk and
//! every query arrives with its vector.

pub mod config;
pub mod error;
pub mod wire;
