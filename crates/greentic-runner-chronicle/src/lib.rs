//! Chronicle-backed long-term memory and knowledge (document-RAG) for the
//! Greentic runner, as `AgentRuntimeExtension`s.
//!
//! Both modules were `greentic-runner-host`'s, behind its `long-term-chronicle`
//! and `knowledge-chronicle` cargo features. They moved here in
//! greentic-runner#787 for one reason: `cargo publish` refuses a crate whose
//! dependency has a git source and no registry version, even an optional one,
//! and the Chronicle crates are private. Keeping the features there made the
//! whole runner workspace unpublishable on the `1.2.0-dev` crates.io lane.
//!
//! What the host kept is the seam. Every mount always installed a trait object
//! (`Arc<dyn LongTermMemory>` / `Arc<dyn Knowledge>`), so the two backends are
//! now registered at startup instead of compiled in:
//!
//! ```ignore
//! greentic_runner_host::runner::runtime_ext::register_agent_runtime_extension(
//!     std::sync::Arc::new(greentic_runner_chronicle::ChronicleLongTermMemory),
//! );
//! ```
//!
//! [`register_all`] does both, in the order the host's feature-gated sites used
//! (long-term memory, then knowledge), and the `greentic-runner-full` binary in
//! this crate is that call plus `greentic_runner::cli_main()`.
//!
//! Behaviour is unchanged: both are env-driven and fail-open, reading
//! `GREENTIC_CHRONICLE_*` and `GREENTIC_KNOWLEDGE_*` respectively, and a
//! runtime whose environment is absent or whose connection fails passes through
//! untouched.

mod knowledge_mount;
mod long_term_memory;

pub use knowledge_mount::ChronicleKnowledge;
pub use long_term_memory::ChronicleLongTermMemory;

/// Register both backends with the host, long-term memory first.
///
/// Order is the host's: extensions attach in registration order and each wraps
/// the previous one, which is what the two feature-gated call sites did when
/// they ran back to back.
pub fn register_all() {
    use greentic_runner_host::runner::runtime_ext::register_agent_runtime_extension;

    register_agent_runtime_extension(std::sync::Arc::new(ChronicleLongTermMemory));
    register_agent_runtime_extension(std::sync::Arc::new(ChronicleKnowledge));
}
