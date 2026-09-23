//! `greentic-runner` plus the Chronicle memory and knowledge backends.
//!
//! The published `greentic-runner` binary carries neither, because the crates
//! behind them are private (see this crate's docs). This binary is the same CLI
//! — `greentic_runner::cli_main()` is `greentic-runner`'s whole `main` — with
//! both backends registered before it starts, which is what
//! `--features knowledge-chronicle,long-term-chronicle` used to produce.
//!
//! Registration must happen BEFORE the host boots: the corpus ingest and every
//! runtime construction read the registry as they run.

#[greentic_types::telemetry::main(service_name = "greentic-runner")]
async fn main() {
    greentic_runner_chronicle::register_all();
    greentic_runner::cli_main().await;
}
