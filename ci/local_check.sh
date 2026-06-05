#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect empty virtual workspace (no members yet); skip cargo commands that
# require at least one package target.
MEMBER_COUNT=$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['workspace_members']))" \
    2>/dev/null || echo 0)

if [[ "$MEMBER_COUNT" -eq 0 ]]; then
    echo "local_check: workspace has no members — skipping fmt/clippy/test (vacuous pass)"
    exit 0
fi

cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
