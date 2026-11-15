#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_DIAGNOSTIC_SOURCE=research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json \
TERMINAL_SLM_DIAGNOSTIC_OUTPUT=research/results/tiny-aya-source-preservation \
TERMINAL_SLM_DIAGNOSTIC_JOB_NAME=terminal-slm-weighted-source-preservation-v1 \
bash scripts/run_tiny_aya_action_diagnostic.sh
