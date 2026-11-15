#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_INTERVENTION_PREPARER=training.agenttune.prepare_observation_metadata \
TERMINAL_SLM_INTERVENTION_OUTPUT=research/results/tiny-aya-observation-metadata \
TERMINAL_SLM_INTERVENTION_JOB_NAME=terminal-slm-observation-metadata-parity-v1 \
bash scripts/run_tiny_aya_observation_intervention.sh
