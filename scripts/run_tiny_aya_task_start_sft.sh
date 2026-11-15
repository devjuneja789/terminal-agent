#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_SFT_PREPARER=training.agenttune.prepare_task_start_training \
TERMINAL_SLM_ADAPTER_REPO=devjuneja789/terminal-slm-tiny-aya-task-start-620-v1 \
TERMINAL_SLM_SFT_CONFIG="$repo_root/configs/tiny_aya_task_start_sft.json" \
TERMINAL_SLM_SFT_JOB_NAME=terminal-slm-task-start-620-v1 \
TERMINAL_SLM_SFT_RESULT_DIR=research/results/tiny-aya-task-start-sft \
TERMINAL_SLM_SFT_TIMEOUT=60m \
bash scripts/run_tiny_aya_sft_smoke.sh
