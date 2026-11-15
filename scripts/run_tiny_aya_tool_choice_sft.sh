#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_SFT_PREPARER=training.agenttune.prepare_tool_choice_training \
TERMINAL_SLM_ADAPTER_REPO="${TERMINAL_SLM_ADAPTER_REPO:-devjuneja789/terminal-slm-tiny-aya-tool-choice-60-v1}" \
TERMINAL_SLM_SFT_CONFIG="$repo_root/configs/tiny_aya_tool_choice_sft.json" \
TERMINAL_SLM_SFT_JOB_NAME=terminal-slm-tool-choice-60-v2-scale256 \
TERMINAL_SLM_SFT_RESULT_DIR=research/results/tiny-aya-tool-choice-sft \
bash scripts/run_tiny_aya_sft_smoke.sh
