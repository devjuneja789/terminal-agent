#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_SFT_PREPARER=training.agenttune.prepare_joint_training \
TERMINAL_SLM_ADAPTER_REPO="${TERMINAL_SLM_ADAPTER_REPO:-devjuneja789/terminal-slm-tiny-aya-joint-60-v1}" \
TERMINAL_SLM_SFT_CONFIG="$repo_root/configs/tiny_aya_joint_choice_sft.json" \
TERMINAL_SLM_SFT_JOB_NAME=terminal-slm-joint-decision-60-v1 \
TERMINAL_SLM_SFT_RESULT_DIR=research/results/tiny-aya-joint-sft \
bash scripts/run_tiny_aya_sft_smoke.sh
