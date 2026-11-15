#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
result_pointer="$(mktemp /tmp/terminal-slm-weighted-result.XXXXXX)"
trap 'rm -f -- "$result_pointer"' EXIT
TERMINAL_SLM_ADAPTER_REPO="${TERMINAL_SLM_ADAPTER_REPO:-devjuneja789/terminal-slm-tiny-aya-weighted-64-v1}" \
TERMINAL_SLM_SFT_CONFIG="$repo_root/configs/tiny_aya_weighted_sft.json" \
TERMINAL_SLM_SFT_JOB_NAME=terminal-slm-weighted-64-v2 \
TERMINAL_SLM_SFT_RESULT_DIR=research/results/tiny-aya-weighted-sft \
TERMINAL_SLM_SFT_RESULT_POINTER="$result_pointer" \
bash scripts/run_tiny_aya_sft_smoke.sh
result_path="$(cat "$result_pointer")"
TERMINAL_SLM_REPLAY_TRAINING_RESULT="$result_path" \
TERMINAL_SLM_REPLAY_RESULT_DIR=research/results/tiny-aya-weighted-replay \
TERMINAL_SLM_REPLAY_JOB_NAME=terminal-slm-weighted-replay-v1 \
bash scripts/run_tiny_aya_paired_replay.sh
