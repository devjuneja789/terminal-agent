#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
TERMINAL_SLM_REPLAY_PREPARER=training.agenttune.prepare_fit300_support \
TERMINAL_SLM_REPLAY_RESULT_DIR=research/results/tiny-aya-fit300-support \
TERMINAL_SLM_REPLAY_JOB_NAME=terminal-slm-joint300-task-start-support-v1 \
bash scripts/run_tiny_aya_paired_replay.sh
