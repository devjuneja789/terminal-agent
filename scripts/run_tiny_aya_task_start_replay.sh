#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${TERMINAL_SLM_REPLAY_TRAINING_RESULT:?Set the completed curriculum training result}"
TERMINAL_SLM_REPLAY_PREPARER=training.agenttune.prepare_task_start_replay \
TERMINAL_SLM_REPLAY_RESULT_DIR=research/results/tiny-aya-task-start-replay \
TERMINAL_SLM_REPLAY_JOB_NAME=terminal-slm-task-start-620-support-replay-v1 \
bash scripts/run_tiny_aya_paired_replay.sh
