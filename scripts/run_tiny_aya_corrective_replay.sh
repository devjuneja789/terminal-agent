#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${TERMINAL_SLM_REPLAY_TRAINING_RESULT:?Set the completed corrective training result path}"
TERMINAL_SLM_REPLAY_PREPARER=training.agenttune.prepare_corrective_replay \
TERMINAL_SLM_REPLAY_RESULT_DIR=research/results/tiny-aya-corrective-replay \
TERMINAL_SLM_REPLAY_JOB_NAME=terminal-slm-corrective-native-replay-v1 \
bash scripts/run_tiny_aya_paired_replay.sh
