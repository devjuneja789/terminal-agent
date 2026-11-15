#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${TERMINAL_SLM_CONTEXT_TRAINING_RESULT:?Set the completed joint training result path}"
TERMINAL_SLM_CONTEXT_PREPARER=training.agenttune.prepare_joint_contexts \
TERMINAL_SLM_CONTEXT_RESULT_DIR=research/results/tiny-aya-joint-contexts \
TERMINAL_SLM_CONTEXT_JOB_NAME=terminal-slm-joint-choice-contexts-v1 \
bash scripts/run_tiny_aya_corrective_contexts.sh
