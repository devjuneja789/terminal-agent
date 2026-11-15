#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hf_bin="$(command -v hf || true)"
if [[ -z "$hf_bin" ]]; then hf_bin="$repo_root/.venv/bin/hf"; fi
"$hf_bin" auth whoami
"$hf_bin" jobs hardware
bundle_dir="$(mktemp -d /tmp/terminal-slm-action-diagnostic.XXXXXX)"
trap 'rm -rf -- "$bundle_dir"' EXIT
python3 -m training.agenttune.prepare_action_diagnostic --destination "$bundle_dir" \
  --training-result "${TERMINAL_SLM_DIAGNOSTIC_SOURCE:-research/results/tiny-aya-memorization/20261114T145152Z.result.json}"
output_dir="${TERMINAL_SLM_DIAGNOSTIC_OUTPUT:-research/results/tiny-aya-action-diagnostic}"
mkdir -p "$output_dir"
log_path="$output_dir/$(date -u +%Y%m%dT%H%M%SZ).log"
"$hf_bin" jobs run --name "${TERMINAL_SLM_DIAGNOSTIC_JOB_NAME:-terminal-slm-action-label-diagnostic-v1}" --flavor t4-small --timeout 30m \
  -s HF_TOKEN --volume "$bundle_dir:/workspace" python:3.12 \
  -- bash /workspace/training/agenttune/run_action_diagnostic_job.sh 2>&1 | tee "$log_path"
python3 scripts/extract_action_diagnostic.py "$log_path" --output-dir "$output_dir"
echo "Full job log: $log_path"
