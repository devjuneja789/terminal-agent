#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hf_bin="$(command -v hf || true)"
if [[ -z "$hf_bin" ]]; then hf_bin="$repo_root/.venv/bin/hf"; fi
"$hf_bin" auth whoami
"$hf_bin" jobs hardware
bundle_dir="$(mktemp -d /tmp/terminal-slm-observation-intervention.XXXXXX)"
trap 'rm -rf -- "$bundle_dir"' EXIT
python3 -m "${TERMINAL_SLM_INTERVENTION_PREPARER:-training.agenttune.prepare_observation_intervention}" --destination "$bundle_dir"
output_dir="${TERMINAL_SLM_INTERVENTION_OUTPUT:-research/results/tiny-aya-observation-intervention}"
mkdir -p "$output_dir"
log_path="$output_dir/$(date -u +%Y%m%dT%H%M%SZ).log"
"$hf_bin" jobs run --name "${TERMINAL_SLM_INTERVENTION_JOB_NAME:-terminal-slm-live-source-intervention-v1}" --flavor t4-small --timeout 30m \
  -s HF_TOKEN --volume "$bundle_dir:/workspace" python:3.12 \
  -- bash /workspace/training/agenttune/run_observation_intervention_job.sh 2>&1 | tee "$log_path"
python3 scripts/extract_observation_intervention.py "$log_path" --output-dir "$output_dir"
echo "Full job log: $log_path"
