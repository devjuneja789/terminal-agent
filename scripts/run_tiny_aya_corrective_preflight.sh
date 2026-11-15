#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hf_bin="$(command -v hf || true)"
if [[ -z "$hf_bin" ]]; then hf_bin="$repo_root/.venv/bin/hf"; fi
"$hf_bin" auth whoami
"$hf_bin" jobs hardware
bundle_dir="$(mktemp -d /tmp/terminal-slm-corrective-preflight.XXXXXX)"
trap 'rm -rf -- "$bundle_dir"' EXIT
python3 -m training.agenttune.prepare_corrective_preflight --destination "$bundle_dir"
output_dir=research/results/tiny-aya-corrective-preflight
mkdir -p "$output_dir"
log_path="$output_dir/$(date -u +%Y%m%dT%H%M%SZ).log"
"$hf_bin" jobs run --name terminal-slm-corrective-label-stop-preflight-v1 --flavor t4-small --timeout 30m \
  -s HF_TOKEN --volume "$bundle_dir:/workspace" python:3.12 \
  -- bash /workspace/training/agenttune/run_corrective_preflight_job.sh 2>&1 | tee "$log_path"
python3 scripts/extract_corrective_preflight.py "$log_path" --output-dir "$output_dir"
echo "Full job log: $log_path"
