#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hf_bin="$repo_root/.venv/bin/hf"
"$hf_bin" auth whoami
"$hf_bin" jobs hardware
bundle_dir="$(mktemp -d /tmp/terminal-slm-final-heldout.XXXXXX)"
trap 'rm -rf -- "$bundle_dir"' EXIT
python3 -m training.agenttune.prepare_final_heldout --destination "$bundle_dir"
output_dir=research/results/tiny-aya-final-heldout
mkdir -p "$output_dir"
log_path="$output_dir/$(date -u +%Y%m%dT%H%M%SZ).log"
"$hf_bin" jobs run --name terminal-slm-final-paired-heldout-v1 --flavor t4-small --timeout 50m \
  -s HF_TOKEN --volume "$bundle_dir:/workspace" python:3.12 \
  -- bash /workspace/training/agenttune/run_final_heldout_job.sh 2>&1 | tee "$log_path"
python3 scripts/extract_paired_replay.py "$log_path" --output-dir "$output_dir" --expected-count 20 --split held-out-eval
echo "Full final evaluation log: $log_path"
