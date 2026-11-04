#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI 'hf' is missing. Install/upgrade huggingface_hub, then retry." >&2
  exit 2
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "No Hugging Face login found. Run 'hf auth login', then retry." >&2
  exit 2
fi

echo "Check the live price and availability of t4-small before launch:"
hf jobs hardware

run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$repo_root/research/results/tiny-aya-baseline"
log_path="$output_dir/${run_tag}.log"
bundle_dir="$(mktemp -d /tmp/terminal-slm-heldout.XXXXXX)"
cleanup() { rm -rf -- "$bundle_dir"; }
trap cleanup EXIT
mkdir -p "$output_dir"
python "$repo_root/training/agenttune/prepare_hf_eval_bundle.py" --destination "$bundle_dir"
if [[ ! -f "$bundle_dir/training/agenttune/run_baseline_job.sh" || -e "$bundle_dir/evals/benchmark/train_support.jsonl" ]]; then
  echo "Held-out bundle preflight failed: missing job script or support tasks present" >&2
  exit 2
fi

set +e
hf jobs run \
  --name terminal-slm-tiny-aya-baseline \
  --flavor t4-small \
  --timeout 40m \
  -s HF_TOKEN \
  --volume "$bundle_dir:/workspace" \
  python:3.12 \
  -- bash /workspace/training/agenttune/run_baseline_job.sh 2>&1 | tee "$log_path"
job_status=${PIPESTATUS[0]}
set -e
if (( job_status != 0 )); then
  echo "HF baseline job failed (exit $job_status); full log: $log_path" >&2
  exit "$job_status"
fi
python "$repo_root/scripts/extract_hf_baseline.py" "$log_path" --output-dir "$output_dir"
echo "Full HF Job log: $log_path"
