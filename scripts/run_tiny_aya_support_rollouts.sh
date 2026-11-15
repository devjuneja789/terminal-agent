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
output_dir="$repo_root/research/results/tiny-aya-support-rollouts"
log_path="$output_dir/${run_tag}.log"
bundle_dir="$(mktemp -d /tmp/terminal-slm-support.XXXXXX)"
cleanup() { rm -rf -- "$bundle_dir"; }
trap cleanup EXIT
mkdir -p "$output_dir"
task_id_args=()
if [[ -n "${TERMINAL_SLM_SUPPORT_TASK_IDS:-}" ]]; then
  IFS=',' read -r -a task_ids <<< "$TERMINAL_SLM_SUPPORT_TASK_IDS"
  for task_id in "${task_ids[@]}"; do
    task_id_args+=(--task-id "$task_id")
  done
fi
python3 "$repo_root/training/agenttune/prepare_hf_support_bundle.py" \
  --destination "$bundle_dir" "${task_id_args[@]}"
if [[ -e "$bundle_dir/evals/benchmark/heldout_eval.jsonl" || -e "$bundle_dir/configs/tiny_aya_eval.json" ]]; then
  echo "Support bundle preflight failed: held-out task/config data is present" >&2
  exit 2
fi
if [[ ! -f "$bundle_dir/training/agenttune/run_support_job.sh" ]]; then
  echo "Support bundle preflight failed: cloud runner is missing" >&2
  exit 2
fi

set +e
hf jobs run \
  --name terminal-slm-tiny-aya-support-rollout-v6 \
  --flavor t4-small \
  --timeout 40m \
  -s HF_TOKEN \
  --volume "$bundle_dir:/workspace" \
  python:3.12 \
  -- bash /workspace/training/agenttune/run_support_job.sh 2>&1 | tee "$log_path"
job_status=${PIPESTATUS[0]}
set -e
if (( job_status != 0 )); then
  echo "HF support rollout job failed (exit $job_status); full log: $log_path" >&2
  exit "$job_status"
fi
python3 "$repo_root/scripts/extract_hf_baseline.py" "$log_path" \
  --output-dir "$output_dir" --kind support
echo "Full HF Job log: $log_path"
