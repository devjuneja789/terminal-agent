#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${TERMINAL_SLM_ADAPTER_REPO:?Set a NEW private artifact repo as namespace/name}"
hf_bin="$(command -v hf || true)"
if [[ -z "$hf_bin" ]]; then hf_bin="$repo_root/.venv/bin/hf"; fi
"$hf_bin" auth whoami
"$hf_bin" jobs hardware
bundle_dir="$(mktemp -d /tmp/terminal-slm-sft-bundle.XXXXXX)"
trap 'rm -rf -- "$bundle_dir"' EXIT
config_args=()
if [[ -n "${TERMINAL_SLM_SFT_CONFIG:-}" ]]; then config_args+=(--config "$TERMINAL_SLM_SFT_CONFIG"); fi
python3 -m "${TERMINAL_SLM_SFT_PREPARER:-training.agenttune.prepare_sft_bundle}" \
  --destination "$bundle_dir" --artifact-repo "$TERMINAL_SLM_ADAPTER_REPO" "${config_args[@]}"
output_dir="${TERMINAL_SLM_SFT_RESULT_DIR:-research/results/tiny-aya-sft-smoke}"
mkdir -p "$output_dir"
log_path="$output_dir/$(date -u +%Y%m%dT%H%M%SZ).log"
"$hf_bin" jobs run --name "${TERMINAL_SLM_SFT_JOB_NAME:-terminal-slm-approved-sft-smoke-v2}" \
  --flavor t4-small --timeout "${TERMINAL_SLM_SFT_TIMEOUT:-25m}" -s HF_TOKEN \
  --volume "$bundle_dir:/workspace" python:3.12 \
  -- bash /workspace/training/agenttune/run_sft_job.sh 2>&1 | tee "$log_path"
pointer_args=()
if [[ -n "${TERMINAL_SLM_SFT_RESULT_POINTER:-}" ]]; then pointer_args+=(--result-pointer "$TERMINAL_SLM_SFT_RESULT_POINTER"); fi
python3 scripts/extract_sft_training.py "$log_path" --output-dir "$output_dir" "${pointer_args[@]}"
