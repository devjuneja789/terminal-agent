#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI 'hf' is missing. Install/upgrade huggingface_hub, then retry." >&2
  exit 2
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "No Hugging Face login found. Run 'hf auth login', then retry." >&2
  exit 2
fi

echo "Current HF GPU hardware and pricing (confirm t4-small before launch):"
hf jobs hardware

bundle_dir="$(mktemp -d /tmp/terminal-slm-hf-smoke.XXXXXX)"
cleanup() {
  rm -rf -- "$bundle_dir"
}
trap cleanup EXIT

python "$repo_root/training/agenttune/prepare_hf_smoke_bundle.py" --destination "$bundle_dir"

# Hugging Face lists t4-small at $0.40/hour and bills Jobs by the minute.
# The 20-minute job timeout caps the GPU portion at about $0.14 at that rate.
hf jobs run \
  --name terminal-slm-phase1-tiny-aya-smoke \
  --flavor t4-small \
  --timeout 20m \
  -s HF_TOKEN \
  --volume "$bundle_dir:/workspace" \
  python:3.12 \
  -- bash /workspace/training/agenttune/run_smoke_job.sh
