#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Jobs mounts the submitted project volume read-only. Copy it to
# writable ephemeral storage before uv installs/runs the smoke script.
python -m pip install --quiet uv
cp -a /workspace /tmp/terminal-slm-phase1
cd /tmp/terminal-slm-phase1
uv run \
  --python 3.12 \
  --with "torch==2.11.0" \
  --with "transformers==5.22.1" \
  --with "accelerate==1.15.0" \
  --with "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@1.1.3" \
  python training/agenttune/hf_gpu_smoke.py
