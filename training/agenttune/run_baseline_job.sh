#!/usr/bin/env bash
set -euo pipefail

# Hugging Face mounts submitted volumes read-only; evaluate from ephemeral storage.
python -m pip install --quiet uv
cp -a /workspace /tmp/terminal-slm-baseline
cd /tmp/terminal-slm-baseline
uv run \
  --python 3.12 \
  --with "torch==2.11.0" \
  --with "transformers==5.22.1" \
  --with "accelerate==1.15.0" \
  --with "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@1.1.3" \
  python training/agenttune/evaluate_heldout.py
