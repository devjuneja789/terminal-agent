#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet uv
cp -a /workspace /tmp/terminal-slm-support
cd /tmp/terminal-slm-support
export TERMINAL_SLM_CONFIG=/tmp/terminal-slm-support/configs/tiny_aya_support_rollout.json
uv run \
  --python 3.12 \
  --with "torch==2.11.0" \
  --with "transformers==5.22.1" \
  --with "accelerate==1.15.0" \
  --with "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@1.1.3" \
  python training/agenttune/generate_support_rollouts.py
