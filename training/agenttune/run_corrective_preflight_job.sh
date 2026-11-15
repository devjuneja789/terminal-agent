#!/usr/bin/env bash
set -euo pipefail
python -m pip install --quiet uv
cp -a /workspace /tmp/terminal-slm-corrective-preflight
cd /tmp/terminal-slm-corrective-preflight
uv run --python 3.12 \
  --with 'torch==2.11.0' --with 'transformers==5.22.1' --with 'accelerate==1.15.0' \
  --with 'trl==1.7.1' --with 'peft==0.21.3' --with 'bitsandbytes==0.50.2' --with 'datasets==5.2.0' \
  --with 'agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@d4db4c12d52342227c4113a70183d0f64aa2f345' \
  python training/agenttune/corrective_preflight.py
