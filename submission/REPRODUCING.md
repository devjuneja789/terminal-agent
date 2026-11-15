# Reproduce Terminal-SLM

The active hackathon model is **CohereLabs/tiny-aya-global**, pinned to
`af89d219b53ed9b13b8a4645f9c8028973510324`. The standalone local CLI is not used to produce the reported cloud runs. All real model inference and post-training use
Hugging Face GPU Jobs. No GPU is required to inspect results, run the core
unit tests, or rebuild the recorded walkthrough.

## Local checks and recorded auditing

```sh
python3 -m unittest discover -v
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements-audit.txt
# Use the actual final JSONL path recorded in submission/RESULTS.md:
.venv/bin/python -m evals.auditkit_report PATH_TO_FINAL_JSONL --output-prefix submission/auditkit/final
python3 scripts/build_submission_demo.py PATH_TO_FINAL_JSONL --output submission/demo.html
```

AuditKIT is pinned at `9d4e21bac240e8108a22c2c137d179d6070868d5` (installed
core version 1.1.12). The core imports existing AgentTune EventLogs and evaluates
them with its recorded AgentEvalRunner. The oracle independently recomputes
the frozen contract from recorded observations and executable validator output;
it does not accept the model's success claim as ground truth. No local model,
terminal commands, remote judge or new validation execution is invoked here.
Its structural scores cover emitted actions, including `finish`; harness attempt
metrics additionally cover rejected model outputs and blocked repeats. These
metrics have different denominators and must remain separately labelled.

## Cloud reproduction

HF CLI authentication, approved access to Tiny Aya Global, GPU credits and
network access are necessary. Credentials are supplied by the job secret
`HF_TOKEN`; never add them to a config, command log or public dataset. Install
the CLI with `python -m pip install huggingface_hub` in a suitable environment
and run `hf auth login`. The provided launchers expect `.venv/bin/hf`.

The fixed production recipe is NF4/double quantization with FP16 compute,
eager attention and seed 42. Trainable LoRA parameters use FP32 with FP16 AMP
initial scale 256. See the immutable experiment configs for full settings.
Cloud jobs pin Python 3.12, torch 2.11.0, transformers 5.22.1, AgentTune commit
`d4db4c12d52342227c4113a70183d0f64aa2f345`, TRL 1.7.1, PEFT 0.21.3,
bitsandbytes 0.50.2 and accelerate 1.15.0.

```sh
# Expensive: reproduces the preregistered final fit on exactly 15 approved targets.
bash scripts/run_tiny_aya_fit300.sh
# Use the freshly extracted result path; do not substitute a different recipe.
TERMINAL_SLM_CONTEXT_TRAINING_RESULT=PATH_TO_TRAINING_RESULT_JSON bash scripts/run_tiny_aya_fit300_contexts.sh
# The committed submission candidate is immutable. Inspect its evidence first.
# Expensive: reproduces the frozen matched comparison, with no training/rewards.
bash scripts/run_tiny_aya_final_heldout.sh
```

Each launcher stages only the required code/data, records its actual job ID,
and retains complete chunked JSONL trajectories. HF Jobs are ephemeral: adapters
are uploaded to a versioned HF artifact repository before the training job exits.
The inference runner reloads the exact artifact commit/subfolder in the freeze.
Missing trajectory chunks are extraction errors; never substitute partial runs
or rerun a model merely to recover a log that HF can still provide.

## Data and evaluation boundaries

The benchmark has 50 deterministic repository tasks: 40 train-support and ten
held-out. Permanent held-out IDs are B005, B010, B015, B020, B025, B030, B035,
B040, B045 and B050. No held-out task supplies training targets or support
rewards. The final candidate is selected from support evidence before final
held-out evaluation, then training stops. Earlier historical held-out results
exist; this is a fixed split, not a claim that nobody has ever seen its outcomes.

Only four reviewed corrective demonstrations were separately admitted, giving
15 continuation targets. Failed prefixes are context, never training labels.
The two approved truthful-failure reports are audit-only and permanently excluded.
The target hash is
`03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`.

The final evaluation reruns base and adapter under identical quantization,
prompts, schemas, limits and stop IDs `[3,6,261001]`. It does not reuse the older
FP16 baseline as its control. Correct stop tokens reduce wasted generation;
that is a runtime correction, not an SFT task-success gain. The small benchmark,
keyword-based answer checks and narrow validators limit the conclusions.
Tasks share one small synthetic miniapp, so the permanent task split does not
provide repository-level separation.

## Execution boundary

Task workspaces are disposable, workspace-relative file APIs reject escapes,
and shell commands have logging, time/output limits and filtered environment.
The shell guardrails are not an operating-system security sandbox. For arbitrary
untrusted repositories, use an isolated container/VM with restricted credentials
and network. No public live terminal execution service is included.
