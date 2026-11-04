# AgentTune and Hugging Face GPU workflow

Phase 1 uses a Hugging Face GPU Job for Tiny Aya Global inference. The local
machine only stages the code bundle and submits the job; model weights are
downloaded inside the job. The bundle includes the workspace-scoped `tools/`
code and train-support task B002. It excludes `.venv`, model weights,
credentials, raw results, and held-out task records.

The smoke runs one real AgentTune `Project` episode using `ReActStrategy` and
`DictToolHarness`. Tiny Aya uses AgentTune's documented Cohere fallback format
and receives tool results as user turns. The existing `Toolbox` executes tools
inside a temporary copy of the miniapp snapshot. The smoke validates the final
response, a successful required `grep` call, and the selected task's validation command.
It is not training or a formal benchmark run.

## Phase 2 frozen held-out evaluation setup

The formal baseline config is `configs/tiny_aya_eval.json`. It pins Tiny Aya's
model revision, cloud package versions, decoding, prompt, schema hash, and the
ten permanent held-out task IDs. The evaluator is
`training/agenttune/evaluate_heldout.py`; it uses the existing benchmark
snapshot-copy/validation runner and workspace-scoped `Toolbox`. It requires
CUDA and refuses to start if the model revision, benchmark IDs, runtime pins,
prompt hash, or schemas differ.

The held-out job bundle is built by
`training/agenttune/prepare_hf_eval_bundle.py`. It carries only the held-out
task file, a manifest containing only held-out IDs, the required miniapp
snapshot, and runtime code. It explicitly excludes `train_support.jsonl`.
Keep this separation for the baseline; future rollout/training preparation
must use a separate support-only bundle.

To reproduce the Phase 3 baseline, check the live price and launch from the
repository root:

```sh
hf auth whoami
hf jobs hardware
./scripts/run_tiny_aya_baseline.sh
```

The job has a 40-minute hard timeout. The launcher preserves the complete HF
Job output and extracts `research/results/tiny-aya-baseline/<run-id>.jsonl`,
`<run-id>.metadata.json`, and `<run-id>.summary.json`. Phase 3 is complete; the
first run and separate trace-recovery run are recorded in EXP-004. Both scored
0/10; the recovery run preserves all ten trajectories. Do not merge the two
runs' records or metrics. Phase 4 is support-only rollout/reward validation;
do not use held-out tasks or traces as training examples or reward targets.

## Run the Phase 1 smoke

From the repository root in WSL:

```sh
source .venv/bin/activate
python -m pip install --upgrade huggingface_hub
hf auth login
hf auth whoami
hf jobs hardware
./scripts/run_hf_gpu_smoke.sh
```

`hf auth login` prompts for the token; do not put it in this repository or send
it in chat. The logged-in account must have access to `CohereLabs/tiny-aya-global`.
The launcher forwards the cached token to the Job as the `HF_TOKEN` secret.

The launcher selects `t4-small`, sets a 20-minute job timeout, and prints the
current hardware table first. Hugging Face currently lists T4-small at
$0.40/hour and bills by the minute, so a full 20-minute run is about $0.14 at
that listed rate. Confirm the live price shown by `hf jobs hardware`; the job
timeout caps the smoke run. No exposed endpoint is created.

The Job prints one line beginning `PHASE1_SMOKE_RESULT=` with the actual
trajectory, model output, validation result, GPU, package versions, and outcome.
Keep the Job ID and that record. If the smoke fails, report the full failure
record and Job logs; do not interpret a failed run as a model pass.

The Docker command uses `--` before `bash /workspace/training/agenttune/run_smoke_job.sh`
to pass the command through to the container, as required by the Jobs CLI when
separating options from the container command.

## Current execution status

The GPU Jobs confirm Tiny Aya loads and generates. B006 exposed weak argument
recovery: the model repeatedly used invalid `read_file` line numbers. On B002,
it found the path and emitted a correct plain-text answer after proposing
another read; the old adapter discarded that answer and kept reading. The
runner preserves raw completions, feeds back only the selected action, and now
converts a plain-text tail after an observed tool step into AgentTune's
`finish` action. The B002 GPU smoke passed with this adaptation. See
`research/experiments/EXP-002-tiny-aya-phase1-smoke.md` and raw records under
`research/results/`. Hardware and package versions were not present in the
successful result; collect them for the formal baseline. No local GPU inference
or training was run.
