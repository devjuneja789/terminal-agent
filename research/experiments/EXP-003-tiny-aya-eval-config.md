# EXP-003 — Tiny Aya held-out evaluation configuration

**Status:** Phase 2 setup complete; see [EXP-004](EXP-004-tiny-aya-baseline.md) for the baseline run.
**Purpose:** Freeze the untouched Tiny Aya baseline procedure before using the
held-out evaluation outputs.
**Run kind:** Configuration and runner preparation only. No inference,
training, or benchmark result was produced in this phase.

## Hypothesis

A fixed cloud runtime and the existing isolated benchmark runner can evaluate
Tiny Aya Global with AgentTune's Cohere fallback tool protocol while preserving
the Terminal-SLM workspace boundary and the permanent held-out split.

## Frozen configuration

The canonical configuration is [`configs/tiny_aya_eval.json`](../../configs/tiny_aya_eval.json).
It pins model revision `af89d219b53ed9b13b8a4645f9c8028973510324`, FP16,
Python 3.12, PyTorch 2.11.0, Transformers 5.22.1, Accelerate 1.15.0, and
AgentTune 1.1.3. It also fixes greedy decoding, an 8-step tool limit, the
per-task time and output limits, prompt, schemas, and metric definitions.
The model is loaded only inside a CUDA Hugging Face Job. The first completed
baseline metrics are in EXP-004; full raw trajectory recovery is pending due
to console log truncation.

The held-out IDs are frozen as:

`B005, B010, B015, B020, B025, B030, B035, B040, B045, B050`.

`training/agenttune/prepare_hf_eval_bundle.py` creates a job input containing
only `heldout_eval.jsonl`, the held-out-only manifest, required evaluator/tool
code, config, and miniapp snapshot. It explicitly omits `train_support.jsonl`.
The existing benchmark loader verifies snapshot hashes and split membership.
This prevents this baseline runner from reading support examples; training
rollout generation must use a separate support-only bundle.

## Evaluation procedure

1. Confirm Hugging Face CLI authentication, model access, and the live GPU Job
   price. No credentials are written to the repository.
2. Stage a held-out-only input bundle.
3. Start one `t4-small` Hugging Face Job with a 40-minute hard timeout.
4. Verify exact runtime package pins, frozen prompt/schema hashes, CUDA
   availability, benchmark ID, ordered task IDs, and snapshot hashes before
   evaluation.
5. Load Tiny Aya once and run each task in a fresh disposable copy through the
   existing benchmark `run_task` validation path.
6. Save every task's full AgentTune events, raw model outputs/prompts, tool
   actions/results, Terminal-SLM trajectory, validator output, token counts,
   latency, hardware, and package metadata.
7. Extract machine-readable JSONL and metadata from the attached job log.

Metrics include executable task success, micro-averaged tool selection,
argument/schema validity, invalid action rate (including malformed outputs),
tool/model steps, tokens, latency, and failure categories. A task only passes
when its declared response/tool/validation conditions and final-answer gate
pass.

## Artifacts and result state

- Frozen config: `configs/tiny_aya_eval.json`
- GPU evaluator: `training/agenttune/evaluate_heldout.py`
- Held-out bundle builder: `training/agenttune/prepare_hf_eval_bundle.py`
- Cloud job script: `training/agenttune/run_baseline_job.sh`
- Launcher: `scripts/run_tiny_aya_baseline.sh`
- Log extractor: `scripts/extract_hf_baseline.py`
- Results: EXP-004 records the baseline metrics and artifact limitation.

## Launch attempt note (2026-11-04)

The first submitted HF Job (`6aeba75ec78863f9c632fbbd`) failed before runtime
setup because the bundle manifest omitted `run_baseline_job.sh`, while the
container command referenced it. No model was loaded and no task was evaluated.
The builder now includes that script, and the local launcher checks for the
script and absence of support data before upload. A rebuilt bundle was verified
to contain all ten held-out tasks, the executable job script, and no support
task file. The baseline must be relaunched; the failed job is not an experiment
result.

A second launch (`6aeba7dec78863f9c632fc36`) successfully installed the pinned
runtime, loaded the model on a Tesla T4, and entered task evaluation. The
evaluator then crashed while classifying the first `BenchmarkRun`: it referenced
an `agent_completed` attribute that the runner does not expose. The runner
records `agent_stop_reason` instead; the evaluator now uses that field. The job
did execute model inference, but crashed before printing/saving its first task
record, so this attempt has no usable baseline trajectory or task outcome and
must not be counted as a result. Review actual account usage before relaunching.

A third launch (`6aeba8df0e0338d1bb803dd9`) loaded the model and reached result
serialization for the first task. It exposed that `BenchmarkRun.trajectory` is
already a dictionary, not an `AgentTrajectory` object; the evaluator now stores
the dictionary directly. As with the preceding attempt, it failed before
emitting a task record. The runtime and GPU path work; evaluator/runner type
assumptions are being corrected before another full launch.

## Conclusion and next experiment

The Phase 2 harness and split isolation are prepared. The first baseline
completed but showed 0/10 task success; two long records were truncated by the
HF console. The chunked record output is now ready for a separate same-config
artifact-recovery run. See EXP-004 for results and the current decision gate.
