# Terminal-SLM: learning terminal actions versus completing terminal tasks

**Track:** 3 — Agentic Fine-tuning. **Problem statement:** Other — reviewed
corrective SFT for executable terminal-tool use. Related to reliable tool
calling and efficient trajectories; this project implements SFT, not RL.

**One-line claim:** Tiny Aya learns reviewed action choices, but the observed
conditional gain does not establish reliable end-to-end repository repair.

Terminal-SLM studies whether a small Cohere model can become a more reliable
terminal agent through AgentTune post-training. We implement five bounded,
workspace-scoped terminal tools, complete trajectory capture and a deterministic
50-task repository benchmark with a permanent 40/10 support/evaluation split.
All model inference and training run on Hugging Face T4 GPUs.

After unsuccessful support rollouts, four owner-reviewed corrective
demonstrations are separately admitted as 15 continuation-only targets. Failed
prefixes remain context; truthful-failure examples remain excluded from
training. A fixed joint QLoRA objective combines tool-choice emphasis with
weighted complete-action supervision. Label-boundary, actual gradient,
optimizer-update and adapter-reload checks make the training process auditable.

A preregistered 300-update fit improves correct tool choices on these known
contexts from 1/15 for the untouched base to 15/15, including four finish
choices and two exact complete-file edits. Exact full actions improve 1/15
to 6/15. Candidate selection uses support evidence only, then freezes the
checkpoint before a matched held-out comparison.

Frozen held-out contract scores rise 10% to 20%, but trace inspection exposes
an inaccurate explanation in the additional pass. Both models complete zero
of five repair/update tasks. We therefore do not claim reliable task-success
improvement. The unchanged benchmark and this rubric counterexample remain
public evidence rather than being silently relabelled.

AuditKIT genuinely evaluates all twenty recorded AgentTune episodes using an
independent executable-evidence contract oracle and structural scorers. The
release contains configs, reviewed-data provenance, adapters, raw trajectories,
failure analysis and a recorded walkthrough. The result distinguishes learning
on supplied contexts from transfer to complete agent conversations, providing
a reproducible foundation for broader verified support data and stronger
future evaluation.

Public collection and source links must be filled from the reviewed release
manifest after publication; none are claimed published by this abstract.
