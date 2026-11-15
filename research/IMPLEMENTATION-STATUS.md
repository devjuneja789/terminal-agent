# Terminal-SLM hackathon work estimate

Estimate updated 2026-11-14; based on the current repository's Track 3 plan.

Approximately **40–50% of the total planned submission work remains**,
despite most infrastructure being built.
Interpret this as roughly **50–60% complete**, not a measured progress metric
or probability of a successful submission/model result.

Available: workspace tools, CLI/harness, frozen benchmark/splits, cloud model
access, real baseline/trajectories, AgentTune SFT and adapter reload, pinned
inference/training jobs, extensive tests and diagnostics. Current limitation:
the last full agent replay is still 0/4. Conditional edit gains do not replace
a reliable end-to-end tuned agent.

| Remaining work | Rough focused effort |
|---|---:|
| Pinned label/stopping preflight and corrective-dataset wiring | 1–3 hours |
| One or two bounded training/support-validation iterations | 4–8 hours |
| AuditKIT integration and final paired held-out evaluation/report | 4–8 hours |
| Public HF artifacts/demo, documentation and submission packaging | 4–7 hours |

With debugging/review buffer, budget **16–30 focused hours**, roughly **2–4
working days** at 6–8 hours/day. This includes hands-on engineering and review,
not just GPU job duration. It assumes the next bounded learning experiment
produces useful behavior; persistent failures can extend the schedule.

The largest uncertainty is model behavior, followed by the remaining AuditKIT
and public-demo work. More formatting experiments are not the planned next
step. Prioritize owner-reviewed corrective support data, a bounded learning
gate, and a defensible before/after evaluation. DPO, GRPO, GGUF and a richer UI
remain deferred under the hackathon plan.

Owner review, admission and the pinned token/label preflight are complete. Four
task-completing demonstrations supply 15 audited continuation-only targets;
two truthful failure reports remain audit-only. No further owner action is
currently needed. EXP-018/019 confirm and correct the stopping-efficiency issue; next is
a research decision on the remaining finish/read and argument-learning blockers.
EXP-019 has now validated the separately versioned stop union on all eight
support episodes (38/38 generations stop correctly). Both models remain 0/4;
EXP-020 subsequently trained the 15 admitted targets for 60 updates/four
epochs, passing all pipeline gates but still scoring 0/4 with zero edits.
No held-out evaluation ran. The learning/coverage limitation remains unresolved.

EXP-021 now supplies the exact 15 reviewed contexts: adapter exact actions
3/15, write choices 0/3 and finish choices 0/4. All lineage/mask/stop checks
pass. The current limitation includes conditional action learning, beyond
runtime coverage; no further training has run.

EXP-022 completed a bounded tool-name-only objective isolation with a recorded
AMP startup failure/correction. Valid choices rise to 8/15 and writes to 2/3,
but finish stays 0/4 and exact actions 3/15. The fixed criterion fails; no agent
replay follows. Finishing, complete reads and valid edit arguments remain blockers.

EXP-023 completed joint first-decision/full-completion training and exact-context
comparison. All pipeline checks pass, but the 15 outputs remain identical to
EXP-022 (choices 8/15, finish 0/4). No full-agent/held-out run or task-success
gain. Optimization adequacy remained unresolved at the end of EXP-023.

EXP-024 completed the final 300-update fit and exact-context comparison. All
checks pass; correct choices 15/15, finish 4/4, exact actions 6/15. Joint-300 wins
the declared support-only rule and is frozen at 07:14:40 UTC. No further training.
EXP-025 completed as Job `6af95c1129dd05ff0368583f`, run `20261115T071710Z`.
Matched contract passes are 1/10 base and 2/10 adapter. The extra B050 pass
contains an inaccurate explanation, and both models remain 0/5 repairs with
no file changes. No reliable end-to-end improvement is established. All twenty
raw records and independent contract/aggregate checks pass; actual AuditKIT
reports evaluate twenty episodes with zero method errors or unknown outcomes.
The recorded walkthrough, report and explicit release payloads are prepared
for review. No further research/training is planned for this submission.


EXP-027 completed 2026-11-15. The task-start curriculum trained for 620 updates
and passed adapter reload. On the fixed four-task support replay, base scored
0/4 and the adapter scored 2/4 (B021, B041). Auditing the recorded full-file
write contents confirmed exact reviewed edits, preserved unchanged AST nodes,
and passing named plus regression tests for both successes. B026 used the
wrong test path; B046 emitted an invalid finish action. This is support-only
evidence and does not establish held-out generalization. See
`research/experiments/EXP-027-task-start-curriculum-plan.md`. The older estimate
and statement that no further training was planned predate this follow-up.
