# Submission-day plan — 2026-11-15

The user approved this plan. Execution status and actual results are recorded
in EXP-024 onward; the time windows below remain planning targets.

## Verified deadline and minimum deliverables

The current [official page](https://hackathon.lexsi.ai/#submit) says Nov 15,
11:59 PM AoE, equivalent to Nov 16, 11:59 UTC. Earlier repository references
to Nov 14 are historical. Keep the user's local midnight as our internal cutoff.

The page requests public model/adapter and dataset/log collections on HF,
public code, a reproducible write-up and before/after metrics. A GitHub README
is accepted as the article fallback, and a public repository can satisfy public
deployment. A live GPU Space is not necessary for the minimum package. Track 3
has an exception to the two-repository minimum; AuditKIT remains a desirable
evaluation integration, not a reason to miss the submission.

## Current evidence

Training, adapter reload, isolated tools, data provenance and diagnostic audits
work. EXP-025 remains the latest matched held-out result: 1/10 base and 2/10
adapter contract passes, with no repair success for either model and an
inaccurate explanation in the adapter's extra pass. EXP-027 is a later
support-only follow-up: its task-start adapter passes 2/4 fixed support tasks
versus 0/4 for the base, with recorded edits independently checked against the
reviewed full-file references. This does not establish held-out generalization
or replace the EXP-025 held-out comparison. Report both with these distinctions.

## Twelve-hour working schedule

Times below are user-local planning targets, not a conversion of the AoE deadline.

| Window | Work and exit condition |
|---|---|
| 12:00–15:00 | One final support-only fit-budget diagnostic. Proposed 300 updates/20 epochs on the same 15 admitted targets, same joint objective, fresh base, seed and numeric settings; 40-minute GPU cap. No sweep, new model, new data or prompt changes. Evaluate exact contexts. Prepare release documentation while GPU work runs. Freeze the candidate by 15:00. |
| 15:00–18:00 | Final paired base/adapter held-out evaluation under one explicitly versioned native-stop/NF4 configuration, preserving all held-out tasks/validators. Integrate genuine AuditKIT reporting with a strict implementation timebox. Keep the original baseline as historical. |
| 18:00–20:00 | Publish the prepared release code and permitted adapter/data/log artifacts with reproduction instructions, provenance and two public HF collections. Create a simple real-trajectory walkthrough; live inference hosting is optional. |
| 20:00–22:00 | Finish the abstract/write-up: actual method, before/after metrics, failure analysis, limitations, compute and links. Verify every claim against saved evidence. |
| 22:00–23:00 | Open every public link while logged out, validate the collections/download instructions, check the final submission fields and submit. |
| 23:00–00:00 | Buffer for broken links, upload delays or submission corrections. No new training. |

The 300-update proposal is an optimization-adequacy test, not a promised fix.
Prior 60-update runs did not fit the critical decisions. Record its protocol
before launching; do not adapt its budget or objective after seeing outcomes.
Freeze selection using support results only. Before launch, declare a
lexicographic comparison against the completed joint-60 checkpoint: exact
full-action matches, valid finish choices, total schema-valid outputs, then
valid tool choices on the same 15 contexts. Keep joint-60 on an exact tie or
if the new run fails. This is a support-only selection rule, not a task-success
claim. Do not select models
or change training after inspecting final held-out outcomes.

## Non-negotiable reporting boundaries

- Distinguish reviewed-context action accuracy from executable task success.
- Attribute the EOS efficiency correction to runtime configuration, not learning.
- Report missing end-to-end improvement plainly; public rules emphasize meaningful
  improvement, so competitiveness of a negative-result entry is uncertain.
- Never label recorded trajectories as live inference or claim AuditKIT usage
  unless it actually evaluates the records.
- No DPO/GRPO, new architecture, model switch or benchmark expansion today.
- Prepare public artifacts and the final statement for review before release;
  user handles the registration identity/final submission fields as needed.

Main remaining risk: spending the entire day on optimization and leaving no
time for an accessible submission. Research stops at 15:00 even if results fail.


## Updated status (2026-11-15)

The original time-window schedule above is historical and superseded by EXP-027.
The bounded task-start training and support replay completed; no more training
or inference is recommended before submission. Remaining work is release and
submission QA: verify actual public links/accessibility, consistency of result
claims with EXP-025/027, and complete the hackathon submission fields. Do not
claim that the support-only improvement is held-out performance.
