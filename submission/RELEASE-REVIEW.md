# Release review

This directory is the submission package under preparation. No file here
asserts that an artifact or submission has already been made public.

The public source URL is expected to be
https://github.com/devjuneja789/terminal-agent . Local uncommitted work must be
included before a public link is considered reproducible. The recorded demo is
standalone `demo.html`, not a live terminal endpoint.

## Concrete publication scope

1. Publish the final code, experiment records, configs, immutable review
   provenance, real results, write-up and walkthrough. Exclude virtualenvs,
   local model weights, credentials/config caches, scratch `my-project`,
   `.terminal-slm`, `.agents`, `.codex`, `.aws`, bytecode and verbose job logs.
2. Publish model cards for the frozen EXP-025 adapter and the EXP-027
   support-only adapter. Do not redistribute the gated base weights. Retain
   Tiny Aya attribution, CC-BY-NC-4.0 terms and acceptable-use requirements.
   Pin the artifact commits/subfolders even after adding model cards.
3. Publish the focused `support-data` payload as
   `devjuneja789/terminal-slm-reviewed-support`. It contains the admitted
   15-target training data and the approved 31-target EXP-027 curriculum with
   provenance. Rejected and pending review candidates are excluded.
4. Publish focused `evaluation-logs` as
   `devjuneja789/terminal-slm-evaluation-evidence`. It contains the final
   matched held-out evidence and EXP-027 evidence only. Held-out traces are
   visibly evaluation-only; EXP-027 is marked support-only and not
   held-out-evaluated.
5. Create two public HF collections: models/adapters and data/evaluation.
   Include a link to the upstream base model, never its weights. Check all
   links without authentication before submitting the form.

`scripts/prepare_submission_release.py` prepares these payloads and a file
checksum manifest after the final evidence index exists. It performs no
network calls or public writes. Review its output before publication.

## Owner decisions and final form

The owner approved the focused release scope: two adapters, approved support
data, final matched evaluation, and EXP-027 evidence; rejected/pending support
history and other diagnostic adapters are omitted. The owner selected MIT for
source code and CC-BY-NC-4.0 for project data. Base-model derivative terms and
acceptable-use policy still apply and cannot be replaced. Remaining owner
actions are to authenticate in WSL, perform the uploads, publish the article,
and copy the actual public URLs into the form. No account credentials are
needed in chat. See `HF-PUBLISH-STEPS.md` for the exact manual workflow.

The official deadline remains Nov 15, 2026, 23:59 AoE (Nov 16, 11:59 UTC);
keep the user's earlier midnight cutoff. Source:
https://hackathon.lexsi.ai/#submit . A functioning submission does not imply
that the judges will accept negative task-success results as meaningful gain.
