# Corrective demonstration admission

The owner approved all six review decisions. A separate validated admission
step has now admitted **only the first four**, exporting **15 unique action
targets** to `data/sft/tiny_aya_corrective_v1.jsonl`.

The two truthful failure reports remain permanently excluded and have zero
training targets. Approval of their factual accuracy never means successful
task completion or admission.

| Artifact | Meaning |
|---|---|
| `support_corrective.v1.jsonl` | Immutable original candidate records, including their creation-time pending status |
| `support_corrective.v1.owner-review-receipt.json` | Immutable owner decisions as originally recorded, before application/admission |
| `support_corrective.v1.review.pending.jsonl` | Preserved pre-application sidecar, hash-matched to the original receipt |
| `support_corrective.v1.review.jsonl` | Applied six approvals, retaining review scopes and owner reasons |
| `support_corrective.v1.admitted.jsonl` | Four derived admitted records with review/admission provenance |
| `support_corrective.v1.admission.json` | Current admission result, export hashes and permanent exclusions |
| `support_corrective.v1.revalidation.jsonl` | Four fresh execution checks in disposable snapshots |
| `support_corrective.v1.manifest.json` | Historical creation-time batch manifest; not current admission status |

Use the dedicated corrective exporter and its explicit continuation boundaries.
The generic `prepare_sft.action_rows()` walks every assistant message and must
not export these transcripts: archived assistant actions are context only.
The admitted SFT rows each contain one corrected completion; all prompt turns
must remain masked in completion-only training.

The new export is separate from the old approved gold data. No merge or extra
weighting occurred, no exact duplicates were found, and neither audit-only
record appears in the export. All four tasks are train-support tasks.

The later EXP-018 pinned token-level audit now passes all 15 rows, including
masking, native closing tokens, weights and sequence lengths. Its separate gate
is `data/sft/tiny_aya_corrective_v1.preflight.json`; historical admission manifests
remain unchanged. The old launchers retain their historical gold-data behavior; EXP-020 adds a
separate corrective training launcher.
EXP-019 now passes the versioned native-stop support replay gate; see
`configs/tiny_aya_native_stopping_v1.json` and its hashed runtime gate.
EXP-020 has now trained the separate 15-target export for 60 updates/four
epochs. Actual trainer labels/weights and adapter reload pass, but the full
support comparison remains 0/4 with no edits. No replay outputs are admitted;
source/review/admission/export files remain unchanged. Admission itself
performed no inference/training; EXP-018 performed only inference and label
inspection, and EXP-019 performed support inference only.
