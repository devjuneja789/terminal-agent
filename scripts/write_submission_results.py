"""Build factual submission tables/index from independently checked final evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_final_heldout import audit


def write(prefix: Path, auditkit_prefix: Path) -> None:
    verified=audit(prefix);summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    metadata=json.loads(prefix.with_suffix('.metadata.json').read_text())
    freeze_path=ROOT/'configs/submission_candidate.json';frozen=json.loads(freeze_path.read_text())
    # Require actual SDK output, not merely a planned integration or export.
    audit_paths=[auditkit_prefix.with_name(auditkit_prefix.name+f'.{arm}.auditkit.json') for arm in ('base','adapter')]
    audit_paths += [auditkit_prefix.with_suffix('.md'),auditkit_prefix.with_suffix('.provenance.json')]
    for path in audit_paths:
        if not path.is_file():raise FileNotFoundError('real AuditKIT report missing: '+str(path))
    provenance=json.loads(auditkit_prefix.with_suffix('.provenance.json').read_text())
    if provenance['source_sha256']!=verified['source_records_sha256'] or provenance['episodes']!=20:
        raise ValueError('AuditKIT reports do not describe these final records')
    training_path=Path(frozen['selected_training_result'])
    training=json.loads(training_path.read_text())
    m=summary['metrics'];base=m['base'];adapter=m['adapter']
    def percent(value):return 'unavailable (no attempts)' if value is None else f'{value:.1%}'
    rows=[('Contract successes',f"{base['successful_tasks']}/10",f"{adapter['successful_tasks']}/10"),
        ('Task success rate',f"{base['task_success_rate']:.1%}",f"{adapter['task_success_rate']:.1%}"),
        ('Tool selection proxy',f"{base['tool_selection_accuracy']:.1%}",f"{adapter['tool_selection_accuracy']:.1%}"),
        ('Argument/schema validity',percent(base['argument_schema_validity']),percent(adapter['argument_schema_validity'])),
        ('Invalid call rate',f"{base['invalid_tool_call_rate']:.1%}",f"{adapter['invalid_tool_call_rate']:.1%}"),
        ('Average tool steps',f"{base['average_tool_steps_per_task']:.2f}",f"{adapter['average_tool_steps_per_task']:.2f}"),
        ('Average model turns',f"{base['average_model_turns_per_task']:.2f}",f"{adapter['average_model_turns_per_task']:.2f}"),
        ('Average tokens (input + output)',f"{base['average_tokens_per_task']:.1f}",f"{adapter['average_tokens_per_task']:.1f}"),
        ('Average agent latency (seconds)',f"{base['average_latency_seconds']:.2f}",f"{adapter['average_latency_seconds']:.2f}")]
    lines=['# Terminal-SLM: final matched evaluation','',f"Actual run: `{summary['run_id']}`. All 20 raw records pass the independent freeze, contract and aggregate checks.",'',
        '## Question and method','',
        'Can a small Cohere model learn reliable terminal actions from reviewed executable support demonstrations using AgentTune?',
        'The terminal harness executes five workspace-scoped tools; AgentTune supplies the native Cohere JSON-action loop and training integration. AuditKIT evaluates actual recorded episodes with an independent contract oracle and structural scorers.',
        '',f"Selected adapter: `{frozen['adapter_repo']}` at `{frozen['adapter_revision']}`, subfolder `{frozen['adapter_subfolder']}`.",
        f"Frozen before this run at `{frozen['frozen_at_utc']}` from the preregistered support-only rule. No training or model selection follows this evaluation.",'',
        f"Base: `{frozen['model_id']}` at `{frozen['model_revision']}`.",
        'Both models use identical NF4/double quantization, FP16 compute, eager attention, seed 42, greedy decoding, native stop union `[3,6,261001]`, prompts, schemas, limits and permanent held-out tasks.',
        '', '## Results','', '| Metric | Untouched base | Selected adapter |','|---|---:|---:|']
    lines[-2:-2]=['**Factual caveat:** the additional adapter contract pass, B050, contains an inaccurate explanation. Neither model completes a code repair (0/5 each). The nominal score change does not establish reliable task-success improvement. See [the trace review](TRACE-REVIEW.md).','']
    lines += [f'| {name} | {b} | {a} |' for name,b,a in rows]
    delta=adapter['task_success_rate']-base['task_success_rate']
    lines += ['',f'Task-success difference: **{delta*100:+.1f} percentage points**.',
        'Contract success is a narrow deterministic benchmark measure and still requires factual trace review. Tool-selection accuracy is the existing required-tool proxy, not an optimal-action oracle. Tokens include repeated input prompts; agent latency excludes initial model loading and the separate validator.',
        '', '## Failure categories','', '```json',json.dumps({a:m[a]['failure_categories'] for a in ('base','adapter')},indent=2),'```','',
        '## Training and provenance','',
        f"The selected training run used {training['action_target_count']} approved continuation targets from {training['trajectory_count']} demonstrations for {training['training_audit']['global_step']} updates/{training['training_audit']['epoch']:g} epochs.",
        'The objective combines 75% first tool-name decision loss and 25% weighted completion loss, emphasizing changed code and JSON closure. Trainable LoRA weights use FP32; the frozen base uses NF4 with FP16 compute. Actual label, all-batch gradient, optimizer-update and reload audits are saved.',
        f"Rows SHA-256: `{frozen['rows_sha256']}`. Failed prefixes remain context only; the two truthful-failure demonstrations remain permanently excluded.",
        'The benchmark contains 50 tasks split 40 support/10 held-out. No held-out targets or rewards were generated. Older held-out runs exist; this is a fixed split, not a claim of newly unseen tasks. Final task file bytes match the committed benchmark.',
        '', '## Compute','',
        f"Training: {training['training_metrics']['train_runtime']:.3f} seconds on `{training['gpu']}`; peak CUDA allocation {training['peak_cuda_memory_bytes']} bytes. This is the selected run, not cumulative project spend.",
        'Final evaluation hardware and runtime versions:', '```json',json.dumps(metadata['hardware'],indent=2),'```',
        'Actual credit balance and cumulative billing are not inferred from runtime. No closed-model cost/quality comparison was performed.',
        '', '## AuditKIT and limitations','',
        'AuditKIT runs locally over real imported AgentTune EventLogs, without model inference, external judges or new tool execution. The outcome oracle independently recomputes the frozen contract from recorded validator and observation evidence. Its tool-validity and loop scores cover emitted actions, including finish; rejected outputs and blocked attempts are reported by the separate harness metrics.',
        'These ten tasks and four demonstrations are too small to support a broad capability claim. Tasks share a small synthetic miniapp; the split is by task ID, not by unrelated repository. The benchmark uses keyword checks and narrow executable validators; passing may miss unrelated regressions. Correct END_RESPONSE stopping is a runtime efficiency correction, not evidence of SFT improvement. No DPO/GRPO or production safety guarantee is claimed.',
        '', '## Evidence and reproduction','',
        f"- [Complete paired records](../{prefix.relative_to(ROOT)}.jsonl)",
        f"- [Frozen metadata](../{prefix.relative_to(ROOT)}.metadata.json)",
        f"- [Independent audit](../{prefix.relative_to(ROOT)}.audit.json)",
        f"- [AuditKIT report](../{auditkit_prefix.relative_to(ROOT)}.md)",
        '- [Recorded walkthrough](demo.html)', '- [Reproduction instructions](REPRODUCING.md)',
        '- [Final protocol](../research/experiments/EXP-025-final-matched-heldout.md)','']
    output=ROOT/'submission/RESULTS.md';output.write_text('\n'.join(lines))
    evidence=[prefix.with_suffix(s) for s in ('.jsonl','.metadata.json','.summary.json','.audit.json','.md')]
    evidence+=audit_paths+[output,ROOT/'submission/ABSTRACT.md',ROOT/'submission/RELEASE-REVIEW.md',ROOT/'submission/REPRODUCING.md',ROOT/'submission/demo.html',ROOT/'submission/TRACE-REVIEW.md',
        ROOT/'research/experiments/EXP-024-final-fit-budget.md',ROOT/'research/experiments/EXP-025-final-matched-heldout.md',training_path]
    for candidate in frozen['candidates']:
        cp=Path(candidate['context_prefix'])
        evidence += [cp.with_suffix(s) for s in ('.context.jsonl','.prediction.jsonl','.metadata.json','.summary.json','.analysis.json')]
    paths=sorted({str(p.resolve().relative_to(ROOT)) for p in evidence})
    for path in paths:
        if not (ROOT/path).is_file():raise FileNotFoundError(path)
    index={'run_id':summary['run_id'],'candidate_freeze_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        'records_path':str(prefix.with_suffix('.jsonl').relative_to(ROOT)),
        'release_evidence_paths':paths,'published':False}
    (ROOT/'submission/final_artifacts.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--auditkit-prefix',type=Path,required=True);args=parser.parse_args()
    write(args.prefix.resolve(),args.auditkit_prefix.resolve())
