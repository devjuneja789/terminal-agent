"""Build a standalone recorded walkthrough from complete real held-out results."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def build(records_path: Path, output: Path) -> None:
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    expected = {(arm, f'B{i:03d}') for arm in ('base', 'adapter') for i in range(5, 51, 5)}
    actual = {(r['arm'], r['task_id']) for r in records}
    if (len(records)!=20 or actual!=expected or any(r['split']!='held-out-eval'
        or r['run_kind']!='final_paired_heldout' or r.get('training_use_prohibited') is not True for r in records)):
        raise ValueError('walkthrough requires all 20 real final held-out records')
    def pre(value):
        text = value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False)
        return '<pre>'+html.escape(text)+'</pre>'
    summaries=[];cards=[]
    for arm in ('base','adapter'):
        values=[r for r in records if r['arm']==arm]
        summaries.append(f'<li>{arm}: {sum(r["success"] for r in values)}/10 frozen task contracts passed</li>')
    for record in records:
        arm=record['arm'];task=record['task_id'];trajectory=record['trajectory']
        title=f'{arm} · {task} · '+('PASS' if record['success'] else 'FAIL')
        cards.append('<details data-arm="'+arm+'"><summary>'+html.escape(title)+'</summary>'
            +'<h3>User task</h3>'+pre(trajectory['user_request'])
            +'<h3>Final response</h3>'+pre({'source':record.get('final_response_source'),
                'text':record['final_response'],'stop':record['agent_stop_reason'],
                'contract_failures':record['criterion_failures']})
            +'<h3>Model requests, raw responses and executed actions</h3>'+pre(record['agenttune_trace'])
            +'<h3>Executable validation</h3>'+pre(record['validation'])
            +'<h3>Native trajectory</h3>'+pre(trajectory)+'</details>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Terminal-SLM — recorded evaluation</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;background:#10151d;color:#e1e8ef}
a{color:#80c7ff}button{padding:8px 16px;margin:0 8px 20px 0;cursor:pointer}details{border:1px solid #435168;border-radius:8px;padding:16px;margin:12px 0}
summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1a2433;padding:14px;border-radius:6px;font-size:13px}h3{font-size:16px}
</style><h1>Terminal-SLM</h1><p>A small terminal agent, executable evaluation and reviewed support-only post-training.</p>
<p><strong>Recorded evaluation, not live inference.</strong> These are actual saved GPU runs. Passing a narrow task contract does not establish general coding reliability. Failures and harness-generated final text remain visible and attributed.</p>
<p><strong>Factual review:</strong> B050's additional adapter contract pass contains explanation errors. Neither model completes a code repair. <a href="TRACE-REVIEW.md">Read the trace review</a> before interpreting the nominal scores.</p>
<ul>'''+''.join(summaries)+'''</ul><p><a href="https://github.com/devjuneja789/terminal-agent">Code, method and reproduction instructions</a></p>
<nav aria-label="Filter episodes"><button onclick="filter('all')">Both models</button><button onclick="filter('base')">Base</button><button onclick="filter('adapter')">Adapter</button></nav>
'''+''.join(cards)+'''<script>function filter(arm){document.querySelectorAll('details[data-arm]').forEach(e=>e.hidden=arm!=='all'&&e.dataset.arm!==arm)}</script></html>'''
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records',type=Path);parser.add_argument('--output',type=Path,default=Path('submission/demo.html'))
    args=parser.parse_args();build(args.records,args.output)
