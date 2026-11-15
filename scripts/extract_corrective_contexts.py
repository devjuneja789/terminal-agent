"""Recover exactly 15 context audits and 30 predictions from bounded chunks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def extract(log: Path,destination: Path) -> dict:
    chunks={};metadata=summary=None
    for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
        for marker in ('CORRECTIVE_CONTEXT_CHUNK=','CORRECTIVE_CONTEXT_ENVIRONMENT=','CORRECTIVE_CONTEXT_SUMMARY='):
            if marker not in line:continue
            value=json.loads(line.split(marker,1)[1])
            if marker=='CORRECTIVE_CONTEXT_ENVIRONMENT=':metadata=value
            elif marker=='CORRECTIVE_CONTEXT_SUMMARY=':summary=value
            else:
                key=(value['kind'],value['record_id']);part,total=value['part'],value['parts']
                if not 1<=part<=total:raise ValueError('invalid chunk number')
                entry=chunks.setdefault(key,{'parts':total,'data':{}})
                if entry['parts']!=total or part in entry['data'] and entry['data'][part]!=value['data']:
                    raise ValueError('conflicting chunks')
                entry['data'][part]=value['data']
            break
    if metadata is None or summary is None:raise ValueError('missing context metadata/summary')
    context_ids=[f'context:{i:02d}' for i in range(15)]
    prediction_ids=[f'{arm}:{i:02d}' for arm in ('base','adapter') for i in range(15)]
    expected={('context',key) for key in context_ids}|{('prediction',key) for key in prediction_ids}
    if (metadata['context_ids']!=context_ids or metadata['prediction_ids']!=prediction_ids
        or summary['context_count']!=15 or summary['prediction_count']!=30 or set(chunks)!=expected
        or metadata['run_id']!=summary['run_id']):raise ValueError('expected exactly 45 complete diagnostic records')
    values={'context':[],'prediction':[]}
    for key in sorted(expected):
        entry=chunks[key]
        if set(entry['data'])!=set(range(1,entry['parts']+1)):raise ValueError('incomplete context record')
        value=json.loads(base64.b64decode(''.join(entry['data'][i] for i in range(1,entry['parts']+1)),validate=True))
        if value['record_id']!=key[1] or value['kind']!=key[0] or value['source_split']!='train-support':
            raise ValueError('identity or split mismatch')
        values[key[0]].append(value)
    destination.mkdir(parents=True,exist_ok=True);prefix=destination/summary['run_id']
    for kind,rows in values.items():
        prefix.with_suffix(f'.{kind}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    prefix.with_suffix('.provenance.json').write_text(json.dumps({'source_log':str(log),
        'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'all_45_records_recovered':True},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(extract(args.log,args.output_dir),indent=2))
