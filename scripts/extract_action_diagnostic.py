"""Recover all 16 label audits and 32 action predictions from HF logs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def extract(log: Path,destination: Path) -> dict:
    chunks={};metadata=summary=None
    for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
        for marker in ('ACTION_DIAGNOSTIC_CHUNK=','ACTION_DIAGNOSTIC_ENVIRONMENT=','ACTION_DIAGNOSTIC_SUMMARY='):
            if marker not in line:continue
            value=json.loads(line.split(marker,1)[1])
            if marker=='ACTION_DIAGNOSTIC_ENVIRONMENT=':metadata=value
            elif marker=='ACTION_DIAGNOSTIC_SUMMARY=':summary=value
            else:
                key=(value['kind'],value['record_id']);part=value['part'];total=value['parts']
                if not 1<=part<=total:raise ValueError('invalid chunk numbering')
                entry=chunks.setdefault(key,{'parts':total,'data':{}})
                if entry['parts']!=total or (part in entry['data'] and entry['data'][part]!=value['data']):
                    raise ValueError('conflicting diagnostic chunks')
                entry['data'][part]=value['data']
            break
    if metadata is None or summary is None:raise ValueError('missing diagnostic metadata or summary')
    expected={('boundary',key) for key in metadata['boundary_ids']}|{('prediction',key) for key in metadata['record_ids']}
    if len(expected)!=48 or len(metadata['boundary_ids'])!=16 or len(metadata['record_ids'])!=32 or set(chunks)!=expected:
        raise ValueError('expected all 48 diagnostic records')
    values={'boundary':[],'prediction':[]}
    for key in sorted(expected):
        entry=chunks[key]
        if set(entry['data'])!=set(range(1,entry['parts']+1)):raise ValueError('incomplete diagnostic record')
        row=json.loads(base64.b64decode(''.join(entry['data'][i] for i in range(1,entry['parts']+1)),validate=True))
        if row['record_id']!=key[1] or row['source_split']!='train-support':raise ValueError('record identity or split mismatch')
        values[key[0]].append(row)
    destination.mkdir(parents=True,exist_ok=True)
    prefix=destination/summary['run_id']
    for kind,rows in values.items():
        prefix.with_suffix(f'.{kind}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    prefix.with_suffix('.provenance.json').write_text(json.dumps({'source_log':str(log),
        'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'all_48_records_recovered':True},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(extract(args.log,args.output_dir),indent=2))
