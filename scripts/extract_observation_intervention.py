"""Recover all twelve matched live-history predictions from bounded HF chunks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def extract(log: Path, destination: Path) -> dict:
    chunks={};metadata=summary=None
    for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
        for suffix in ('CHUNK','ENVIRONMENT','SUMMARY'):
            marker='OBSERVATION_INTERVENTION_'+suffix+'='
            if marker not in line:continue
            value=json.loads(line.split(marker,1)[1])
            if suffix=='ENVIRONMENT':metadata=value
            elif suffix=='SUMMARY':summary=value
            else:
                part,total=value['part'],value['parts']
                if not 1<=part<=total:raise ValueError('invalid chunk numbering')
                entry=chunks.setdefault(value['record_id'],{'parts':total,'data':{}})
                if entry['parts']!=total or (part in entry['data'] and entry['data'][part]!=value['data']):
                    raise ValueError('conflicting chunks')
                entry['data'][part]=value['data']
            break
    if metadata is None or summary is None:raise ValueError('incomplete metadata/summary')
    expected=metadata['record_ids']
    if len(expected)!=12 or len(set(expected))!=12 or set(chunks)!=set(expected) or summary['record_ids']!=expected:
        raise ValueError('expected all twelve prediction records')
    records=[]
    for identity in expected:
        entry=chunks[identity]
        if set(entry['data'])!=set(range(1,entry['parts']+1)):raise ValueError('incomplete prediction chunks')
        record=json.loads(base64.b64decode(''.join(entry['data'][i] for i in range(1,entry['parts']+1)),validate=True))
        if record['record_id']!=identity or record['source_split']!='train-support':raise ValueError('record identity/split mismatch')
        records.append(record)
    destination.mkdir(parents=True,exist_ok=True)
    prefix=destination/summary['run_id']
    prefix.with_suffix('.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    prefix.with_suffix('.provenance.json').write_text(json.dumps({'source_log':str(log),'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'all_12_records_recovered':True},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(extract(args.log,args.output_dir),indent=2))
