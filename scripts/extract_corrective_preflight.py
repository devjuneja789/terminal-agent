"""Recover all fifteen label audits and twelve stopping comparisons."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def extract(log: Path,destination: Path) -> dict:
    chunks={};meta=summary=None
    for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
        for suffix in ('CHUNK','ENVIRONMENT','SUMMARY'):
            marker='CORRECTIVE_PREFLIGHT_'+suffix+'='
            if marker not in line:continue
            value=json.loads(line.split(marker,1)[1])
            if suffix=='ENVIRONMENT':meta=value
            elif suffix=='SUMMARY':summary=value
            else:
                part,total=value['part'],value['parts']
                if not 1<=part<=total:raise ValueError('invalid chunk numbering')
                entry=chunks.setdefault(value['record_id'],{'parts':total,'data':{}})
                if entry['parts']!=total or (part in entry['data'] and entry['data'][part]!=value['data']):raise ValueError('conflicting chunks')
                entry['data'][part]=value['data']
            break
    if meta is None or summary is None:raise ValueError('missing preflight metadata/summary')
    expected=meta['record_ids']
    if len(expected)!=27 or len(set(expected))!=27 or set(chunks)!=set(expected) or summary['record_ids']!=expected:
        raise ValueError('expected all 27 preflight records')
    groups={'label':[],'generation':[]}
    for identity in expected:
        entry=chunks[identity]
        if set(entry['data'])!=set(range(1,entry['parts']+1)):raise ValueError('incomplete preflight record')
        record=json.loads(base64.b64decode(''.join(entry['data'][i] for i in range(1,entry['parts']+1)),validate=True))
        if record['record_id']!=identity or record['source_split']!='train-support' or record['kind'] not in groups:
            raise ValueError('unexpected record identity/split')
        groups[record['kind']].append(record)
    if len(groups['label'])!=15 or len(groups['generation'])!=12:raise ValueError('wrong record counts')
    destination.mkdir(parents=True,exist_ok=True);prefix=destination/summary['run_id']
    for kind,records in groups.items():prefix.with_suffix('.'+kind+'.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    prefix.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    prefix.with_suffix('.provenance.json').write_text(json.dumps({'source_log':str(log),
        'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'all_27_records_recovered':True},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(extract(args.log,args.output_dir),indent=2))
