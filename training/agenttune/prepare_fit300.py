"""Stage the final fixed 300-update fit-budget experiment without new data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.agenttune.prepare_joint_training import prepare as joint_prepare
from training.agenttune.prepare_sft import ROOT,digest

PROFILE=ROOT/'configs/tiny_aya_joint_fit300.json'


def prepare(destination: Path,artifact_repo: str,config_path: Path | None=None) -> dict:
    config=json.loads(PROFILE.read_text())
    if config_path and json.loads(config_path.read_text())!=config:raise ValueError('final fit-budget profile is fixed')
    manifest=joint_prepare(destination,artifact_repo);config['artifact_repo']=artifact_repo
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest.update(config_sha256=digest(config),max_steps=300,expected_epochs=20)
    (destination/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--artifact-repo',required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.artifact_repo,args.config)))
