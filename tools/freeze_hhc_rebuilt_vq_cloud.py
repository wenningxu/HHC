"""Freeze a val-selected rebuilt VQ as an immutable cloud-only hardlink."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.hhc_frozen_vq import file_sha,normalization_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot-snapshot',action='store_true');args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud only; never acquire a local checkpoint')
    args.run=args.run.resolve();args.output=args.output.resolve()
    assert args.run.is_relative_to(ROOT/'output') and args.output.is_relative_to(ROOT/'output')
    manifest=json.loads((args.run/'training_manifest.json').read_text())
    if not args.pilot_snapshot:assert manifest['status']=='complete','Wait for VQ completion, or explicitly request a labelled pilot snapshot'
    assert manifest['contract']['origin']=='fresh_cloud_rebuild_not_historical185k'
    args.output.mkdir(exist_ok=False)
    path=args.output/'best_fid.pth';os.link(args.run/'best_fid.pth',path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['step']>0 and (args.run/f'validation_{saved["step"]:06d}.json').exists()
    assert saved['manifest']['contract_sha256']==manifest['contract_sha256']
    validation=args.output/'selected_validation.json'
    validation.write_bytes((args.run/f'validation_{saved["step"]:06d}.json').read_bytes())
    spec={'schema':'hhc_frozen_cloud_vq_v1','origin':'fresh_cloud_rebuild_not_historical185k',
        'selection_scope':'intermediate_val_selected_pilot' if args.pilot_snapshot else 'completed_training_val_best',
        'checkpoint':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path),'step':saved['step'],
        'training_contract_sha256':manifest['contract_sha256'],
        'selected_validation':str(validation),'selected_validation_sha256':file_sha(validation),
        'normalization_sha256':normalization_sha(saved['mean'],saved['std']),
        'source_run':str(args.run),'source_run_status_at_freeze':manifest['status'],
        'quality_claim':'VQ selection only; not an HHC generation result or SOTA claim'}
    (args.output/'frozen_vq_spec.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps(spec))


if __name__=='__main__':main()
