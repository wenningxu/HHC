"""Read-only, standard-library prerequisite inventory. Does not launch training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory(vq_run, native_data):
    clip_candidates=[ROOT/'runtime_assets/clip/ViT-B-32.pt',Path.home()/'.cache/clip/ViT-B-32.pt']
    training_clip=next((p for p in clip_candidates if p.is_file()),clip_candidates[0])
    required = {
        'frozen_vq185k': vq_run / 'best_fid.pth',
        'native_data_manifest': native_data / 'manifest.json',
        'native_motions': native_data / 'motions.h5',
        'native_train_split': native_data / 'split/train.txt',
        'native_captions': native_data / 'texts_processed',
        'val_manifest': ROOT / 'output/interhuman_validation_requests/manifest.json',
        'val_mm_requests': ROOT / 'output/interhuman_validation_mm_requests/validation_mm.json',
        'test_main_manifest': ROOT / 'output/interhuman_fair_requests_v3/manifest.json',
        'test_mm_manifest': ROOT / 'output/interhuman_fair_mm_requests/manifest.json',
        'smpl_female': ROOT / 'body_models/smpl/SMPL_FEMALE.pkl',
        'evaluator': ROOT / 'third_party/InterMask/checkpoints/eval_model/interclip.ckpt',
        'training_clip': training_clip,
    }
    records = {k: {'path':str(p), 'exists':p.exists(),
                   'bytes':p.stat().st_size if p.is_file() else None} for k,p in required.items()}
    missing = [k for k,v in records.items() if not v['exists']]
    return {'status':'missing_assets' if missing else 'paths_exist_not_launch_validated',
            'training_launched':False, 'performance_improvement_measured':False,
            'required':records, 'missing':missing,
            'limits':'Existence only. Before launch verify VQ step/hash/mean/std, native data manifest, frozen request contents, evaluator/source hashes and device availability. Absolute paths in frozen manifests must resolve through staging; never change request content.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--vq-run',type=Path,default=Path('/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004'))
    parser.add_argument('--native-data',type=Path,default=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=inventory(args.vq_run,args.native_data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'missing':result['missing'],'training_launched':False}))
