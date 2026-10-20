"""Recover the previously inaccessible historical VQ; do not alter active bindings."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import sys
import time

ROOT=Path('/workspace/HHC')
ORIGINAL=Path('/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004')
OUT=ROOT/'output/historical_vq185k_recovered_20261020'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert Path(__file__).resolve().parents[1]==ROOT
    sys.path.insert(0,str(ROOT))
    import torch
    from models.vqvae_interhuman_smpl_raw import VQSMPLRaw
    from utils.hhc_frozen_vq import normalization_sha
    torch.set_num_threads(2)
    frozen=ROOT/'output/hhc_vq_final_frozen_20261016/frozen_vq_spec.json'
    current=json.loads(frozen.read_text())
    before=sha(frozen)
    training=json.loads((ORIGINAL/'training_manifest.json').read_text())
    assert training['status']=='complete' and training['best_step']==185000
    validation=json.loads((ORIGINAL/'validation_185000.json').read_text())
    assert validation['samples']==480 and validation['fid']==training['best_fid']
    source=ORIGINAL/'best_fid.pth'
    source_hash=sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=False)
    assert saved['step']==185000
    net=VQSMPLRaw(SimpleNamespace(nb_code=12288,mu=.99))
    net.load_state_dict(saved['net'],strict=True)
    assert all(torch.isfinite(v).all() for v in saved['net'].values() if torch.is_tensor(v))
    mean,std=saved['mean'],saved['std']
    assert mean.shape==std.shape==(276,) and mean.dtype==std.dtype==torch.float32
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
    OUT.mkdir(exist_ok=True)
    copied={}
    for name in ['best_fid.pth','normalization.npz','training_manifest.json',
                 'validation_185000.json','validation_panel.json']:
        src,dst=ORIGINAL/name,OUT/name
        if dst.exists():assert sha(src)==sha(dst),name
        else:
            temporary=OUT/(name+'.partial')
            assert not temporary.exists()
            shutil.copyfile(src,temporary)
            assert sha(src)==sha(temporary)
            temporary.rename(dst)
        copied[name]={'bytes':dst.stat().st_size,'sha256':sha(dst)}
    source_matches={name:sha(ROOT/name)==expected if (ROOT/name).is_file() else None
                    for name,expected in training['sources'].items()}
    assert sha(source)==source_hash and sha(frozen)==before and not torch.cuda.is_initialized()
    result={'status':'historical_vq185k_recovered_cpu_strict_load_verified',
            'step':185000,'source':str(source),'snapshot':str(OUT/'best_fid.pth'),
            'sha256':source_hash,'state_tensors':len(saved['net']),
            'normalization_sha256':normalization_sha(mean,std),
            'same_normalization_as_active_vq':normalization_sha(mean,std)==current['normalization_sha256'],
            'same_checkpoint_as_active_vq':source_hash==current['sha256'],
            'historical_validation_report_not_rerun':validation,
            'original_training_source_matches_current':source_matches,
            'files':copied,'active_vq_spec_sha256_unchanged':before,
            'active_vq_step':current['step'],'active_hhc_saved_step':4000,
            'active_binding_changed':False,'new_quality_measurement':False,
            'cuda_initialized':False,'time_unix':time.time()}
    (OUT/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
