"""Freeze the MGT-eligible train-only empirical paired-shape prior on cloud CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    args.output=args.output.resolve();assert args.output.is_relative_to(ROOT/'output')
    args.output.mkdir(exist_ok=False);os.nice(10)
    import h5py
    import numpy as np
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
    manifest_sha=sha(native/'manifest.json')
    assert manifest_sha=='7468c59cbe010e438227ec1a13e30d1fd9a03a8fc1e8c9d0d4bfb971e02c5dc5'
    split_paths={k:native/'split'/f'{k}.txt' for k in ('train','val','test')}
    identifiers={k:p.read_text().splitlines() for k,p in split_paths.items()}
    assert not (set(identifiers['train']) & (set(identifiers['val']) | set(identifiers['test'])))
    rows=[];names=[];exclusions={};text_digest=hashlib.sha256();started=time.time()
    with h5py.File(native/'motions.h5','r') as source:
        for name in identifiers['train']:
            reason=None
            if name not in source or 'motion30' not in source[name]:reason='missing_motion'
            elif not 30<=len(source[name]['motion30'])<1000:reason='length_outside_MGT_cache'
            else:
                path=native/'texts_processed'/f'{name}.txt';raw=path.read_bytes()
                text_digest.update(name.encode()+b'\0'+raw+b'\0')
                texts=[line.strip() for line in raw.decode().splitlines()]
                if not texts:reason='no_text'
            if reason:
                exclusions[reason]=exclusions.get(reason,0)+1
                continue
            beta=np.asarray(source[name]['betas'][:],dtype=np.float32).reshape(2,10)
            assert np.isfinite(beta).all()
            rows.append(beta);names.append(name)
    assert len(set(names))==len(names)>100
    beta=np.stack(rows);flat=beta.reshape(len(beta),20).astype(np.float64)
    mean=flat.mean(0);std=flat.std(0);norms=np.linalg.norm(flat,axis=1)
    active=std>1e-6
    distances=np.linalg.norm((flat[:,active]-mean[active])/std[active],axis=1)
    zero_distance=float(np.linalg.norm(mean[active]/std[active]))
    array=args.output/'train_shape_pairs.npz'
    np.savez_compressed(array,identifiers=np.asarray(names),betas=beta)
    with np.load(array,allow_pickle=False) as recovered:
        assert np.array_equal(recovered['identifiers'],names) and np.array_equal(recovered['betas'],beta)
    spec={'schema':'hhc_train_only_paired_shape_prior_v1','status':'complete',
        'array':str(array),'array_bytes':array.stat().st_size,'array_sha256':sha(array),
        'rows':len(beta),'shape':[len(beta),2,10],'dtype':'float32',
        'data_manifest_sha256':manifest_sha,'split_sha256':{k:sha(p) for k,p in split_paths.items()},
        'eligible_train_names_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        'eligible_text_digest':text_digest.hexdigest(),'excluded_counts':exclusions,
        'selection':'Exact native MGT build_tokens support: train IDs in split order, motion30 present, 30<=length<1000, nonempty splitlines text; interaction-uniform empirical pairs, preserve person order',
        'beta_mean_pair':mean.reshape(2,10).tolist(),'beta_std_pair':std.reshape(2,10).tolist(),
        'beta_l2_pair_quantiles':{str(q):float(np.quantile(norms,q)) for q in (0,.05,.5,.95,1)},
        'unique_pair_rows':len(np.unique(flat,axis=0)),'active_dimensions':int(active.sum()),
        'zero_standardized_distance_to_train_mean':zero_distance,
        'train_standardized_distance_quantiles':{str(q):float(np.quantile(distances,q)) for q in (0,.05,.5,.95,1)},
        'zero_distance_empirical_percentile':float(np.mean(distances<=zero_distance)),
        'validation_test_beta_arrays_read':False,'motion_arrays_read':False,
        'gpu_work':False,'local_checkpoint_downloads':False,'builder_sha256':sha(Path(__file__)),
        'mgt_cache_source_sha256':sha(ROOT/'utils/interhuman_smpl_mgt.py'),
        'scope':'Training-shape statistics and frozen prior only; not evidence of generation improvement or SOTA',
        'proposed_validation_factors':['zero (existing primary text-only reference)','fixed train mean (deterministic constant)','empirical train pair (unconditional latent shape drawn using independent per-request seed)'],
        'elapsed_s':time.time()-started}
    (args.output/'prior_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ('status','rows','unique_pair_rows','active_dimensions','zero_standardized_distance_to_train_mean','zero_distance_empirical_percentile','array_bytes','gpu_work')}),flush=True)


if __name__=='__main__':main()
