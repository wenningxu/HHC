"""Cloud CPU audit of completed baseline validation reports and embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.interhuman_evaluator_wrapper import load_official_metrics_module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def large_sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reference=None;rows=[];metrics=load_official_metrics_module()
    for path in args.reports:
        report=json.loads(path.read_text());assert report['status']=='complete' and not report['is_formal_test']
        assert report['samples']==480 and report['split']=='validation'
        cache=path.parent/'validation480.h5';assert large_sha(cache)==report['cache_sha256']
        with h5py.File(cache) as handle:
            assert handle.attrs['status']=='complete'
            generation_seconds=float(handle.attrs.get('generation_seconds',handle.attrs.get('elapsed_s',float('nan'))))
            assert np.isfinite(generation_seconds) and generation_seconds>0
        embeddings=path.with_suffix('.embeddings.npz');assert sha(embeddings)==report['embeddings_sha256']
        with np.load(embeddings) as arrays:
            gt,text=arrays['GT'],arrays['text'];assert gt.shape==text.shape==(480,512)
            if reference is None:reference=(report,gt.copy(),text.copy())
            first,ref_gt,ref_text=reference
            assert report['request_sha256']==first['request_sha256']
            assert report['evaluator']==first['evaluator']
            gt_diff=float(np.max(np.abs(gt-ref_gt)));text_diff=float(np.max(np.abs(text-ref_text)))
            assert gt_diff<=1e-6 and text_diff<=1e-6
            assert np.isfinite(arrays['raw']).all()
            per_match=[];per_top=[]
            for start in range(0,480,96):
                distances=metrics.euclidean_distance_matrix(text[start:start+96],arrays['raw'][start:start+96])
                per_match.extend(np.diag(distances).tolist())
                per_top.extend(metrics.calculate_top_k(np.argsort(distances,axis=1),top_k=3).tolist())
            per_match=np.asarray(per_match);per_top=np.asarray(per_top)
            assert abs(per_match.mean()-report['metrics']['matching'])<1e-6
            assert np.allclose(per_top.mean(0),[report['metrics'][k] for k in ('top1','top2','top3')],rtol=0,atol=1e-12)
        differences=np.array([r['gt']-r['generated'] for r in report['lengths']])
        lengths=np.array([r['gt'] for r in report['lengths']]);length_groups={}
        for label,support in [('lt30',lengths<30),('30to99',(lengths>=30)&(lengths<100)),('ge100',lengths>=100)]:
            length_groups[label]={'count':int(support.sum()),'matching':float(per_match[support].mean()),
                                  'top1_top2_top3':per_top[support].mean(0).tolist()}
        rows.append({'method':report['method'],'report_sha256':sha(path),
            'same_gt_embedding_max_abs_error':gt_diff,'same_text_embedding_max_abs_error':text_diff,
            'gt_minus_generated_length_histogram':{str(int(k)):int((differences==k).sum()) for k in np.unique(differences)},
            'length_difference_mean':float(differences.mean()),'sampler':report['metadata']['sampler'],
            'generation_loop_wall_seconds':generation_seconds,
            'processed_length_diagnostics_original_batch96_candidates':length_groups,
            'metrics':report['metrics'],'embeddings_sha256':report['embeddings_sha256']})
    value={'status':'passed','samples':480,'request_sha256':reference[0]['request_sha256'],
        'reports':rows,'is_formal_test':False,'confidence_intervals':'not estimated',
        'generation_timing_context':'Single cloud RTX4090, batch1, coexisting pre-existing InterGen training; loop wall time excludes model initialization. Not a dedicated inference speed benchmark.',
        'limitations':'Same frozen requests, GT/text embeddings and evaluator. Native method sampling budgets and generated lengths differ; not a controlled equal-cost/equal-length experiment or a SOTA claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
