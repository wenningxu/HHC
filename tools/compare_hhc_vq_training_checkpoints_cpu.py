"""Paired identifier-cluster bootstrap across native VQ training checkpoints.

Intervals describe validation sampling for these fixed models, not uncertainty
over training seeds, generation repetitions, or state-of-the-art performance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset,sha256_file
from tools.score_interhuman_vq_geometry import error_sums,physical_sums


def metrics(values):
    frames,full,root,tp,fp,fn,foot_distance,foot_count=values.sum(0)
    return np.asarray([1000*full/(22*frames),1000*root/frames,2*tp/(2*tp+fp+fn),1000*foot_distance/foot_count])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control-step',type=int,required=True);parser.add_argument('--candidate-step',type=int,required=True)
    parser.add_argument('--control-spec',type=Path,required=True);parser.add_argument('--candidate-spec',type=Path,required=True)
    args=parser.parse_args()
    assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert args.candidate_step>args.control_step>0
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if os.name!='nt':os.nice(10)
    data=InterHumanValidationRequestDataset();entry=data.manifest['request'];datasets=[]
    for path in (args.control,args.candidate):
        dataset=InterHumanGeneratedH5Dataset(path,data.request_path,request_entry=entry,formal=False)
        assert dataset.metadata['status']=='complete' and dataset.metadata['split']=='val';datasets.append(dataset)
    specs=[json.loads(path.read_text()) for path in (args.control_spec,args.candidate_spec)]
    for dataset,step,spec,path in zip(datasets,(args.control_step,args.candidate_step),specs,(args.control_spec,args.candidate_spec)):
        assert spec['step']==step and spec['sha256']==dataset.metadata['vq_checkpoint_sha256']
        assert sha256_file(path)==dataset.metadata['vq_spec_sha256']
        assert spec['origin']==dataset.metadata['vq_origin']=='fresh_cloud_rebuild_not_historical185k'
        assert dataset.metadata['method']=='hhc_native_smpl_vq_reconstruction'
        assert dataset.metadata['vq_step']==step
        assert dataset.metadata['vq_selection_scope']=='intermediate_val_selected_pilot'
    assert datasets[0].metadata['native_manifest_sha256']==datasets[1].metadata['native_manifest_sha256']
    for key in ('training_contract_sha256','source_run','normalization_sha256'):
        assert specs[0][key]==specs[1][key]
    groups=sorted({r['identifier'] for r in data.requests});indices={name:i for i,name in enumerate(groups)}
    values=np.zeros((2,len(groups),8),dtype=np.float64)
    for i,r in enumerate(data.requests):
        name,text,a,b,n=data[i]
        for arm,dataset in enumerate(datasets):
            pn,pt,p,q,length=dataset[i];assert (name,text,n)==(pn,pt,length)
            first=error_sums(a[None],p[None],torch.tensor([n]),data.adapter.mean,data.adapter.std)
            second=error_sums(b[None],q[None],torch.tensor([n]),data.adapter.mean,data.adapter.std)
            phys=physical_sums(a[None],b[None],p[None],q[None],torch.tensor([n]),data.adapter.mean,data.adapter.std,.1)
            values[arm,indices[name]]+=np.asarray([2*n,first['full_distance_sum']+second['full_distance_sum'],
                first['root_distance_sum']+second['root_distance_sum'],phys['contact_tp'],phys['contact_fp'],phys['contact_fn'],
                phys['pred_foot_displacement_sum_m'],phys['gt_contact_foot_transitions']])
    assert np.array_equal(values[0,:,0],values[1,:,0]) and np.array_equal(values[0,:,-1],values[1,:,-1])
    observed=np.stack([metrics(v) for v in values]);rng=np.random.default_rng(20261016);deltas=[]
    for _ in range(100):
        draw=rng.integers(0,len(groups),size=(100,len(groups)));means=[]
        for arm in range(2):
            totals=values[arm][draw].sum(1);frames,full,root,tp,fp,fn,distance,count=totals.T
            means.append(np.stack([1000*full/(22*frames),1000*root/frames,2*tp/(2*tp+fp+fn),1000*distance/count],axis=1))
        deltas.append(means[1]-means[0])
    delta=np.concatenate(deltas);names=['full_mpjpe_mm','root_position_error_mm','geometry_contact_f1','sliding_mm_per_frame']
    result={'status':'complete','scope':f'Fixed native VQ{args.control_step} vs VQ{args.candidate_step}, same architecture/data/training contract, +{args.candidate_step-args.control_step} training updates; validation identifier-cluster uncertainty only, NOT matched-budget algorithm gain, training-seed or generation/SOTA CI',
        'samples':480,'identifier_clusters':len(groups),'bootstrap_repetitions':10000,'bootstrap_seed':20261016,
        'metrics':{name:{'control':float(observed[0,j]),'candidate':float(observed[1,j]),
            'delta_candidate_minus_control':float(observed[1,j]-observed[0,j]),
            'delta_95_ci':np.quantile(delta[:,j],[.025,.975]).tolist()} for j,name in enumerate(names)},
        'control_sha256':sha256_file(args.control),'candidate_sha256':sha256_file(args.candidate),
        'control_spec_sha256':sha256_file(args.control_spec),'candidate_spec_sha256':sha256_file(args.candidate_spec),
        'script_sha256':sha256_file(Path(__file__)),'geometry_contact_is_proxy':True,'is_generation_evaluation':False}
    for dataset in datasets:dataset.close()
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
