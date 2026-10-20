"""CPU acceptance checks for the real train-only prior and frozen-val shape draws."""
import ast
import json
import os
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from utils.hhc_train_shape_prior import TrainShapePrior,file_sha


def main():
    assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    spec=ROOT/'output/hhc_train_shape_prior_20261016/prior_spec.json'
    prior=TrainShapePrior(spec,file_sha(ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004/manifest.json'),
                         file_sha(ROOT/'utils/interhuman_smpl_mgt.py'))
    payload=json.loads((ROOT/'output/interhuman_validation_requests/validation.json').read_text())
    requests=payload['requests'];assert len(requests)==480
    assert not (set(prior.identifiers) & {r['identifier'] for r in requests})
    def rng_signature():
        state=np.random.get_state()
        return (state[0],state[1].tobytes(),state[2:],random.getstate())
    before=rng_signature();counts={}
    for policy in ('train_mean','empirical_train_pair'):
        full,indices=prior.request_betas(requests,policy)
        reordered,_=prior.request_betas(list(reversed(requests)),policy)
        chunked={}
        for i in range(0,480,37):chunked.update(prior.request_betas(requests[i:i+37],policy)[0])
        assert all(np.array_equal(full[k],reordered[k]) and np.array_equal(full[k],chunked[k]) for k in full)
        assert all(np.array_equal(v,prior.mean if indices[k] is None else prior.betas[indices[k]]) for k,v in full.items())
        counts[policy]=len(set(indices.values()))
        if policy=='empirical_train_pair':
            changed=prior.request_betas(requests,policy,replication=1)[1]
            assert any(changed[k]!=indices[k] for k in indices)
    assert before==rng_signature()
    class RestrictedRequest(dict):
        def __getitem__(self,key):
            assert key in ('identifier','sample_seed'),'GT/caption field accessed: '+key
            return super().__getitem__(key)
    prior.request_betas([RestrictedRequest(r) for r in requests],'empirical_train_pair')
    failures=[]
    for kwargs in ({'expected_data_sha':'invalid'},{'expected_cache_source_sha':'invalid'}):
        try:TrainShapePrior(spec,**kwargs)
        except ValueError:failures.append('mismatched_'+next(iter(kwargs)))
        else:raise AssertionError('Mismatched prior accepted')
    try:prior.request_betas([requests[0],requests[0]],'train_mean')
    except ValueError:failures.append('duplicate_identifiers')
    else:raise AssertionError('Duplicate identifiers silently overwritten')
    from tools.export_hhc_mgt_train_prior_validation_cloud import TrainPriorScorer
    from unittest.mock import patch
    import torch
    betas,_=prior.request_betas(requests[:4],'empirical_train_pair')
    # Real CPU FK constructor; forbid any H5 access so GT-beta loading cannot pass.
    with patch('h5py.File',side_effect=AssertionError('Forbidden GT H5 read')):
        scorer=TrainPriorScorer(None,None,None,None,None,None,None,None,None,'cpu','cpu',betas)
    assert all(torch.equal(scorer.betas[k],torch.from_numpy(v)) for k,v in betas.items())
    baseline=ast.parse((ROOT/'tools/export_hhc_mgt_validation_cloud.py').read_text())
    candidate=ast.parse((ROOT/'tools/export_hhc_mgt_train_prior_validation_cloud.py').read_text())
    def generation_loop(tree):
        return next(node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='begin')
    assert ast.dump(generation_loop(baseline))==ast.dump(generation_loop(candidate))
    assert TrainPriorScorer.generate is __import__('utils.interhuman_smpl_mgt',fromlist=['StreamingScorer']).StreamingScorer.generate
    config=json.loads((ROOT/'migration/hhc-text-only-validation.json').read_text())
    assert all(file_sha(ROOT/p)==digest for p,digest in config['sources'].items())
    assert not torch.cuda.is_initialized()
    result={'status':'passed_cpu_policy_and_constructor_audit','samples':480,'train_prior_rows':prior.spec['rows'],
        'sampled_row_counts':counts,'global_rng_unchanged':True,'batch_order_chunking_invariant':True,
        'only_identifier_and_seed_request_fields_accessed':True,'train_val_identifiers_disjoint':True,
        'constructor_reads_no_GT_H5':True,'real_cpu_FK_constructor':True,'generation_loop_AST_identical':True,
        'inherited_generation_function_unchanged':True,'guard_rejections':failures,'existing_23_sources_unchanged':True,
        'gpu_work':False,'full_MGT_generation_executed':False,'is_quality_evaluation':False,
        'spec_sha256':file_sha(spec),'sources':{p:file_sha(ROOT/p) for p in
        ('tools/audit_hhc_train_shape_prior_cloud.py','tools/export_hhc_mgt_train_prior_validation_cloud.py','utils/hhc_train_shape_prior.py')}}
    destination=ROOT/'output/hhc_sota_20261016/train_shape_prior_cpu_audit.json'
    assert not destination.exists();destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
