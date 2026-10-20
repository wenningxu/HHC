"""Read an existing cloud checkpoint's names; no model inference or CUDA."""
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
    parser.add_argument('--source-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert str(ROOT)=='/workspace/HHC','Checkpoints stay on cloud'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only'
    assert not args.output.exists(),'Preserve previous evidence'
    source=json.loads(args.source_audit.read_text())
    assert source['status']=='pass'
    comparison=source['intergen_source_comparison']
    assert comparison['all_model_ast_equal_after_explicit_identifier_renaming']
    assert comparison['same_download_ids']
    checkpoint=ROOT/'third_party/intergen_base_20261008/checkpoints/intergen.ckpt'
    assert checkpoint.stat().st_size==2701617525
    digest=sha(checkpoint)
    assert digest=='bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d'
    import torch
    torch.set_num_threads(2)
    os.nice(10)
    started=time.time()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state={k.replace('model.','') if 'model' in k else k:v for k,v in payload['state_dict'].items()}
    assert all(v.device.type=='cpu' for v in state.values())
    aliases={'clip_transformer.':'bert_transformer.',
             'clipTransEncoder.':'bertTransEncoder.','clip_ln.':'bert_ln.'}
    rows=[]
    for old,new in aliases.items():
        names=sorted(k for k in state if k.startswith(old))
        expected=[new+k[len(old):] for k in names]
        rows.append({'intergen_prefix':old,'coshmdm_expected_prefix':new,
                     'stored_intergen_keys':len(names),
                     'stored_coshmdm_keys':sum(k.startswith(new) for k in state),
                     'tensor_elements':sum(state[k].numel() for k in names),
                     'expected_keys_absent_from_checkpoint':[k for k in expected if k not in state],
                     'intergen_keys':names})
    assert not torch.cuda.is_initialized()
    result={'schema':'coshmdm_public_checkpoint_namespace_v1','status':'complete_cpu_key_audit',
        'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
        'checkpoint_sha256':digest,'state_tensor_count':len(state),'prefixes':rows,
        'expected_coshmdm_keys_missing':sum(len(r['expected_keys_absent_from_checkpoint']) for r in rows),
        'source_audit_sha256':sha(args.source_audit),'script_sha256':sha(Path(__file__)),
        'checkpoint_modified':False,'gpu_work':False,'local_checkpoint_downloads':False,
        'full_model_load_executed':False,'inference_executed':False,
        'scope':'Stored cloud checkpoint names and AST-verified source alias comparison only. Not a new CoShMDM checkpoint, runtime reproduction, quality score or proof of paper implementation.',
        'conclusion':'Public source renames CLIP module attributes to bert_*; its infer strict=False can skip learned text layers, while eval strict=True requires those names. Remapping these aliases yields the same source architecture/weights as InterGen, not an independent paper baseline.',
        'elapsed_s':time.time()-started}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','state_tensor_count','expected_coshmdm_keys_missing','gpu_work','full_model_load_executed')}),flush=True)


if __name__=='__main__':main()
