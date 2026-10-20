"""Isolated HHC text CFG ablation; historical entrypoint is unchanged."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.mgt_interhuman_smpl import SMPLMGT
from models.mgt_interhuman_smpl_cfg import configure_conditioning
from models.interhuman_evaluator_wrapper import InterHumanEvaluatorWrapper,load_official_metrics_module,evaluator_provenance
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from dataset.interhuman_mgt import InterHumanMGTDrawBatchSampler
from dataset.interhuman_token_windows import InterHumanTokenWindows
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset,InterHumanFrozenRequestDataset,canonical_sha256
from utils.interhuman_smpl_mgt import load_vq,build_tokens,encode_text,StreamingScorer,sha,DATA,VQ_RUN
import utils.interhuman_smpl_mgt as smpl_assets
from utils.hhc_frozen_vq import load_frozen_vq
from utils.hhc_cpu_text_cache import encode_text_cpu_float32
from utils.motion_masks import sample_historical_mmm_corruption_with_kinematic_ratios
from utils.interhuman_mgt_training import interhuman_mgt_objective
from utils.runtime import generate_src_mask,preserving_rng_state,evaluating
from tools.reconstruct_interhuman_raw_body import load_model


def atomic_json(path,value):
    pending=path.with_suffix('.tmp'); pending.write_text(json.dumps(value,indent=2)+'\n'); pending.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def model_state(net):
    return {k:v for k,v in net.state_dict().items() if not k.startswith('trans_base.vqvae.')}


def restore_model(net,state):
    result=net.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and all(k.startswith('trans_base.vqvae.') for k in result.missing_keys)


def main():
    global DATA, VQ_RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--eval-device',default='cpu')
    parser.add_argument('--wandb-mode',choices=['disabled','online','offline'],default='disabled')
    parser.add_argument('--gpu-memory-fraction',type=float,default=.5)
    parser.add_argument('--conditioning-arm',choices=['historical','matched'],required=True)
    parser.add_argument('--token-window',choices=['historical','uniform'],default='historical')
    parser.add_argument('--steps',type=int,default=40000)
    parser.add_argument('--seed',type=int,default=114514)
    parser.add_argument('--formal-test',action='store_true')
    parser.add_argument('--native-data',type=Path,default=DATA)
    parser.add_argument('--vq-run',type=Path,default=VQ_RUN)
    parser.add_argument('--vq-spec',type=Path,help='Explicit immutable spec for a newly rebuilt VQ; historical mode remains default')
    args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU experiments run only on the attributed cloud checkout')
    if args.device!='cuda:0' or not 0<args.gpu_memory_fraction<=.5:raise ValueError('Use cloud cuda:0 and at most half its memory')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,0)
    if args.steps < 10000 or args.steps % 10000: parser.error('--steps must be a positive multiple of 10000')
    DATA=smpl_assets.DATA=args.native_data.resolve()
    VQ_RUN=smpl_assets.VQ_RUN=args.vq_run.resolve()
    vq_spec=json.loads(args.vq_spec.read_text()) if args.vq_spec else None
    if vq_spec:
        VQ_RUN=smpl_assets.VQ_RUN=Path(vq_spec['checkpoint']).parent.resolve()
    assert 64%args.microbatch==0
    args.output.mkdir(parents=True,exist_ok=args.resume)
    manifest_path=args.output/'training_manifest.json'
    if not args.smoke and shutil.disk_usage(args.output).free<9*1024**3 and not args.resume:
        raise RuntimeError('Require 9 GiB free for optimizer/latest atomic replacement plus best MGT')
    def log(row):
        row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**row}
        print(json.dumps(row),flush=True)
        with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
    def status(state):
        atomic_json(manifest_path,state)
        (args.output/'STATUS.md').write_text(
            f"# 原生SMPL InterHuman MGT\n\n状态：{state['status']}\n\n"
            f"更新：{time.strftime('%Y-%m-%d %H:%M:%S %z')}\n\n"
            f"训练 step：{state.get('step',0)}/{args.steps}；最佳 val step/FID：{state.get('best_step')} / {state.get('best_fid')}\n\n"
            f"正式评测已完成 repetition：{state.get('completed_test_reps',[])}\n\n"
            "协议与指标详见 training_manifest.json、metrics.jsonl、test/summary.json。未完成不能当作已验收。\n")
    sources=['tools/train_hhc_smpl_cfg_ablation.py','models/mgt_interhuman_smpl_cfg.py','models/mgt_interhuman_smpl.py','utils/interhuman_smpl_mgt.py',
             'models/two_person_transformer.py','models/motion_transformer.py','models/interhuman_transformer.py',
             'models/vqvae_interhuman_smpl_raw.py','models/vqvae_sep3_shape.py','utils/losses.py','utils/motion_masks.py','dataset/interhuman_fair_requests.py',
             'dataset/interhuman_hhc_adapter.py','dataset/interhuman_mgt.py','dataset/interhuman_token_windows.py','utils/interhuman_mgt_training.py','utils/hhc_frozen_vq.py','utils/hhc_cpu_text_cache.py']
    contract={'schema':'native_smpl_text_cfg_ablation_v1','conditioning_arm':args.conditioning_arm,'text_drop_probability':.2,'shape_drop_probability':.2,'vq_checkpoint':str(VQ_RUN/'best_fid.pth'),
              'vq_sha256':sha(VQ_RUN/'best_fid.pth'),'vq_step':vq_spec['step'] if vq_spec else 185000,
              'vq_origin':vq_spec['origin'] if vq_spec else 'historical185k',
              'vq_selection_scope':vq_spec['selection_scope'] if vq_spec else 'historical_val_best185k',
              'vq_spec_sha256':sha(args.vq_spec) if args.vq_spec else None,'data_sha256':sha(DATA/'manifest.json'),
              'sources':{p:sha(ROOT/p) for p in sources},'token_window_policy':args.token_window,'seed':args.seed,'batch':64,'microbatch':args.microbatch,
              'eval_device':args.eval_device,'gpu_memory_fraction':args.gpu_memory_fraction,'wandb_mode':args.wandb_mode,
              'steps':args.steps,'architecture':{'embed':1024,'heads':16,'local_layers':2,'base_adaln_layers':7,'head_layers':9,
              'dropout':.1,'codebooks':[12288,12288],'latent_dims':[96,64],'shape':True,'block_size':76},
              'optimizer':{'name':'AdamW','lr':.0002,'betas':[.5,.9],'weight_decay':1e-6,'milestone':150000,'gamma':.05,
                           'warmup':False,'gradient_clip':1.,'precision':'bf16'},
              'mask':'historical_pipeline_paper_kinematic_ratio','pkeep':.5,'objective':'historical_all_valid',
              'sampling':{'steps':50,'temperature':.5,'cfg':3.5,'cascade':['root','body'],'ranking_noise':0.,'conditioning':'paper_text_cfg'},
              'differences_from_interx':['Text arm: matched jointly drops word/global text; historical drops global only. Independent shape dropout .2 and all-valid NLL retained.',
                 'Token window policy is explicitly recorded; compare each factor separately before combining.',
                 'No hand stream; native SMPL shape-conditioned frozen VQ identified by explicit step/origin/hash.',
                 'Capacity76 instead of38 accommodates InterHuman300-frame requests; all other widths/layers preserved.',
                 'Frozen val-only selection and isolated RNG replace historical test-side-effects.',
                 'Native generation ceil((official_length+1)/4) covers FK preprocessing frame loss; no valid zero tail.',
                 'Formal score is shape-conditioned using requested raw betas, unlike unshaped InterMask; disclose this advantage.',
                 'Stream scoring stores embeddings and hashes, not full generated-motion H5.'],
              'requests':{p:sha(ROOT/p) for p in ['output/interhuman_validation_requests/manifest.json',
                       'output/interhuman_validation_mm_requests/validation_mm.json','output/interhuman_fair_requests_v3/manifest.json',
                       'output/interhuman_fair_mm_requests/manifest.json']}}
    contract_hash=canonical_sha256(contract)
    state={'status':'initializing','pid':os.getpid(),'step':0,'best_step':None,'best_fid':None,'contract':contract,'contract_sha256':contract_hash,'formal_test_requested':args.formal_test}
    if args.resume:
        prior=json.loads(manifest_path.read_text()); assert prior['contract_sha256']==contract_hash
        state={**prior,'pid':os.getpid(),'status':'resuming','formal_test_requested':args.formal_test}
    status(state)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.vq_spec:
        vq,mean,std,verified_spec=load_frozen_vq(args.vq_spec,args.device,DATA/'manifest.json')
        assert verified_spec==vq_spec
    else:vq,mean,std=load_vq(args.device)
    before_vq={k:v.detach().cpu().clone() for k,v in vq.state_dict().items()}
    data=build_tokens(vq,mean,std,args.device,args.output/'tokens.pt')
    data=InterHumanTokenWindows(data,args.token_window)
    clip_model,_=clip.load('ViT-B/32',device=args.eval_device,jit=False)
    clip_model.eval().requires_grad_(False)
    captions=sorted({text for row in data.rows for text in row['texts']})
    text_cache={}
    for begin in range(0,len(captions),128):
        names=captions[begin:begin+128]
        clip_parameter=next(clip_model.parameters())
        if clip_parameter.device.type=='cpu' and clip_parameter.dtype==torch.float32:
            global_text,word=encode_text_cpu_float32(clip_model,names)
        else:
            global_text,word=encode_text(clip_model,names,args.eval_device)
        for name,g,w in zip(names,global_text.cpu(),word.cpu()): text_cache[name]=(g,w)
    torch.manual_seed(args.seed)
    net=configure_conditioning(SMPLMGT(vq).to(args.device),args.conditioning_arm)
    assert all(torch.equal(v.detach().cpu(),before_vq[k]) for k,v in vq.state_dict().items())
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.0002,betas=(.5,.9),weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150000],gamma=.05)
    start=0
    if args.resume and (args.output/'latest.pth').exists():
        saved=torch.load(args.output/'latest.pth',map_location='cpu',weights_only=False)
        assert saved['contract_sha256']==contract_hash
        restore_model(net,saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start=saved['step']
        state.update(step=start,best_fid=saved['best_fid'],best_step=saved['best_step'])
    evaluator=InterHumanEvaluatorWrapper.from_official_checkpoint(args.eval_device)
    metrics=load_official_metrics_module(); adapter=OfficialInterHumanAdapter()
    scorer=StreamingScorer(net,vq,mean,std,clip_model,evaluator,metrics,adapter,load_model(),args.device,args.eval_device)
    validation=InterHumanValidationRequestDataset(adapter=adapter)
    mm_validation=json.loads((ROOT/'output/interhuman_validation_mm_requests/validation_mm.json').read_text())
    mm_validation['mm_repeats']=mm_validation['repeats']
    run=None
    if not args.smoke and args.wandb_mode!='disabled':
        import wandb
        run=wandb.init(project='NewInter',group='mgt_interhuman_smpl_cfg_ablation',name=args.output.name,dir=str(args.output),
                       mode=args.wandb_mode,id=state.get('wandb_run_id'),resume='allow',config=contract)
        state['wandb_run_id']=run.id
    state['status']='smoke_running' if args.smoke else 'training'; status(state)
    total_steps=2 if args.smoke else args.steps
    sampler=InterHumanMGTDrawBatchSampler(data,64,total_steps,args.seed,start_step=start)
    loader=DataLoader(data,batch_sampler=sampler,num_workers=0)
    iterator=iter(loader)
    # Evaluator initialization and DataLoader iterator construction consume
    # RNG; restore only after them, immediately before the next update.
    if args.resume and start: restore_rng(saved['rng'])
    log({'event':'ready','samples':len(data),'captions':len(captions),'trainable_parameters':sum(p.numel() for p in params),'microbatch':args.microbatch})
    net.train(); vq.eval(); started=time.monotonic()
    for step,(target,lengths,text,beta) in enumerate(iterator,start+1):
        optimizer.zero_grad(set_to_none=True)
        totals={}
        for offset in range(0,64,args.microbatch):
            sl=slice(offset,offset+args.microbatch)
            y=target[sl].to(args.device); b=beta[sl].to(args.device)
            valid=generate_src_mask(76,lengths[sl].to(args.device))
            attention=generate_src_mask(76,lengths[sl].to(args.device)+1)
            corrupted,masked,_,_=sample_historical_mmm_corruption_with_kinematic_ratios(y,valid[:,:,None].expand_as(y),12288,pkeep=.5)
            g=torch.stack([text_cache[t][0] for t in text[sl]]).to(args.device)
            w=torch.stack([text_cache[t][1] for t in text[sl]]).to(args.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net.forward_function(corrupted,g,attention,word_emb=w,betas=b,gender=torch.ones(len(b),2,dtype=torch.long,device=args.device))
                objective=interhuman_mgt_objective(logits,{'target':y,'valid_mask':valid[:,:,None].expand_as(y),'masked_mask':masked})
                loss=objective['loss']*(args.microbatch/64)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss step{step}')
            loss.backward()
            for k,v in objective.items(): totals[k]=totals.get(k,0)+v.item()*args.microbatch/64
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if step<=2 or step%200==0:
            row={'event':'train','step':step,'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],
                 'elapsed_s':time.monotonic()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated(args.device)/1024**3,**totals}
            log(row)
            if run: wandb.log({'train/'+k:v for k,v in row.items() if isinstance(v,(float,int))},step=step)
        is_best=False
        if step%10000==0 or args.smoke and step==total_steps:
            with preserving_rng_state(),evaluating(net):
                configure_conditioning(net,args.conditioning_arm,sampling=True)
                result=scorer.main_score(validation,steps=50)
                if args.smoke:
                    mm_payload={**mm_validation,'requests':mm_validation['requests'][:2]}
                else: mm_payload=mm_validation
                result['multimodality']=scorer.mm_score(mm_payload)
                configure_conditioning(net,args.conditioning_arm)
            if not all(np.isfinite(value) for value in result.values()):
                raise FloatingPointError(f'Nonfinite validation step{step}: {result}')
            vq.eval()
            is_best=state['best_fid'] is None or result['fid']<state['best_fid']
            if is_best: state.update(best_fid=result['fid'],best_step=step)
            log({'event':'validation','step':step,'best_fid':state['best_fid'],'best_step':state['best_step'],**result})
            atomic_json(args.output/f'validation_{step:06d}.json',result)
            if run: wandb.log({'val/'+k:v for k,v in result.items()},step=step)
        state['step']=step
        if not args.smoke and step%1000==0:
            if shutil.disk_usage(args.output).free<4*1024**3: raise RuntimeError('Disk <4GiB before atomic checkpoint')
            payload={'model':model_state(net),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'rng':rng_state(),'step':step,'best_fid':state['best_fid'],'best_step':state['best_step'],'contract_sha256':contract_hash}
            pending=args.output/'checkpoint.tmp'; torch.save(payload,pending); pending.replace(args.output/'latest.pth')
            if is_best:
                torch.save({'model':model_state(net),'step':step,'contract_sha256':contract_hash},pending)
                pending.replace(args.output/'best_fid.pth')
            status(state)
    assert all(torch.equal(v.detach().cpu(),before_vq[k]) for k,v in vq.state_dict().items())
    if args.smoke:
        state['status']='smoke_complete'; status(state); return
    for relative,digest in {**contract['sources'],**contract['requests']}.items():
        if sha(ROOT/relative)!=digest:
            raise RuntimeError(f'Frozen source/request changed during training: {relative}')
    assert sha(VQ_RUN/'best_fid.pth')==contract['vq_sha256']
    if not args.formal_test:
        state['status']='pilot_complete_validation_only'; status(state)
        if run: run.finish()
        return
    state['status']='training_complete_test_running'; status(state)
    selected=torch.load(args.output/'best_fid.pth',map_location='cpu',weights_only=False)
    restore_model(net,selected['model']); assert selected['step']==state['best_step']
    selected_hash=sha(args.output/'best_fid.pth')
    net.eval(); configure_conditioning(net,args.conditioning_arm,sampling=True)
    out=args.output/'test'; out.mkdir(exist_ok=True)
    mm_manifest=json.loads((ROOT/'output/interhuman_fair_mm_requests/manifest.json').read_text())
    identity={'contract_sha256':contract_hash,'checkpoint_sha256':selected_hash,'selected_step':selected['step'],
              'vq_checkpoint_sha256':contract['vq_sha256'],'evaluator':evaluator_provenance(include_checkpoint_sha256=True),
              'shape_condition':'requested original betas, fixed within MM prompt; no GT motion supplied',
              'storage':'streaming embeddings; generated trajectories not retained','length_bridge':'ceil((L+1)/4) native frames then exact L online features'}
    atomic_json(out/'identity.json',identity)
    for replication in range(20):
        path=out/f'replication_{replication:02d}.json'
        if path.exists():
            previous=json.loads(path.read_text()); assert previous['identity']==identity and previous['status']=='complete'
            assert all(sha(out/name)==digest for name,digest in previous['embedding_sha256'].items())
            continue
        dataset=InterHumanFrozenRequestDataset(replication,adapter=adapter)
        main_result=scorer.main_score(dataset,seed=1234+replication,embedding_path=out/f'main_{replication:02d}.npz')
        entry=mm_manifest['replications'][replication]
        assert sha(entry['path'])==entry['sha256']
        payload=json.loads(Path(entry['path']).read_text())
        assert canonical_sha256(payload['requests'])==entry['request_sha256']
        mm=scorer.mm_score(payload,replication,embedding_path=out/f'mm_{replication:02d}.npz')
        result={'status':'complete','replication':replication,'identity':identity,'main':main_result,'multimodality':mm,
                'request_sha256':dataset.payload['request_sha256'],'mm_request_sha256':entry['request_sha256'],
                'embedding_sha256':{name:sha(out/name) for name in (f'main_{replication:02d}.npz',f'mm_{replication:02d}.npz')}}
        if not all(np.isfinite(value) for value in [*main_result.values(),mm]):
            raise FloatingPointError(f'Nonfinite test repetition {replication}')
        atomic_json(path,result)
        log({'event':'test_replication','replication':replication,**main_result,'multimodality':mm})
        state['completed_test_reps']=list(range(replication+1)); status(state)
    records=[json.loads((out/f'replication_{i:02d}.json').read_text()) for i in range(20)]
    summary={'status':'complete','replications':20,'identity':identity,'ci':'1.96*population_std/sqrt(20)','metrics':{}}
    for key in ('fid','matching','top1','top2','top3','diversity','diversity_gt','multimodality'):
        values=np.array([r[key] if key=='multimodality' else r['main'][key] for r in records])
        assert np.isfinite(values).all()
        summary['metrics'][key]={'mean':float(values.mean()),'ci95':float(1.96*values.std()/np.sqrt(20)),'values':values.tolist()}
    atomic_json(out/'summary.json',summary)
    report=['# InterHuman SMPL + shape MGT 正式评测','',
            f"选定 checkpoint：{selected['step']}（仅按冻结val FID选模）；训练seed{args.seed}；arm={args.conditioning_arm}；20次test repetitions。",'',
            '| 指标 | mean ± 95% CI |','|---|---:|']
    report += [f"| {key} | {value['mean']:.6f} ± {value['ci95']:.6f} |" for key,value in summary['metrics'].items()]
    report += ['', 'InterCLIP FID，不是joint FID；使用请求的真实betas，不能称与无shape模型完全同条件。',
               '完整manifest、checkpoint/request/evaluator/embedding hashes见identity.json及逐rep JSON。',
               '未保存全量生成动作H5；可按固定种子重生成。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    state['status']='complete'; state['completed_test_reps']=list(range(20)); status(state)
    log({'event':'all_complete','summary':str(out/'summary.json')})
    if run: run.finish()


if __name__=='__main__':
    try: main()
    except Exception as error:
        if '--output' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--output')+1]); path=root/'training_manifest.json'
            if path.exists():
                state=json.loads(path.read_text())
                if state.get('pid')==os.getpid():
                    state.update(status='failed',error=repr(error),traceback=traceback.format_exc())
                    atomic_json(path,state)
                    (root/'STATUS.md').write_text('# 任务异常停止\n\n'+repr(error)+'\n\n最后有效checkpoint保留；详情见training_manifest.json。\n')
        raise
