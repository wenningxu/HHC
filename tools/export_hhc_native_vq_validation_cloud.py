"""Export native 276D VQ reconstruction using the complete official val requests.

The native training monitor uses a different center crop. This independent
path preserves every official request and pads the raw input to a multiple of
four by repeating its last frame, then discards the padding before conversion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import MEAN_SHA256,STD_SHA256,InterHumanGeneratedH5Dataset
from dataset.interhuman_smpl_pair import body192_from_motion
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK
from tools.reconstruct_interhuman_raw_body import load_model
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.hhc_decoder_override import load_decoder_override


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--decoder-spec',type=Path,help='Explicit separately identified decoder-only pilot')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('All GPU work is cloud only')
    args.output=args.output.resolve();assert args.output.is_relative_to(ROOT/'output')
    args.output.mkdir(exist_ok=False)
    state={'status':'initializing','pid':os.getpid(),'started':time.time(),'is_generation_quality':False,
           'protocol':'complete official frozen 480 val; native raw request crop; repeat-last pad to 4; trim raw padding before official processing'}
    status=args.output/'STATUS.json';write(status,state)
    sources=['tools/export_hhc_native_vq_validation_cloud.py','utils/hhc_frozen_vq.py',
             'models/vqvae_interhuman_smpl_raw.py','dataset/interhuman_smpl_pair.py',
             'dataset/interhuman_fair_requests.py','dataset/interhuman_hhc_adapter.py']
    sources.append('utils/hhc_decoder_override.py')
    hashes={p:file_sha(ROOT/p) for p in sources}
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
        torch.cuda.set_per_process_memory_fraction(.2,0)
        native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
        net,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
        decoder=load_decoder_override(net,args.decoder_spec,spec) if args.decoder_spec else None
        before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        data=InterHumanValidationRequestDataset();assert len(data)==480
        fk=NativeSMPLFK().cuda();smpl=load_model()
        temp=args.output/'reconstructions.partial.h5';destination=args.output/'reconstructions.h5'
        identity_position_max=0.;identity_other_max=0.;identity_contact_mismatches=0
        padding_counts={str(k):0 for k in range(4)}
        def official(motion,beta,request):
            body=body192_from_motion(motion.cpu(),beta.cpu().reshape(2,10),smpl).numpy()
            if request.get('left_right_swap',False):raise ValueError('Frozen val must not introduce a mirror')
            return data.adapter.process_loaded_pair(*body,anchor_person=request['anchor_person'],
                   output_person_swap=request['output_person_swap'],normalize=True)
        with h5py.File(native/'motions.h5') as archive,h5py.File(temp,'x') as out,torch.inference_mode():
            out.attrs.update(schema='interhuman_fair_generation_v1',status='running',split='val',
                method='hhc_native_smpl_vq_reconstruction',request_file_sha256=data.manifest['request']['sha256'],
                request_content_sha256=data.payload['request_sha256'],normalization_mean_sha256=MEAN_SHA256,
                normalization_std_sha256=STD_SHA256,vq_checkpoint_sha256=spec['sha256'],vq_step=spec['step'],
                vq_origin=spec['origin'],vq_selection_scope=spec['selection_scope'],
                vq_spec_sha256=file_sha(args.vq_spec),exporter_sha256=hashes[sources[0]],
                reconstruction_protocol=state['protocol'],native_manifest_sha256=file_sha(native/'manifest.json'))
            if decoder:
                out.attrs.update(method='hhc_native_smpl_vq_decoder_only_pilot',decoder_override_sha256=decoder['sha256'],
                                 decoder_updates=decoder['decoder_updates'],decoder_arm=decoder['arm'],
                                 decoder_spec_sha256=file_sha(args.decoder_spec))
            strings=h5py.string_dtype('utf-8')
            for key in ('names','texts'):out.create_dataset(key,(480,),dtype=strings)
            out.create_dataset('lengths',(480,),dtype=np.int64)
            for key in ('motion1','motion2'):
                out.create_dataset(key,(480,300,262),dtype=np.float32,chunks=(1,300,262),compression='lzf')
            for i,request in enumerate(data.requests):
                name,text,gt1,gt2,length=data[i]
                assert name==request['identifier'] and length+1==request['raw_frames']
                group=archive[name];start=request['crop_start'];n=request['raw_frames']
                raw=torch.from_numpy(group['motion30'][start:start+n]).cuda();assert len(raw)==n
                beta=torch.from_numpy(group['betas'][:]).cuda();gender=torch.ones(1,2,dtype=torch.long,device='cuda')
                x=fk.from_archive(raw,beta);pad=(-n)%4;padding_counts[str(pad)]+=1
                padded=torch.cat([x,x[-1:].expand(pad,-1,-1)]) if pad else x
                pred,_,_=net(((padded.reshape(1,-1,276)-mean)/std),beta.reshape(1,20),gender)
                decoded=(pred[0,:n]*std+mean).reshape(n,23,12)
                generated=official(fk.to_archive(decoded,beta),beta,request)
                roundtrip=official(fk.to_archive(x,beta),beta,request)
                for p,g,key,reference in zip(generated,(gt1,gt2),('motion1','motion2'),roundtrip):
                    assert p.shape==(length,262) and np.isfinite(p).all()
                    delta=(reference-g[:length].numpy())*data.adapter.std
                    identity_position_max=max(identity_position_max,float(np.abs(delta[:,:66]).max()))
                    identity_other_max=max(identity_other_max,float(np.abs(delta[:,66:258]).max()))
                    ref_contact=reference[:,258:]*data.adapter.std[258:]+data.adapter.mean[258:]
                    gt_contact=g[:length,258:].numpy()*data.adapter.std[258:]+data.adapter.mean[258:]
                    identity_contact_mismatches+=int((np.abs(ref_contact-gt_contact)>.5).sum())
                    block=np.zeros((300,262),dtype=np.float32);block[:length]=p;out[key][i]=block
                # Conversion identity threshold is derived from the prior native
                # restoration audit, not a target floor/contact constraint.
                if identity_position_max>1e-4:raise ValueError('Native identity diverges from official GT coordinates')
                out['names'][i]=name;out['texts'][i]=text;out['lengths'][i]=length
                if (i+1)%96==0:
                    state.update(status='exporting',completed=i+1,identity_position_max_m=identity_position_max)
                    write(status,state);print(json.dumps(state),flush=True)
            assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in net.state_dict().items())
            for p,digest in hashes.items():assert file_sha(ROOT/p)==digest,'Source changed: '+p
            out.attrs['status']='complete'
        os.replace(temp,destination)
        validated=InterHumanGeneratedH5Dataset(destination,data.request_path,request_entry=data.manifest['request'],formal=False)
        assert len(validated)==480;validated.close()
        state.update(status='complete',ended=time.time(),samples=480,vq_step=spec['step'],vq_sha256=spec['sha256'],
            vq_selection_scope=spec['selection_scope'],request_content_sha256=data.payload['request_sha256'],
            h5=str(destination),h5_bytes=destination.stat().st_size,h5_sha256=file_sha(destination),
            sources=hashes,raw_padding_counts=padding_counts,identity_position_max_m=identity_position_max,
            identity_feature_66_258_max=identity_other_max,identity_contact_mismatches=identity_contact_mismatches,
            identity_contact_values=2*sum(r['processed_length'] for r in data.requests)*4,
            frozen_vq_unchanged=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
            decoder_override=decoder,
            shape_conditioning='Requested GT betas; disclose additional condition relative to unshaped VQs')
        write(status,state);print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
