"""Explicit immutable VQ specifications for new HHC cloud rebuilds."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from models.vqvae_interhuman_smpl_raw import VQSMPLRaw


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def normalization_sha(mean,std):
    return hashlib.sha256(mean.cpu().contiguous().numpy().tobytes()+std.cpu().contiguous().numpy().tobytes()).hexdigest()


def load_frozen_vq(spec_path,device,data_manifest):
    spec=json.loads(Path(spec_path).read_text())
    assert spec['schema']=='hhc_frozen_cloud_vq_v1' and spec['origin']=='fresh_cloud_rebuild_not_historical185k'
    assert spec['selection_scope'] in ('completed_training_val_best','intermediate_val_selected_pilot')
    assert isinstance(spec['step'],int) and spec['step']>0,'An untrained recovery is not a trained tokenizer'
    assert file_sha(spec['selected_validation'])==spec['selected_validation_sha256']
    validation=json.loads(Path(spec['selected_validation']).read_text())
    assert validation['samples']==480 and validation['protocol']=='fixed_val_first480_center_max300_raw_smpl_v1'
    assert torch.isfinite(torch.tensor([validation['fid'],validation['body_mpjpe_mm']])).all()
    checkpoint=Path(spec['checkpoint'])
    assert checkpoint.name=='best_fid.pth' and checkpoint.stat().st_size==spec['bytes']
    assert file_sha(checkpoint)==spec['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==spec['step']
    assert saved['manifest']['contract_sha256']==spec['training_contract_sha256']
    contract=saved['manifest']['contract']
    assert contract['origin']==spec['origin'] and contract['dataset_manifest_sha256']==file_sha(data_manifest)
    assert contract['codebook_size']==12288 and contract['latent_dims']==[96,64]
    mean,std=saved['mean'],saved['std']
    assert mean.dtype==std.dtype==torch.float32 and mean.shape==std.shape==(276,)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
    assert normalization_sha(mean,std)==spec['normalization_sha256']
    net=VQSMPLRaw(SimpleNamespace(nb_code=12288,mu=.99))
    net.load_state_dict(saved['net'],strict=True)
    net.requires_grad_(False).eval().to(device)
    return net,mean.to(device),std.to(device),spec
