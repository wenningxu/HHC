"""Load a separately identified decoder pilot without altering its base VQ."""
import json
from pathlib import Path
import torch
from utils.hhc_frozen_vq import file_sha


def load_decoder_override(net,spec_path,base_spec):
    value=json.loads(Path(spec_path).read_text())
    assert value['schema']=='hhc_decoder_only_override_v1'
    assert value['base_vq_sha256']==base_spec['sha256'] and value['base_vq_step']==base_spec['step']
    checkpoint=Path(value['checkpoint'])
    assert checkpoint.stat().st_size==value['bytes'] and file_sha(checkpoint)==value['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['base_vq_sha256']==base_spec['sha256']
    assert saved['updates']==value['decoder_updates']==1000 and saved['arm']==value['arm']
    assert saved['contract_sha256']==value['contract_sha256']
    before={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('decoder.')}
    net.decoder.load_state_dict(saved['decoder'],strict=True)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in net.state_dict().items() if k in before)
    net.requires_grad_(False).eval()
    return value
