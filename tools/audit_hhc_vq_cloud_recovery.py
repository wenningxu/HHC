"""CPU-only check of an immutable cloud recovery hardlink; no downloads."""
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.hhc_frozen_vq import file_sha
from utils.hhc_resumable_batches import ResumableWindowBatches


def main():
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud recovery remains cloud only')
    torch.set_num_threads(2)
    run=ROOT/'output/vq_hhc_native_rebuild_seed123_20261016'
    control=ROOT/'output/hhc_vq_rebuild_control_20261016'
    snapshot=control/f'recovery_audit_{os.getpid()}.pth';os.link(run/'latest.pth',snapshot)
    saved=torch.load(snapshot,map_location='cpu',weights_only=False)
    contract=json.loads((run/'training_contract.json').read_text())
    import hashlib
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert saved['contract_sha256']==contract_sha
    assert all(torch.isfinite(v).all() for v in saved['net'].values() if v.is_floating_point())
    optimizer=saved['optimizer']['state'];assert optimizer
    steps=[]
    for values in optimizer.values():
        for value in values.values():
            if torch.is_tensor(value):assert torch.isfinite(value).all()
        steps.append(int(values['step']))
    assert min(steps)==max(steps)==saved['update']
    assert saved['step']==max(0,saved['update']-999)
    for values in saved['quantizer_training_state'].values():
        assert values['init']
        for key in ('code_sum','code_count','usage'):assert torch.isfinite(values[key]).all()
    order=saved['batch_order'];batches=ResumableWindowBatches(order['size'],order['batch'],order['seed']);batches.load_state_dict(order)
    expected=batches.draw();again=ResumableWindowBatches(order['size'],order['batch'],order['seed']);again.load_state_dict(order)
    assert expected==again.draw()
    result={'status':'passed','device':'cpu','update':saved['update'],'main_step':saved['step'],
        'recovery_bytes':snapshot.stat().st_size,'recovery_sha256':file_sha(snapshot),
        'model_tensors':len(saved['net']),'optimizer_states':len(optimizer),'optimizer_steps_all_match_update':True,
        'model_optimizer_ema_finite':True,'batch_epoch':order['epoch'],'batch_cursor':order['cursor'],
        'next_batch_from_saved_order_repeatable':True,'contract_sha256':contract_sha,
        'checkpoint_downloaded':False,'is_quality_evaluation':False}
    (control/'recovery_audit.json').write_text(json.dumps(result,indent=2)+'\n');snapshot.unlink();print(json.dumps(result))


if __name__=='__main__':main()
