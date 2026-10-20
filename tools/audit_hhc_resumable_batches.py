import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hhc_resumable_batches import ResumableWindowBatches

def main():
    checks = {}
    n,b,seed = 1031,256,123
    a=ResumableWindowBatches(n,b,seed); full=[]
    for _ in range((n//b)*3): full.append(a.draw())
    checks['drop_last_and_no_repeat'] = all(len(x)==b for x in full) and all(len(set(x))==b for x in full[:n//b])
    checks['tail_dropped'] = set(sum(full[:n//b],[])).issubset(set(range(n))) and len(set(sum(full[:n//b],[])))== (n//b)*b
    x=ResumableWindowBatches(n,b,seed); prefix=[x.draw() for _ in range(5)]; state=x.state_dict(); y=ResumableWindowBatches(n,b,seed); y.load_state_dict(state)
    checks['interrupted_restored_next_batches'] = [x.draw() for _ in range(12)] == [y.draw() for _ in range(12)]
    gstate=torch.get_rng_state(); z=ResumableWindowBatches(n,b,seed); z.draw(); checks['global_torch_rng_unchanged']=torch.equal(gstate,torch.get_rng_state())
    checks['cross_epoch'] = len(set(sum([a.draw() for _ in range(n//b)],[]))) == (n//b)*b
    errors=0
    for bad in ({'schema':'x'}, {'schema':x.SCHEMA,'size':n+1,'batch':b,'seed':seed,'epoch':0,'cursor':0,'permutation':None,'generator_state':torch.Generator().get_state()}, {'schema':x.SCHEMA,'size':n,'batch':b,'seed':seed,'epoch':0,'cursor':1,'permutation':None,'generator_state':torch.Generator().get_state()}):
        try: y.load_state_dict(bad)
        except ValueError: errors += 1
    checks['invalid_contracts_rejected'] = errors == 3
    out={'schema':'hhc_resumable_batches_audit_v1','parameters':{'dataset_size':n,'batch_size':b,'seed':seed},'checks':checks,'all_pass':all(checks.values()),'note':'Batch order is deterministic/resumable and differs from historical RandomSampler order; it preserves random-permutation distribution, not bitwise full-training equivalence.'}
    path=Path('output/hhc_sota_20261016/resumable_batches_audit.json'); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
