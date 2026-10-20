"""CPU audit against the real historical TokenData; no trained model or GPU."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.interhuman_smpl_mgt import TokenData
from dataset.interhuman_token_windows import InterHumanTokenWindows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=[{'name':str(n),'tokens':torch.arange(n)[:,None].expand(-1,2),
           'texts':['caption A','caption B'],'beta':torch.arange(20).float()} for n in (7,75,76,180)]
    base=TokenData(rows);historical=InterHumanTokenWindows(base);uniform=InterHumanTokenWindows(base,'uniform')
    starts=set();saw_tail=False
    for seed in range(1024):
        for i in range(len(rows)):
            original=base[(i,seed)];same=historical[(i,seed)];changed=uniform[(i,seed)]
            assert torch.equal(original[0],same[0]) and original[1:3]==same[1:3] and torch.equal(original[3],same[3])
            assert changed[1:3]==original[1:3] and torch.equal(changed[3],original[3])
            assert torch.equal(changed[0],uniform[(i,seed)][0])
            assert torch.equal(changed[0][changed[1]],torch.tensor([12288,12288]))
            if i<2:assert torch.equal(changed[0],original[0])
            if i==3:
                starts.add(int(changed[0][0,0]));saw_tail|=bool((changed[0][:75]>75).any())
    assert saw_tail and min(starts)==0 and max(starts)>=104
    report={'status':'passed','device':'cpu','trained_checkpoint_loaded':False,'seeds':1024,
        'historical_exactly_matches_real_TokenData':True,'short_sequences_unchanged':True,
        'caption_length_beta_and_eos_preserved':True,'restart_deterministic':True,
        'long_window_distinct_starts':len(starts),'start_min':min(starts),'start_max':max(starts),
        'training_quality_test':False,'notes':'Same frozen token rows; only long-window position differs. Does not recover excluded short/long source sequences.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
