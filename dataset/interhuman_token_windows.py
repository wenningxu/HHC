"""Single-factor token-window policy for native HHC MGT ablations."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class InterHumanTokenWindows(Dataset):
    def __init__(self,base,policy='historical'):
        if policy not in ('historical','uniform'):raise ValueError(policy)
        self.base=base;self.policy=policy
        self.rows=base.rows;self.base_seed=base.base_seed;self.records=base.records

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        if self.policy=='historical':return self.base[index]
        i,seed=index;row=self.rows[i];tokens=row['tokens']
        # Preserve the original caption and first/last-token augmentation RNGs.
        py=random.Random(seed);rng=np.random.default_rng(seed)
        text=py.choice(row['texts'])
        if rng.choice([False,False,True]) and len(tokens)>1:
            tokens=tokens[:-1] if rng.choice([True,False]) else tokens[1:]
        start=0
        if len(tokens)>75:
            window_rng=np.random.default_rng(np.random.SeedSequence([int(seed),20261016,75]))
            start=int(window_rng.integers(len(tokens)-75+1))
        tokens=tokens[start:start+75]
        target=torch.full((76,2),12289,dtype=torch.long)
        target[:len(tokens)]=tokens;target[len(tokens)]=12288
        return target,len(tokens),text,row['beta']
