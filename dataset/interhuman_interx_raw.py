"""In-memory view of archived SMPL, using the original Inter-X raw slots."""
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.interhuman_smpl_pair import SCHEMA


class InterHumanInterXRaw(Dataset):
    def __init__(self, root, split, fk, window=64, stats=None):
        self.root = Path(root)
        self.window = window
        self.ids,self.motions,self.betas = [],[],[]
        with h5py.File(self.root/'motions.h5','r') as h:
            assert h.attrs['schema'] == SCHEMA and h.attrs['status'] == 'complete'
            for name in (self.root/'split'/f'{split}.txt').read_text().splitlines():
                if name not in h or 'motion30' not in h[name]:
                    continue
                motion = torch.from_numpy(h[name]['motion30'][:])
                if len(motion) < (window if window else 16):
                    continue
                beta = torch.from_numpy(h[name]['betas'][:]).float()
                with torch.no_grad():
                    motion = fk.from_archive(motion,beta).reshape(-1,276)
                self.ids.append(name)
                self.motions.append(motion)
                self.betas.append(beta.reshape(20))
        if stats is None:
            assert split == 'train'
            count = sum(len(x) for x in self.motions)
            mean = sum(x.double().sum(0) for x in self.motions)/count
            variance = sum(x.double().square().sum(0) for x in self.motions)/count-mean.square()
            std = variance.clamp_min(0).sqrt()
            std[std<1e-6] = 1
            self.mean,self.std = mean.float(),std.float()
        else:
            self.mean,self.std = stats
        # Match the historical Inter-X window enumeration, including its
        # omitted final start and duplicated first start; don't change sampler.
        self.cumsum = np.cumsum([0]+[max(0,len(x)-window) for x in self.motions]) if window else None

    def __len__(self):
        return int(self.cumsum[-1]) if self.window else len(self.motions)

    def __getitem__(self,index):
        if self.window:
            motion_id = int(np.searchsorted(self.cumsum,index)-1) if index else 0
            start = int(index-self.cumsum[motion_id]-1) if index else 0
            motion = self.motions[motion_id][start:start+self.window]
        else:
            motion_id,start = index,0
            motion = self.motions[index]
        return (motion-self.mean)/self.std,self.betas[motion_id],torch.ones(2,dtype=torch.long)
