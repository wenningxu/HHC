import os

import numpy as np
import torch


def load_chi3d_group_skeletons(base_dir="dataset/chi3d/betas_group", groups=(2, 3, 4)):
    p1_list = []
    p2_list = []
    for group in groups:
        arr = np.load(os.path.join(base_dir, f"{group}.npy"))
        p1_list.append(arr[0])
        p2_list.append(arr[1])
    p1 = torch.from_numpy(np.stack(p1_list, axis=0)).float()
    p2 = torch.from_numpy(np.stack(p2_list, axis=0)).float()
    return p1, p2
