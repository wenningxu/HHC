import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import torch
import numpy as np
from tools.test_data import _compute_geometry, smplx_2_amass
import codecs as cs
from train.train_t2i_trans import MaskGITLightning
from os.path import join as pjoin
import os
from time import time
from tqdm import tqdm

if __name__ == '__main__':
    # Load the Lightning checkpoint by constructing the model from its saved
    # hyper-parameters and applying the state_dict manually. This avoids
    # MaskGITLightning.load_from_checkpoint, which resolves the class via the
    # pickled module path (train_t2i_trans.MaskGITLightning) that no longer
    # matches after the module moved into the train/ subpackage.
    ckpt_path = 'output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt'
    device = 'cuda:1'
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('hyper_parameters') or ckpt.get('hparams')
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    # Build the model structure without triggering __init__'s own resume
    # loading — the full trained weights are restored from this ckpt's
    # state_dict below, so the standalone resume_pth/resume_trans paths saved
    # in the hyper-parameters are not needed (and may not exist locally).
    args.resume_pth = None
    args.resume_trans = None
    if not getattr(args, 'device', None):
        args.device = device
    model = MaskGITLightning(args, eval_wrapper=None, val_loader=None)
    # Load the full Lightning state_dict (net *, trans_encoder *, clip_model *)
    # to restore the exact trained weights, matching what load_from_checkpoint did.
    state_dict = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'checkpoint loaded with {len(missing)} missing and {len(unexpected)} unexpected keys')
    model = model.to(device)
    model.eval()
    text_folder = 'dataset/interx/texts_processed'
    betas_folder = 'dataset/interx/betas'
    generate_name = 'generate/generate.txt'
    id_list = []
    with cs.open(generate_name, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())

    for name in id_list:
        beta_data = np.load(pjoin(betas_folder, name + '.npz'), allow_pickle=True)
        betas = beta_data['betas'].reshape(1, -1)
        gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
        gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]
        gender_codes = torch.tensor(gender_codes).unsqueeze(0)
        with cs.open(os.path.join(text_folder, name + '.txt')) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]

        model.generate([caption], torch.from_numpy(betas), gender_codes, [150], 'generate/gen_results', [name])

    # print(f"Total generation time for {len(id_list)} samples: {end_time - start_time:.2f} seconds")

    # for file in tqdm(os.listdir('generate/gen_results')):
    #     data = np.load(os.path.join('generate/gen_results', file), allow_pickle=True)
    #     data = dict(data)
    #     amass_data = smplx_2_amass(data)
    #     _, joints = _compute_geometry(amass_data)
    #     root_transl = data['trans']
    #     root_transl[:, 1] = root_transl[:, 1] - np.min(joints[:, :, 1])  # min-value align to the ground
    #     np.savez(os.path.join('generate/gen_results_ground', file),
    #              poses=data['poses'], trans=root_transl, betas=data['betas'], gender=data['gender'],
    #              mocap_framerate=30)

