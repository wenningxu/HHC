import torch
import numpy as np
from test_data import _compute_geometry, smplx_2_amass
import codecs as cs
from train_t2i_trans import MaskGITLightning
from os.path import join as pjoin
import os
from time import time
from tqdm import tqdm

if __name__ == '__main__':
    model = MaskGITLightning.load_from_checkpoint(
        'output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt', eval_wrapper=None, val_loader=None, map_location="cuda:1")
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

