import smplx
from VolumetricSMPL import attach_volume
from optimize.optimize_full_body import smplx_2_amass
import numpy as np
import os
import torch
import codecs as cs
from tqdm import tqdm
from optimization.losses import DyadicInteractionLoss, vertex_normals
from optimization.text_2_regions import ContactExtractor
from utils.paramUtil import CANDIDATES
from optimization.human_body_prior.body_model.body_model import BodyModel
import pickle as pkl

def create_init_dict(data_dict):
    """
    创建用于正则化的初始状态字典。
    1. 将 numpy 数组转为 tensor，转 float32，上 GPU，去梯度。
    2. 特殊处理 'gender' 字段，确保它是 Python 原生字符串。
    """
    init_dict = {}

    # 定义哪些键是需要参与计算的数值型数据
    tensor_keys = ['pose_body', 'pose_lhand', 'pose_rhand', 'root_orient', 'trans', 'betas']

    for k, v in data_dict.items():
        # --- 情况 A: 数值型数据 (Pose, Trans, Betas) ---
        if k in tensor_keys:
            data = v
            # 1. 如果是 numpy，转 tensor
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # 2. 如果已经是 tensor (或刚转完)，处理类型和设备
            if isinstance(data, torch.Tensor):
                # .float() 强转 float32
                # .to(device) 上 GPU
                # .detach() 确保不带梯度（作为参考值）
                init_dict[k] = data.clone().float().to(device).detach()

        # --- 情况 B: 字符串数据 (Gender) ---
        elif k == 'gender':
            # .npz 加载出来的字符串通常比较怪，可能是 numpy.str_, bytes, 或者数组
            val = v

            # 1. 如果是 numpy 数组 (例如 array('male', dtype='<U4'))
            if isinstance(val, np.ndarray):
                # 0维数组 (scalar) 直接取 item
                if val.ndim == 0:
                    val = val.item()
                # 1维数组 取第一个元素
                elif val.size > 0:
                    val = val[0]

            # 2. 如果是 bytes (例如 b'male')，解码
            if isinstance(val, (bytes, np.bytes_)):
                val = val.decode('utf-8')

            # 3. 强制转为 Python 原生 str
            init_dict[k] = str(val)

    return init_dict


def get_rid_to_parts_mapping(body_model, rid_to_vid_dict):
    """
    从外部提取 RID 与 VolumetricSMPL 零件索引的对应关系
    :param body_model: 已经 attach_volume 后的 SMPL-X 模型实例
    :param rid_to_vid_dict: 你的语义区域到顶点索引的字典 {rid_int: tensor([vids])}
    :return: {rid_int: [part_index_1, part_index_2, ...]}
    """
    # 1. 访问内部的 partitioner (通常在 .volume 属性下)
    # 根据你的 Traceback，VolumetricSMPL 实例在 model.volume
    vol_module = body_model.volume if hasattr(body_model, 'volume') else body_model
    partitioner = vol_module.partitioner

    # 2. 获取零件的顶点选择器 [num_parts, max_verts_per_part]
    # 注意：这个 tensor 通常是填充过的 (padded)
    tight_vert_selector = partitioner.tight_vert_selector
    num_parts = tight_vert_selector.shape[0]

    # 3. 建立顶点到零件的快速反向查找表
    # 考虑到一个顶点理论上只属于一个 tight part
    vid_to_part = {}
    for p_idx in range(num_parts):
        vids = tight_vert_selector[p_idx].unique()  # 去掉 padding 重复值
        for v in vids.tolist():
            vid_to_part[v] = p_idx

    # 4. 计算每个 RID 覆盖了哪些零件
    rid_to_parts = {}
    for rid, vid_tensor in rid_to_vid_dict.items():
        v_list = vid_tensor.tolist()

        # 找出该区域所有顶点所属的零件索引，并去重
        matched_parts = set()
        for v in v_list:
            if v in vid_to_part:
                matched_parts.add(vid_to_part[v])

        # 排序后存储
        rid_to_parts[rid] = sorted(list(matched_parts))

        # 科研打印：确认覆盖情况
        # print(f"RID {rid}: Covers {len(v_list)} verts -> Parts {rid_to_parts[rid]}")

    return rid_to_parts
# Create a SMPL body and extend it with volumetric functionalities
# Supports SMPL, SMPLH, and SMPL-X
if __name__ == "__main__":
    SMPL_MODEL_DIR = 'body_models/'
    data_folder = 'final_demo/intergen_results_ground'
    val_file = 'dataset/interx/splits/val.txt'
    text_dir = 'dataset/interx/texts_processed'
    contact_pairs_dir = 'dataset/interx/contact_pairs'
    pairs = os.listdir(contact_pairs_dir)

    device = 'cuda:0'
    # body_model_male = smplx.create(SMPL_MODEL_DIR, "smplx", gender='male', use_pca=False).to(device)
    # body_model_female = smplx.create(SMPL_MODEL_DIR, "smplx", gender='female', use_pca=False).to(device)
    #
    # body_model_male = attach_volume(body_model_male, pretrained=True, device=device)
    # body_model_female = attach_volume(body_model_female, pretrained=True, device=device)
    # smpl_all = {'male': body_model_male, 'female': body_model_female}
    loss_module = DyadicInteractionLoss(device=device).to(device)
    id_list = []
    collision_all = []
    contact_scores = []

    with cs.open(val_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())

    MY_API_KEY = "AIzaSyBasqVApLHbY8tRevAnU9EnraMJvpBuW0s"

    # 1. 实例化
    # text_list[0] = 'Two people wrestle with their hands, sitting face to face. The first person firmly grips the other persons left hand with his left hand. '
    extractor = ContactExtractor(api_key=MY_API_KEY)

    for name in tqdm(id_list[:135]):
        try:
            data_p1 = np.load(os.path.join(data_folder, name+'_pred_p1.npz'), allow_pickle=True)
            data_p2 = np.load(os.path.join(data_folder, name+'_pred_p2.npz'), allow_pickle=True)
            data_p1 = dict(data_p1)
            data_p2 = dict(data_p2)
        except:
            print(f'no file for {name}')
            continue
        data_p1 = dict(data_p1)
        data_p2 = dict(data_p2)

        if len(data_p1.keys()) < 6:
            data_p1 = smplx_2_amass(data_p1)
            data_p1 = create_init_dict(data_p1)
        if len(data_p2.keys()) < 6:
            data_p2 = smplx_2_amass(data_p2)
            data_p2 = create_init_dict(data_p2)

        text_list = []
        with cs.open(os.path.join(text_dir, name + '.txt')) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_list.append(caption)
        # print(len(data_p1.keys()))
        if name + '.pkl' not in pairs:
            contacts = extractor.get_contacts(text_list[0], CANDIDATES)
            save_path = os.path.join(contact_pairs_dir, name + '.pkl')
            with open(save_path, 'wb') as f:
                pkl.dump(contacts, f)
        else:
            with open(os.path.join(contact_pairs_dir, name + '.pkl'), 'rb') as f:
                contacts = pkl.load(f)

        contact_score, pene = loss_module.forward_semantic_metric(data_p1, data_p2, contacts)
        contact_scores.append(contact_score.cpu().item())
        collision_all.append(pene.cpu().item())

    print('collision_all', np.array(collision_all).mean())
    print('contact_scores', np.array(contact_scores).mean())