from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM
import numpy as np
from tqdm import tqdm
import os
import torch

from optimization.human_body_prior.body_model.body_model import BodyModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
neutral_bm_path = 'body_models/smplx/SMPLX_NEUTRAL.npz'
male_bm_path = 'body_models/smplx/SMPLX_MALE.npz'
female_bm_path = 'body_models/smplx/SMPLX_FEMALE.npz'
neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=10)
male_bm = BodyModel(bm_fname=male_bm_path, num_betas=10)
female_bm = BodyModel(bm_fname=female_bm_path, num_betas=10)

def smplx_2_amass(dict):
    pose = dict['poses']  # [T, 72]
    trans = dict['trans']  # [T, 3]
    betas = dict['betas']
    gender = dict['gender']
    root_orient = pose[:, 0]
    pose_body = pose[:, 1:22]
    pose_lhand = pose[:, 25:40]
    pose_rhand = pose[:, 40:55]
    amass_dict = {
        'root_orient': root_orient,
        'pose_body': pose_body,
        'pose_lhand': pose_lhand,
        'pose_rhand': pose_rhand,
        'trans': trans,
        'betas': betas,
        'gender': gender,
    }
    return amass_dict

def _compute_geometry(params):
    body_pose = torch.from_numpy(params['pose_body']).float()
    left_hand_pose = torch.from_numpy(params['pose_lhand']).float()
    right_hand_pose = torch.from_numpy(params['pose_rhand']).float()
    root_transl = torch.from_numpy(params['trans']).float()
    global_orient = torch.from_numpy(params['root_orient']).float()
    betas = torch.from_numpy(params['betas']).float()
    frame_number = body_pose.shape[0]
    # jaw_pose = np.zeros((frame_number, 3))
    # leye_pose = np.zeros((frame_number, 3))
    # reye_pose = np.zeros((frame_number, 3))

    if params['gender'] == 'male':
        bm = male_bm  # gender is ignored for training/evaluation
    elif params['gender'] == 'female':
        bm = female_bm
    else:
        bm = neutral_bm

    body_pose = body_pose.reshape([frame_number, -1])
    hand_pose = torch.cat([left_hand_pose, right_hand_pose], dim=1).reshape(
        [frame_number, -1])
    betas = betas.repeat(frame_number, 1)
    # print('body_pose', body_pose.shape, 'hand_pose', hand_pose.shape,'global_orient', global_orient.shape, 'betas', betas.shape, 'root_transl', root_transl.shape)
    body = bm(pose_body=body_pose, pose_hand=hand_pose, root_orient=global_orient,
              betas=betas, trans=root_transl)  # betas is ignored for training/evaluation

    verts = body.v
    joints = body.Jtr
    faces = bm.f
    return verts, joints.cpu().numpy()

# rsults_dir = 'final_demo/intergen_results'
# out_put_dir = rsults_dir + '_ground'
# os.makedirs(out_put_dir, exist_ok=True)
#
# for file in tqdm(os.listdir(rsults_dir)):
#     data = np.load(os.path.join(rsults_dir, file), allow_pickle=True)
#     data = dict(data)
#     amass_data = smplx_2_amass(data)
#     _, joints = _compute_geometry(amass_data)
#     root_transl = data['trans']
#     root_transl[:, 1] = root_transl[:, 1] - np.min(joints[:, :, 1])  # min-value align to the ground
#     np.savez(os.path.join(out_put_dir, file),
#              poses=data['poses'], trans=root_transl, betas=data['betas'], gender=data['gender'],
#              mocap_framerate=30)


def calculate_contact_probs_numpy(
        indices_dir,
        contact_dir,
        codebook_size=12288,
        save_path='checkpoints/vq/code_contact_probs.npy'
):
    """
    Args:
        indices_dir: 存放 indices 的文件夹，文件格式 .npy, shape (T, 3)
        contact_dir: 存放 contact labels 的文件夹，文件格式 .npy, shape (T,)
        codebook_size: 码本大小 (例如 512 或 1024)
    """

    # 1. 获取文件列表并排序，确保一一对应
    indices_files = sorted([f for f in os.listdir(indices_dir) if f.endswith('.npy')])

    # 初始化计数器
    # shape: (3, codebook_size) -> (Body, Hand, Root) 三个通道独立统计
    total_counts = np.zeros((3, codebook_size), dtype=np.float64)
    contact_counts = np.zeros((3, codebook_size), dtype=np.float64)

    print(f"[INFO] 开始处理 {len(indices_files)} 个文件...")
    print(f"[INFO] Codebook Size: {codebook_size}, Latent Parts: 3")

    for f_name in tqdm(indices_files):
        # 2. 构建完整路径
        idx_path = os.path.join(indices_dir, f_name)
        cnt_path = os.path.join(contact_dir, f_name)

        # 检查对应的 label 文件是否存在
        if not os.path.exists(cnt_path):
            print(f"[WARNING] 缺少对应的 Contact Label: {f_name}, 跳过。")
            continue

        # 3. 加载数据
        # indices: (T, 3)
        # contact: (T,) or (T, 1)
        indices = np.load(idx_path).astype(np.int32)
        contact = np.load(cnt_path).astype(np.float32)

        # 展平 contact 以防是 (T, 1)
        if contact.ndim > 1:
            contact = contact.flatten()

        # 安全检查：长度对齐
        T = indices.shape[0]
        if contact.shape[0] != T:
            # 简单的截断或补齐策略 (通常预处理好的数据应该是一致的)
            min_len = min(T, contact.shape[0])
            indices = indices[:min_len]
            contact = contact[:min_len]

        # 4. 核心统计 (Vectorized Scatter Add)
        # 我们遍历 3 个部位 (Body, Hand, Root)
        for part_idx in range(3):
            # 取出当前部位的所有 code indices: shape (T,)
            part_codes = indices[:, part_idx]

            # 统计出现次数
            # equivalent to: total_counts[part_idx][code] += 1
            np.add.at(total_counts[part_idx], part_codes, 1)

            # 统计接触次数 (contact 为 0 或 1，也可以是概率)
            # equivalent to: contact_counts[part_idx][code] += contact[t]
            np.add.at(contact_counts[part_idx], part_codes, contact)

    # 5. 计算概率
    # 防止除以 0
    epsilon = 1e-8
    probs = contact_counts / (total_counts + epsilon)

    # 处理 Dead Codes (从未出现过的 Code)
    # 将其概率设为 0 (视为无交互) 或者 0.5 (视为不确定)
    # 这里设为 0，防止采样时误选
    probs[total_counts == 0] = -1.0
    # 如果你想标记死码，可以使用一个特殊值，例如 -1 或 0.5
    # probs[total_counts == 0] = -1.0  # 或者 0.5

    # 6. 保存结果
    # np.save(save_path, probs)
    print(f"[SUCCESS] 结果已保存至: {save_path}")
    print(f"[INFO] 输出形状: {probs.shape}")  # 应该是 (3, codebook_size)

    # 简单的统计打印
    for i, name in enumerate(['Body', 'Hand', 'Root']):
        high_contact = np.sum(probs[i] > 0.8)
        low_contact = np.sum(probs[i] < 0.2)
        dead_codes = np.sum(probs[i] == -1.0)
        print(f"  - {name} Codebook: 高接触Codes(>0.8)={high_contact}, 低接触Codes(<0.2)={low_contact}, 死码={dead_codes}")

from scipy import linalg
def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()
# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    # 请修改为你的实际路径
    # indices_path = "checkpoints/vq/codebook"
    # contact_path = "checkpoints/vq/contact_token_label"
    #
    # calculate_contact_probs_numpy(
    #     indices_dir=indices_path,
    #     contact_dir=contact_path,
    #     codebook_size=12288  # 或者是 512，取决于你的 VQVAE 设置
    # )
    # prob_path = "checkpoints/vq/code_contact_probs.npy"
    # # 统计不为0的概率数量
    # for part_idx, part_name in enumerate(['Body', 'Hand', 'Root']):
    #     probs = np.load(prob_path)[part_idx]
    #     non_zero_count = np.sum(probs > 0)
    #     print(f"{part_name} 部位的非零概率数量: {non_zero_count} / 12288")
    mm_list = []
    mm_folder = 'final_demo/mm'
    for file in os.listdir(mm_folder):
        data = np.load(os.path.join(mm_folder, file))
        mm_list.append(data)

    mm_all = np.concatenate(mm_list, axis=1)
    print(mm_all.shape)
    mm_num = calculate_multimodality(mm_all, 10)
    print(mm_num)
