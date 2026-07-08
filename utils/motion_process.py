import os

import torch
import torch.nn.functional as F
from utils.quaternion import quaternion_to_cont6d, qrot, qinv
from utils.rotation_conversions import *
import numpy as np

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def recover_interx(data):

    data = data.reshape(-1, 56, 12)
    pose1 = data[..., :-1, :6]
    trans1 = data[..., -1, :3].reshape(-1, 3).cpu().detach().numpy()
    pose2 = data[..., :-1, 6:12]
    trans2 = data[..., -1, 6:9].reshape(-1, 3).cpu().detach().numpy()
    pose1 = matrix_to_axis_angle(rotation_6d_to_matrix(pose1.reshape(-1, 55, 6))).cpu().detach().numpy()
    pose2 = matrix_to_axis_angle(rotation_6d_to_matrix(pose2.reshape(-1, 55, 6))).cpu().detach().numpy()

    return trans1, pose1, trans2, pose2


class MotionNormalizerTorchInterX():
    def __init__(self, motion_rep='vel'):
        if motion_rep == 'vel':
            mean = np.load("./dataset/interx/Interx_Mean_root.npy")
            std = np.load("./dataset/interx/Interx_Std_root.npy")
        else:
            mean = np.load("./dataset/interx/Interx_Mean.npy")
            std = np.load("./dataset/interx/Interx_Std.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x


class MotionNormalizerTorchChi3D():
    def __init__(self):
        mean = np.load("./dataset/chi3d/stats/Chi3D_Mean_root.npy")
        std = np.load("./dataset/chi3d/stats/Chi3D_Std_root.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()

    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x

class MotionNormalizerTorchInterXSingle():
    def __init__(self):
        mean = np.load("./dataset/interx/Mean.npy")
        std = np.load("./dataset/interx/Std.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x

def swap_left_right_hhi(raw_data):
    data = raw_data.copy()

    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    hhi_left_hand_chain = [37, 38, 39, 25, 26, 27, 28, 29, 30, 34, 35, 36, 31, 32, 33]
    hhi_right_hand_chain = [52, 53, 54, 40, 41, 42, 43, 44, 45, 49, 50, 51, 46, 47, 48]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, hhi_right_hand_chain]
        data[:, hhi_right_hand_chain] = data[:, hhi_left_hand_chain]
        data[:, hhi_left_hand_chain] = tmp

    scale_vec = np.array([1, -1, -1])
    data[:, :-1, :3] = data[:, :-1, :3] * scale_vec
    data[:, :-1, 3:] = data[:, :-1, 3:] * scale_vec
    data[:, -1, 0] *= -1
    data[:, -1, 3] *= -1

    return data


def prepare_root_features(pos_a, pos_b):
    """
    pos_a/b: (Batch, Frames, 3)
    返回: (Batch, Frames, 12) 的增强表达
    """
    # 计算全局速度 (填充首帧)
    vel_a = pos_a[1:, :] - pos_a[:-1, :]
    # 使用 torch.cat 和 torch.zeros 替换 numpy 实现
    vel_a = torch.cat([torch.zeros((1, 3), device=pos_a.device, dtype=pos_a.dtype), vel_a], dim=0)  # 首帧速度设为0
    vel_b = pos_b[1:, :] - pos_b[:-1, :]
    vel_b = torch.cat([torch.zeros((1, 3), device=pos_b.device, dtype=pos_b.dtype), vel_b], dim=0)  # 首帧速度设为0

    # 计算相对特征
    rel_pos = pos_b - pos_a
    rel_vel = vel_b - vel_a

    # 组合 12 维特征:
    # [vel_a(3), vel_b(3), rel_pos(3), rel_vel(3)]
    combined = torch.cat([vel_a, vel_b, rel_pos, rel_vel], dim=-1)

    return combined


def prepare_interx(data, swap=False):
    data = data.reshape(-1, 56, 12)
    pos_a = data[..., :6] if not swap else data[..., 6:]
    pos_b = data[..., 6:] if not swap else data[..., :6]
    root_feature = prepare_root_features(pos_a[:, -1, :3], pos_b[:, -1, :3])
    body_feature = torch.cat([pos_a[:, :-1, :], pos_b[:, :-1, :]], dim=-1)
    # np.newaxis 对应 None 或 .unsqueeze()，这里使用 None 保持简洁
    return torch.cat([body_feature, root_feature[:, None, :]], dim=1).reshape(-1, 56 * 12)


def recover_root_interx(root_features):
    """
    root_features: (Frames, 12)
    返回: pos_a, pos_b (Frames, 3) 的位置估计
    注意：由于速度是相对的，这个函数只能恢复出一个位置序列，另一个需要根据初始条件推断。
    这里我们假设 pos_a 的初始位置为0，并通过累积速度来恢复 pos_a，然后根据相对位置恢复 pos_b。
    """
    vel_a = root_features[:, :3]
    rel_pos = root_features[:, 6:9]

    # 累积速度恢复 pos_a
    pos_a = torch.cumsum(vel_a, dim=0)  # 从速度恢复位置，假设初始位置为0

    # 根据相对位置恢复 pos_b
    pos_b = pos_a + rel_pos

    return pos_a, pos_b


def recover_interx_from_features(features):
    """
    body_feature: (Batch, Frames, 55, 6)
    root_feature: (Batch, Frames, 12)
    返回: pos_a, pos_b (Batch, Frames, 3) 的位置估计
    """
    # 从 body_feature 中提取 pose 信息
    features = features.reshape(-1, 56, 12)
    interx_features = torch.zeros_like(features, device=features.device, dtype=features.dtype)
    pos_a, pos_b = recover_root_interx(features[:, -1, :])
    interx_features[:, :-1, :] = features[:, :-1, :]
    interx_features[:, -1, :3] = pos_a
    interx_features[:, -1, 6:9] = pos_b

    return interx_features.reshape(-1, 56 * 12)


def recover_root_interx_batch(root_features):
    """
    root_features: (Batch, Frames, 12)
    返回: pos_a, pos_b (Batch, Frames, 3)
    """
    vel_a = root_features[:, :, :3]
    rel_pos = root_features[:, :, 6:9]

    # 在时间维度 (dim=1) 进行累积求和，恢复 Batch 中每个样本的轨迹
    pos_a = torch.cumsum(vel_a, dim=1)

    # 根据相对位置恢复 pos_b
    pos_b = pos_a + rel_pos

    return pos_a, pos_b


def recover_interx_from_features_batch(features):
    """
    features: (Batch, Frames, 56 * 12) 或 (Batch * Frames, 56 * 12)
    """
    # 如果输入是平铺的 (N, 672)，需要先还原 Batch 和 Frames
    features = features.reshape(features.shape[0], features.shape[1], 56, 12)

    B, T, J, C = features.shape
    interx_features = torch.zeros_like(features, device=features.device, dtype=features.dtype)

    # 传入 (B, T, 12) 进行恢复
    pos_a, pos_b = recover_root_interx_batch(features[:, :, -1, :])

    # 填充 body 旋转信息
    interx_features[:, :, :-1, :] = features[:, :, :-1, :]

    # 填充恢复的 root 位置信息
    interx_features[:, :, -1, :3] = pos_a
    interx_features[:, :, -1, 6:9] = pos_b

    return interx_features.reshape(B, T, 56 * 12)
# def get_rollout_history_global(last_primitive, history_length, motion_normalizer):
#     """
#     last_primitive: [B, T, D] 上一段生成的动作，包含 Global Position
#     """
#     # 1. 取出最后一段历史长度的数据
#     # 假设 global position 在最后3维
#     raw_history = last_primitive[:, -history_length:, :]
#
#     # 2. 反归一化 (还原到真实米单位)
#     # 必须反归一化，因为直接在归一化空间做减法是错误的
#     raw_history_denorm = self.train_dataset.denormalize(raw_history)
#
#     # 3. 提取锚点 (Anchor)
#     # 通常取第一帧作为当前窗口的"局部原点"
#     # shape: [B, 1, 3]
#     anchor_pos = raw_history_denorm[:, 0:1, root_pos_idx]
#
#     # 如果还需要处理旋转，也要在这里提取 anchor_rot
#     # anchor_rot = raw_history_denorm[:, 0:1, root_rot_idx]
#
#     # 4. 制造"窗口相对"的全局坐标 (Window-Relative Global)
#     # 这一步是关键！让数据变成以0开始的轨迹
#     relative_history = raw_history_denorm.clone()
#     relative_history[..., root_pos_idx] = raw_history_denorm[..., root_pos_idx] - anchor_pos
#
#     model_input = self.train_dataset.normalize(relative_history)
#
#     return model_input, anchor_pos  # 返回 anchor 以便生成后加回去

def get_spatial_min_interaction_label_single(
        p1_pos,
        p2_pos,
        strict_thresh=0.1,  # 严格阈值：小于此距离直接视为真接触
        loose_thresh=0.3,  # 宽松阈值：允许“隔空交互”发生的最远距离
        spatial_tolerance=0.05,
        smooth_window=3,  # 空间容差：在全局最小值的基础上，允许放宽多少距离？
        mask=None
):
    """
    针对单样本 (Single Sequence) 的交互标注。
    基于“水位线”原理：以全局最小值为基准，向上浮动 spatial_tolerance 的范围均视为交互。

    Args:
        p1_pos: (T, J1, 3)
        p2_pos: (T, J2, 3)
    Returns:
        labels: (T,) 0或1
        min_distances: (T,) 每一帧的最短距离
    """
    T, J1, _ = p1_pos.shape
    _, J2, _ = p2_pos.shape

    diff = p1_pos.unsqueeze(2) - p2_pos.unsqueeze(1)
    dist_matrix = torch.norm(diff, dim=-1)  # (T, J1, J2)
    if mask is not None:
        dist_matrix = dist_matrix + mask.unsqueeze(0)

    frame_min_dists = dist_matrix.min(dim=2).values.min(dim=1).values  # (T,)

    # 初始化标签
    labels = torch.zeros(T, dtype=torch.long, device=p1_pos.device)

    # 2. 找到全局最小值 (谷底)
    global_min_val, global_min_idx = frame_min_dists.min(dim=0)
    global_min_val = global_min_val.item()

    # === 逻辑分支 ===

    # Case A: 发生了真接触 (Real Contact)
    # 哪怕只有一帧碰到了，我们认为这就是一个接触动作
    # 标注逻辑：凡是小于 strict_thresh 的都标，或者也应用 spatial 逻辑？
    # 通常真接触直接切阈值最准。
    if global_min_val < strict_thresh:
        labels = (frame_min_dists < strict_thresh).long()

    # Case B: 踢空气/隔空交互 (Near Miss / Air Interaction)
    # 全局最小值没达到 strict，但在 loose 范围内
    elif global_min_val < loose_thresh:
        # 核心逻辑：空间回溯 (Spatial Backtracking)
        # 定义动态水位线：最低点 + 容差
        # 比如最低是 0.25m，容差 0.05m，那么 [0.25, 0.30] 范围内的帧都算
        dynamic_limit = global_min_val + spatial_tolerance

        # 生成 Mask
        candidates = (frame_min_dists <= dynamic_limit)

        # [可选优化] 连通域过滤 (Connectivity Check)
        # 防止数据噪声导致 100 帧以外有个点稍微抖动了一下也满足条件
        # 我们只保留 包含全局最小值索引 的那一段连续区间

        # 简单做法：直接用 candidates.long()
        # 严谨做法：只取 global_min_idx 周围相连的 True
        # labels = filter_connected_component(candidates, global_min_idx)
        labels = candidates.long()

    # Case C: 离得太远 (No Interaction)
    else:
        # labels 保持全0
        pass

    if smooth_window > 1:
        # 使用中值滤波或简单的滑动窗口平均
        # 这里使用滑动窗口平均后重新取阈值作为平滑手段
        padding = smooth_window // 2
        labels_float = labels.float().view(1, 1, -1)
        smoothed = torch.nn.functional.avg_pool1d(labels_float, kernel_size=smooth_window, stride=1, padding=padding)
        labels = (smoothed > 0.5).long().view(-1)[:labels.shape[0]]  # 超过一半的帧判定为靠近，才标为1

    return labels, frame_min_dists


def keep_longest_and_ramp(labels, ramp_len=3):
    """
    只保留最长接触段，并在段内应用梯形/三角形权重。
    段外权重严格为 0。
    """
    # 1. 找最长段
    # Pad是为了处理边界，diff=1是起点，-1是终点
    padded = F.pad(labels.float(), (1, 1))
    diff = padded[1:] - padded[:-1]
    starts, ends = (diff == 1).nonzero(), (diff == -1).nonzero()

    if len(starts) == 0: return torch.zeros_like(labels, dtype=torch.float)

    # 获取最长段的起止点 [s, e)
    idx = torch.argmax(ends - starts)
    s, e = starts[idx].item(), ends[idx].item()

    # 2. 生成段内权重 (Trapezoidal/Triangular Ramp)
    weights = torch.zeros_like(labels, dtype=torch.float)

    # 局部时间索引: [0, 1, 2, ... len-1]
    t = torch.arange(e - s, device=labels.device).float()

    # 计算到最近边缘的距离: min(距离起点, 距离终点)
    # 例如 len=5 -> [0, 1, 2, 1, 0]
    dists = torch.minimum(t, (e - s - 1) - t)

    # 归一化并截断: (dist + 1) / ramp_len
    # +1 是为了保证边缘帧也有权重 (如 0.1) 而不是 0
    # 如果段长够长，中间会 saturate 到 1.0 (梯形)
    # 如果段长不够，中间升不到 1.0 就降下来了 (三角形，保持对称)
    ramp = torch.clamp((dists + 1) / ramp_len, max=1.0)

    weights[s:e] = ramp
    return weights
