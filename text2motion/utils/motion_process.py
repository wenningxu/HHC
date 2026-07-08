import torch
from common.quaternion import quaternion_to_cont6d, qrot, qinv
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
    def __init__(self):
        mean = np.load("../dataset/interx/Interx_Mean.npy")
        std = np.load("../dataset/interx//Interx_Std.npy")

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