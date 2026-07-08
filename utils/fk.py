"""Based on Daniel Holden code from:
   A Deep Learning Framework for Character Motion Synthesis and Editing
   (http://www.ipab.inf.ed.ac.uk/cgvu/motionsynthesis.pdf)
"""

import os

import numpy as np
import torch
import torch.nn as nn
from utils.rotation_conversions import euler_angles_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix


class ForwardKinematicsLayer(nn.Module):
    """ Forward Kinematics Layer Class """

    def __init__(self, args=None, parents=None, positions=None, device=None):
        super().__init__()
        self.b_idxs = None
        if device is None:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if parents is None and positions is None:
            # Load SMPL skeleton (their joint order is different from the one we use for bvh export)
            smpl_fname = os.path.join(args.smpl.smpl_body_model, args.data.gender, 'model.npz')
            smpl_data = np.load(smpl_fname, encoding='latin1')
            self.parents = torch.from_numpy(smpl_data['kintree_table'][0].astype(np.int32)).to(self.device)
            self.parents = self.parents.long()
            self.positions = torch.from_numpy(smpl_data['J'].astype(np.float32)).to(self.device)
            self.positions[1:] -= self.positions[self.parents[1:]]
        else:
            self.parents = torch.from_numpy(parents).to(self.device)
            self.parents = self.parents.long()
            self.positions = torch.from_numpy(positions).to(self.device)
            self.positions = self.positions.float()
        self.positions[0] = 0

    def rotate(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def identity_rotation(self, rotations):
        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).to(self.device)
        diagonal = torch.reshape(
            diagonal, torch.Size([1] * len(rotations.shape[:2]) + [4, 4]))
        ts = diagonal.repeat(rotations.shape[:2] + torch.Size([1, 1]))
        return ts

    def make_fast_rotation_matrices(self, positions, rotations):
        if len(rotations.shape) == 4 and rotations.shape[-2:] == torch.Size([3, 3]):
            rot_matrices = rotations
        elif rotations.shape[-1] == 3:
            rot_matrices = euler_angles_to_matrix(rotations, convention='XYZ')
        elif rotations.shape[-1] == 4:
            rot_matrices = quaternion_to_matrix(rotations)
        elif rotations.shape[-1] == 6:
            rot_matrices = rotation_6d_to_matrix(rotations)
        else:
            raise NotImplementedError(f'Unimplemented rotation representation in FK layer, shape of {rotations.shape}')

        rot_matrices = torch.cat([rot_matrices, positions[..., None]], dim=-1)
        zeros = torch.zeros(rot_matrices.shape[:-2] + torch.Size([1, 3])).to(self.device)
        ones = torch.ones(rot_matrices.shape[:-2] + torch.Size([1, 1])).to(self.device)
        zerosones = torch.cat([zeros, ones], dim=-1)
        rot_matrices = torch.cat([rot_matrices, zerosones], dim=-2)
        return rot_matrices

    def rotate_global(self, parents, positions, rotations):
        locals = self.make_fast_rotation_matrices(positions, rotations)
        globals = self.identity_rotation(rotations)

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        b_size = positions.shape[0]
        if self.b_idxs is None:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)
        elif self.b_idxs.shape[-1] != b_size:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)

        for i in range(1, positions.shape[1]):
            globals[:, i] = self.rotate(
                globals[self.b_idxs, parents[i]], locals[:, i])

        return globals

    def get_tpose_joints(self, offsets, parents):
        num_joints = len(parents)
        joints = [offsets[:, 0]]
        for j in range(1, len(parents)):
            joints.append(joints[parents[j]] + offsets[:, j])

        return torch.stack(joints, dim=1)

    def canonical_to_local(self, canonical_xform, global_orient=None):
        """
        Args:
            canonical_xform: (B, J, 3, 3)
            global_orient: (B, 3, 3)

        Returns:
            local_xform: (B, J, 3, 3)
        """
        local_xform = torch.zeros_like(canonical_xform)

        if global_orient is None:
            global_xform = canonical_xform
        else:
            global_xform = torch.matmul(global_orient.unsqueeze(1), canonical_xform)
        for i in range(global_xform.shape[1]):
            if i == 0:
                local_xform[:, i] = global_xform[:, i]
            else:
                local_xform[:, i] = torch.bmm(torch.linalg.inv(global_xform[:, self.parents[i]]), global_xform[:, i])

        return local_xform

    def global_to_local(self, global_xform):
        """
        Args:
            global_xform: (B, J, 3, 3)

        Returns:
            local_xform: (B, J, 3, 3)
        """
        local_xform = torch.zeros_like(global_xform)

        for i in range(global_xform.shape[1]):
            if i == 0:
                local_xform[:, i] = global_xform[:, i]
            else:
                local_xform[:, i] = torch.bmm(torch.linalg.inv(global_xform[:, self.parents[i]]), global_xform[:, i])

        return local_xform

    def forward(self, rotations, positions=None):
        """
        Args:
            rotations (B, J, D)

        Returns:
            The global position of each joint after FK (B, J, 3)
        """
        # Get the full transform with rotations for skinning
        b_size = rotations.shape[0]
        if positions is None:
            positions = self.positions.repeat(b_size, 1, 1)
        transforms = self.rotate_global(self.parents, positions, rotations)
        coordinates = transforms[:, :, :3, 3] / transforms[:, :, 3:, 3]

        return coordinates, transforms


class ForwardKinematicsLayerBatch(nn.Module):
    """
    Forward Kinematics Layer Class
    【【!! 已修改以支持 (B, T, J, D) 的时序批次输入 !!】】
    """

    def __init__(self, parents, device=None):
        # 【【!! 高亮：修改 !!】】
        # 简化 __init__。我们不再需要 args, gender, positions 等。
        # 这个类现在只负责一件事：根据固定的拓扑(parents)进行FK计算。
        # 骨骼长度 (positions) 将在 forward 时作为参数动态传入。
        super().__init__()

        if device is None:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 只存储固定的拓扑结构
        self.parents = torch.from_numpy(parents).to(self.device).long()
        self.num_joints = self.parents.shape[0]

    def rotate(self, t0s, t1s):
        # 这个方法是通用的矩阵乘法，不需要修改
        return torch.matmul(t0s, t1s)

    def identity_rotation(self, rotations):
        # 【【!! 高亮：修改 !!】】
        # 为什么修改?
        # 原始代码假设 rotations.shape[:2] 是 (B,)，
        # 但现在 rotations.shape[:3] 是 (B, T, J)。
        # 我们需要创建一个 (B, T, J, 4, 4) 的单位矩阵批次。

        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).to(self.device)

        # 原始代码:
        # diagonal = torch.reshape(
        #     diagonal, torch.Size([1] * len(rotations.shape[:2]) + [4, 4]))
        # ts = diagonal.repeat(rotations.shape[:2] + torch.Size([1, 1]))

        # 修正后:
        # 1. 创建 (1, 1, 1, 4, 4) 的基础矩阵
        diagonal = torch.reshape(
            diagonal, torch.Size([1] * len(rotations.shape[:3]) + [4, 4]))

        # 2. repeat 以匹配 (B, T, J)
        # rotations.shape[:3] 是 (B, T, J)
        ts = diagonal.repeat(rotations.shape[:3] + torch.Size([1, 1]))
        # ts 的形状现在是 (B, T, J, 4, 4)
        return ts

    def make_fast_rotation_matrices(self, positions, rotations):
        # 【【!! 无需修改 !!】】
        # 为什么?
        # 这个函数内部的索引 (如 [..., None], dim=-1, shape[:-2])
        # 都是“相对”的，它们能自动处理 (B, J, ...) 和 (B, T, J, ...)
        # 两种情况，写得非常好。

        if len(rotations.shape) == 5 and rotations.shape[-2:] == torch.Size([3, 3]):  # 适配(B,T,J,3,3)
            rot_matrices = rotations
        elif rotations.shape[-1] == 3:
            rot_matrices = euler_angles_to_matrix(rotations, convention='XYZ')
        elif rotations.shape[-1] == 4:
            rot_matrices = quaternion_to_matrix(rotations)
        elif rotations.shape[-1] == 6:
            rot_matrices = rotation_6d_to_matrix(rotations)  # (B,T,J,6) -> (B,T,J,3,3)
        else:
            raise NotImplementedError(f'Unimplemented rotation representation in FK layer, shape of {rotations.shape}')

        # positions 是 (B, T, J, 3) -> positions[..., None] 是 (B, T, J, 3, 1)
        # rot_matrices 是 (B, T, J, 3, 3)
        # 拼接后是 (B, T, J, 3, 4)
        rot_matrices = torch.cat([rot_matrices, positions[..., None]], dim=-1)

        # shape[:-2] 是 (B, T, J)
        zeros = torch.zeros(rot_matrices.shape[:-2] + torch.Size([1, 3])).to(self.device)  # (B,T,J,1,3)
        ones = torch.ones(rot_matrices.shape[:-2] + torch.Size([1, 1])).to(self.device)  # (B,T,J,1,1)
        zerosones = torch.cat([zeros, ones], dim=-1)  # (B,T,J,1,4)

        # 拼接后是 (B, T, J, 4, 4)
        rot_matrices = torch.cat([rot_matrices, zerosones], dim=-2)
        return rot_matrices

    def rotate_global(self, parents, positions, rotations):
        # locals 形状 (B, T, J, 4, 4)
        locals = self.make_fast_rotation_matrices(positions, rotations)

        # 【【!! 高亮：全新修正 !!】】
        # 1. 创建一个 Python 列表来存储每个关节的全局变换
        globals_list = []
        num_joints = parents.shape[0]

        # 2. 循环遍历所有 J 关节
        for i in range(num_joints):
            if i == 0:
                # 根关节(i=0)的全局变换就是它的局部变换
                # (B, T, 4, 4)
                current_global = locals[..., 0, :, :]
            else:
                # 3. 获取父关节的索引
                parent_idx = parents[i]

                # 4. 从 *已经计算好* 的列表中获取父关节的全局变换
                #    注意：这里是从 Python 列表（而不是张量）中读取
                #    parent_globals 形状 (B, T, 4, 4)
                parent_globals = globals_list[parent_idx]

                # 5. 获取当前关节(i)的局部变换
                child_locals = locals[..., i, :, :]  # (B, T, 4, 4)

                # 6. 计算当前关节(i)的全局变换
                current_global = self.rotate(parent_globals, child_locals)

            # 7. 将当前计算出的 (B, T, 4, 4) 张量添加到列表中
            globals_list.append(current_global)

        # 8. 循环结束后，将列表中的 J 个 (B, T, 4, 4) 张量
        #    沿着新的 J 维度 (dim=2) 堆叠起来
        #    结果形状为 (B, T, J, 4, 4)
        globals = torch.stack(globals_list, dim=2)

        return globals

    # ... (get_tpose_joints 方法不需要修改，因为它不被 forward 使用) ...
    # ... (canonical_to_local 和 global_to_local 也有同样的问题，如果需要使用它们，也必须按类似方式修正) ...

    def forward(self, rotations, positions):
        # 【【!! 高亮：修改 !!】】
        # 为什么修改?
        # 1. __init__ 简化了，不再加载默认 positions。
        # 2. 传入的 positions 是 (B, J, 3)，而 rotations 是 (B, T, J, D)。
        #    我们需要在这里将 positions 扩展以匹配 rotations 的时间维度。

        """
        Args:
            rotations (B, T, J, D_rot)  <-- 包含时间
            positions (B, J, 3)         <-- 不含时间
        Returns:
            The global position of each joint after FK (B, T, J, 3)
        """
        b_size = rotations.shape[0]
        t_size = rotations.shape[1]  # 获取时间维度 T

        # 原始代码:
        # if positions is None:
        #     positions = self.positions.repeat(b_size, 1, 1)

        # 修正后:
        # 1. 将 (B, J, 3) -> (B, 1, J, 3)
        positions_expanded = positions.unsqueeze(1)
        # 2. 将 (B, 1, J, 3) -> (B, T, J, 3)
        #    使用 expand 来高效地 "重复" T 次，以匹配时间维度
        positions_expanded = positions_expanded.expand(-1, t_size, -1, -1)

        # 现在 positions_expanded 和 rotations 都有 (B, T, J, ...) 维度
        transforms = self.rotate_global(self.parents, positions_expanded, rotations)

        # 提取坐标，使用 ... 以适应 (B, T, J, 4, 4)
        coordinates = transforms[..., :3, 3] / transforms[..., 3:, 3]

        return coordinates, transforms


class FK_Pair_Layer(nn.Module):
    """一个封装了一对FK层的模块，用于处理一个特定的骨架对。"""

    def __init__(self, parents, skeleton_data_p1, skeleton_data_p2):
        super().__init__()
        skeleton_data_p1[1:] -= skeleton_data_p1[parents[1:]]
        skeleton_data_p2[1:] -= skeleton_data_p2[parents[1:]]
        self.fk_p1 = ForwardKinematicsLayer(parents=parents,
                                            positions=skeleton_data_p1)
        self.fk_p2 = ForwardKinematicsLayer(parents=parents,
                                            positions=skeleton_data_p2)

    def forward(self, motion_data_pair):
        # 将输入的二人动作数据拆分
        d_half = motion_data_pair.shape[-1] // 2
        p1_motion = motion_data_pair[..., :-1, :d_half]
        p2_motion = motion_data_pair[..., :-1, d_half:]

        p1_trans = motion_data_pair[..., -1:, :3]
        p2_trans = motion_data_pair[..., -1:, 6:9]

        # 分别计算FK
        p1_coords, _ = self.fk_p1(p1_motion)
        p2_coords, _ = self.fk_p2(p2_motion)

        # 添加平移信息
        p1_coords += p1_trans
        p2_coords += p2_trans


        return p1_coords, p2_coords

class FK_Pair_Layer_Batch(nn.Module):
    """一个封装了一对FK层的模块，用于处理一个特定的骨架对。"""

    def __init__(self, parents, skeleton_data_p1, skeleton_data_p2, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        skeleton_data_p1[:, 1:] -= skeleton_data_p1[:, parents[1:]]
        skeleton_data_p2[:, 1:] -= skeleton_data_p2[:, parents[1:]]
        self.fk_p1 = ForwardKinematicsLayerBatch(parents=parents, device=self.device)
        self.fk_p2 = ForwardKinematicsLayerBatch(parents=parents, device=self.device)
        skeleton_data_p1 = skeleton_data_p1.reshape(skeleton_data_p1.shape[0], -1)
        skeleton_data_p2 = skeleton_data_p2.reshape(skeleton_data_p2.shape[0], -1)

        self.p1_offset = skeleton_data_p1.shape[0]
        combined_skeleton_data = torch.cat([skeleton_data_p1, skeleton_data_p2], dim=0)
        self.unified_positions_lookup = nn.Embedding.from_pretrained(
            combined_skeleton_data,
            freeze=True
        )

    def forward(self, motion_data_pair, group_ids_batch):
        # 将输入的二人动作数据拆分
        d_half = motion_data_pair.shape[-1] // 2
        p1_motion = motion_data_pair[..., :-1, :d_half]
        p2_motion = motion_data_pair[..., :-1, d_half:]

        p1_trans = motion_data_pair[..., -1:, :3]
        p2_trans = motion_data_pair[..., -1:, 6:9]

        # 分别计算FK
        p1_source_flag = group_ids_batch[:, 0]  # (B,)
        p1_idx = group_ids_batch[:, 1]  # (B,)

        p2_source_flag = group_ids_batch[:, 2]  # (B,)
        p2_idx = group_ids_batch[:, 3]  # (B,)

        # 计算 P1 的真实全局索引
        # 如果 flag 是 0: real_idx = idx + 0
        # 如果 flag 是 1: real_idx = idx + self.p1_offset
        p1_real_idx = p1_idx + p1_source_flag * self.p1_offset

        # 计算 P2 的真实全局索引
        p2_real_idx = p2_idx + p2_source_flag * self.p1_offset

        # 执行查询 (一次性矢量化查询)
        p1_positions_flat = self.unified_positions_lookup(p1_real_idx)
        p2_positions_flat = self.unified_positions_lookup(p2_real_idx)

        # Reshape 回 (B, J, 3)
        p1_positions_batch = p1_positions_flat.reshape(
            p1_positions_flat.shape[0], self.fk_p1.num_joints, 3
        )
        p2_positions_batch = p2_positions_flat.reshape(
            p2_positions_flat.shape[0], self.fk_p2.num_joints, 3
        )

         # 计算FK

        p1_coords, _ = self.fk_p1(p1_motion, p1_positions_batch)
        p2_coords, _ = self.fk_p2(p2_motion, p2_positions_batch)

        # 添加平移信息
        p1_coords += p1_trans
        p2_coords += p2_trans


        return p1_coords, p2_coords

class FK_Single_Layer_Batch(nn.Module):
    """一个封装了一对FK层的模块，用于处理一个特定的骨架对。"""

    def __init__(self, parents, skeleton_data_p1, skeleton_data_p2, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        skeleton_data_p1[:, 1:] -= skeleton_data_p1[:, parents[1:]]
        skeleton_data_p2[:, 1:] -= skeleton_data_p2[:, parents[1:]]
        self.fk = ForwardKinematicsLayerBatch(parents=parents, device=self.device)
        skeleton_data_p1 = skeleton_data_p1.reshape(skeleton_data_p1.shape[0], -1)
        skeleton_data_p2 = skeleton_data_p2.reshape(skeleton_data_p2.shape[0], -1)

        self.p1_offset = skeleton_data_p1.shape[0]
        combined_skeleton_data = torch.cat([skeleton_data_p1, skeleton_data_p2], dim=0)
        self.unified_positions_lookup = nn.Embedding.from_pretrained(
            combined_skeleton_data,
            freeze=True
        )

    def forward(self, motion_data, group_ids_batch):
        # 将输入的二人动作数据拆分
        motion = motion_data[..., :-1, :]

        trans = motion_data[..., -1:, :3]

        # 分别计算FK
        p1_source_flag = group_ids_batch[:, 0]  # (B,)
        p1_idx = group_ids_batch[:, 1]  # (B,)

        # 计算 P1 的真实全局索引
        # 如果 flag 是 0: real_idx = idx + 0
        # 如果 flag 是 1: real_idx = idx + self.p1_offset
        p1_real_idx = p1_idx + p1_source_flag * self.p1_offset
        # 执行查询 (一次性矢量化查询)
        p1_positions_flat = self.unified_positions_lookup(p1_real_idx)

        # Reshape 回 (B, J, 3)
        p1_positions_batch = p1_positions_flat.reshape(
            p1_positions_flat.shape[0], self.fk.num_joints, 3
        )

         # 计算FK

        coords, _ = self.fk(motion, p1_positions_batch)

        # 添加平移信息
        coords += trans


        return coords