"""INVALID experimental loss rewrite retained only for diagnosis; do not import."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import time

# ==========================================
# 1. 基础几何工具函数
# ==========================================
from optimization.human_body_prior.body_model.body_model import BodyModel
from optimization.foot_contact_constraints import (
    freeze_initial_contact_mask,
    released_dynamic_skate_loss,
    skate_loss_with_mask,
)
from optimization.metric_aligned_foot_sliding import (
    freeze_initial_grounded_foot_mask,
    initial_or_current_grounded_horizontal_mean_speed,
)
from optimization.contact_loss_reduction import (
    CONTACT_LOSS_REDUCTION_POLICIES,
    reduce_contact_pair_sums,
)
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from utils.motion_process import get_spatial_min_interaction_label_single

def vertex_normals(vertices, faces):
    """
    计算顶点法线 (Batch化)
    vertices: [B, V, 3]
    faces: [B, F, 3]
    """
    device = vertices.device
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int64)).to(device)
    faces = faces.unsqueeze(0).repeat(vertices.shape[0], 1, 1)
    bs, nv = vertices.shape[:2]
    # print(bs, nv)
    bs, nf = faces.shape[:2]

    normals = torch.zeros(bs * nv, 3).to(device)

    # 展开 Faces 索引以适配扁平化的 Vertices

    faces_expanded = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]

    # 扁平化索引

    verts_flat = vertices.reshape(bs * nv, 3)
    faces_flat = faces_expanded.view(-1, 3).long()

    # 获取三角形顶点
    v0 = verts_flat[faces_flat[:, 0]]
    v1 = verts_flat[faces_flat[:, 1]]
    v2 = verts_flat[faces_flat[:, 2]]

    # 计算面法线 (未归一化，模长即为2倍面积，实现面积加权)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    # 累加到顶点
    normals.index_add_(0, faces_flat[:, 0], face_normals)
    normals.index_add_(0, faces_flat[:, 1], face_normals)
    normals.index_add_(0, faces_flat[:, 2], face_normals)

    # 归一化
    normals = F.normalize(normals, eps=1e-6, dim=1)
    return normals.reshape((bs, nv, 3))


def point2point_signed(x, y, x_normals=None, y_normals=None, return_vector=False):
    """
    计算点云 x 到 y 的有符号距离 (x -> y)
    x, y: [B, N, 3]
    """
    # 尝试导入高效库，否则报错或需替换为暴力实现
    import chamfer_distance as chd
    ch_dist = chd.ChamferDistance()

    # 计算 Chamfer Distance
    # x_near: y中离x最近的点 (shape like x)
    # y_near: x中离y最近的点 (shape like y)
    # xidx_near: 索引
    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y, x_normals=x_normals, y_normals=y_normals)

    # 我们主要关心 x 到 y 的距离 (x 中的点是否贴合/穿透 y)
    # 注意：ch_dist 返回的 x_near 是 float dist^2 还是坐标？
    # 通常 ch_dist 返回的是 (dist_x2y, dist_y2x, idx_x2y, idx_y2x)
    # 这里假设 ch_dist 行为符合您之前的代码逻辑

    # 为了严谨，这里重新gather坐标计算向量
    N, P1, D = x.shape
    P2 = y.shape[1]

    # 找到 x 中每个点对应的 y 中的最近点索引
    # xidx_near is [B, P1] containing indices in range [0, P2)
    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).long()
    y_target_for_x = y.gather(1, xidx_near_expanded)  # x 对应的 y 表面最近点

    # 向量: x -> y_target
    x2y_vec = x - y_target_for_x

    # 距离模长
    dist_unsigned = torch.norm(x2y_vec, dim=-1)

    # 符号判定
    signed_dist = dist_unsigned
    if y_normals is not None:
        # 获取对应点的法线
        y_nn = y_normals.gather(1, xidx_near_expanded)
        # dot(vector, normal)
        # 向量是 x -> y_surface，法线是 y_surface -> out
        # 如果 x 在外，向量指向内，法线指向外 -> dot < 0
        # 如果 x 在内，向量指向外，法线指向外 -> dot > 0
        # (注：具体符号取决于您的 vector 定义方向，这里沿用通用逻辑：内负外正)
        # 这里为了防穿模，通常定义：穿透为负。

        # 修正逻辑以匹配 point2point 的直觉：
        # vector = query_point - surface_point
        # normal = surface_normal
        # dot > 0 => query point is aligned with normal => Outside (Positive)
        # dot < 0 => query point is against normal => Inside (Negative)

        dot = torch.sum(x2y_vec * y_nn, dim=-1)
        sign = torch.sign(dot)
        signed_dist = dist_unsigned * sign

    if return_vector:
        return signed_dist, xidx_near, x2y_vec, y_target_for_x
    return signed_dist


# def point2point_signed(x, y, x_normals=None, y_normals=None, return_vector=False):
#     """
#     纯 PyTorch 实现的点对点有符号距离 (无需 pytorch3d/chamfer_distance)
#     x: [B, N, 3] (Source)
#     y: [B, M, 3] (Target)
#     """
#     B, N, _ = x.shape
#     _, M, _ = y.shape
#
#     # 1. 计算距离矩阵 (Brute-force distance matrix)
#     # 显存优化提示: 如果 N*M*B 太大导致 OOM，可以在这里加一个 for b in range(B) 循环
#     # [B, N, M]
#     dist_mat = torch.cdist(x, y, p=2)
#
#     # 2. 找到 y 中离 x 最近的点
#     # min_val: [B, N] (无符号距离)
#     # min_idx: [B, N] (最近点索引)
#     dist_unsigned, xidx_near = torch.min(dist_mat, dim=2)
#
#     # 3. 获取最近点的坐标
#     # [B, N] -> [B, N, 3]
#     # 我们需要构建 batch 索引来正确 gather
#     batch_indices = torch.arange(B, device=x.device).view(-1, 1).expand(-1, N)
#     y_target = y[batch_indices, xidx_near]  # Gather 操作
#
#     # 4. 计算向量和有符号距离
#     x2y_vec = x - y_target
#
#     signed_dist = dist_unsigned
#     if y_normals is not None:
#         # 获取最近点的法线
#         y_nn = y_normals[batch_indices, xidx_near]
#
#         # 投影: dot(vector, normal)
#         # 假设: vector 指向外 + normal 指向外 = 正 (外部)
#         #       vector 指向内 + normal 指向外 = 负 (内部/穿透)
#         dot = torch.sum(x2y_vec * y_nn, dim=-1)
#         sign = torch.sign(dot)
#         signed_dist = dist_unsigned * sign
#
#     if return_vector:
#         return signed_dist, xidx_near, x2y_vec, y_target
#     return signed_dist

# ==========================================
# 2. 核心 Loss 类 (Explicit Control)
# ==========================================

class DyadicInteractionLoss(nn.Module):
    def __init__(
        self,
        device='cuda:1',
        penetration_shallow_threshold=0.05,
        penetration_interaction_threshold=10.0,
        signed_distance_backend='volumetric_smpl',
        volumetric_smpl_seed=1234,
        volumetric_smpl_query_chunk=1024,
        volumetric_smpl_frame_batch_size=1,
        volumetric_smpl_backend=None,
        foot_contact_mask_policy='released_dynamic',
        contact_loss_reduction_policy='paper_literal_raw_sum',
    ):
        """
        :param smpl_model: SMPL/SMPL-X 模型实例
        """
        super().__init__()

        num_betas = 10
        downsample = 4

        neutral_bm_path = 'body_models/smplx/SMPLX_NEUTRAL.npz'
        male_bm_path = 'body_models/smplx/SMPLX_MALE.npz'
        female_bm_path = 'body_models/smplx/SMPLX_FEMALE.npz'
        self.neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas).to(device)
        self.male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas).to(device)
        self.female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas).to(device)

        self.device = device
        if foot_contact_mask_policy not in (
            'released_dynamic',
            'frozen_initial',
            'initial_or_current_grounded_horizontal_mean_speed',
        ):
            raise ValueError(
                'foot_contact_mask_policy must be released_dynamic, '
                'frozen_initial, or '
                'initial_or_current_grounded_horizontal_mean_speed'
            )
        self.foot_contact_mask_policy = foot_contact_mask_policy
        if contact_loss_reduction_policy not in CONTACT_LOSS_REDUCTION_POLICIES:
            raise ValueError(
                'contact_loss_reduction_policy must be one of '
                f'{CONTACT_LOSS_REDUCTION_POLICIES}'
            )
        self.contact_loss_reduction_policy = contact_loss_reduction_policy
        # Stored in the original input-person order.  ``forward`` reorders
        # these masks together with the people if semantic role assignment
        # swaps P1/P2.
        self.frozen_foot_contact_masks = None
        self.penetration_shallow_threshold = float(
            penetration_shallow_threshold
        )
        self.penetration_interaction_threshold = float(
            penetration_interaction_threshold
        )
        if signed_distance_backend not in ('volumetric_smpl', 'nearest_vertex'):
            raise ValueError(
                'signed_distance_backend must be volumetric_smpl or nearest_vertex'
            )
        self.signed_distance_backend = signed_distance_backend
        self.volumetric_smpl_seed = int(volumetric_smpl_seed)
        self.volumetric_smpl_frame_batch_size = int(
            volumetric_smpl_frame_batch_size
        )
        if self.volumetric_smpl_frame_batch_size < 1:
            raise ValueError('volumetric_smpl_frame_batch_size must be positive')
        self.volumetric_smpl = None
        if signed_distance_backend == 'volumetric_smpl':
            if volumetric_smpl_backend is None:
                self.volumetric_smpl = VolumetricSMPLBackend(
                    model_root='body_models',
                    device=device,
                    max_query_points_per_call=volumetric_smpl_query_chunk,
                )
            else:
                if not isinstance(
                    volumetric_smpl_backend, VolumetricSMPLBackend
                ):
                    raise TypeError(
                        'volumetric_smpl_backend must be a '
                        'VolumetricSMPLBackend instance'
                    )
                requested_device = torch.device(device)
                if volumetric_smpl_backend.device != requested_device:
                    raise ValueError(
                        'shared VolumetricSMPL device differs from the loss '
                        f'device: {volumetric_smpl_backend.device} vs '
                        f'{requested_device}'
                    )
                if (
                    volumetric_smpl_backend.max_query_points_per_call
                    != int(volumetric_smpl_query_chunk)
                ):
                    raise ValueError(
                        'shared VolumetricSMPL query chunk differs from the '
                        'requested optimization contract'
                    )
                self.volumetric_smpl = volumetric_smpl_backend

        # --- 默认权重配置 (可在外部修改) ---
        self.weights = {
            'attract': 1.0,  # 接触吸引力
            'pene': 5.0,  # 全局防穿模
            'reg': 1.0,  # 正则化
            'smooth': 0.1,  # 平滑
            'skate': 100.0,  # 脚步防滑
            'ground': 20.0  # 地面穿透
        }

        # 辅助索引 (脚部用于防滑)
        self.lfoot_idx = [7, 10]
        self.rfoot_idx = [8, 11]
        raw_rid2vid = pkl.load(open('body_models/flickrci3ds_r75_rid_to_smplx_vid.pkl', 'rb'))

        self.rid_to_vid_tensors = {}

        for rid, vid_list in raw_rid2vid.items():
            # 强制转换为 int 作为 key
            rid_int = int(rid)
            self.rid_to_vid_tensors[rid_int] = torch.tensor(vid_list, dtype=torch.long).to(device)

        self.swap_determined = False  # 是否已经执行过角色匹配
        self.swapped = False  # 是否决定交换 (即 B是P1)

        self.is_anchored = False  # 是否进入锚定阶段
        self.fixed_pairs = {}  # 存储锚定点

        self.current_winner_regions = {}

    def _surface_signed_distance(
        self,
        query_vertices,
        target_vertices,
        target_normals,
        target_volume=None,
    ):
        """Return the selected backend's signed distance to a target body."""
        if self.signed_distance_backend == 'volumetric_smpl':
            if target_volume is None:
                raise RuntimeError('VolumetricSMPL target output is required')
            target_output, target_gender = target_volume
            return self.volumetric_smpl.signed_distance(
                query_vertices,
                target_output,
                target_gender,
            )
        return point2point_signed(
            query_vertices,
            target_vertices,
            y_normals=target_normals,
        )

    def _compute_geometry(self, params):
        body_pose = params['pose_body']
        left_hand_pose = params['pose_lhand']
        right_hand_pose = params['pose_rhand']
        root_transl = params['trans']
        global_orient = params['root_orient']
        betas = params['betas']
        frame_number = body_pose.shape[0]
        # jaw_pose = np.zeros((frame_number, 3))
        # leye_pose = np.zeros((frame_number, 3))
        # reye_pose = np.zeros((frame_number, 3))

        if (params['gender'] == 0) or (params['gender'] == 'male'):
            bm = self.male_bm  # gender is ignored for training/evaluation
        elif (params['gender'] == 1) or (params['gender'] == 'female'):
            bm = self.female_bm
        else:
            bm = self.neutral_bm

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
        # print('verts', verts.shape, 'joints', joints.shape, 'faces', faces.shape)
        normals = vertex_normals(verts, faces)
        return verts, joints, normals

    def _determine_role_assignment(
        self,
        vA,
        vB,
        nA,
        nB,
        interaction_pairs,
        temporal_mask,
        volume_A=None,
        volume_B=None,
    ):
        """
        在 Epoch 0 运行一次：比较 (A->P1) 和 (B->P1) 的初始语义 Loss，决定是否交换。
        """
        # print("[DyadicLoss] Auto-matching roles based on text description...")

        # 使用 no_grad，因为这只是为了做决定，不需要反传
        with torch.no_grad():
            # 方案 1: A 是 P1
            loss_1, _ = self._calc_semantic_contact_loss(
                vA, vB, nA, nB, interaction_pairs, temporal_mask,
                volume_A=volume_A, volume_B=volume_B,
            )

            # 方案 2: B 是 P1 (交换输入)
            loss_2, _ = self._calc_semantic_contact_loss(
                vB, vA, nB, nA, interaction_pairs, temporal_mask,
                volume_A=volume_B, volume_B=volume_A,
            )

        if loss_2 < loss_1:
            # print(f"  > Swap Detected! Loss(B=P1):{loss_2:.3f} < Loss(A=P1):{loss_1:.3f}")
            # print(f"  > Assigning: Character B -> Person 1, Character A -> Person 2")
            self.swapped = True
        else:
            # print(f"  > Order Kept. Loss(A=P1):{loss_1:.3f} < Loss(B=P1):{loss_2:.3f}")
            # print(f"  > Assigning: Character A -> Person 1, Character B -> Person 2")
            self.swapped = False

        self.swap_determined = True
    # --------------------------------------------------
    #  Module 1: Explicit Contact Loss (用户指定区域和时间)
    # --------------------------------------------------
    def freeze_contact_candidates(self, verts_A, verts_B, norms_A, norms_B, interaction_pairs, temporal_mask):
        """Freeze Eq. 5 unique bidirectional nearest-vertex pairs."""
        if self.is_anchored: return

        print(">>> Freezing Anchors with One-to-One Constraint <<<")
        self.fixed_pairs = {}

        for i, _pair in enumerate(interaction_pairs):
            g_idx_A, g_idx_B = self.current_winner_regions.get(i, (None, None))
            if g_idx_A is None or g_idx_B is None:
                print(f"  Pair {i}: Skipped (No cached search result).")
                continue
            if temporal_mask is not None and temporal_mask.sum() > 0:
                peak_frame_idx = torch.argmax(temporal_mask)
            else:
                peak_frame_idx = 0
            vA_sub = verts_A[peak_frame_idx:peak_frame_idx + 1, g_idx_A]
            vB_sub = verts_B[peak_frame_idx:peak_frame_idx + 1, g_idx_B]
            d_mat = torch.cdist(vA_sub, vB_sub).squeeze(0)
            nearest_b = torch.argmin(d_mat, dim=1)
            nearest_a = torch.argmin(d_mat, dim=0)
            candidates = {
                (a, int(nearest_b[a])) for a in range(d_mat.shape[0])
            }
            candidates.update({
                (int(nearest_a[b]), b) for b in range(d_mat.shape[1])
            })
            candidates = sorted(
                candidates,
                key=lambda indices: (
                    float(d_mat[indices[0], indices[1]]),
                    indices[0],
                    indices[1],
                ),
            )
            used_A = set()
            used_B = set()
            valid_pairs_A = []
            valid_pairs_B = []
            for idx_a_local, idx_b_local in candidates:
                if idx_a_local in used_A or idx_b_local in used_B:
                    continue
                valid_pairs_A.append(g_idx_A[idx_a_local])
                valid_pairs_B.append(g_idx_B[idx_b_local])
                used_A.add(idx_a_local)
                used_B.add(idx_b_local)
            self.fixed_pairs[i] = {
                'idx_A': torch.stack(valid_pairs_A).long().to(self.device),
                'idx_B': torch.stack(valid_pairs_B).long().to(self.device),
            }
            print(
                f"  Pair {i}: Locked {len(valid_pairs_A)} unique "
                "bidirectional nearest pairs."
            )

        self.is_anchored = True


    def _calc_explicit_contact_loss(self,
                                    verts_A, norms_A, idx_A,
                                    verts_B, norms_B, idx_B,
                                    temporal_mask):
        """
        计算显式接触 Loss
        :param idx_A: [N_a] A 的接触顶点索引
        :param idx_B: [N_b] B 的接触顶点索引
        :param temporal_mask: [T] 每一帧的接触权重 (0=无接触, 1=强接触)
        """
        # 提取区域顶点
        # vA_contact: [T, N_a, 3]
        vA_contact = verts_A[:, idx_A, :]
        nA_contact = norms_A[:, idx_A, :]
        vB_contact = verts_B[:, idx_B, :]
        nB_contact = norms_B[:, idx_B, :]

        # 计算 A -> B 的距离 (通常双向计算更稳，或者只算主动方->被动方)
        # dist_A2B: A 中的点到 B 表面最近点的有符号距离
        dist_A2B, idx_nearest_on_B, vec_A2B, _ = point2point_signed(
            vA_contact, vB_contact,
            x_normals=nA_contact, y_normals=nB_contact,
            return_vector=True
        )

        # --- A. 距离吸引 (Attraction) ---
        # 只在 temporal_mask > 0 的帧生效
        # temporal_mask shape: [T] -> [T, 1]
        t_mask = temporal_mask.view(-1, 1)

        # 加权距离: abs(dist) * frame_weight
        # print('dist_A2B', dist_A2B.device, 't_mask', t_mask.device)
        weighted_dist = torch.abs(dist_A2B) * t_mask

        # 归一化 Loss (除以有效接触的总权重，防止 Mask 只有一帧时 Loss 过小)
        # 避免除以 0
        total_weight = torch.sum(t_mask) + 1e-6
        loss_attract = torch.sum(torch.mean(weighted_dist, dim=1)) / total_weight  # Mean over vertices, Sum over time

        # --- B. 法线对齐 (Alignment) ---
        # 强制接触区域法线相反
        # 获取 A 中每个点对应的 B 上的最近点的法线
        # idx_nearest_on_B: [T, N_a]
        idx_long = idx_nearest_on_B.long().unsqueeze(-1).expand(-1, -1, 3)
        nB_target = torch.gather(nB_contact, 1, idx_long)

        # Cosine Alignment: dot(nA, nB_target) should be -1
        alignment = torch.sum(nA_contact * nB_target, dim=-1)  # [T, N_a]

        # Loss = (dot + 1)^2
        align_error = (alignment + 1.0) ** 2
        weighted_align = align_error * t_mask
        loss_align = torch.sum(torch.mean(weighted_align, dim=1)) / total_weight

        return loss_attract, loss_align

    def _calc_fine_region_contact_loss(
        self,
        verts_A,
        verts_B,
        norms_A,
        norms_B,
        fine_region_pairs,
        temporal_mask,
        volume_A=None,
        volume_B=None,
    ):
        """
        Chi3D 专用的细粒度 region-pair 接触 loss。
        输入的 pair 已经是 75-region 小区域对，因此只跳过“候选区域竞赛”这一步；
        后续的 top-k、双向吸引、法线对齐、anchor freeze 逻辑与 semantic 分支保持一致。
        """
        if not fine_region_pairs or (temporal_mask is not None and torch.sum(temporal_mask) == 0):
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        total_loss_attract = 0.0
        total_loss_align = 0.0

        area_ratio = 0.10
        align_weight = 5.0

        if temporal_mask is None:
            t_weights = torch.ones(verts_A.shape[0], 1, device=self.device)
            mask_sum = verts_A.shape[0]
            active_weight_sum = verts_A.shape[0]
        else:
            t_weights = temporal_mask.view(-1, 1)
            mask_sum = torch.sum(temporal_mask) + 1e-6
            active_weight_sum = torch.sum(temporal_mask)

        if self.is_anchored:
            pair_lock_sums = []
            locked_vertex_pairs = []
            total_loss_align = 0.0
            count = 0

            for i, pair in enumerate(fine_region_pairs):
                if i not in self.fixed_pairs:
                    continue

                anchors = self.fixed_pairs[i]
                idx_A = anchors['idx_A']
                idx_B = anchors['idx_B']
                pts_A = verts_A[:, idx_A, :]
                pts_B = verts_B[:, idx_B, :]

                pair_distance_sq = ((pts_A - pts_B) ** 2).sum(dim=-1)
                loss_lock = torch.sum(pair_distance_sq * t_weights)
                pair_lock_sums.append(loss_lock)
                locked_vertex_pairs.append(idx_A.numel())

                nA = norms_A[:, idx_A, :]
                nB = norms_B[:, idx_B, :]
                dot = torch.sum(nA * nB, dim=-1)
                align_err = ((dot + 1.0) ** 2).mean(dim=1, keepdim=True)
                total_loss_align += torch.sum(align_err * t_weights) / mask_sum * 0.5
                count += 1

            total_loss_lock = (
                reduce_contact_pair_sums(
                    pair_lock_sums,
                    active_weight_sum,
                    self.contact_loss_reduction_policy,
                    elements_per_pair=locked_vertex_pairs,
                )
                if pair_lock_sums else 0.0
            )
            return total_loss_lock, total_loss_align

        valid_pairs = 0
        pair_attract_sums = []
        for i, pair in enumerate(fine_region_pairs):
            rid_a = int(pair["p1_rid"])
            rid_b = int(pair["p2_rid"])

            if rid_a not in self.rid_to_vid_tensors or rid_b not in self.rid_to_vid_tensors:
                continue

            idx_A = self.rid_to_vid_tensors[rid_a]
            idx_B = self.rid_to_vid_tensors[rid_b]
            sub_A = verts_A[:, idx_A, :]
            sub_B = verts_B[:, idx_B, :]
            nA = norms_A[:, idx_A, :]
            nB = norms_B[:, idx_B, :]

            self.current_winner_regions[i] = (idx_A, idx_B)

            _, idx_nearest_B, _, _ = point2point_signed(
                sub_A, sub_B.detach(), x_normals=nA, y_normals=nB.detach(), return_vector=True
            )
            dist_signed_A = self._surface_signed_distance(
                sub_A, sub_B.detach(), nB.detach(), target_volume=volume_B
            )
            dist_abs_A = torch.abs(dist_signed_A)
            k_A = max(1, int(len(idx_A) * area_ratio))
            topk_vals_A, topk_indices_A = torch.topk(dist_abs_A, k=k_A, dim=1, largest=False)
            loss_attract_A = topk_vals_A.mean(dim=1, keepdim=True)

            idx_gather_A = topk_indices_A.unsqueeze(-1).expand(-1, -1, 3)
            nA_selected = torch.gather(nA, 1, idx_gather_A)
            idx_nearest_B_selected = torch.gather(idx_nearest_B, 1, topk_indices_A)
            idx_gather_B = idx_nearest_B_selected.unsqueeze(-1).expand(-1, -1, 3)
            nB_selected = torch.gather(nB, 1, idx_gather_B)
            loss_align_A = ((torch.sum(nA_selected * nB_selected, dim=-1) + 1.0) ** 2).mean(dim=1, keepdim=True) * align_weight

            _, idx_nearest_A, _, _ = point2point_signed(
                sub_B, sub_A.detach(), x_normals=nB, y_normals=nA.detach(), return_vector=True
            )
            dist_signed_B = self._surface_signed_distance(
                sub_B, sub_A.detach(), nA.detach(), target_volume=volume_A
            )
            dist_abs_B = torch.abs(dist_signed_B)
            k_B = max(1, int(len(idx_B) * area_ratio))
            topk_vals_B, topk_indices_B = torch.topk(dist_abs_B, k=k_B, dim=1, largest=False)
            loss_attract_B = topk_vals_B.mean(dim=1, keepdim=True)

            idx_gather_B_own = topk_indices_B.unsqueeze(-1).expand(-1, -1, 3)
            nB_selected_2 = torch.gather(nB, 1, idx_gather_B_own)
            idx_nearest_A_selected = torch.gather(idx_nearest_A, 1, topk_indices_B)
            idx_gather_A_target = idx_nearest_A_selected.unsqueeze(-1).expand(-1, -1, 3)
            nA_selected_2 = torch.gather(nA, 1, idx_gather_A_target)
            loss_align_B = ((torch.sum(nB_selected_2 * nA_selected_2, dim=-1) + 1.0) ** 2).mean(dim=1, keepdim=True) * align_weight

            pair_attract_sums.append(torch.sum(
                (loss_attract_A + loss_attract_B) * t_weights
            ))
            total_loss_align += torch.sum(
                (loss_align_A + loss_align_B) * t_weights
            )
            valid_pairs += 1

        if pair_attract_sums:
            total_loss_attract = reduce_contact_pair_sums(
                pair_attract_sums,
                active_weight_sum,
                self.contact_loss_reduction_policy,
            )
        return total_loss_attract, total_loss_align

    def _calc_semantic_contact_loss(
        self,
        verts_A,
        verts_B,
        norms_A,
        norms_B,
        interaction_pairs,
        temporal_mask,
        volume_A=None,
        volume_B=None,
    ):
        """
        根据 LLM 输出的 Contact Area 和 Normal Alignment 动态计算语义接触 Loss。
        """
        # 0. 基础检查
        if not interaction_pairs or (temporal_mask is not None and torch.sum(temporal_mask) == 0):
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)



        # --- 1. 定义物理属性映射表 (Hyperparameter Mapping) ---

        # 映射 Contact Area -> Top-K Ratio
        # 决定了“拉近多少点”：点越多，接触面积越大
        area_map = {'point': 0.02, 'flat': 0.10, 'wrap': 0.15, 'hug': 0.20, 'default': 0.02}

        # 映射 Normal Alignment -> Alignment Weight
        # 决定了“法线对齐的强弱”：权重越高，表面越平行
        align_map = {
            'strict': 5.0,  # 强强制对齐 (击掌、扶墙)
            'adaptive': 0.5,  # 弱对齐，允许弯曲 (抓握)
            'none': 0.0,  # 不对齐 (点击)
            'default': 1.0
        }

        total_loss_attract = 0.0
        total_loss_align = 0.0

        # 预处理时间 Mask
        if temporal_mask is None:
            t_weights = torch.ones(verts_A.shape[0], 1, device=self.device)
            mask_sum = verts_A.shape[0]
            active_weight_sum = verts_A.shape[0]
        else:
            t_weights = temporal_mask.view(-1, 1)
            mask_sum = torch.sum(temporal_mask) + 1e-6
            active_weight_sum = torch.sum(temporal_mask)

        if self.is_anchored:
            pair_lock_sums = []
            locked_vertex_pairs = []
            total_loss_align = 0.0

            count = 0
            for i, pair in enumerate(interaction_pairs):
                if i not in self.fixed_pairs: continue
                # 取出冻结的索引对 (一一对应关系)
                # idx_A 和 idx_B 长度都是 K，且 idx_A[n] 应与 idx_B[n] 重合
                anchors = self.fixed_pairs[i]
                idx_A = anchors['idx_A']
                idx_B = anchors['idx_B']

                # 提取点坐标 [T, K, 3]
                pts_A = verts_A[:, idx_A, :]
                pts_B = verts_B[:, idx_B, :]

                # Paper Eq. 5 sums squared Euclidean distance once for every
                # frozen one-to-one pair, allowing gradients to both bodies.
                pair_distance_sq = ((pts_A - pts_B) ** 2).sum(dim=-1)
                loss_lock = torch.sum(pair_distance_sq * t_weights)
                pair_lock_sums.append(loss_lock)
                locked_vertex_pairs.append(idx_A.numel())
                # --- 法线对齐 (可选) ---
                # 获取对齐权重
                a_type = pair.get('align_type', 'default')
                align_map = {'strict': 0.5, 'adaptive': 0.5, 'none': 0.0, 'default': 0.5}
                w_align = align_map.get(a_type, 0.5)

                if w_align > 0:
                    nA = norms_A[:, idx_A, :]
                    nB = norms_B[:, idx_B, :]
                    # Dot product
                    dot = torch.sum(nA * nB, dim=-1)
                    align_err = ((dot + 1.0) ** 2).mean(dim=1, keepdim=True)
                    loss_align = torch.sum(align_err * t_weights) / mask_sum
                    total_loss_align += loss_align * w_align

                count += 1

            # 返回锁定 Loss
            # 注意：这里的 loss_lock 替代了之前的 loss_attract
            total_loss_lock = (
                reduce_contact_pair_sums(
                    pair_lock_sums,
                    active_weight_sum,
                    self.contact_loss_reduction_policy,
                    elements_per_pair=locked_vertex_pairs,
                )
                if pair_lock_sums else 0.0
            )
            return total_loss_lock, total_loss_align


        pair_attract_sums = []
        for i, pair in enumerate(interaction_pairs):
            c_type = pair.get('contact_type', 'default')
            a_type = pair.get('align_type', 'default')
            curr_top_k_ratio = area_map.get(c_type, area_map['default'])
            curr_align_w = align_map.get(a_type, align_map['default'])
            w_p1 = pair.get('w_p1', 0.5)
            w_p2 = pair.get('w_p2', 0.5)

            # --- [修改] 获取候选区域 ID 列表 (List of Ints) ---
            # 外部 parser 应该已经把 'Right Leg' 转成了 [35, 36, 37] 这种 ID 列表
            p1_rids = pair.get('p1_ids', [])
            p2_rids = pair.get('p2_ids', [])

            # =========================================================
            #  Phase 1: 区域竞赛 (RID Competition)
            # =========================================================

            best_dist_val = float('inf')
            best_sub_A = None
            best_sub_B = None
            best_nA = None
            best_nB = None
            best_idx_A_len = 0
            best_idx_B_len = 0
            best_idx_A_global = None
            best_idx_B_global = None

            # 遍历 RID (Int)
            for rid1 in p1_rids:
                if rid1 not in self.rid_to_vid_tensors: continue
                idx1 = self.rid_to_vid_tensors[rid1]  # 获取 GPU Tensor
                sub1 = verts_A[:, idx1, :]

                for rid2 in p2_rids:
                    if rid2 not in self.rid_to_vid_tensors: continue
                    idx2 = self.rid_to_vid_tensors[rid2]
                    sub2 = verts_B[:, idx2, :]

                    # 快速估算距离 (无梯度)
                    with torch.no_grad():
                        d = point2point_signed(sub1, sub2)
                        current_dist_score = torch.abs(d).min()

                    if current_dist_score < best_dist_val:
                        best_dist_val = current_dist_score
                        # 记录胜出者
                        best_sub_A = sub1
                        best_sub_B = sub2
                        best_idx_A_global = idx1
                        best_idx_B_global = idx2
                        best_nA = norms_A[:, idx1, :]
                        best_nB = norms_B[:, idx2, :]
                        best_idx_A_len = len(idx1)
                        best_idx_B_len = len(idx2)

            # =========================================================
            # 方向 1: A -> B (拉动 A 去贴合 B)
            # =========================================================
            # 计算距离和最近邻
            if best_sub_A is None: continue
            self.current_winner_regions[i] = (best_idx_A_global, best_idx_B_global)

            # --- A -> B ---
            _, idx_nearest_B, vec_A2B, _ = point2point_signed(best_sub_A, best_sub_B.detach(), x_normals=best_nA,
                                                              y_normals=best_nB.detach(), return_vector=True)
            dist_signed_A = self._surface_signed_distance(
                best_sub_A,
                best_sub_B.detach(),
                best_nB.detach(),
                target_volume=volume_B,
            )
            dist_abs_A = torch.abs(dist_signed_A)

            k_A = max(1, int(best_idx_A_len * curr_top_k_ratio))
            topk_vals_A, topk_indices_A = torch.topk(dist_abs_A, k=k_A, dim=1, largest=False)
            loss_attract_A = topk_vals_A.mean(dim=1, keepdim=True) * w_p1

            if curr_align_w > 0:
                idx_gather_A = topk_indices_A.unsqueeze(-1).expand(-1, -1, 3)
                nA_selected = torch.gather(best_nA, 1, idx_gather_A)
                idx_nearest_B_selected = torch.gather(idx_nearest_B, 1, topk_indices_A)
                idx_gather_B = idx_nearest_B_selected.unsqueeze(-1).expand(-1, -1, 3)
                nB_selected = torch.gather(best_nB, 1, idx_gather_B)
                loss_align_A = ((torch.sum(nA_selected * nB_selected, dim=-1) + 1.0) ** 2).mean(dim=1,
                                                                                                keepdim=True) * curr_align_w
            else:
                loss_align_A = torch.zeros_like(loss_attract_A)

            # --- B -> A ---
            _, idx_nearest_A, vec_B2A, _ = point2point_signed(best_sub_B, best_sub_A.detach(), x_normals=best_nB,
                                                              y_normals=best_nA.detach(), return_vector=True)
            dist_signed_B = self._surface_signed_distance(
                best_sub_B,
                best_sub_A.detach(),
                best_nA.detach(),
                target_volume=volume_A,
            )
            dist_abs_B = torch.abs(dist_signed_B)
            k_B = max(1, int(best_idx_B_len * curr_top_k_ratio))
            topk_vals_B, topk_indices_B = torch.topk(dist_abs_B, k=k_B, dim=1, largest=False)
            loss_attract_B = topk_vals_B.mean(dim=1, keepdim=True) * w_p2

            if curr_align_w > 0:
                idx_gather_B_own = topk_indices_B.unsqueeze(-1).expand(-1, -1, 3)
                nB_selected_2 = torch.gather(best_nB, 1, idx_gather_B_own)
                idx_nearest_A_selected = torch.gather(idx_nearest_A, 1, topk_indices_B)
                idx_gather_A_target = idx_nearest_A_selected.unsqueeze(-1).expand(-1, -1, 3)
                nA_selected_2 = torch.gather(best_nA, 1, idx_gather_A_target)
                loss_align_B = ((torch.sum(nB_selected_2 * nA_selected_2, dim=-1) + 1.0) ** 2).mean(dim=1,
                                                                                                    keepdim=True) * curr_align_w
            else:
                loss_align_B = torch.zeros_like(loss_attract_B)

            # =========================================================
            # 3. 汇总与加权
            # =========================================================
            # 应用时间 Mask 并累加
            current_pair_attract = torch.sum((loss_attract_A + loss_attract_B) * t_weights)
            current_pair_align = torch.sum((loss_align_A + loss_align_B) * t_weights)

            pair_attract_sums.append(current_pair_attract)
            total_loss_align += current_pair_align

        if pair_attract_sums:
            total_loss_attract = reduce_contact_pair_sums(
                pair_attract_sums,
                active_weight_sum,
                self.contact_loss_reduction_policy,
            )
        return total_loss_attract, total_loss_align
    # --------------------------------------------------
    #  Module 2: Penetration (Global or ROI)
    # --------------------------------------------------
    def _calc_penetration_loss_impl(
        self,
        verts_A,
        verts_B,
        norms_A,
        norms_B,
        volume_A=None,
        volume_B=None,
        temporal_mask=None,
        ratio=1.0,
        normalize_by_num_penetrating=False,
        thresh_shallow=0.2,
        thresh_deep=0.5,
    ):
        """
        [改进版] 防穿模 Loss / Metric：
        - 非交互期 (Mask=0): 仅修复浅层穿透 (<5cm)，忽略深层错误，防止梯度爆炸。
        - 交互期 (Mask=1): 修复所有穿透，包括严重穿透，确保物理真实。
        - Paper Eq. 6 sums squared signed distances over active penetrating
          vertices.  The normalization option is retained only for explicitly
          requested legacy diagnostics and is disabled by the paper path.
        """
        if self.signed_distance_backend == 'volumetric_smpl':
            if volume_A is None or volume_B is None:
                raise RuntimeError('VolumetricSMPL body outputs are required')
            if normalize_by_num_penetrating:
                raise ValueError(
                    'normalized penetration is a legacy nearest-vertex '
                    'diagnostic, not paper Eq. 6'
                )
            output_A, gender_A = volume_A
            output_B, gender_B = volume_B
            loss_B2A = self.volumetric_smpl.penetration_squared_sum(
                verts_B,
                output_A,
                gender_A,
                temporal_mask,
                shallow_threshold=thresh_shallow,
                interaction_threshold=thresh_deep,
                frame_batch_size=self.volumetric_smpl_frame_batch_size,
            )
            loss_A2B = self.volumetric_smpl.penetration_squared_sum(
                verts_A,
                output_B,
                gender_B,
                temporal_mask,
                shallow_threshold=thresh_shallow,
                interaction_threshold=thresh_deep,
                frame_batch_size=self.volumetric_smpl_frame_batch_size,
            )
            return (loss_B2A + loss_A2B) * ratio

        # --- 1. 计算双向距离 ---
        d_B2A = self._surface_signed_distance(
            verts_B, verts_A, norms_A, target_volume=volume_A
        )
        d_A2B = self._surface_signed_distance(
            verts_A, verts_B, norms_B, target_volume=volume_B
        )

        # --- 2. 准备动态阈值 ---
        # print(temporal_mask[38:48])
        # 构建 Batch 级别的阈值向量 [B, 1]
        if temporal_mask is not None:
            # 展平 mask: [B, 1]
            t_m = temporal_mask.view(-1, 1)
            # 如果 temporal_mask > 0.1 (认为是交互帧)，使用深层阈值，否则使用浅层阈值
            current_thresh = torch.where(
                t_m > 0.1,
                torch.tensor(thresh_deep, device=self.device),
                torch.tensor(thresh_shallow, device=self.device)
            )
        else:
            # 如果没传 mask，默认使用浅层模式 (或者你可以默认深层，看需求)
            current_thresh = torch.tensor(thresh_shallow, device=self.device)

        # --- 3. 定义计算逻辑 ---
        def compute_weighted_loss(dist_signed):
            # dist_signed: [B, N]

            # 扩展阈值以匹配点数 [B, 1] -> [B, N]
            # 这样第 b 个 batch 的所有点都共享同一个阈值
            thresh_expanded = current_thresh  # broadcasting 会自动处理 [B, 1] vs [B, N]

            # Mask:
            # 1. 距离 < 0 (穿进去了)
            # 2. 绝对深度 < 当前阈值 (交互时阈值很大=全选; 非交互时阈值很小=只选表层)
            valid_pene_mask = (dist_signed < 0) & (torch.abs(dist_signed) < thresh_expanded)
            # print(torch.sum(valid_pene_mask))
            # Paper Eq. 6: sum Phi(v)^2 over the active set.  ``ratio`` is a
            # legacy experimental multiplier; the paper path always uses 1.
            loss = torch.sum((dist_signed[valid_pene_mask] ** 2)) * ratio

            num_penetrating = valid_pene_mask.sum()

            # 如果完全没有穿模，直接返回 0 (避免除以 0)
            if num_penetrating < 1:
                return dist_signed.sum() * 0.0

            if normalize_by_num_penetrating:
                # 优化口径：除以穿模点数量，避免大面积穿模时梯度爆炸。
                return loss / num_penetrating.float()

            # 指标口径：直接返回总穿透，不做归一化。
            return loss

        # --- 4. 计算总 Loss ---
        loss_pene_B2A = compute_weighted_loss(d_B2A)
        loss_pene_A2B = compute_weighted_loss(d_A2B)

        return loss_pene_B2A + loss_pene_A2B

    def _calc_penetration_loss(
        self,
        verts_A,
        verts_B,
        norms_A,
        norms_B,
        volume_A=None,
        volume_B=None,
        temporal_mask=None,
        ratio=1.0,
    ):
        """
        Paper Eq. 6 penetration loss (unnormalized sum over active vertices).
        """
        return self._calc_penetration_loss_impl(
            verts_A, verts_B, norms_A, norms_B,
            volume_A=volume_A,
            volume_B=volume_B,
            temporal_mask=temporal_mask,
            ratio=ratio,
            normalize_by_num_penetrating=False,
            # Paper Eq. 6: 0.05 m outside interaction and 10 m during
            # interaction, so every penetrating vertex is corrected there.
            thresh_shallow=self.penetration_shallow_threshold,
            thresh_deep=self.penetration_interaction_threshold,
        )

    def _calc_penetration_metric(
        self,
        verts_A,
        verts_B,
        norms_A,
        norms_B,
        volume_A=None,
        volume_B=None,
        temporal_mask=None,
        ratio=1.0,
    ):
        """
        统计专用 pene metric：不除以 num_penetrating，返回总穿透量。
        """
        return self._calc_penetration_loss_impl(
            verts_A, verts_B, norms_A, norms_B,
            volume_A=volume_A,
            volume_B=volume_B,
            temporal_mask=temporal_mask,
            ratio=ratio,
            normalize_by_num_penetrating=False,
            thresh_shallow=10.0,
            thresh_deep=10.0,
        )

    # --------------------------------------------------
    #  Module 3: Aux Losses (Regularization)
    # --------------------------------------------------
    def _calc_aux_losses(
        self, params_A, params_B, params_A_init, params_B_init,
        joints_A, joints_B, verts_A, verts_B, temporal_mask=None,
        foot_contact_masks=None,
    ):
        # 1. Reg Pose
        mask_penalty_ratio = 3.0

        T = params_A['pose_body'].shape[0]
        device = params_A['pose_body'].device

        if temporal_mask is not None:
            # [T, 1]
            # normalized mask (0~1) -> weights (1.0 ~ ratio)
            # mask=0 -> weight=1.0
            # mask=1 -> weight=mask_penalty_ratio
            w_base = 1.0 + (mask_penalty_ratio - 1.0) * temporal_mask.view(-1, 1, 1)
        else:
            w_base = torch.ones(T, 1, device=device)

        # 归一化因子 (用于替代 mean 的分母)
        # 注意：为了保持 Loss 量级一致，我们除以 sum(w) 而不是 T
        # 或者为了简单起见，我们保持 mean 的逻辑，只是在分子上加权 (这样 Loss 会变大，有助于强调)
        # 这里采用：加权后的 Mean (即 sum(loss * w) / T )，这样交互区的梯度会变大

        # --- 1. Reg Pose (正则化) ---
        def calc_split_reg(curr, init, weights):
            loss = 0.0

            # 辅助函数：计算加权均方误差
            def weighted_mse(pred, target, w):
                # pred: [T, D], w: [T, 1]
                diff = (pred - target) ** 2
                # 广播 w 到 D 维度，计算加权平均
                return torch.mean(diff * w)

            # 身体姿态
            if 'pose_body' in curr:
                # print(curr['pose_body'].shape, init['pose_body'].shape, weights.shape)
                loss += weighted_mse(curr['pose_body'], init['pose_body'], weights)

            # 手部姿态 (手部比较灵活，即便在接触时，也允许一定微调，所以权重系数稍微给小点?)
            # 策略：保持 mask 的强化，但基础系数保持 0.5
            if 'pose_lhand' in curr:
                loss += 0.5 * weighted_mse(curr['pose_lhand'], init['pose_lhand'], weights)
            if 'pose_rhand' in curr:
                loss += 0.5 * weighted_mse(curr['pose_rhand'], init['pose_rhand'], weights)

            # 全局旋转 (非常重要，接触时根节点乱转会导致严重穿模)
            if 'root_orient' in curr:
                loss += 2.0 * weighted_mse(curr['root_orient'], init['root_orient'], weights)

            # 位移 (非常重要，接触时不能乱飘)
            if 'trans' in curr:
                loss += 5.0 * weighted_mse(curr['trans'], init['trans'], weights)

            return loss

        loss_reg = calc_split_reg(params_A, params_A_init, w_base) + \
                   calc_split_reg(params_B, params_B_init, w_base)

        # --- 2. Smoothness & Acc (平滑) ---

        def calc_smooth_acc(joints, weights):
            # joints: [T, J, 3]
            # weights: [T, 1]

            # velocity: [T-1, J, 3]
            velocity = joints[1:] - joints[:-1]
            # 对应的权重: 取 t=1 到 T 的权重 (或者 t=0 到 T-1，这里取后半段比较合理，表示"进入该帧的速度")
            w_vel = weights[1:].unsqueeze(-1)  # [T-1, 1, 1] 用于广播到 J, 3

            # 加权平滑 Loss
            loss_smooth = torch.mean((velocity ** 2) * w_vel)

            # acceleration: [T-2, J, 3]
            if joints.shape[0] >= 3:
                acc = joints[2:] - 2 * joints[1:-1] + joints[:-2]
                w_acc = weights[2:].unsqueeze(-1)  # [T-2, 1, 1]
                loss_acc = torch.sum((acc ** 2) * w_acc) / (w_acc.sum().float()+1e-6)
            else:
                loss_acc = torch.tensor(0.0, device=device)

            return loss_smooth, loss_acc

        loss_smooth_A, loss_acc_A = calc_smooth_acc(joints_A, w_base)
        loss_smooth_B, loss_acc_B = calc_smooth_acc(joints_B, w_base)
        loss_smooth = loss_smooth_A + loss_smooth_B
        loss_acc = loss_acc_A + loss_acc_B

        # 3. Foot skating.  The default path is an exact factorization of the
        # released predicate.  The preregistered ablation changes only when
        # that same predicate is evaluated: once on iteration 0, then frozen.
        if self.foot_contact_mask_policy == 'released_dynamic':
            if foot_contact_masks is not None:
                raise ValueError(
                    'released_dynamic must not receive frozen contact masks'
                )
            loss_skate = (
                released_dynamic_skate_loss(joints_A)
                + released_dynamic_skate_loss(joints_B)
            )
        elif self.foot_contact_mask_policy == 'frozen_initial':
            if foot_contact_masks is None or len(foot_contact_masks) != 2:
                raise RuntimeError(
                    'frozen_initial requires two iteration-0 contact masks'
                )
            loss_skate = (
                skate_loss_with_mask(joints_A, foot_contact_masks[0])
                + skate_loss_with_mask(joints_B, foot_contact_masks[1])
            )
        else:
            if foot_contact_masks is None or len(foot_contact_masks) != 2:
                raise RuntimeError(
                    'metric-aligned foot loss requires two iteration-0 '
                    'grounded masks'
                )
            loss_skate = (
                initial_or_current_grounded_horizontal_mean_speed(
                    joints_A, foot_contact_masks[0]
                )
                + initial_or_current_grounded_horizontal_mean_speed(
                    joints_B, foot_contact_masks[1]
                )
            )
        #
        # # 4. Ground Penetration
        loss_ground = torch.mean(F.relu(-verts_A[:, :, 1])) + torch.mean(F.relu(-verts_B[:, :, 1]))

        return loss_reg, loss_smooth, loss_acc, loss_skate, loss_ground

    # ==========================================
    #           Forward (用户主控)
    # ==========================================
    def forward(self,
                params_A, params_B,
                params_A_init, params_B_init,
                temporal_mask,  # [Tensor T] 时间掩码/权重
                epoch,
                contact_indices_A=None,  # [List or Tensor] A 的接触顶点索引
                contact_indices_B=None,  # [List or Tensor] B 的接触顶点索引
                interaction_pairs=None,  # 可选的语义接触对
                fine_region_pairs=None,  # Chi3D 已标注好的 75-region 精细接触对
                freeze_epoch = 100
                ):
        """
        :param contact_indices_A: Tensor of vertex indices for Person A involved in contact
        :param contact_indices_B: Tensor of vertex indices for Person B involved in contact
        :param temporal_mask: Tensor [T], weight of contact for each frame (0.0 to 1.0+)
        """

        # 1. Geometry Calculation
        # tick = time.time()
        v1, j1, n1 = self._compute_geometry(params_A)
        v2, j2, n2 = self._compute_geometry(params_B)
        if self.foot_contact_mask_policy in (
            'frozen_initial',
            'initial_or_current_grounded_horizontal_mean_speed',
        ):
            if self.frozen_foot_contact_masks is None:
                if int(epoch) != 0:
                    raise RuntimeError(
                        'frozen_initial must first be called at iteration 0'
                    )
                if self.foot_contact_mask_policy == 'frozen_initial':
                    freeze_mask = freeze_initial_contact_mask
                else:
                    freeze_mask = freeze_initial_grounded_foot_mask
                self.frozen_foot_contact_masks = (
                    freeze_mask(j1),
                    freeze_mask(j2),
                )
        volume_1 = volume_2 = None
        contact_volume_1 = contact_volume_2 = None
        if self.volumetric_smpl is not None:
            self.volumetric_smpl.begin_iteration(
                seed=self.volumetric_smpl_seed + int(epoch)
            )
            output_1, gender_1 = self.volumetric_smpl.body(params_A)
            output_2, gender_2 = self.volumetric_smpl.body(params_B)
            volume_1 = (output_1, gender_1)
            volume_2 = (output_2, gender_2)
            # Eq. 4 role weights act on the source character.  Detaching the
            # paired body preserves that directional optimization contract.
            contact_volume_1 = (
                self.volumetric_smpl.detached_output(output_1), gender_1
            )
            contact_volume_2 = (
                self.volumetric_smpl.detached_output(output_2), gender_2
            )
        # print('time for geometry calculation: {:.3f}s'.format(time.time() - tick))

        role_pairs = interaction_pairs
        if role_pairs is None and fine_region_pairs is not None:
            role_pairs = [
                {
                    'p1_ids': [int(pair['p1_rid'])],
                    'p2_ids': [int(pair['p2_rid'])],
                    'contact_type': 'flat',
                    'align_type': 'strict',
                    'w_p1': 0.5,
                    'w_p2': 0.5,
                }
                for pair in fine_region_pairs
            ]

        # 2. Explicit Contact Loss (Attract + Align)
        # 完全由用户输入的 indices 和 mask 决定
        if epoch == 0 and not self.swap_determined and role_pairs:
            self._determine_role_assignment(
                v1, v2, n1, n2, role_pairs, temporal_mask,
                volume_A=contact_volume_1,
                volume_B=contact_volume_2,
            )

        if self.swapped:
            # 交换：B 是 P1，A 是 P2
            vA, vB = v2, v1
            nA, nB = n2, n1
            jA, jB = j2, j1
            params_A, params_B = params_B, params_A
            params_A_init, params_B_init = params_B_init, params_A_init
            volume_A, volume_B = volume_2, volume_1
            contact_volume_A, contact_volume_B = (
                contact_volume_2, contact_volume_1
            )
            foot_contact_masks = (
                None if self.frozen_foot_contact_masks is None else
                (
                    self.frozen_foot_contact_masks[1],
                    self.frozen_foot_contact_masks[0],
                )
            )
            # 如果有显式索引，也要交换 (假设 idx_A 对应 params_A)
            # contact_indices_A, contact_indices_B = contact_indices_B, contact_indices_A
        else:
            # 保持：A 是 P1，B 是 P2
            vA, vB = v1, v2
            nA, nB = n1, n2
            jA, jB = j1, j2
            volume_A, volume_B = volume_1, volume_2
            contact_volume_A, contact_volume_B = (
                contact_volume_1, contact_volume_2
            )
            foot_contact_masks = self.frozen_foot_contact_masks
            # params_P1, params_P2 = params_A, params_B
            # init_P1, init_P2 = params_A_init, params_B_init
            # idx_P1, idx_P2 = contact_indices_A, contact_indices_B

        # ``freeze_epoch`` is the first fine-stage iteration.  Candidate
        # regions have already been selected by the preceding coarse step, so
        # freeze before computing this iteration's contact loss (50/50 for
        # the paper's iterations 0..49 and 50..99).
        if epoch == freeze_epoch and not self.is_anchored and role_pairs:
            with torch.no_grad():
                self.freeze_contact_candidates(
                    vA, vB, nA, nB, role_pairs, temporal_mask
                )

        if interaction_pairs is not None:
            loss_attract, loss_align = self._calc_semantic_contact_loss(
                vA, vB, nA, nB,  # <--- 传入法线
                interaction_pairs,
                temporal_mask=temporal_mask,
                volume_A=contact_volume_A,
                volume_B=contact_volume_B,
            )
        elif fine_region_pairs is not None:
            loss_attract, loss_align = self._calc_fine_region_contact_loss(
                vA, vB, nA, nB,
                fine_region_pairs,
                temporal_mask=temporal_mask,
                volume_A=contact_volume_A,
                volume_B=contact_volume_B,
            )
        elif contact_indices_A is not None and contact_indices_B is not None:
            loss_attract, loss_align = self._calc_explicit_contact_loss(
                vA, nA, contact_indices_A,
                vB, nB, contact_indices_B,
                temporal_mask
            )
        else:
            loss_attract, loss_align = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # 3. Penetration Loss (Global or Sub-sampled)
        # 建议传入 contact indices 或更大的区域来加速
        # 这里为了安全，演示计算接触区域附近的防穿模，或者您可以传入专门的 pene_indices
        # loss_pene = self._calc_penetration_loss(vA, vB, nA, nB, contact_indices_A, contact_indices_B)
        # 注意：如果只计算 contact_indices 的穿模，非接触区域穿模会被忽略。
        # 如果显存允许，建议随机采样全身计算 pene:
        # loss_pene = torch.tensor(0, device=self.device)
        loss_pene = self._calc_penetration_loss(
            vA, vB, nA, nB,
            volume_A=volume_A,
            volume_B=volume_B,
            temporal_mask=temporal_mask,
        )

        # 4. Auxiliary Losses
        l_reg, l_smooth, loss_acc, loss_skate, loss_ground = self._calc_aux_losses(
            params_A, params_B, params_A_init, params_B_init, jA, jB, vA,
            vB, temporal_mask=temporal_mask,
            foot_contact_masks=foot_contact_masks,
        )
        # l_reg, l_smooth, loss_acc, loss_skate, loss_ground = torch.tensor(0, device=self.device) ,torch.tensor(0, device=self.device) ,torch.tensor(0, device=self.device)\
        #     ,torch.tensor(0, device=self.device) ,torch.tensor(0, device=self.device)
        # 5. Total Sum
        total_loss = (self.weights['reg'] * l_reg) + \
                     (self.weights['smooth'] * l_smooth) + \
                     (self.weights['smooth'] * loss_acc) + \
                     (self.weights['skate'] * loss_skate) + \
                     (self.weights['pene'] * loss_pene) + \
                     (self.weights['ground'] * loss_ground) + \
        (self.weights['attract'] * loss_attract)
        # (self.weights['align'] * loss_align)


        loss_dict = {
            'total': total_loss,
            'attract': loss_attract,
            'align': loss_align,
            'pene': loss_pene,
            'reg': l_reg,
            'smooth': l_smooth,
            'acc': loss_acc,
            'skate': loss_skate,
            'ground': loss_ground
        }

        return total_loss, loss_dict

    def forward_semantic_metric(self, paramA, paramB, interaction_pairs, temporal_mask=None):
        verts_A, joints_A, n_A  = self._compute_geometry(paramA)
        verts_B, joints_B, n_B  = self._compute_geometry(paramB)
        volume_A = volume_B = None
        if self.volumetric_smpl is not None:
            self.volumetric_smpl.begin_iteration(seed=self.volumetric_smpl_seed)
            output_A, gender_A = self.volumetric_smpl.body(paramA)
            output_B, gender_B = self.volumetric_smpl.body(paramB)
            volume_A = (output_A, gender_A)
            volume_B = (output_B, gender_B)

        if (temporal_mask is None) and (interaction_pairs is not None):
            labels, min_dists = get_spatial_min_interaction_label_single(joints_A,
                                                                         joints_B,
                                                                         strict_thresh=0.01, loose_thresh=0.3,
                                                                         spatial_tolerance=0.1, smooth_window=5)
            # labels = keep_longest_and_ramp(labels)
            temporal_mask = labels.to(self.device)

        self._determine_role_assignment(
            verts_A, verts_B, n_A, n_B, interaction_pairs, temporal_mask,
            volume_A=volume_A,
            volume_B=volume_B,
        )

        if self.swapped:
            # 交换：B 是 P1，A 是 P2
            verts_A, verts_B = verts_B, verts_A
            n_A, n_B = n_B, n_A
            volume_A, volume_B = volume_B, volume_A



        loss_attract, loss_align = self._calc_semantic_contact_loss(
            verts_A, verts_B, n_A, n_B,
            interaction_pairs,
            temporal_mask=temporal_mask,
            volume_A=volume_A,
            volume_B=volume_B,
        )

        loss_pene = self._calc_penetration_loss(
            verts_A, verts_B, n_A, n_B,
            volume_A=volume_A,
            volume_B=volume_B,
            temporal_mask=temporal_mask,
        )
        return loss_attract, loss_pene
