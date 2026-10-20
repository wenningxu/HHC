"""INVALID experimental optimizer rewrite retained only for diagnosis; do not import."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm  # 推荐安装 tqdm 用来显示进度条
from optimization.losses import DyadicInteractionLoss
from optimization.text_2_regions import ContactExtractor
import copy
import os
import codecs as cs
from os.path import join as pjoin
from utils.paramUtil import CANDIDATES
from utils.motion_process import get_spatial_min_interaction_label_single, keep_longest_and_ramp
from tools.test_data import _compute_geometry
from optimization.optimization_variable_policy import (
    OPTIMIZATION_VARIABLE_POLICIES,
    configure_optimization_variables,
    optimization_variable_diagnostics,
    project_optimization_variables,
)
import time

PAPER_OPTIMIZATION_ITERATIONS = 100
PAPER_OPTIMIZATION_LR = 2e-2
PAPER_FINE_STAGE_START = 50


def _volumetric_smpl_diagnostics(loss_module):
    """Return backend provenance without assuming VolumetricSMPL is active."""
    backend = loss_module.volumetric_smpl
    if backend is None:
        return {}, {
            'status': 'not_used',
            'reason': 'nearest_vertex signed-distance backend',
        }
    return (
        dict(backend.checkpoint_records),
        dict(backend.representation_contract),
    )


def optimize_full_body(
        initial_data_A,  # 角色A的初始数据字典 {'pose':..., 'trans':..., 'beta':...}
        initial_data_B,  # 角色B的初始数据字典
        contacts_A=None,  # 角色A的接触点信息
        contacts_B=None,  # 角色B的接触点信息
        temporal_mask=None,  # 时间掩码，指定哪些帧需要优化
        interaction_pairs=None,  # 可选，指定哪些部位之间有交互
        fine_region_pairs=None,  # Chi3D 已标注好的 75-region 精细接触对
        num_epochs=PAPER_OPTIMIZATION_ITERATIONS,
        lr=PAPER_OPTIMIZATION_LR,
        device='cuda',
        loss_weights=None,
        freeze_epoch=None,
        volumetric_smpl_seed=1234,
        volumetric_smpl_query_chunk=1024,
        volumetric_smpl_frame_batch_size=1,
        volumetric_smpl_backend=None,
        signed_distance_backend='volumetric_smpl',
        foot_contact_mask_policy='released_dynamic',
        contact_loss_reduction_policy='paper_literal_raw_sum',
        optimization_variable_policy='released_all_per_frame',
        return_diagnostics=False,
):
    """
    执行 High-Five 动作优化
    """
    print(f"Start Optimization on {device}...")

    # ==========================================
    # 1. 准备优化变量 (Variables to Optimize)
    # ==========================================
    # 关键：我们需要对这些变量求导，所以必须 set requires_grad=True
    # 我们使用 .clone().detach() 创建副本，不破坏原始数据

    # --- Person A ---
    # 假设 pose 包含了 global_orient 和 body_pose (根据您的SMPL接口调整)
    # 如果您的接口是分开的，请分别创建变量并放入 params 字典
    # ==========================================
    # 1. 准备优化变量 (Variables to Optimize)
    # ==========================================
    # 辅助函数：快速创建可优化的 Tensor
    def create_optim_vars(data_dict):
        vars_dict = {}
        # 需要优化的参数列表
        keys_to_optimize = ['root_orient', 'trans', 'pose_body', 'pose_lhand', 'pose_rhand']

        for key in keys_to_optimize:
            if key in data_dict:
                data = data_dict[key]

                # 1. 如果是 numpy 数组，先转为 tensor
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)

                # 2. 统一转换为 float32 (SMPL通常只需要单精度)
                # 3. 移动到设备
                # 4. 开启梯度
                vars_dict[key] = data.clone().float().to(device).requires_grad_(True)

        # Betas 处理 (通常不需要优化)
        if 'betas' in data_dict:
            data = data_dict['betas']

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # Betas 不需要 requires_grad=True
            vars_dict['betas'] = data.float().to(device)


        if 'gender' in data_dict:
            data = data_dict['gender']
            if isinstance(data, np.ndarray):
                if data.size != 1:
                    raise ValueError('each optimized person requires one gender')
                data = data.reshape(-1)[0].item()
            if isinstance(data, torch.Tensor):
                if data.numel() != 1:
                    raise ValueError('each optimized person requires one gender')
                data = data.detach().cpu().item()
            if isinstance(data, (bytes, np.bytes_)):
                data = data.decode('utf-8')

            if data in ('male', 0):
                gender_code = 0
            elif data in ('female', 1):
                gender_code = 1
            else:
                raise ValueError(
                    'paper VolumetricSMPL optimization requires male/female '
                    f'weights, got {data!r}'
                )

            # Betas 不需要 requires_grad=True
            vars_dict['gender'] = torch.tensor(gender_code).long().to(device)

        return vars_dict

    # 创建变量字典
    vars_A = create_optim_vars(initial_data_A)
    vars_B = create_optim_vars(initial_data_B)
    if optimization_variable_policy not in OPTIMIZATION_VARIABLE_POLICIES:
        raise ValueError(
            'optimization_variable_policy must be one of '
            f'{OPTIMIZATION_VARIABLE_POLICIES}'
        )
    variable_policy_hook_handles = (
        configure_optimization_variables(vars_A, optimization_variable_policy)
        + configure_optimization_variables(vars_B, optimization_variable_policy)
    )
    if temporal_mask is not None:
        temporal_mask = torch.as_tensor(
            temporal_mask, dtype=torch.float32, device=device
        ).reshape(-1)
        frame_count = vars_A['trans'].shape[0]
        if temporal_mask.shape[0] != frame_count:
            raise ValueError(
                'temporal_mask length must equal the motion frame count: '
                f'{temporal_mask.shape[0]} vs {frame_count}'
            )

    # 收集所有需要梯度的参数给 Optimizer
    optimizer_params = []
    for v in vars_A.values():
        if v.requires_grad: optimizer_params.append(v)
    for v in vars_B.values():
        if v.requires_grad: optimizer_params.append(v)

    # ==========================================
    # 2. 设置优化器与 Loss
    # ==========================================
    optimizer = optim.Adam(optimizer_params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180], gamma=0.1)
    # 实例化 Loss (确保使用最新版的 DyadicInteractionLoss)
    loss_module = DyadicInteractionLoss(
        device=device,
        signed_distance_backend=signed_distance_backend,
        volumetric_smpl_seed=volumetric_smpl_seed,
        volumetric_smpl_query_chunk=volumetric_smpl_query_chunk,
        volumetric_smpl_frame_batch_size=volumetric_smpl_frame_batch_size,
        volumetric_smpl_backend=volumetric_smpl_backend,
        foot_contact_mask_policy=foot_contact_mask_policy,
        contact_loss_reduction_policy=contact_loss_reduction_policy,
    ).to(device)
    if loss_weights is not None:
        unknown = sorted(set(loss_weights) - set(loss_module.weights))
        if unknown:
            raise ValueError(f'unknown optimization loss weights: {unknown}')
        loss_module.weights.update({
            key: float(value) for key, value in loss_weights.items()
        })
    if freeze_epoch is None:
        # Paper supplement: 100 iterations, half coarse and half fine contact.
        freeze_epoch = num_epochs // 2
    if not 0 <= freeze_epoch < num_epochs:
        raise ValueError('freeze_epoch must be in [0, num_epochs)')

    # 准备“初始状态”作为正则化参考 (detach)
    # 直接深度拷贝一份 detach 的字典
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

    params_A_init = create_init_dict(initial_data_A)
    params_B_init = create_init_dict(initial_data_B)

    if (temporal_mask is None) and ((interaction_pairs is not None) or (fine_region_pairs is not None)):
        _, joints_A, _ = loss_module._compute_geometry(params_A_init)
        _, joints_B, _ = loss_module._compute_geometry(params_B_init)
        labels, min_dists = get_spatial_min_interaction_label_single(joints_A,
                                                                     joints_B,
                                                                     strict_thresh=0.01, loose_thresh=0.3,
                                                                     spatial_tolerance=0.1, smooth_window=5)
        # labels = keep_longest_and_ramp(labels)
        temporal_mask = labels.to(device)
    elif (temporal_mask is None) and (interaction_pairs is None):
        temporal_mask = torch.ones(len(params_A_init['trans'])).to(device)

    # ==========================================
    # 3. 优化循环 (Optimization Loop)
    # ==========================================
    loop = tqdm(range(num_epochs))
    tick = time.time()
    loss_history = []
    for epoch in loop:
        # A. 清空梯度
        optimizer.zero_grad()

        # B. 组装当前参数字典 (不需要再手动组装了，vars_A 本身就是字典)
        # 只要 vars_A 里的 tensor 是 requires_grad 的，计算图就会建立
        # 注意：这里的 Key 必须与 loss_module._compute_geometry 中 smpl() 接收的参数名一致

        # C. 计算 Loss
        loss, loss_dict = loss_module(
            vars_A, vars_B,  # 当前参数 (带梯度)
            params_A_init, params_B_init,  # 初始参数 (无梯度)
            temporal_mask,
            epoch,
            contact_indices_A =contacts_A,
            contact_indices_B = contacts_B,
            interaction_pairs=interaction_pairs,
            fine_region_pairs=fine_region_pairs,
            freeze_epoch=freeze_epoch,
        )
        loss_history.append({
            'iteration': int(epoch),
            'contact_stage': (
                'fine_one_to_one' if loss_module.is_anchored else 'coarse'
            ),
            **{
                key: float(value.detach().cpu())
                if isinstance(value, torch.Tensor) else float(value)
                for key, value in loss_dict.items()
            },
        })

        # D. 反向传播
        loss.backward()

        # E. 更新参数
        optimizer.step()
        project_optimization_variables(
            vars_A, params_A_init, optimization_variable_policy
        )
        project_optimization_variables(
            vars_B, params_B_init, optimization_variable_policy
        )
        # scheduler.step()
        # F. 日志
        #         loss_dict = {
        #             'total': total_loss,
        #             'attract': loss_attract,
        #             'align': loss_align,
        #             'pene': loss_pene,
        #             'reg': l_reg,
        #             'smooth': l_smooth,
        #             'acc': loss_acc,
        #             'skate': loss_skate,
        #             'ground': loss_ground
        #         }
        loop.set_postfix({
            'Total': f"{loss.item():.3f}",
            'Attr': f"{loss_dict.get('attract', 0):.4f}",
            'Pene': f"{loss_dict.get('pene', 0):.4f}",
            'Reg': f"{loss_dict.get('reg', 0):.4f}",
            'align': f"{loss_dict.get('align', 0):.4f}",
            'smooth': f"{loss_dict.get('smooth', 0):.4f}",
            'acc': f"{loss_dict.get('acc', 0):.4f}",
            'skate': f"{loss_dict.get('skate', 0):.4f}",
            'ground': f"{loss_dict.get('ground', 0):.4f}",

        })

    elapsed_seconds = time.time() - tick
    print("Optimization Finished.")
    print('time taken: {:.3f} seconds'.format(elapsed_seconds))

    # ==========================================
    # 4. 返回结果 (Return Results)
    # ==========================================
    # 返回优化后的数据 (detach去掉梯度，转回cpu或numpy方便后续保存)
    def detach_dict(vars_dict):
        res = {}
        for k, v in vars_dict.items():
            res[k] = v.detach().cpu().numpy()  # 去掉梯度，准备保存
        return res

    final_data_A = detach_dict(vars_A)
    final_data_B = detach_dict(vars_B)

    if return_diagnostics:
        (
            volumetric_smpl_checkpoints,
            volumetric_smpl_representation,
        ) = _volumetric_smpl_diagnostics(loss_module)
        diagnostics = {
            'iterations': int(num_epochs),
            'learning_rate': float(lr),
            'fine_stage_start': int(freeze_epoch),
            'elapsed_seconds': float(elapsed_seconds),
            'loss_weights': dict(loss_module.weights),
            'used_fine_contact_stage': bool(loss_module.is_anchored),
            'frozen_contact_region_pairs': len(loss_module.fixed_pairs),
            'frozen_contact_vertex_pairs': int(sum(
                anchors['idx_A'].numel()
                for anchors in loss_module.fixed_pairs.values()
            )),
            'signed_distance_backend': loss_module.signed_distance_backend,
            'volumetric_smpl_seed': int(volumetric_smpl_seed),
            'volumetric_smpl_query_chunk': int(volumetric_smpl_query_chunk),
            'volumetric_smpl_frame_batch_size': int(
                volumetric_smpl_frame_batch_size
            ),
            'volumetric_smpl_backend_reused': bool(
                volumetric_smpl_backend is not None
            ),
            'foot_contact_mask_policy': loss_module.foot_contact_mask_policy,
            'contact_loss_reduction_policy': (
                loss_module.contact_loss_reduction_policy
            ),
            'optimization_variable_policy': optimization_variable_policy,
            'optimization_variable_invariants': {
                'p1': optimization_variable_diagnostics(
                    vars_A, params_A_init, optimization_variable_policy
                ),
                'p2': optimization_variable_diagnostics(
                    vars_B, params_B_init, optimization_variable_policy
                ),
            },
            'volumetric_smpl_checkpoints': volumetric_smpl_checkpoints,
            'volumetric_smpl_representation': volumetric_smpl_representation,
            'loss_history': loss_history,
        }
        return final_data_A, final_data_B, diagnostics
    return final_data_A, final_data_B


def create_trapezoidal_mask(total_frames, t_start, t_peak_start, t_peak_end, t_end, device='cuda'):
    """
    生成梯形时间 Mask
    :param total_frames: 总帧数 (例如 100)
    :param t_start: 开始变大的帧 (73) -> value=0
    :param t_peak_start: 完全接触开始帧 (76) -> value=1
    :param t_peak_end: 开始分离帧 (83) -> value=1
    :param t_end: 结束帧 (86) -> value=0
    """
    mask = torch.zeros(total_frames, device=device)

    # 1. 渐入阶段 (Rise): t_start -> t_peak_start
    # math: (t - t_start) / (t_peak_start - t_start)
    rise_len = t_peak_start - t_start
    if rise_len > 0:
        # linspace 生成从 0 到 1 的序列
        # 比如 73-76 是 3帧，我们生成 0.0, 0.33, 0.66 (不包括最后的1.0，留给plateau)
        rise_values = torch.linspace(0, 1, steps=rise_len + 1)[:-1]
        mask[t_start:t_peak_start] = rise_values.to(device)

    # 2. 保持阶段 (Plateau): t_peak_start -> t_peak_end
    # math: 1.0
    # 注意 python 切片是左闭右开，所以要 +1 才能包含 t_peak_end
    mask[t_peak_start: t_peak_end + 1] = 1.0

    # 3. 渐出阶段 (Fall): t_peak_end -> t_end
    # math: 1.0 - (t - t_peak_end) / (t_end - t_peak_end)
    fall_len = t_end - t_peak_end
    if fall_len > 0:
        # linspace 生成从 1 到 0 的序列
        fall_values = torch.linspace(1, 0, steps=fall_len + 1)[1:]  # [1:] 去掉开头的1.0(已经在plateau了)
        mask[t_peak_end + 1: t_end + 1] = fall_values.to(device)

    return mask


# ==========================================
# 使用示例
# ==========================================

# ==========================================
# 5. 调用示例 (Usage Example)
# ==========================================
def smplx_2_amass(dict):
    pose = dict['poses']  # [T, 72]
    print(pose.shape)
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



if __name__ == "__main__":
    # 假设您已经有了 smpl, 和加载好的数据 data_a, data_b
    # data_a 结构示例: {'pose': tensor[T, 72], 'trans': tensor[T, 3], 'beta': ...}
    data_dir = 'results/vq_mask_gen_4_ablation_adalnshape_1_cfg_val_ground'
    val_file = 'dataset/interx/val.txt'
    text_dir = 'dataset/interx/texts_processed'
    contact_dir = ''
    id_list = ['G057T001A005R004']

    # with cs.open(val_file, 'r') as f:
    #     for line in f.readlines():
    #         id_list.append(line.strip())
    # 确保 smpl 模型参数被冻结 (我们只优化输入 pose，不优化模型本身的权重)
    for name in id_list:
        try:
            data_p1 = np.load(os.path.join(data_dir, name+'_pred_p1.npz'), allow_pickle=True)
            data_p2 = np.load(os.path.join(data_dir, name+'_pred_p2.npz'), allow_pickle=True)
            data_p1 = dict(data_p1)
            data_p2 = dict(data_p2)
        except:
            print(f'no file for {name}')
            continue

        text_list = []
        with cs.open(os.path.join(text_dir, name+'.txt')) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_list.append(caption)
        # print(len(data_p1.keys()))
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('set GEMINI_API_KEY before running the live LLM example')

        # 1. 实例化
        # text_list[0] = 'Two people wrestle with their hands, sitting face to face. The first person firmly grips the other persons left hand with his left hand. '
        extractor = ContactExtractor(api_key=api_key)
        contacts = extractor.get_contacts(text_list[0], CANDIDATES)
        print(text_list[0], contacts)

        if len(data_p1.keys()) < 6:
            data_p1 = smplx_2_amass(data_p1)
        if len(data_p2.keys()) < 6:
            data_p2 = smplx_2_amass(data_p2)
         # 假设接触点索引已经准备好
        _, joints = _compute_geometry(data_p1)
        root_transl = data_p1['trans']
        root_transl[:, 1] = root_transl[:, 1] - np.min(joints[:, :, 1])  # min-value align to the ground
        data_p1['trans'] = root_transl

        _, joints = _compute_geometry(data_p2)
        root_transl = data_p2['trans']
        root_transl[:, 1] = root_transl[:, 1] - np.min(joints[:, :, 1])  # min-value align to the ground
        # root_transl[:, 0] -= 0.1
        data_p2['trans'] = root_transl

        T = len(data_p1['trans'])  # 假设总帧数 (请根据你的实际数据修改)
        DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        temporal_mask = create_trapezoidal_mask(104, 50, 50, 52, 52, device=DEVICE)


        refined_A, refined_B = optimize_full_body(
            initial_data_A=data_p1,
            initial_data_B=data_p2,
            temporal_mask=temporal_mask,
            interaction_pairs=None,
            num_epochs=PAPER_OPTIMIZATION_ITERATIONS,
            lr=PAPER_OPTIMIZATION_LR,
            device=DEVICE
        )

        for k,v in refined_A.items():
            print(f"refined_A[{k}]: shape={v.shape}, dtype={v.dtype}")
        # 这里的 refined_A['pose'] 就是修复好接触、无穿模的新动作
        print("Optimization done! Saving results...")
        gender_list = ['male', 'female', 'neutral']
        save_path = f'results/optimization_pipeline_gt/{name}'
        os.makedirs(save_path, exist_ok=True)
        for i, data in enumerate([refined_A, refined_B]):
            pose_hml = np.zeros([len(refined_A['trans']), 165]).reshape(-1, 55, 3)
            pose_hml[:, 0, :] = data['root_orient']
            pose_hml[:, 1:22, :] = data['pose_body']
            pose_hml[:, 25:40, :] = data['pose_lhand']
            pose_hml[:, 40:55, :] = data['pose_rhand']


            # np.savez(os.path.join(save_path, f'refined_person_{i+1}.npz'),
            #          poses=pose_hml, trans = data['trans'], betas=data['betas'], gender=gender_list[int(data['gender'])], mocap_framerate=30)
    # np.savez('refined_motion.npz', pose_a=refined_A['pose'].cpu().numpy(), ...)
