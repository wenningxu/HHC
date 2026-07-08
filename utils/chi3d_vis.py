import os

import numpy as np
import torch

from utils.motion_process import MotionNormalizerTorchChi3D, recover_interx


_CHI3D_NORMALIZER = MotionNormalizerTorchChi3D()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def denormalize_chi3d_motion(motion):
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)
    return _CHI3D_NORMALIZER.backward(motion)


def _to_numpy_motion(motion):
    if isinstance(motion, torch.Tensor):
        return motion.detach().cpu()
    return torch.from_numpy(motion)


def _load_chi3d_betas(group_id, base_dir="dataset/chi3d/betas"):
    betas = np.load(os.path.join(base_dir, f"{int(group_id)}.npy")).astype(np.float32)
    if betas.shape == (2, 10):
        return betas
    return np.zeros((2, 10), dtype=np.float32)


def _extract_sample_group_id(group_ids):
    if isinstance(group_ids, torch.Tensor):
        group_ids = group_ids.detach().cpu()
        if group_ids.dim() > 1:
            group_ids = group_ids[0]
        group_ids = group_ids.tolist()

    if isinstance(group_ids, (list, tuple)):
        if len(group_ids) == 4:
            return int(group_ids[1]) + 2
        return int(group_ids[0])

    return int(group_ids)


def export_chi3d_motion_npz(motion_denorm, out_dir, stem, group_id, fps=30):
    ensure_dir(out_dir)
    motion_denorm = _to_numpy_motion(motion_denorm)
    if motion_denorm.dim() == 3:
        if motion_denorm.shape[0] != 1:
            raise ValueError("export_chi3d_motion_npz expects a single sample")
        motion_denorm = motion_denorm[0]

    trans1, pose1, trans2, pose2 = recover_interx(motion_denorm)
    betas = _load_chi3d_betas(group_id)

    np.savez(
        os.path.join(out_dir, f"{stem}_p1.npz"),
        poses=pose1,
        trans=trans1,
        betas=betas[0],
        gender="neutral",
        mocap_framerate=fps,
    )
    np.savez(
        os.path.join(out_dir, f"{stem}_p2.npz"),
        poses=pose2,
        trans=trans2,
        betas=betas[1],
        gender="neutral",
        mocap_framerate=fps,
    )


def run_vq_sample_export(net, sample_batch, out_dir, fps=30):
    motion, group_ids = sample_batch
    device = next(net.parameters()).device
    motion = motion.to(device).float()
    group_ids = group_ids.to(device)
    sample_group_id = _extract_sample_group_id(group_ids)
    with torch.no_grad():
        recon_motion, _, _ = net(motion, group_ids)
    motion_denorm = denormalize_chi3d_motion(motion[:1])
    recon_denorm = denormalize_chi3d_motion(recon_motion[:1])
    export_chi3d_motion_npz(motion_denorm, out_dir, "sample_gt", sample_group_id, fps=fps)
    export_chi3d_motion_npz(recon_denorm, out_dir, "sample_recon", sample_group_id, fps=fps)


def build_fixed_chi3d_templates(action_to_id, groups=(2, 3, 4), max_templates=3):
    templates = []
    for action_name in sorted(action_to_id.keys()):
        for group in groups:
            templates.append(
                {
                    "action_name": action_name,
                    "action_id": action_to_id[action_name],
                    "group_id": group,
                    "name": f"{action_name}_G{group}",
                }
            )
            if len(templates) >= max_templates:
                return templates
    return templates


def run_mgt_sample_export(trans_net, vq_net, template_conditions, out_dir, m_length=64, fps=30):
    ensure_dir(out_dir)
    device = next(trans_net.parameters()).device
    action_ids = torch.tensor([item["action_id"] for item in template_conditions], device=device, dtype=torch.long)
    group_ids = torch.tensor([item["group_id"] for item in template_conditions], device=device, dtype=torch.long)
    lengths = torch.tensor([m_length for _ in template_conditions], device=device, dtype=torch.long)

    with torch.no_grad():
        if hasattr(trans_net, "sample_intra_cas"):
            code_idx = trans_net.sample_intra_cas(action_ids, group_ids, m_length=lengths)
        else:
            code_idx = trans_net.sample(action_ids, group_ids, m_length=lengths)
        code_idx = sanitize_code_idx(code_idx, vq_net.num_code)
        pred_motion = vq_net(code_idx, to_chi3d_vq_group(group_ids), type="decode")

    pred_denorm = denormalize_chi3d_motion(pred_motion)
    for idx, item in enumerate(template_conditions):
        export_chi3d_motion_npz(pred_denorm[idx:idx + 1], out_dir, f"{item['name']}_pred", item["group_id"], fps=fps)


def to_chi3d_vq_group(group_ids):
    group_idx = group_ids - 2
    return torch.stack(
        [
            torch.zeros_like(group_idx),
            group_idx,
            torch.ones_like(group_idx),
            group_idx,
        ],
        dim=1,
    ).long()


def sanitize_code_idx(code_idx, num_code):
    code_idx = code_idx.clone()
    code_idx = torch.where(code_idx >= num_code, torch.zeros_like(code_idx), code_idx)
    code_idx = torch.clamp(code_idx, min=0, max=num_code - 1)
    return code_idx
