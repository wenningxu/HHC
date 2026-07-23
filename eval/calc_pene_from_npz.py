import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from optimization.human_body_prior.body_model.body_model import BodyModel
from utils.paramUtil import CANDIDATES


def vertex_normals(vertices, faces):
    device = vertices.device
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int64)).to(device)
    faces = faces.unsqueeze(0).repeat(vertices.shape[0], 1, 1)

    bs, nv = vertices.shape[:2]
    normals = torch.zeros(bs * nv, 3, device=device)

    faces_expanded = faces + (torch.arange(bs, dtype=torch.int32, device=device) * nv)[:, None, None]
    verts_flat = vertices.reshape(bs * nv, 3)
    faces_flat = faces_expanded.view(-1, 3).long()

    v0 = verts_flat[faces_flat[:, 0]]
    v1 = verts_flat[faces_flat[:, 1]]
    v2 = verts_flat[faces_flat[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals.index_add_(0, faces_flat[:, 0], face_normals)
    normals.index_add_(0, faces_flat[:, 1], face_normals)
    normals.index_add_(0, faces_flat[:, 2], face_normals)
    normals = F.normalize(normals, eps=1e-6, dim=1)
    return normals.reshape(bs, nv, 3)


def point2point_signed(x, y, y_normals=None):
    import chamfer_distance as chd

    ch_dist = chd.ChamferDistance()
    _, _, xidx_near, _ = ch_dist(x, y, y_normals=y_normals)

    batch_size, num_points, dims = x.shape
    xidx_near_expanded = xidx_near.view(batch_size, num_points, 1).expand(batch_size, num_points, dims).long()
    y_target = y.gather(1, xidx_near_expanded)
    x2y_vec = x - y_target
    dist_unsigned = torch.norm(x2y_vec, dim=-1)

    if y_normals is None:
        return dist_unsigned

    y_nn = y_normals.gather(1, xidx_near_expanded)
    dot = torch.sum(x2y_vec * y_nn, dim=-1)
    sign = torch.sign(dot)
    return dist_unsigned * sign


def normalize_gender(gender):
    if isinstance(gender, np.ndarray):
        if gender.ndim == 0:
            gender = gender.item()
        elif gender.size > 0:
            gender = gender.reshape(-1)[0]
    if isinstance(gender, (bytes, np.bytes_)):
        gender = gender.decode("utf-8")
    return str(gender)


def npz_to_amass_dict(npz_dict):
    if {"root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans", "betas", "gender"}.issubset(npz_dict.keys()):
        return {
            "root_orient": np.asarray(npz_dict["root_orient"]),
            "pose_body": np.asarray(npz_dict["pose_body"]),
            "pose_lhand": np.asarray(npz_dict["pose_lhand"]),
            "pose_rhand": np.asarray(npz_dict["pose_rhand"]),
            "trans": np.asarray(npz_dict["trans"]),
            "betas": np.asarray(npz_dict["betas"]),
            "gender": normalize_gender(npz_dict["gender"]),
        }

    if "poses" not in npz_dict:
        raise KeyError("npz must contain either AMASS-style keys or a `poses` key.")

    poses = np.asarray(npz_dict["poses"])
    if poses.ndim == 3 and poses.shape[1:] == (55, 3):
        root_orient = poses[:, 0, :]
        pose_body = poses[:, 1:22, :]
        pose_lhand = poses[:, 25:40, :]
        pose_rhand = poses[:, 40:55, :]
    elif poses.ndim == 2 and poses.shape[1] == 55 * 3:
        poses = poses.reshape(poses.shape[0], 55, 3)
        root_orient = poses[:, 0, :]
        pose_body = poses[:, 1:22, :]
        pose_lhand = poses[:, 25:40, :]
        pose_rhand = poses[:, 40:55, :]
    else:
        raise ValueError(f"Unsupported poses shape: {poses.shape}")

    return {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_lhand": pose_lhand,
        "pose_rhand": pose_rhand,
        "trans": np.asarray(npz_dict["trans"]),
        "betas": np.asarray(npz_dict["betas"]),
        "gender": normalize_gender(npz_dict["gender"]),
    }


def to_torch_params(param_dict, device):
    out = {}
    for key in ["root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans", "betas"]:
        out[key] = torch.as_tensor(param_dict[key], dtype=torch.float32, device=device)
    out["gender"] = param_dict["gender"]
    return out


class PeneCalculator:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.neutral_bm = BodyModel(bm_fname="body_models/smplx/SMPLX_NEUTRAL.npz", num_betas=10).to(self.device)
        self.male_bm = BodyModel(bm_fname="body_models/smplx/SMPLX_MALE.npz", num_betas=10).to(self.device)
        self.female_bm = BodyModel(bm_fname="body_models/smplx/SMPLX_FEMALE.npz", num_betas=10).to(self.device)
        self.lfoot_idx = [7, 10]
        self.rfoot_idx = [8, 11]
        raw_rid2vid = pkl.load(open("body_models/flickrci3ds_r75_rid_to_smplx_vid.pkl", "rb"))
        self.rid_to_vid_tensors = {}
        for rid, vid_list in raw_rid2vid.items():
            self.rid_to_vid_tensors[int(rid)] = torch.tensor(vid_list, dtype=torch.long, device=self.device)

    def _select_body_model(self, gender):
        if gender in (0, "male"):
            return self.male_bm
        if gender in (1, "female"):
            return self.female_bm
        return self.neutral_bm

    def _compute_geometry(self, params):
        body_pose = params["pose_body"]
        left_hand_pose = params["pose_lhand"]
        right_hand_pose = params["pose_rhand"]
        root_transl = params["trans"]
        global_orient = params["root_orient"]
        betas = params["betas"]

        frame_number = body_pose.shape[0]
        bm = self._select_body_model(params["gender"])

        body_pose = body_pose.reshape(frame_number, -1)
        hand_pose = torch.cat([left_hand_pose, right_hand_pose], dim=1).reshape(frame_number, -1)
        betas = betas.reshape(1, -1).repeat(frame_number, 1)

        body = bm(
            pose_body=body_pose,
            pose_hand=hand_pose,
            root_orient=global_orient,
            betas=betas,
            trans=root_transl,
        )

        verts = body.v
        joints = body.Jtr
        normals = vertex_normals(verts, bm.f)
        return verts, joints, normals

    def _calc_penetration_metric_per_frame(self, verts_a, verts_b, norms_a, norms_b, temporal_mask=None, ratio=1.0):
        d_b2a = point2point_signed(verts_b, verts_a, y_normals=norms_a)
        d_a2b = point2point_signed(verts_a, verts_b, y_normals=norms_b)

        thresh_shallow = 10.0
        thresh_deep = 10.0

        if temporal_mask is not None:
            current_thresh = torch.where(
                temporal_mask.view(-1, 1) > 0.1,
                torch.tensor(thresh_deep, device=self.device),
                torch.tensor(thresh_shallow, device=self.device),
            )
        else:
            current_thresh = torch.tensor(thresh_shallow, device=self.device)

        def compute_weighted_loss(dist_signed):
            valid_pene_mask = (dist_signed < 0) & (torch.abs(dist_signed) < current_thresh)
            weight_val = 2.0 * ratio if ratio < 1 else 2.0
            weights = torch.zeros_like(dist_signed)
            weights[valid_pene_mask] = weight_val
            return torch.sum(weights * (dist_signed ** 2), dim=1)

        return compute_weighted_loss(d_b2a) + compute_weighted_loss(d_a2b)

    def _calc_skate_metric(self, joints):
        foot_indices = self.lfoot_idx + self.rfoot_idx
        feet_vel = joints[1:, foot_indices, :] - joints[:-1, foot_indices, :]
        feet_height = joints[:-1, foot_indices, 1]

        vel_thresh = torch.tensor([0.001, 0.001, 0.001, 0.001], device=joints.device)
        height_thresh = torch.tensor([0.12, 0.05, 0.12, 0.05], device=joints.device)

        feet_speed_sq = (feet_vel[..., 0] ** 2) + (feet_vel[..., 1] ** 2) + (feet_vel[..., 2] ** 2)
        contact = (feet_speed_sq < vel_thresh) & (feet_height < height_thresh)

        if contact.any():
            feet_speed = torch.norm(feet_vel, dim=-1)
            return torch.sum(feet_speed[contact])
        return torch.tensor(0.0, device=joints.device)

    def _calc_contact_mismatch_per_frame(self, verts_a, verts_b, norms_a, norms_b, interaction_pairs):
        if not interaction_pairs:
            zero = torch.zeros(verts_a.shape[0], device=self.device)
            return zero, torch.zeros(verts_a.shape[0], dtype=torch.bool, device=self.device)

        area_map = {"point": 0.02, "flat": 0.10, "wrap": 0.15, "hug": 0.20, "default": 0.02}
        total_mismatch = torch.zeros(verts_a.shape[0], device=self.device)
        frame_has_mismatch = torch.zeros(verts_a.shape[0], dtype=torch.bool, device=self.device)
        valid_pairs = 0

        for pair in interaction_pairs:
            p1_rids = pair.get("p1_ids", [])
            p2_rids = pair.get("p2_ids", [])
            curr_top_k_ratio = area_map.get(pair.get("contact_type", "default"), area_map["default"])

            best_dist_val = float("inf")
            best_sub_a = None
            best_sub_b = None
            best_n_a = None
            best_n_b = None
            best_idx_a_len = 0
            best_idx_b_len = 0

            for rid1 in p1_rids:
                if rid1 not in self.rid_to_vid_tensors:
                    continue
                idx1 = self.rid_to_vid_tensors[rid1]
                sub1 = verts_a[:, idx1, :]

                for rid2 in p2_rids:
                    if rid2 not in self.rid_to_vid_tensors:
                        continue
                    idx2 = self.rid_to_vid_tensors[rid2]
                    sub2 = verts_b[:, idx2, :]
                    with torch.no_grad():
                        d = point2point_signed(sub1, sub2)
                        current_dist_score = torch.abs(d).min()
                    if current_dist_score < best_dist_val:
                        best_dist_val = current_dist_score
                        best_sub_a = sub1
                        best_sub_b = sub2
                        best_n_a = norms_a[:, idx1, :]
                        best_n_b = norms_b[:, idx2, :]
                        best_idx_a_len = len(idx1)
                        best_idx_b_len = len(idx2)

            if best_sub_a is None:
                continue

            dist_signed_a = point2point_signed(best_sub_a, best_sub_b, y_normals=best_n_b)
            dist_signed_b = point2point_signed(best_sub_b, best_sub_a, y_normals=best_n_a)

            # 只有该帧该区域的双向 signed distance 全为正，才认为是“纯漂移无穿透”。
            valid_frame_a = (dist_signed_a > 0).all(dim=1)
            valid_frame_b = (dist_signed_b > 0).all(dim=1)
            valid_frame = valid_frame_a & valid_frame_b

            dist_abs_a = torch.relu(dist_signed_a)
            dist_abs_b = torch.relu(dist_signed_b)

            k_a = max(1, int(best_idx_a_len * curr_top_k_ratio))
            k_b = max(1, int(best_idx_b_len * curr_top_k_ratio))
            topk_vals_a = torch.topk(dist_abs_a, k=k_a, dim=1, largest=False).values.mean(dim=1)
            topk_vals_b = torch.topk(dist_abs_b, k=k_b, dim=1, largest=False).values.mean(dim=1)

            pair_mismatch = topk_vals_a + topk_vals_b
            mismatch_frame = valid_frame & (pair_mismatch > 0)
            pair_mismatch = pair_mismatch * mismatch_frame.float()

            total_mismatch += pair_mismatch
            frame_has_mismatch |= mismatch_frame
            valid_pairs += 1

        if valid_pairs == 0:
            zero = torch.zeros(verts_a.shape[0], device=self.device)
            return zero, torch.zeros(verts_a.shape[0], dtype=torch.bool, device=self.device)
        return total_mismatch / valid_pairs, frame_has_mismatch

    def load_interaction_pairs(self, contact_path):
        with open(contact_path, "rb") as f:
            return pkl.load(f)

    @torch.no_grad()
    def compute_from_npz(self, p1_path, p2_path, metric="pene", interaction_pairs=None):
        data_p1 = npz_to_amass_dict(dict(np.load(p1_path, allow_pickle=True)))
        data_p2 = npz_to_amass_dict(dict(np.load(p2_path, allow_pickle=True)))
        params_p1 = to_torch_params(data_p1, self.device)
        params_p2 = to_torch_params(data_p2, self.device)

        max_frames = min(params_p1["trans"].shape[0], params_p2["trans"].shape[0], 150)
        for params in (params_p1, params_p2):
            for key in ["root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans"]:
                params[key] = params[key][:max_frames]

        verts_p1, joints_p1, norms_p1 = self._compute_geometry(params_p1)
        verts_p2, joints_p2, norms_p2 = self._compute_geometry(params_p2)

        if metric == "pene":
            pene_per_frame = self._calc_penetration_metric_per_frame(verts_p1, verts_p2, norms_p1, norms_p2)
            return pene_per_frame.mean()
        if metric == "skate":
            return self._calc_skate_metric(joints_p1) + self._calc_skate_metric(joints_p2)
        if metric == "mismatch":
            mismatch_per_frame, frame_has_mismatch = self._calc_contact_mismatch_per_frame(
                verts_p1, verts_p2, norms_p1, norms_p2, interaction_pairs or []
            )
            num_mismatch_frames = frame_has_mismatch.sum()
            if num_mismatch_frames < 1:
                return torch.tensor(0.0, device=self.device)
            return mismatch_per_frame.sum() / num_mismatch_frames.float()

        raise ValueError(f"Unsupported metric: {metric}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute motion metrics directly from p1.npz and p2.npz.")
    parser.add_argument("p1", type=Path, nargs="?", help="Path to p1.npz")
    parser.add_argument("p2", type=Path, nargs="?", help="Path to p2.npz")
    parser.add_argument("--folder", type=Path, help="Folder containing *_p1.npz and *_p2.npz pairs")
    parser.add_argument("--metric", choices=["pene", "skate", "mismatch"], default="pene", help="Metric to compute")
    parser.add_argument("--contact-dir", type=Path, default=Path("dataset/interx/contact_pairs"), help="Directory containing sample_name.pkl contact files")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N samples in folder mode")
    return parser.parse_args()


def main():
    args = parse_args()
    calculator = PeneCalculator(device=args.device)

    if args.folder is not None:
        p1_files = sorted(args.folder.glob("*_pred_p1.npz"))
        if not p1_files:
            raise FileNotFoundError(f"No *_pred_p1.npz files found in {args.folder}")

        values = []
        failed = []
        progress = tqdm(p1_files, total=len(p1_files), desc="Processing NPZ files", unit="sample")
        for idx, p1_path in enumerate(progress, start=1):
            p2_path = Path(str(p1_path).replace("_p1.npz", "_p2.npz"))
            if not p2_path.exists():
                failed.append((p1_path.name, "missing_p2"))
                progress.set_postfix(ok=len(values), fail=len(failed))
                continue

            try:
                interaction_pairs = None
                if args.metric == "mismatch":
                    contact_path = args.contact_dir / (p1_path.name.replace("_pred_p1.npz", "").replace("_p1.npz", "") + ".pkl")
                    interaction_pairs = calculator.load_interaction_pairs(contact_path)
                metric_value = calculator.compute_from_npz(
                    p1_path, p2_path, metric=args.metric, interaction_pairs=interaction_pairs
                ).item()
                values.append((p1_path.name.replace("_p1.npz", ""), metric_value))
            except Exception as exc:
                failed.append((p1_path.name, repr(exc)))

            if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(p1_files)):
                progress.set_postfix(ok=len(values), fail=len(failed))

        pene_values = np.array([value for _, value in values], dtype=np.float64)
        print("SUMMARY")
        print(f"num_total: {len(p1_files)}")
        print(f"num_success: {len(values)}")
        print(f"num_failed: {len(failed)}")
        if len(pene_values) > 0:
            if args.metric == "pene":
                metric_name = "pene_per_frame"
            elif args.metric == "mismatch":
                metric_name = "mismatch_per_frame"
            else:
                metric_name = "skate"
            print(f"mean_{metric_name}: {pene_values.mean():.8f}")
            print(f"median_{metric_name}: {np.median(pene_values):.8f}")
            print(f"min_{metric_name}: {pene_values.min():.8f}")
            print(f"max_{metric_name}: {pene_values.max():.8f}")
        if failed:
            print(f"first_failed: {failed[:10]}")
        return

    if args.p1 is None or args.p2 is None:
        raise ValueError("Provide both p1 and p2, or use --folder for batch mode.")

    interaction_pairs = None
    if args.metric == "mismatch":
        sample_name = args.p1.name.replace("_pred_p1.npz", "").replace("_p1.npz", "")
        interaction_pairs = calculator.load_interaction_pairs(args.contact_dir / f"{sample_name}.pkl")
    metric_value = calculator.compute_from_npz(args.p1, args.p2, metric=args.metric, interaction_pairs=interaction_pairs)
    if args.metric == "pene":
        print(f"mean_pene_per_frame: {metric_value.item():.8f}")
    elif args.metric == "mismatch":
        print(f"mean_mismatch_per_frame: {metric_value.item():.8f}")
    else:
        print(f"skate: {metric_value.item():.8f}")


if __name__ == "__main__":
    main()
