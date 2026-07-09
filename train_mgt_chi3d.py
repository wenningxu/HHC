import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

import options.option_transformer as option_trans
import utils.utils_model as utils_model
from dataset import dataset_chi3d
from exit.utils import init_save_folder, generate_src_mask
from models.model_tools import build_model
from utils.chi3d_vis import build_fixed_chi3d_templates, ensure_dir, run_mgt_sample_export


def get_acc(cls_pred, target, mask):
    pred_indices = cls_pred.argmax(dim=-1)
    pred_selected = pred_indices[mask]
    target_selected = target[mask]
    right_num = (pred_selected == target_selected).sum()
    total_num = mask.sum().clamp(min=1)
    return (right_num * 100 / total_num).item()


def _load_split_ids(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _to_rot_6d(motion):
    motion = torch.from_numpy(motion.astype(np.float32))
    joints3d = motion[:, -1, :]
    pose = motion[:, :-1, :]
    pose_all = []
    for ii in range(2):
        pose_all.append(
            dataset_chi3d.geometry.matrix_to_rotation_6d(
                dataset_chi3d.geometry.axis_angle_to_matrix(pose[:, :, 3 * ii:3 * ii + 3])
            )
        )
    ret = torch.cat(pose_all, dim=2)
    padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
    for ii in range(2):
        padded_tr[:, 6 * ii:6 * ii + 3] = joints3d[:, 3 * ii:3 * ii + 3]
    ret = torch.cat((ret, padded_tr[:, None]), 1)
    return torch.reshape(ret, (ret.shape[0], -1))


def export_chi3d_tokens_if_needed(vqvae, token_dir, split_file, motion_file, logger):
    ensure_dir(token_dir)
    split_ids = _load_split_ids(split_file)
    missing_ids = [name for name in split_ids if not os.path.exists(os.path.join(token_dir, f"{name}.npy"))]
    if not missing_ids:
        logger.info(f"Using existing chi3d tokens from {token_dir}")
        return

    logger.info(f"Exporting {len(missing_ids)} chi3d token files to {token_dir}")
    normalizer = dataset_chi3d.MotionNormalizerTorchChi3D()
    device = next(vqvae.parameters()).device
    with h5py.File(motion_file, "r") as mf:
        with torch.no_grad():
            for name in tqdm(missing_ids, desc="tokenize_chi3d"):
                motion = mf[name][:].astype("float32")
                motion = _to_rot_6d(motion)
                motion = normalizer.forward(motion).unsqueeze(0).to(device)
                group = int(name.split("_")[0][1:])
                group_idx = group - 2
                group_ids = torch.tensor([[0, group_idx, 1, group_idx]], dtype=torch.long, device=device)
                code_idx = vqvae(motion, group_ids, type="encode")[0].detach().cpu().numpy().astype(np.int64)
                np.save(os.path.join(token_dir, f"{name}.npy"), code_idx)
    logger.info("Finished exporting chi3d tokens")


args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
args.dataname = "chi3d"
args.shape = True
args.num_local_layer = 0
args.motion_rep = "raw"
args.resume_pth = args.resume_pth or "output/vq-chi3d-sep/2026-05-06-18-45-06_exp_debug/checkpoints/net_iter_000030000.pth"
args.resume_trans = args.resume_trans or "output/mgt-chi3d/2026-05-07-04-39-10_exp_debug/checkpoints/net_iter_000015000.pth"
args.eval_iter = 5000
if args.vq_name == "VQVAE":
    args.vq_name = os.path.join(os.path.dirname(args.resume_pth), "tokens_iter_000030000")
args.out_dir = os.path.join(args.out_dir, "mgt-chi3d")
init_save_folder(args)
device = args.device
sample_fps = 50
sample_length = 96

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
os.environ.setdefault("WANDB_DIR", args.out_dir)
os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(args.out_dir, "wandb_cache"))
os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(args.out_dir, "wandb_config"))
ensure_dir(os.environ["WANDB_DIR"])
ensure_dir(os.environ["WANDB_CACHE_DIR"])
ensure_dir(os.environ["WANDB_CONFIG_DIR"])
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
wandb.init(
    project="NewInter",
    config=vars(args),
    group="mgt_chi3d",
    name=os.path.basename(args.out_dir),
    dir=args.out_dir,
)
sample_root = ensure_dir(os.path.join(args.out_dir, "samples_mgt"))
ckpt_root = ensure_dir(os.path.join(args.out_dir, "checkpoints"))

vq_ckpt = torch.load(args.resume_pth, map_location="cpu")
vq_args = vq_ckpt.get("args", {})
for key in ["nb_code", "code_dim", "output_emb_width", "down_t", "stride_t", "width", "depth", "dilation_growth_rate", "vq_act", "vq_norm"]:
    if key in vq_args:
        setattr(args, key, vq_args[key])
logger.info(
    f"Using VQ config from checkpoint | nb_code {args.nb_code} | code_dim {args.code_dim} "
    f"| output_emb_width {args.output_emb_width} | down_t {args.down_t} | stride_t {args.stride_t}"
)

vqvae = build_model(args, "chi3d_vq")
if args.resume_pth:
    vqvae.load_state_dict(vq_ckpt["net"], strict=False)
vqvae.eval().to(device)
for p in vqvae.parameters():
    p.requires_grad = False

export_chi3d_tokens_if_needed(
    vqvae,
    args.vq_name,
    "dataset/chi3d/splits/train.txt",
    "dataset/chi3d/motions/chi3d-shape.h5",
    logger,
)

train_set = dataset_chi3d.Chi3DTokenDataset(
    "dataset/chi3d/splits/train.txt",
    args.vq_name,
    unit_length=2 ** args.down_t,
    codebook_size=args.nb_code,
)
num_actions = train_set.num_actions
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
)
sample_templates = build_fixed_chi3d_templates(train_set.action_to_id)

net = build_model(args, "chi3d_action_transformer", vqvae=vqvae, num_actions=num_actions).to(device)

optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, net, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
start_iter = 1

if args.resume_trans:
    trans_ckpt = torch.load(args.resume_trans, map_location="cpu")
    net.load_state_dict(trans_ckpt["trans"], strict=False)
    if "optimizer" in trans_ckpt:
        optimizer.load_state_dict(trans_ckpt["optimizer"])
    if "scheduler" in trans_ckpt:
        scheduler.load_state_dict(trans_ckpt["scheduler"])
    start_iter = int(trans_ckpt.get("iter", 0)) + 1
    logger.info(f"Resuming MGT from {args.resume_trans} at iter {start_iter}")


def save_checkpoint(iteration, suffix):
    ckpt = {
        "trans": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iter": iteration,
        "args": vars(args),
        "resume_pth": args.resume_pth,
        "sample_templates": sample_templates,
    }
    torch.save(ckpt, os.path.join(ckpt_root, suffix))

for nb_iter in tqdm(range(start_iter, args.total_iter + 1), initial=max(start_iter - 1, 0), total=args.total_iter):
    for _, action_ids, _, m_tokens, m_tokens_len, group_ids in train_loader:
        action_ids = action_ids.to(device)
        group_ids = group_ids.to(device)
        target = m_tokens.to(device).clone()
        m_tokens_len = m_tokens_len.to(device) if hasattr(m_tokens_len, "to") else torch.as_tensor(m_tokens_len, device=device)

        batch_size, max_len, _ = target.shape
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)
        seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)
        mask_bernoulli = torch.bernoulli(args.pkeep * torch.ones(target.shape, device=target.device))
        mask_bernoulli = torch.logical_or(mask_bernoulli, ~seq_mask_no_end_3c).int()
        r_indices = torch.randint_like(target, args.nb_code)
        input_indices = mask_bernoulli * target + (1 - mask_bernoulli) * r_indices
        mask_token_id = args.nb_code + 2

        rand_time = torch.rand(batch_size, device=device)
        rand_mask_probs = torch.cos(rand_time * torch.pi * 0.5).clamp(min=0.1, max=1.0)
        flat_valid_len = m_tokens_len * 3
        num_token_masked = (flat_valid_len.float() * rand_mask_probs).round().clamp(min=1).long()
        batch_rand_score = torch.rand((batch_size, max_len, 3), device=device)
        batch_rand_score = batch_rand_score + (~seq_mask_no_end_3c).float() * 1e9
        batch_randperm = batch_rand_score.view(batch_size, -1).argsort(dim=-1)
        mask_token_flat = batch_randperm < num_token_masked.view(-1, 1)
        mask_token = mask_token_flat.view(batch_size, max_len, 3)
        masked_input_indices = torch.where(mask_token, mask_token_id, input_indices)

        seq_mask_attn = generate_src_mask(max_len, m_tokens_len + 1)
        cls_pred = net(masked_input_indices, action_ids, group_ids, src_mask=seq_mask_attn)

        weights = seq_mask_no_end_3c.float()
        sample_valid_tokens = seq_mask_no_end_3c.sum(dim=(1, 2), keepdim=True)
        weights = weights / sample_valid_tokens.clamp(min=1)
        cls_pred_flat = cls_pred[seq_mask_no_end_3c].reshape(-1, cls_pred.shape[-1])
        target_flat = target[seq_mask_no_end_3c].reshape(-1)
        weights_flat = weights[seq_mask_no_end_3c].reshape(-1)

        loss_cls = F.cross_entropy(cls_pred_flat, target_flat, reduction="none")
        loss_cls = (loss_cls * weights_flat).sum() / batch_size

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()
        scheduler.step()

        if nb_iter % args.print_iter == 0:
            acc = get_acc(cls_pred, target, seq_mask_no_end_3c)
            logger.info(f"Iter {nb_iter} | loss {loss_cls.item():.5f} | acc {acc:.2f}")
            writer.add_scalar("train/loss", loss_cls.item(), nb_iter)
            writer.add_scalar("train/acc", acc, nb_iter)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], nb_iter)
            wandb.log(
                {
                    "Train/Loss": loss_cls.item(),
                    "Train/Acc": acc,
                    "Train/Learning Rate": scheduler.get_last_lr()[0],
                },
                step=nb_iter,
            )
        if nb_iter % args.eval_iter == 0:
            logger.info(f"Exporting MGT sample at iter {nb_iter}")
            save_dir = ensure_dir(os.path.join(sample_root, f"{nb_iter:09d}"))
            run_mgt_sample_export(net, vqvae, sample_templates, save_dir, m_length=sample_length, fps=sample_fps)
            save_checkpoint(nb_iter, f"net_iter_{nb_iter:09d}.pth")
        break

save_checkpoint(args.total_iter, "net_last.pth")
wandb.finish()
