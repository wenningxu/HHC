import json
import os
import sys
from os.path import join as pjoin

# Allow running as `python train/train_vq_interx.py` from repo root by putting
# the repo root on sys.path before importing project packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

import options.option_vq as option_vq
import utils.losses as losses
import utils.utils_model as utils_model
from dataset import dataset_interx
from utils.runtime import init_save_folder
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from models.model_tools import build_model


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
# InterX two-person VQVAE (VQVAE_SEP3_shape) config. dataname='hhi' selects the
# 55-joint SMPLX branch in VQVAE_SEP3_shape.
args.dataname = "hhi"
args.motion_rep = "raw"          # 6-dim root (HHI_ROOT_MASK_OLD); uses Interx_Mean/Std.npy
args.window_size = 96
# nb_code / architecture MUST match the t2i transformer (option_transformer
# defaults: nb_code=12288, code_dim=256). The VQ checkpoint is rebuilt with the
# t2i's args and loaded strict, so keep them identical.
args.nb_code = 12288
args.code_dim = 256
args.output_emb_width = 512  # match option_transformer default (unused by VQVAE_SEP3_shape but kept consistent)
# Default total_iter to 200000 (>= VQ_RESUME_ITER) unless overridden on the CLI.
# option_vq's argparse default is 300000; only shorten for smoke if the user passed --total-iter.
import argparse as _ap
args.total_iter = 200000 if args.total_iter == 300000 else args.total_iter
args.out_dir = os.path.join(args.out_dir, "vq-interx-sep")
init_save_folder(args)
os.makedirs(args.out_dir, exist_ok=True)
device = args.device
sample_fps = 30

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
wandb.init(
    project="NewInter",
    config=vars(args),
    group="vqvae_interx",
    name=os.path.basename(args.out_dir),
)

# t2i transformer loads the trained VQ from checkpoints/vq/net_175000.pth
ckpt_root = "checkpoints/vq"
os.makedirs(ckpt_root, exist_ok=True)
sample_root = os.path.join(args.out_dir, "samples_vq")
os.makedirs(sample_root, exist_ok=True)
VQ_RESUME_ITER = 175000  # iter whose checkpoint t2i expects as net_175000.pth

args.nb_joints = 55
train_loader = dataset_interx.DATALoader(
    args.dataname,
    args.batch_size,
    "dataset/interx/train.txt",
    "dataset/interx/motions/train.h5",
    window_size=args.window_size,
    motion_rep=args.motion_rep,
)
train_loader_iter = dataset_interx.cycle(train_loader)
sample_loader = dataset_interx.DATALoader(
    args.dataname,
    1,
    "dataset/interx/train.txt",
    "dataset/interx/motions/train.h5",
    num_workers=0,
    window_size=args.window_size,
    motion_rep=args.motion_rep,
)
sample_batch = next(iter(sample_loader))

net = build_model(args, "interx_vq")  # -> VQVAE_SEP3_shape
net.train().to(device)

optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=0)
Loss = losses.ReConsLossInter(
    args,
    args.recons_loss,
    args.nb_joints,
    net.fk_layers,
    normalizer=MotionNormalizerTorchInterX(motion_rep=args.motion_rep),
)


def save_checkpoint(iteration, suffix):
    ckpt = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iter": iteration,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(ckpt_root, suffix))


def interx_sample_export(net, batch, save_dir, fps=30):
    """Decode a single sample and save GT + reconstruction as npz (InterX
    dual-person /trans1,pose1,trans2,pose2/)."""
    motion, betas, gender, group = batch
    motion = motion.to(device).float()
    betas = betas.to(device).float()
    gender = gender.to(device)
    norm = MotionNormalizerTorchInterX(motion_rep=args.motion_rep)
    with torch.no_grad():
        recon, _, _ = net(motion, betas, gender, type="full")
    gt_den = norm.backward(motion[:1]).reshape(1, -1, 56, 12)
    rec_den = norm.backward(recon[:1]).reshape(1, -1, 56, 12)
    os.makedirs(save_dir, exist_ok=True)
    t1, p1, t2, p2 = recover_interx(gt_den[0])
    np.savez(pjoin(save_dir, "sample_gt.npz"), trans1=t1, pose1=p1, trans2=t2, pose2=p2, fps=fps)
    t1, p1, t2, p2 = recover_interx(rec_den[0])
    np.savez(pjoin(save_dir, "sample_recon.npz"), trans1=t1, pose1=p1, trans2=t2, pose2=p2, fps=fps)


metric_sums = {
    "loss": 0.0,
    "recons": 0.0,
    "commit": 0.0,
    "joint": 0.0,
    "velocity": 0.0,
    "root": 0.0,
    "rotation": 0.0,
    "foot_contact": 0.0,
    "distance_map": 0.0,
    "perplexity_u": 0.0,
    "perplexity_r": 0.0,
    "perplexity_l": 0.0,
}

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    if nb_iter < args.warm_up_iter:
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    else:
        current_lr = scheduler.get_last_lr()[0]
    motion, betas, gender, group = next(train_loader_iter)
    motion = motion.to(device).float()
    betas = betas.to(device).float()
    gender = gender.to(device)
    group = group.to(device)

    pred_motion, loss_commit, perplexity = net(motion, betas, gender, type="full")
    loss_dict = Loss(pred_motion, motion, betas, gender, group)
    loss = (
        loss_dict["main"] + loss_dict["Joint"] + loss_dict["Velocity"] + loss_dict["Rotation"] +
        loss_dict["Root"] + loss_dict["FootContact"] + loss_dict["DistanceMap"] + args.commit * loss_commit
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    metric_sums["loss"] += loss.item()
    metric_sums["recons"] += loss_dict["main"].item()
    metric_sums["commit"] += loss_commit.item()
    metric_sums["joint"] += loss_dict["Joint"].item()
    metric_sums["velocity"] += loss_dict["Velocity"].item()
    metric_sums["root"] += loss_dict["Root"].item()
    metric_sums["rotation"] += loss_dict["Rotation"].item()
    metric_sums["foot_contact"] += loss_dict["FootContact"].item()
    metric_sums["distance_map"] += loss_dict["DistanceMap"].item()
    metric_sums["perplexity_u"] += perplexity[0].item()
    metric_sums["perplexity_r"] += perplexity[1].item()
    metric_sums["perplexity_l"] += perplexity[2].item()

    if nb_iter % args.print_iter == 0:
        avg_metrics = {key: value / args.print_iter for key, value in metric_sums.items()}
        logger.info(
            f"Train. Iter {nb_iter} : \t Loss. {avg_metrics['loss']:.5f} \t Commit. {avg_metrics['commit']:.5f} "
            f"\t PPL-U. {avg_metrics['perplexity_u']:.2f} \t PPL-R. {avg_metrics['perplexity_r']:.2f} "
            f"\t PPL-L. {avg_metrics['perplexity_l']:.2f} \t Recons. {avg_metrics['recons']:.5f} "
            f"\t LR {scheduler.get_last_lr()[0]:.5f} \t Pos. {avg_metrics['joint']:.5f} "
            f"\t Vel. {avg_metrics['velocity']:.5f} \t Root. {avg_metrics['root']:.5f} "
            f"\t Rot. {avg_metrics['rotation']:.5f} \t FC. {avg_metrics['foot_contact']:.5f} "
            f"\t DistanceMap. {avg_metrics['distance_map']:.5f}"
        )
        writer.add_scalar("train/loss", avg_metrics["loss"], nb_iter)
        writer.add_scalar("train/recons", avg_metrics["recons"], nb_iter)
        writer.add_scalar("train/commit", avg_metrics["commit"], nb_iter)
        writer.add_scalar("train/joint", avg_metrics["joint"], nb_iter)
        writer.add_scalar("train/velocity", avg_metrics["velocity"], nb_iter)
        writer.add_scalar("train/root", avg_metrics["root"], nb_iter)
        writer.add_scalar("train/rotation", avg_metrics["rotation"], nb_iter)
        writer.add_scalar("train/foot_contact", avg_metrics["foot_contact"], nb_iter)
        writer.add_scalar("train/distance_map", avg_metrics["distance_map"], nb_iter)
        writer.add_scalar("train/perplexity_u", avg_metrics["perplexity_u"], nb_iter)
        writer.add_scalar("train/perplexity_r", avg_metrics["perplexity_r"], nb_iter)
        writer.add_scalar("train/perplexity_l", avg_metrics["perplexity_l"], nb_iter)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], nb_iter)
        wandb.log(
            {
                "Train/Loss": avg_metrics["loss"],
                "Train/Reconstruction Loss": avg_metrics["recons"],
                "Train/Commitment Loss": avg_metrics["commit"],
                "Train/Position Loss": avg_metrics["joint"],
                "Train/Velocity Loss": avg_metrics["velocity"],
                "Train/Root Loss": avg_metrics["root"],
                "Train/Rotation Loss": avg_metrics["rotation"],
                "Train/FC Loss": avg_metrics["foot_contact"],
                "Train/Distance Map Loss": avg_metrics["distance_map"],
                "Train/Perplexity_u": avg_metrics["perplexity_u"],
                "Train/Perplexity_r": avg_metrics["perplexity_r"],
                "Train/Perplexity_l": avg_metrics["perplexity_l"],
                "Train/Learning Rate": scheduler.get_last_lr()[0],
                "Warmup/Learning Rate": current_lr if nb_iter < args.warm_up_iter else scheduler.get_last_lr()[0],
            },
            step=nb_iter,
        )
        for key in metric_sums:
            metric_sums[key] = 0.0

    if nb_iter % args.eval_iter == 0:
        logger.info(f"Exporting VQ sample at iter {nb_iter}")
        save_dir = os.path.join(sample_root, f"{nb_iter:09d}")
        interx_sample_export(net, sample_batch, save_dir, fps=sample_fps)
        save_checkpoint(nb_iter, f"net_iter_{nb_iter:09d}.pth")
        if nb_iter == VQ_RESUME_ITER:
            save_checkpoint(nb_iter, "net_175000.pth")

save_checkpoint(args.total_iter, "net_last.pth")
wandb.finish()
