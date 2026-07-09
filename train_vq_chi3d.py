import json
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

import options.option_vq as option_vq
import utils.losses as losses
import utils.utils_model as utils_model
from dataset import dataset_chi3d
from exit.utils import init_save_folder
from models.model_tools import build_model
from utils.chi3d_vis import ensure_dir, run_vq_sample_export


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
args.dataname = "chi3d"
args.motion_rep = "raw"
args.window_size = 96
args.total_iter = 50000
args.out_dir = os.path.join(args.out_dir, "vq-chi3d-sep")
init_save_folder(args)
device = args.device
sample_fps = 50

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
wandb.init(
    project="NewInter",
    config=vars(args),
    group="vqvae_chi3d",
    name=os.path.basename(args.out_dir),
)
sample_root = ensure_dir(os.path.join(args.out_dir, "samples_vq"))
ckpt_root = ensure_dir(os.path.join(args.out_dir, "checkpoints"))

args.nb_joints = 55
train_loader = dataset_chi3d.DATALoader(
    args.dataname,
    args.batch_size,
    "dataset/chi3d/splits/train.txt",
    "dataset/chi3d/motions/chi3d-shape.h5",
    window_size=args.window_size,
)
train_loader_iter = dataset_chi3d.cycle(train_loader)
sample_loader = dataset_chi3d.DATALoader(
    args.dataname,
    1,
    "dataset/chi3d/splits/train.txt",
    "dataset/chi3d/motions/chi3d-shape.h5",
    window_size=args.window_size,
    shuffle=False,
    num_workers=0,
)
sample_batch = next(iter(sample_loader))

net = build_model(args, "chi3d_vq")
net.train().to(device)

optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=0)
Loss = losses.ReConsLossInter(
    args,
    args.recons_loss,
    args.nb_joints,
    net.fk_layers,
    normalizer=dataset_chi3d.MotionNormalizerTorchChi3D(),
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
    motion, group_ids = next(train_loader_iter)
    motion = motion.to(device).float()
    group_ids = group_ids.to(device)

    pred_motion, loss_commit, perplexity = net(motion, group_ids)
    loss_dict = Loss(pred_motion, motion, None, None, group_ids)
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
        save_dir = ensure_dir(os.path.join(sample_root, f"{nb_iter:09d}"))
        run_vq_sample_export(net, sample_batch, save_dir, fps=sample_fps)
        save_checkpoint(nb_iter, f"net_iter_{nb_iter:09d}.pth")

save_checkpoint(args.total_iter, "net_last.pth")
wandb.finish()
