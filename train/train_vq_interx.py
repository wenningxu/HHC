import json
import hashlib
import math
import os
import random
import sys
from contextlib import contextmanager
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
from dataset import dataset_interx, dataset_interx_TM
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.eval_trans import calculate_activation_statistics, calculate_frechet_distance
from utils.word_vectorizer import WordVectorizer
from utils.runtime import evaluating, init_save_folder
from utils.training_telemetry import (
    ThroughputMeter,
    gpu_peak_memory_gib,
    parameter_and_gradient_telemetry,
)
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from utils.quantizer_state import (
    quantizer_training_state,
    restore_quantizer_training_state,
)
from models.model_tools import build_model


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')).hexdigest()


def atomic_json(path, payload):
    temporary = path + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


args = option_vq.get_args_parser()
if args.stride_t != 2 or args.down_t < 1:
    raise ValueError('the current VQ decoder requires stride_t=2 and down_t>=1')
temporal_stride = args.stride_t ** args.down_t
if args.window_size <= 0 or args.window_size % temporal_stride:
    raise ValueError('training window_size must be a positive multiple of the VQ temporal stride')
if args.wandb_mode != 'online':
    raise ValueError('all newly launched VQ training requires --wandb-mode online')
os.environ['HHC_INTERX_STATS_DIR'] = os.path.abspath(args.stats_dir)
os.environ['HHC_INTERX_STATS_PREFIX'] = args.stats_prefix
torch.manual_seed(args.seed)
# InterX two-person VQVAE (VQVAE_SEP3_shape) config. dataname='hhi' selects the
# 55-joint SMPLX branch in VQVAE_SEP3_shape.
args.dataname = "hhi"
args.motion_rep = "raw"          # 6-dim root (HHI_ROOT_MASK_OLD); uses Interx_Mean/Std.npy
args.output_emb_width = 512  # match option_transformer default (unused by VQVAE_SEP3_shape but kept consistent)
args.out_dir = os.path.join(args.out_dir, "vq-interx-sep")
init_save_folder(args)
os.makedirs(args.out_dir, exist_ok=True)
device = args.device
sample_fps = 30

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# t2i transformer loads the trained VQ from checkpoints/vq/net_175000.pth
ckpt_root = args.checkpoint_dir
os.makedirs(ckpt_root, exist_ok=True)
training_manifest_path = os.path.join(ckpt_root, 'training_manifest.json')
if args.resume_pth:
    if not os.path.isfile(training_manifest_path):
        raise RuntimeError('VQ resume requires its original training manifest')
    with open(training_manifest_path, encoding='utf-8') as handle:
        prior_manifest = json.load(handle)
    args.wandb_run_id = prior_manifest.get('observability', {}).get(
        'wandb_run_id'
    )
    if not args.wandb_run_id:
        raise RuntimeError('VQ resume manifest lacks an online W&B run identity')
else:
    if os.path.isfile(training_manifest_path):
        raise RuntimeError(
            'VQ checkpoint directory already has a training manifest; pass '
            '--resume-pth or use a new checkpoint directory'
        )
    args.wandb_run_id = wandb.util.generate_id()
wandb.init(
    project="NewInter",
    config=vars(args),
    group="vqvae_interx",
    name=os.path.basename(args.out_dir),
    mode='online',
    id=args.wandb_run_id,
    resume='allow',
)
stats_mean_path = os.path.join(args.stats_dir, f'{args.stats_prefix}_Mean.npy')
stats_std_path = os.path.join(args.stats_dir, f'{args.stats_prefix}_Std.npy')
manifest_contract = {
    'schema': 'hhc_vq_training_manifest_v2',
    'command': ['python'] + sys.argv,
    'resolved_arguments': {
        key: value for key, value in vars(args).items()
        if key not in {'out_dir', 'resume_pth', 'resume_gpt', 'wandb_run_id'}
    },
    'observability': {
        'wandb_mode': 'online',
        'wandb_project': 'NewInter',
        'wandb_run_id': args.wandb_run_id,
        'metric_profile': 'vq_autoencoder_body_root_hands',
        'telemetry_interval_optimizer_steps': 50,
        'gradient_measurement_point': (
            'after backward, before optimizer step; bf16 uses no GradScaler'
        ),
    },
    'artifacts': {
        'train_split': {
            'path': os.path.abspath(args.train_split_file),
            'sha256': sha256_file(args.train_split_file),
        },
        'train_motion': {
            'path': os.path.abspath(args.train_motion_file),
            'sha256': sha256_file(args.train_motion_file),
        },
        'validation_split': {
            'path': os.path.abspath(args.validation_split_file),
            'sha256': sha256_file(args.validation_split_file),
        },
        'validation_motion': {
            'path': os.path.abspath(args.validation_motion_file),
            'sha256': sha256_file(args.validation_motion_file),
        },
        'normalization_mean': {
            'path': os.path.abspath(stats_mean_path),
            'sha256': sha256_file(stats_mean_path),
        },
        'normalization_std': {
            'path': os.path.abspath(stats_std_path),
            'sha256': sha256_file(stats_std_path),
        },
        'sources': {
            path: sha256_file(path) for path in (
                'train/train_vq_interx.py', 'options/option_vq.py',
                'dataset/dataset_interx.py', 'models/model_tools.py',
                'models/vqvae_sep3_shape.py', 'utils/losses.py',
                'utils/motion_process.py', 'utils/quantizer_state.py',
                'utils/runtime.py', 'utils/training_telemetry.py',
            )
        },
    },
}
manifest_contract['contract_sha256'] = canonical_sha256({
    'schema': manifest_contract['schema'],
    'resolved_arguments': manifest_contract['resolved_arguments'],
    'artifacts': manifest_contract['artifacts'],
    'observability': {
        key: value for key, value in manifest_contract['observability'].items()
        if key != 'wandb_run_id'
    },
})
if os.path.isfile(training_manifest_path):
    with open(training_manifest_path, encoding='utf-8') as handle:
        existing_training_manifest = json.load(handle)
    if (
        existing_training_manifest.get('schema') == manifest_contract['schema']
        and existing_training_manifest.get('contract_sha256')
        != manifest_contract['contract_sha256']
    ):
        raise ValueError(
            'VQ training manifest contract differs from this launch; use a '
            'new checkpoint directory'
        )
else:
    atomic_json(training_manifest_path, {
        **manifest_contract, 'status': 'running'
    })
sample_root = os.path.join(args.out_dir, "samples_vq")
os.makedirs(sample_root, exist_ok=True)
VQ_RESUME_ITER = args.total_iter

args.nb_joints = 55
train_loader = dataset_interx.DATALoader(
    args.dataname,
    args.batch_size,
    args.train_split_file,
    args.train_motion_file,
    window_size=args.window_size,
    motion_rep=args.motion_rep,
)
train_loader_iter = dataset_interx.cycle(train_loader)
# Selecting a fixed visualization sample must not change the training sampler
# order or the subsequently initialized model.  DataLoader's RandomSampler
# consumes the global CPU RNG when its iterator is created.
sample_rng_state = torch.get_rng_state()
sample_loader = dataset_interx.DATALoader(
    args.dataname,
    1,
    args.train_split_file,
    args.train_motion_file,
    num_workers=0,
    window_size=args.window_size,
    motion_rep=args.motion_rep,
)
sample_batch = next(iter(sample_loader))
torch.set_rng_state(sample_rng_state)

validation_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')
validation_loader = dataset_interx_TM.DATALoader(
    args.dataname,
    32,
    args.validation_split_file,
    validation_vectorizer,
    args.validation_motion_file,
    num_workers=0,
    unit_length=temporal_stride,
    shuffle=False,
)
validation_evaluator = EvaluatorModelWrapper(
    get_opt('checkpoints/hhi/Comp_v6_KLD01/opt.txt', device)
)

net = build_model(args, "interx_vq")  # -> VQVAE_SEP3_shape
net.train().to(device)

optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)


def warmup_cosine(step):
    """Paper schedule: 1,000 linear warm-up steps, then cosine to zero."""
    if step < args.warm_up_iter:
        return float(step + 1) / float(max(1, args.warm_up_iter))
    progress = ((step - args.warm_up_iter) /
                max(1, args.total_iter - args.warm_up_iter))
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


if args.schedule_mode == 'historical_separate_warmup_cosine':
    # The successful January MMM run performed 999 warm-up optimizer updates
    # first, then started a fresh 300k-labelled cosine phase.  Its log's
    # iteration counter resets after warm-up, so reproduce that behavior
    # explicitly instead of compressing warm-up into total_iter.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iter, eta_min=0,
    )
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
Loss = losses.ReConsLossInter(
    args,
    args.recons_loss,
    args.nb_joints,
    net.fk_layers,
    normalizer=MotionNormalizerTorchInterX(motion_rep=args.motion_rep),
)
for loader_name, loader in (("train", train_loader), ("sample", sample_loader)):
    dataset_normalizer = loader.dataset.normalizer
    if not (
        torch.equal(dataset_normalizer.motion_mean, Loss.normalizer.motion_mean)
        and torch.equal(dataset_normalizer.motion_std, Loss.normalizer.motion_std)
    ):
        raise RuntimeError(
            f"{loader_name} dataset and reconstruction loss use different "
            f"normalization statistics for motion_rep={args.motion_rep!r}"
        )


start_iter = 0
best_validation_fid = float('inf')
best_validation_iter = -1
if args.resume_pth:
    logger.info(f"Resuming VQ training from {args.resume_pth}")
    resume = torch.load(args.resume_pth, map_location=device)
    net.load_state_dict(resume["net"], strict=True)
    if "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    if "scheduler" in resume:
        scheduler.load_state_dict(resume["scheduler"])
    legacy_ema_state = restore_quantizer_training_state(
        net, resume.get("quantizer_training_state")
    )
    if legacy_ema_state:
        logger.warning(
            "Checkpoint predates complete explicit EMA state. Preserving loaded "
            "codebooks and rebuilding missing accumulators; prior EMA history "
            "is unavailable."
        )
    start_iter = int(resume.get("iter", 0))
    best_validation_fid = float(
        resume.get('best_validation_fid', best_validation_fid)
    )
    best_validation_iter = int(
        resume.get('best_validation_iter', best_validation_iter)
    )


def save_checkpoint(iteration, suffix):
    ckpt = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "quantizer_training_state": quantizer_training_state(net),
        "iter": iteration,
        "args": vars(args),
        "best_validation_fid": best_validation_fid,
        "best_validation_iter": best_validation_iter,
    }
    torch.save(ckpt, os.path.join(ckpt_root, suffix))


@contextmanager
def fixed_validation_rng(seed):
    """Make validation crops repeatable without changing the training stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = []
    if device.startswith('cuda'):
        cuda_devices = [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@torch.no_grad()
def validation_reconstruction_fid():
    """Reconstruction FID on the held-out Inter-X validation split."""
    normalizer = MotionNormalizerTorchInterX(motion_rep='raw')
    gt_embeddings = []
    reconstruction_embeddings = []
    reconstruction_absolute_error = 0.0
    reconstruction_values = 0
    with fixed_validation_rng(args.validation_seed), evaluating(net):
        for batch in validation_loader:
            (word_embeddings, pos_one_hots, _, sent_len, motion, lengths,
             _, _, betas, gender, _) = batch
            motion = motion.to(device).float()
            betas = betas.to(device).float()
            gender = gender.to(device)
            lengths = lengths.to(device)
            reconstruction_raw = torch.zeros_like(motion)
            for frame_length in lengths.unique(sorted=True).tolist():
                selected = lengths == frame_length
                padded_frames = int(math.ceil(int(frame_length) / temporal_stride) * temporal_stride)
                model_input = torch.zeros(
                    int(selected.sum()), padded_frames, motion.shape[-1],
                    dtype=motion.dtype, device=motion.device,
                )
                available_frames = min(padded_frames, motion.shape[1])
                model_input[:, :available_frames] = (
                    motion[selected, :available_frames]
                )
                reconstructed, _, _ = net(
                    model_input,
                    betas[selected], gender[selected], type='full',
                )
                reconstruction_raw[selected, :int(frame_length)] = (
                    normalizer.backward(reconstructed[:, :int(frame_length)])
                )
            raw_motion = normalizer.backward(motion)
            for row, frame_length in enumerate(lengths.tolist()):
                valid_reconstruction = reconstruction_raw[row, :frame_length]
                valid_ground_truth = raw_motion[row, :frame_length]
                reconstruction_absolute_error += float(
                    torch.sum(torch.abs(
                        valid_reconstruction - valid_ground_truth
                    )).item()
                )
                reconstruction_values += valid_ground_truth.numel()
            _, gt_embedding = validation_evaluator.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, raw_motion, lengths.cpu()
            )
            _, reconstruction_embedding = validation_evaluator.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len,
                reconstruction_raw, lengths.cpu(),
            )
            gt_embeddings.append(gt_embedding.cpu())
            reconstruction_embeddings.append(reconstruction_embedding.cpu())
    gt_embeddings = torch.cat(gt_embeddings).numpy()
    reconstruction_embeddings = torch.cat(reconstruction_embeddings).numpy()
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    reconstruction_mu, reconstruction_cov = calculate_activation_statistics(
        reconstruction_embeddings
    )
    return (
        float(calculate_frechet_distance(
            gt_mu, gt_cov, reconstruction_mu, reconstruction_cov
        )),
        reconstruction_absolute_error / max(1, reconstruction_values),
    )


def interx_sample_export(net, batch, save_dir, fps=30):
    """Decode a single sample and save GT + reconstruction as npz (InterX
    dual-person /trans1,pose1,trans2,pose2/)."""
    motion, betas, gender, group = batch
    motion = motion.to(device).float()
    betas = betas.to(device).float()
    gender = gender.to(device)
    norm = MotionNormalizerTorchInterX(motion_rep=args.motion_rep)
    # EMA quantizers update in training mode even under no_grad.  Sample
    # exports must not perturb the codebook or its accumulated statistics.
    with evaluating(net), torch.no_grad():
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
training_throughput = ThroughputMeter()
telemetry_interval_optimizer_steps = 50


def vq_training_update(optimizer_step):
    motion, betas, gender, group = next(train_loader_iter)
    motion = motion.to(device).float()
    betas = betas.to(device).float()
    gender = gender.to(device)
    group = group.to(device)

    autocast_enabled = args.precision == "bf16" and device.startswith("cuda")
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        pred_motion, loss_commit, perplexity = net(
            motion, betas, gender, type="full"
        )
        loss_dict = Loss(pred_motion, motion, betas, gender, group)
        loss = (
            loss_dict["main"] + loss_dict["Joint"]
            + loss_dict["Velocity"] + loss_dict["Rotation"]
            + loss_dict["Root"] + loss_dict["FootContact"]
            + loss_dict["DistanceMap"] + args.commit * loss_commit
        )
    optimizer.zero_grad()
    loss.backward()
    training_throughput.update(motion.shape[0])
    if optimizer_step % telemetry_interval_optimizer_steps == 0:
        telemetry = parameter_and_gradient_telemetry(
            net, fail_on_nonfinite=False
        )
        telemetry.update({
            'train/optimizer_step': float(optimizer_step),
            'train/loss': float(loss.detach()),
            'train/reconstruction_loss': float(loss_dict['main'].detach()),
            'train/commitment_loss': float(loss_commit.detach()),
            'train/perplexity_body': float(perplexity[0].detach()),
            'train/perplexity_root': float(perplexity[1].detach()),
            'train/perplexity_hands': float(perplexity[2].detach()),
            # Historical aliases only: right_hand=root, left_hand=hands.
            'train/perplexity_right_hand': float(perplexity[1].detach()),
            'train/perplexity_left_hand': float(perplexity[2].detach()),
            'train/learning_rate': float(optimizer.param_groups[0]['lr']),
            'train/samples_per_second': (
                training_throughput.measure_and_reset()
            ),
            'train/gpu_peak_memory_gib': gpu_peak_memory_gib(device),
        })
        wandb.log(telemetry, step=optimizer_step)
        if (telemetry['train/nonfinite_gradients']
                or telemetry['train/nonfinite_weights']):
            raise FloatingPointError(
                f'nonfinite VQ training telemetry: {telemetry}'
            )
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    optimizer.step()
    return loss, loss_commit, perplexity, loss_dict


if (
    args.schedule_mode == 'historical_separate_warmup_cosine'
    and start_iter == 0
):
    warmup_sums = {key: 0.0 for key in metric_sums}
    for warmup_iter in tqdm(
        range(1, args.warm_up_iter),
        desc='historical VQ warm-up',
    ):
        warmup_lr = (
            args.lr * (warmup_iter + 1) / (args.warm_up_iter + 1)
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        loss, loss_commit, perplexity, loss_dict = vq_training_update(
            warmup_iter
        )
        warmup_sums["loss"] += loss.item()
        warmup_sums["recons"] += loss_dict["main"].item()
        warmup_sums["commit"] += loss_commit.item()
        warmup_sums["joint"] += loss_dict["Joint"].item()
        warmup_sums["velocity"] += loss_dict["Velocity"].item()
        warmup_sums["root"] += loss_dict["Root"].item()
        warmup_sums["rotation"] += loss_dict["Rotation"].item()
        warmup_sums["foot_contact"] += loss_dict["FootContact"].item()
        warmup_sums["distance_map"] += loss_dict["DistanceMap"].item()
        warmup_sums["perplexity_u"] += perplexity[0].item()
        warmup_sums["perplexity_r"] += perplexity[1].item()
        warmup_sums["perplexity_l"] += perplexity[2].item()
        if warmup_iter % args.print_iter == 0:
            averages = {
                key: value / args.print_iter
                for key, value in warmup_sums.items()
            }
            logger.info(
                f"Warmup. Iter {warmup_iter} : \t Loss. "
                f"{averages['loss']:.5f} \t Recons. "
                f"{averages['recons']:.5f} \t LR {warmup_lr:.8f}"
            )
            wandb.log({
                'Warmup/Loss': averages['loss'],
                'Warmup/Reconstruction Loss': averages['recons'],
                'Warmup/Learning Rate': warmup_lr,
            }, step=warmup_iter)
            for key in warmup_sums:
                warmup_sums[key] = 0.0

main_update_offset = (
    args.warm_up_iter - 1
    if args.schedule_mode == 'historical_separate_warmup_cosine'
    else 0
)
for nb_iter in tqdm(range(start_iter + 1, args.total_iter + 1), initial=start_iter, total=args.total_iter):
    wandb_step = main_update_offset + nb_iter
    current_lr = scheduler.get_last_lr()[0]
    loss, loss_commit, perplexity, loss_dict = vq_training_update(wandb_step)
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
                "train/loss": avg_metrics["loss"],
                "train/reconstruction_loss": avg_metrics["recons"],
                "train/commitment_loss": avg_metrics["commit"],
                "train/perplexity_body": avg_metrics["perplexity_u"],
                "train/perplexity_root": avg_metrics["perplexity_r"],
                "train/perplexity_hands": avg_metrics["perplexity_l"],
                # Historical aliases; use the canonical root/hands fields.
                "train/perplexity_right_hand": avg_metrics["perplexity_r"],
                "train/perplexity_left_hand": avg_metrics["perplexity_l"],
                "train/learning_rate": scheduler.get_last_lr()[0],
            },
            step=wandb_step,
        )
        for key in metric_sums:
            metric_sums[key] = 0.0

    if nb_iter % args.eval_iter == 0:
        logger.info(f"Exporting VQ sample at iter {nb_iter}")
        save_dir = os.path.join(sample_root, f"{nb_iter:09d}")
        interx_sample_export(net, sample_batch, save_dir, fps=sample_fps)
        save_checkpoint(nb_iter, f"net_iter_{nb_iter:09d}.pth")
        validation_fid, validation_loss = validation_reconstruction_fid()
        logger.info(
            f"Validation. Iter {nb_iter} : \t Reconstruction FID. "
            f"{validation_fid:.6f} \t Best. {best_validation_fid:.6f}"
        )
        writer.add_scalar('validation/reconstruction_fid', validation_fid, nb_iter)
        wandb.log({
            'Validation/FID': validation_fid,
            'val/FID': validation_fid,
            'val/loss': validation_loss,
            'Validation/Best FID': min(best_validation_fid, validation_fid),
            'Best Iteration': (
                nb_iter if validation_fid < best_validation_fid
                else best_validation_iter
            ),
        }, step=wandb_step)
        if validation_fid < best_validation_fid:
            best_validation_fid = validation_fid
            best_validation_iter = nb_iter
            save_checkpoint(nb_iter, 'net_best_validation_fid.pth')
        if nb_iter == VQ_RESUME_ITER:
            save_checkpoint(nb_iter, f"net_{VQ_RESUME_ITER}.pth")

save_checkpoint(args.total_iter, "net_last.pth")
if manifest_contract.get('schema') == 'hhc_vq_training_manifest_v2':
    final_checkpoint = os.path.join(ckpt_root, f'net_{args.total_iter}.pth')
    if not os.path.isfile(final_checkpoint):
        final_checkpoint = os.path.join(ckpt_root, 'net_last.pth')
    atomic_json(training_manifest_path, {
        **manifest_contract,
        'status': 'complete',
        'final_optimizer_step': main_update_offset + args.total_iter,
        'final_checkpoint': os.path.abspath(final_checkpoint),
        'final_checkpoint_sha256': sha256_file(final_checkpoint),
        'best_validation_checkpoint': os.path.abspath(
            os.path.join(ckpt_root, 'net_best_validation_fid.pth')
        ),
        'best_validation_fid': best_validation_fid,
        'best_validation_iter': best_validation_iter,
    })
wandb.finish()
