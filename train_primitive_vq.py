import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings

warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from exit.utils import get_model, generate_src_mask, init_save_folder
from models.vqvae_sep import VQVAE_SEP, VQVAE_SEP_shape, VQVAE_SEP_shape_primitive
from models.vqtrans import AutoMldVae
from utils.rotation2xyz import Rotation2xyz_x
from utils.motion_process import MotionNormalizerTorchInterX


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


##### ---- Exp dirs ---- #####
torch.autograd.set_detect_anomaly(False)
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'vq-shape')  # /{args.exp_name}
# args.warm_up_iter = 5
# args.total_iter = 1
# os.makedirs(args.out_dir, exist_ok = True)
init_save_folder(args)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
wandb.init(
    project="NewInter",  # <-- 替换成你的项目名称
    config=args,  # 自动记录所有超参数
    group='vae_baseline_primitive',
    name='vae_shape_primitive_0',
    id='vae_shape_primitive_0'  # (可选) 为这次运行命名，可以基于时间戳或配置
)

w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 21

elif args.dataname == 'hhi':
    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55

else:
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda:1'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
NUM_PRIMITIVE = args.num_primitive
HISTORY_LENGTH = args.history_length
FUTURE_LENGTH = args.future_length
print('num_primitive:', NUM_PRIMITIVE, ' history_length:', HISTORY_LENGTH, ' future_length:', FUTURE_LENGTH)

train_file = 'dataset/interx/train.txt'
train_motion = 'dataset/interx/motions/train.h5'
train_loader = dataset_interx.DATALoaderPrimitive(args.dataname,
                                         args.batch_size,
                                         train_file,
                                         train_motion,
                                        8,
                                         NUM_PRIMITIVE,
                                         HISTORY_LENGTH,
                                         FUTURE_LENGTH,
                                         unit_length=2 ** args.down_t)

train_loader_iter = dataset_interx.cycle(train_loader)

val_file = 'dataset/interx/val.txt'
val_motion = 'dataset/interx/motions/val.h5'
val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                          32,
                                          val_file,
                                          w_vectorizer,
                                          val_motion,
                                          unit_length=2 ** args.down_t)

##### ---- Network ---- #####
args.sep_uplow = False
mld = True
if args.sep_uplow:
    net = VQVAE_SEP_shape_primitive(args,  ## use args to define different parameters in different quantizers
                          args.nb_code,
                          args.code_dim,
                          args.output_emb_width,
                          args.down_t,
                          args.stride_t,
                          args.width,
                          args.depth,
                          args.dilation_growth_rate,
                          args.vq_act,
                          args.vq_norm, )
elif mld:
    net = AutoMldVae(56*12)
else:
    net = vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                           args.nb_code,
                           args.code_dim,
                           args.output_emb_width,
                           args.down_t,
                           args.stride_t,
                           args.width,
                           args.depth,
                           args.dilation_growth_rate,
                           args.vq_act,
                           args.vq_norm,
                           args.levels)

# args.resume_pth = 'output/vq-shape/2025-10-27-13-08-27_vqvae_shape_7_sep_fk_cd64/net_last.pth'
if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=False)
net.train()
net.cuda(1)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=0)
# rot2xyz = Rotation2xyz_x('cuda')
fk_layer = net.fk_layers
Loss = losses.ReConsLossInter(args, args.recons_loss, args.nb_joints, fk_layer)
motion_normalizer = MotionNormalizerTorchInterX()
##### ------ warm-up ------- #####
avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.
for nb_iter in tqdm(range(1, args.warm_up_iter)):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_motion, betas, gender, group = next(train_loader_iter)
    gt_motion = gt_motion.cuda(1).float()  # (bs, 64, dim)
    betas = betas.cuda(1).float()
    gender = gender.cuda(1)
    group = group.cuda(1)

    last_primitive_pred = None
    for primitive_idx in range(NUM_PRIMITIVE):
        # 从大张量中提取出当前基元的批次
        # motion_tensor shape: [B, primitive_length, D_motion]
        motion_tensor = gt_motion[:, primitive_idx, :, :]
        # [B, F, D]
        future_motion_gt = motion_tensor[:, -FUTURE_LENGTH:, :]
        # [B, H, D]
        history_motion_gt = motion_tensor[:, :HISTORY_LENGTH, :]
        # 2. 决定使用哪个历史
        current_history = history_motion_gt  # 默认使用真实历史
        if last_primitive_pred is not None:
            # 模拟 rollout：使用上一轮的预测结果
            # (在真实代码中，你会有 self.get_rollout_history)
            rollout_history = last_primitive_pred[:, -HISTORY_LENGTH:, :]
            current_history = rollout_history
            # print(f"    (idx={primitive_idx}: 使用上一轮的预测作为历史)")

        # pred_motion, loss_commit, perplexity = net(future_motion_gt, betas, gender, current_history)
        latent, dist = net.encode(future_motion_gt, current_history)
        pred_motion = net.decode(latent, current_history, FUTURE_LENGTH, betas, gender)
        losses = Loss(pred_motion, future_motion_gt, betas, gender, group)
        # loss_vel = Loss.forward_joint(pred_motion, gt_motion)
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
        kl_loss = kl_loss.mean()

        loss = losses['main'] + losses['Joint'] + losses['Velocity'] + losses['Root'] + losses[
            'Rotation'] + args.kl * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += losses['main'].item()
        avg_kl += args.kl * kl_loss.item()
        avg_position += losses['Joint'].item()
        avg_velocity += losses['Velocity'].item()
        avg_root += losses['Root'].item()
        avg_rotation += losses['Rotation'].item()
        avg_fc += losses['FootContact'].item()
        # avg_inter_oriental += losses['RelativeOrientation'].item()
        avg_distance_map += losses['DistanceMap'].item()

    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter * NUM_PRIMITIVE
        avg_kl /= args.print_iter * NUM_PRIMITIVE
        avg_position /= args.print_iter * NUM_PRIMITIVE
        avg_velocity /= args.print_iter * NUM_PRIMITIVE
        avg_root /= args.print_iter * NUM_PRIMITIVE
        avg_rotation /= args.print_iter * NUM_PRIMITIVE
        avg_fc /= args.print_iter * NUM_PRIMITIVE
        # avg_inter_oriental /= args.print_iter
        avg_distance_map /= args.print_iter * NUM_PRIMITIVE

        logger.info(f"Train. Iter {nb_iter} : \t  KL. {avg_kl:.5f} \t "
                    f"Recons.  {avg_recons:.5f} \t LR {scheduler.get_last_lr()[0]:.5f} \t Pos. {avg_position:.5f} "
                    f"\t Vel. {avg_velocity:.5f} \t Root. {avg_root:.5f} \t Rot. {avg_rotation:.5f} "
                    f"\t FC. {avg_fc:.5f} "  f"\t DistanceMap. {avg_distance_map:.5f}")
        wandb.log({
            'Warmup/Learning Rate': current_lr,
            'Warmup/KL Loss': avg_kl,
            'Warmup/Reconstruction Loss': avg_recons,
            'Warmup/Position Loss': avg_position,
            'Warmup/Velocity Loss': avg_velocity,
            'Warmup/Root Loss': avg_root,
            'Warmup/Rotation Loss': avg_rotation,
            'Warmup/FC Loss': avg_fc,
            # 'Warmup/Inter Oriental Loss': avg_inter_oriental,
            'Warmup/Distance Map Loss': avg_distance_map
        }, step=nb_iter)
        avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.
fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
    args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0,
    best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, primitive=True)

for nb_iter in tqdm(range(1, args.total_iter + 1)):

    gt_motion, betas, gender, group = next(train_loader_iter)
    gt_motion = gt_motion.cuda(1).float()  # bs, seq_len, feature
    betas = betas.cuda(1).float()
    gender = gender.cuda(1)
    group = group.cuda(1)

    last_primitive = None
    for primitive_idx in range(NUM_PRIMITIVE):
        # 从大张量中提取出当前基元的批次
        # motion_tensor shape: [B, primitive_length, D_motion]
        motion_tensor = gt_motion[:, primitive_idx, :, :]
        # [B, F, D]
        future_motion_gt = motion_tensor[:, -FUTURE_LENGTH:, :]
        # [B, H, D]
        history_motion_gt = motion_tensor[:, :HISTORY_LENGTH, :]
        # 2. 决定使用哪个历史
        current_history = history_motion_gt  # 默认使用真实历史
        if last_primitive is not None:
            # 模拟 rollout：使用上一轮的预测结果
            # (在真实代码中，你会有 self.get_rollout_history)
            rollout_history = last_primitive[:, -HISTORY_LENGTH:, :]
            current_history = rollout_history
        # print(f"    (idx={primitive_idx}: 使用上一轮的预测作为历史)")
        latent, dist = net.encode(future_motion_gt, current_history)
        pred_motion = net.decode(latent, current_history, FUTURE_LENGTH, betas, gender)
        losses = Loss(pred_motion, future_motion_gt, betas, gender, group)
        # loss_vel = Loss.forward_joint(pred_motion, gt_motion)
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
        kl_loss = kl_loss.mean()
        # loss_vel = Loss.forward_joint(pred_motion, gt_motion)

        loss = losses['main'] + losses['Joint'] + losses['Velocity'] + losses['Root'] + losses[
            'Rotation'] + args.kl * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if nb_iter > args.stage1_steps:
            rollout_prob = min(1.0, (nb_iter - args.stage1_steps) / max(
                float(args.stage1_steps), 1e-6))
            if torch.rand(1).item() < rollout_prob:
                last_primitive = pred_motion.detach()

        avg_recons += losses['main'].item()
        avg_kl += args.kl * kl_loss.item()
        avg_position += losses['Joint'].item()
        avg_velocity += losses['Velocity'].item()
        avg_root += losses['Root'].item()
        avg_rotation += losses['Rotation'].item()
        avg_fc += losses['FootContact'].item()
        # avg_inter_oriental += losses['RelativeOrientation'].item()
        avg_distance_map += losses['DistanceMap'].item()

    scheduler.step()

    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter * NUM_PRIMITIVE
        avg_kl /= args.print_iter * NUM_PRIMITIVE
        avg_position /= args.print_iter * NUM_PRIMITIVE
        avg_velocity /= args.print_iter * NUM_PRIMITIVE
        avg_root /= args.print_iter * NUM_PRIMITIVE
        avg_rotation /= args.print_iter * NUM_PRIMITIVE
        avg_fc /= args.print_iter * NUM_PRIMITIVE
        # avg_inter_oriental /= args.print_iter
        avg_distance_map /= args.print_iter * NUM_PRIMITIVE

        # writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        # writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        # writer.add_scalar('./Train/Commit', avg_commit, nb_iter)

        logger.info(f"Train. Iter {nb_iter} : \t  KL. {avg_kl:.5f} \t "
                    f"Recons.  {avg_recons:.5f} \t LR {scheduler.get_last_lr()[0]:.5f} \t Pos. {avg_position:.5f} "
                    f"\t Vel. {avg_velocity:.5f} \t Root. {avg_root:.5f} \t Rot. {avg_rotation:.5f} "
                    f"\t FC. {avg_fc:.5f} "  f"\t DistanceMap. {avg_distance_map:.5f}")

        wandb.log({
            'Train/Reconstruction Loss': avg_recons,
            'Train/KL Loss': avg_kl,
            'Train/Position Loss': avg_position,
            'Train/Velocity Loss': avg_velocity,
            'Train/Root Loss': avg_root,
            'Train/Rotation Loss': avg_rotation,
            'Train/Learning Rate': scheduler.get_last_lr()[0],  # 记录当前学习率
            'Train/FC Loss': avg_fc,
            # 'Train/Inter Oriental Loss': avg_inter_oriental,
            'Train/Distance Map Loss': avg_distance_map
        }, step=nb_iter)

        avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    if nb_iter % args.eval_iter == 0:
        fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2,
            best_top3, best_matching, eval_wrapper=eval_wrapper, primitive=True)
        wandb.log({
            'Validation/FID': fid,
            'Validation/Best FID': best_fid,
            'Validation/Best Diversity': best_div,
            'Validation/Best Top1 Accuracy': best_top1,
            'Validation/Best Matching Score': best_matching,
            'Best Iteration': best_iter  # 记录达到最佳性能的迭代次数
        }, step=nb_iter)
        torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))

wandb.finish()