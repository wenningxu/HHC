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
from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM, dataset_interx_tokenize
from archive.diffusion_mld import eval_diffusion
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from utils.runtime import get_model, generate_src_mask, init_save_folder
from models.vqvae_sep import VQVAE_SEP, VQVAE_SEP_shape, VQVAE_SEP3_shape, VAE_CNN, VAE_SHAPE_CNN
from models.vqvae_hi import HumanVQVAEHI
from archive.diffusion_mld.vqtrans import DualPersonMldVAE, MldVae, MldVaeSeparate
from utils.rotation2xyz import Rotation2xyz_x
def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
torch.autograd.set_detect_anomaly(True)
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
device = args.device

args.out_dir = os.path.join(args.out_dir, f'vae-sep') # /{args.exp_name}
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
    config=args,                  # 自动记录所有超参数
    group='vae_baseline',
    name='vae_shape_cnn_1_small',
    id='vae_shape_cnn_1_small'# (可选) 为这次运行命名，可以基于时间戳或配置
)


w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

if args.dataname == 'kit' :
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 21

elif args.dataname == 'hhi':
    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55

else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
NUM_PRIMITIVE = args.num_primitive
HISTORY_LENGTH = args.history_length
FUTURE_LENGTH = args.future_length

train_file = 'dataset/interx/splits/train.txt'
train_motion = 'dataset/interx/motions/train.h5'
train_loader = dataset_interx.DATALoader(args.dataname,
                                        args.batch_size,
                                        train_file,
                                        train_motion,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_interx.cycle(train_loader)
# print('len_0' , len(train_loader))
val_file = 'dataset/interx/splits/val.txt'
val_motion = 'dataset/interx/motions/val.h5'
val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                    32,
                                    val_file,
                                    w_vectorizer,
                                    val_motion,
                                    unit_length=2**args.down_t)

val_loader_loss = dataset_interx.DATALoader(args.dataname,
                                    args.batch_size,
                                    val_file,
                                    val_motion,
                                    window_size=args.window_size,
                                    unit_length=2**args.down_t)
# val_loader_iter_loss = dataset_interx.cycle(val_loader_loss)
# print('len_1' , len(val_loader))
##### ---- Network ---- #####
args.sep_uplow = False
if args.sep_uplow:
    net = MldVae(56*12)
else:
    net = VAE_SHAPE_CNN(args, ## use args to define different parameters in different quantizers
                        args.code_dim,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm,)

# args.resume_pth = 'output/vq-shape/2025-10-27-13-08-27_vqvae_shape_7_sep_fk_cd64/net_last.pth'
if args.resume_pth :
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=False)
net.train()
net.to(device)



##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=0)
# rot2xyz = Rotation2xyz_x('cuda')
fk_layer = net.fk_layers
Loss = losses.ReConsLossInter(args, args.recons_loss, args.nb_joints, fk_layer)

##### ------ warm-up ------- #####
avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.
fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, continuous=True)
for nb_iter in tqdm(range(1, args.warm_up_iter)):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_motion, betas, gender, group = next(train_loader_iter)
    gt_motion = gt_motion.to(device).float() # (bs, 64, dim)
    betas = betas.to(device).float()
    gender = gender.to(device)
    group = group.to(device)
    m_length = torch.full((gt_motion.shape[0],), gt_motion.shape[1], dtype=torch.long).to(device)

    latent, dist = net.encode(gt_motion, betas, gender)
    pred_motion = net.decode(latent, betas, gender)

    # kl loss trans
    mu_ref = torch.zeros_like(dist.loc)
    scale_ref = torch.ones_like(dist.scale)
    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
    kl_loss = kl_loss.mean()
    # # body
    # mu_ref = torch.zeros_like(dist[1].loc)
    # scale_ref = torch.ones_like(dist[1].scale)
    # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    # kl_loss_body = torch.distributions.kl_divergence(dist[1], dist_ref)
    # kl_loss += kl_loss_body.mean()
    # # hand
    # mu_ref = torch.zeros_like(dist[2].loc)
    # scale_ref = torch.ones_like(dist[2].scale)
    # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    # kl_loss_hand = torch.distributions.kl_divergence(dist[2], dist_ref)
    # kl_loss += kl_loss_hand.mean()

    # kl_loss = kl_loss / 3.0

    losses = Loss(pred_motion, gt_motion, betas, gender, group)
    # loss_vel = Loss.forward_joint(pred_motion, gt_motion)

    loss = losses['main'] + losses['Joint'] + losses['Velocity'] + losses['Rotation'] + losses['Root'] + losses['FootContact'] + losses['DistanceMap'] + args.kl * kl_loss

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

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter
        avg_position /= args.print_iter
        avg_velocity /= args.print_iter
        avg_root /= args.print_iter
        avg_rotation /= args.print_iter
        avg_fc /= args.print_iter
        # avg_inter_oriental /= args.print_iter
        avg_distance_map /= args.print_iter

        logger.info(f"Train. Iter {nb_iter} : \t KL. {avg_kl:.5f} \t"
                    f"Recons.  {avg_recons:.5f} \t LR {scheduler.get_last_lr()[0]:.5f} \t Pos. {avg_position:.5f} "
                    f"\t Vel. {avg_velocity:.5f} \t Root. {avg_root:.5f} \t Rot. {avg_rotation:.5f} "
                    f"\t FC. {avg_fc:.5f} "  f"\t DistanceMap. {avg_distance_map:.5f}")
        wandb.log({
            'Warmup/Learning Rate': current_lr,
            'Warmup/Reconstruction Loss': avg_recons,
            'Warmup/KL Loss': avg_kl,
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
fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, continuous=True)

for nb_iter in tqdm(range(1, args.total_iter + 1)):

    gt_motion, betas, gender, group = next(train_loader_iter)
    gt_motion = gt_motion.to(device).float() # (bs, 64, dim)
    betas = betas.to(device).float()
    gender = gender.to(device)
    group = group.to(device)
    m_length = torch.full((gt_motion.shape[0],), gt_motion.shape[1], dtype=torch.long).to(device)

    latent, dist = net.encode(gt_motion, betas, gender)
    pred_motion = net.decode(latent, betas, gender)

    mu_ref = torch.zeros_like(dist.loc)
    scale_ref = torch.ones_like(dist.scale)
    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
    kl_loss = kl_loss.mean()
    # # body
    # mu_ref = torch.zeros_like(dist[1].loc)
    # scale_ref = torch.ones_like(dist[1].scale)
    # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    # kl_loss_body = torch.distributions.kl_divergence(dist[1], dist_ref)
    # kl_loss += kl_loss_body.mean()
    # # hand
    # mu_ref = torch.zeros_like(dist[2].loc)
    # scale_ref = torch.ones_like(dist[2].scale)
    # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    # kl_loss_hand = torch.distributions.kl_divergence(dist[2], dist_ref)
    # kl_loss += kl_loss_hand.mean()

    # kl_loss = kl_loss / 3.0

    losses = Loss(pred_motion, gt_motion, betas, gender, group)
    # loss_vel = Loss.forward_joint(pred_motion, gt_motion)

    loss = losses['main'] + losses['Joint'] + losses['Velocity'] + losses['Rotation'] + losses['Root'] + losses['FootContact'] + losses['DistanceMap'] + args.kl * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    avg_recons += losses['main'].item()
    avg_kl += args.kl * kl_loss.item()
    avg_position += losses['Joint'].item()
    avg_velocity += losses['Velocity'].item()
    avg_root += losses['Root'].item()
    avg_rotation += losses['Rotation'].item()
    avg_fc += losses['FootContact'].item()
    # avg_inter_oriental += losses['RelativeOrientation'].item()
    avg_distance_map += losses['DistanceMap'].item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter
        avg_position /= args.print_iter
        avg_velocity /= args.print_iter
        avg_root /= args.print_iter
        avg_rotation /= args.print_iter
        avg_fc /= args.print_iter
        # avg_inter_oriental /= args.print_iter
        avg_distance_map /= args.print_iter

        # writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        # writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        # writer.add_scalar('./Train/Commit', avg_commit, nb_iter)

        logger.info(f"Train. Iter {nb_iter} : \t KL. {avg_kl:.5f} \t"
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

    if nb_iter % args.eval_iter==0 :
        # torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
        fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper, continuous=True)
        wandb.log({
            'Validation/FID': fid,
            'Validation/Best FID': best_fid,
            'Validation/Best Diversity': best_div,
            'Validation/Best Top1 Accuracy': best_top1,
            'Validation/Best Matching Score': best_matching,
            'Best Iteration': best_iter  # 记录达到最佳性能的迭代次数
        }, step=nb_iter)

        # validation
        net.eval()
        val_iter_len = len(val_loader_loss)
        avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        for i, val_batch in enumerate(val_loader_loss):
            gt_motion, betas, gender, group = next(train_loader_iter)
            gt_motion = gt_motion.to(device).float() # (bs, 64, dim)
            betas = betas.to(device).float()
            gender = gender.to(device)
            group = group.to(device)
            m_length = torch.full((gt_motion.shape[0],), gt_motion.shape[1], dtype=torch.long).to(device)

            latent, dist = net.encode(gt_motion, betas, gender)
            pred_motion = net.decode(latent, betas, gender)

            # kl loss trans
            mu_ref = torch.zeros_like(dist.loc)
            scale_ref = torch.ones_like(dist.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
            kl_loss = kl_loss.mean()
            # # body
            # mu_ref = torch.zeros_like(dist[1].loc)
            # scale_ref = torch.ones_like(dist[1].scale)
            # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            # kl_loss_body = torch.distributions.kl_divergence(dist[1], dist_ref)
            # kl_loss += kl_loss_body.mean()
            # # hand
            # mu_ref = torch.zeros_like(dist[2].loc)
            # scale_ref = torch.ones_like(dist[2].scale)
            # dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            # kl_loss_hand = torch.distributions.kl_divergence(dist[2], dist_ref)
            # kl_loss += kl_loss_hand.mean()

            # kl_loss = kl_loss / 3.0

            losses = Loss(pred_motion, gt_motion, betas, gender, group)

            avg_recons += losses['main'].item()
            avg_kl += args.kl * kl_loss.item()
            avg_position += losses['Joint'].item()
            avg_velocity += losses['Velocity'].item()
            avg_root += losses['Root'].item()
            avg_rotation += losses['Rotation'].item()
            avg_fc += losses['FootContact'].item()
            # avg_inter_oriental += losses['RelativeOrientation'].item()
            avg_distance_map += losses['DistanceMap'].item()

        net.train()
        avg_recons /= val_iter_len
        avg_kl /= val_iter_len
        avg_position /= val_iter_len
        avg_velocity /= val_iter_len
        avg_root /= val_iter_len
        avg_rotation /= val_iter_len
        avg_fc /= val_iter_len
        # avg_inter_oriental /= args.print_iter
        avg_distance_map /= val_iter_len
        # writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        # writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        # writer.add_scalar('./Train/Commit', avg_commit, nb_iter
        logger.info(
            f"Validation. Iter {nb_iter} : \t KL. {avg_kl:.5f} \t"
            f"Recons.  {avg_recons:.5f} \t LR {scheduler.get_last_lr()[0]:.5f} \t Pos. {avg_position:.5f} "
            f"\t Vel. {avg_velocity:.5f} \t Root. {avg_root:.5f} \t Rot. {avg_rotation:.5f} "
            f"\t FC. {avg_fc:.5f} "  f"\t DistanceMap. {avg_distance_map:.5f}")
        wandb.log({
            'Validation_loss/Reconstruction Loss': avg_recons,
            'Validation_loss/KL Loss': avg_kl,
            'Validation_loss/Position Loss': avg_position,
            'Validation_loss/Velocity Loss': avg_velocity,
            'Validation_loss/Root Loss': avg_root,
            'Validation_loss/Rotation Loss': avg_rotation,
            'Validation_loss/Learning Rate': scheduler.get_last_lr()[0],  # 记录当前学习率
            'Validation_loss/FC Loss': avg_fc,
            # 'Train/Inter Oriental Loss': avg_inter_oriental,
            'Validation_loss/Distance Map Loss': avg_distance_map
        }, step=nb_iter)
        torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
        avg_recons, avg_kl, avg_position, avg_velocity, avg_root, avg_rotation, avg_fc, avg_inter_oriental, avg_distance_map = 0., 0., 0., 0., 0., 0., 0., 0., 0.




wandb.finish()