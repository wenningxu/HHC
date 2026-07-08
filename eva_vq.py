import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import numpy as np

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
from models.vqvae_sep import VQVAE_SEP, VQVAE_SEP_shape, VQVAE_SEP_shape_primitive, VQVAE_SEP3_shape, VQVAE_SEP3_shape_single, VAE_SHAPE_CNN, VQVAE_SEP3
from models.vqvae_hi import HumanVQVAEHI
from models.vqtrans import MldVae, DualPersonMldVAE, AutoMldVae
from models.vqvae import HumanVQVAE
from utils.motion_process import recover_interx, MotionNormalizerTorchInterX

def get_latent_scale(model, batch):
    """
    get the scale of the latent space
        model: model or model_avg
    """
    original_mode = model.training
    model.eval()

    future_length = 12
    history_length = 3
    num_primitive = 4
    primitive_idx = 0

    with torch.no_grad():
        motion_tensor = batch[:, primitive_idx, :, :]
        # [B, F, D]
        future_motion_gt = motion_tensor[:, -future_length:, :]
        # [B, H, D]
        history_motion_gt = motion_tensor[:, :history_length, :]

        latent, dist = net.encode(future_motion_gt, history_motion_gt)

        all_mean = latent.mean()
        all_std = (latent - all_mean).pow(2).mean().sqrt()
        model.register_buffer("latent_mean", all_mean)
        model.register_buffer("latent_std", all_std)
        print(f"latent mean: {all_mean}, latent std: {all_std}")

    model.train(original_mode)
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
device = args.device

args.out_dir = os.path.join(args.out_dir, f'vae-sep') # /{args.exp_name}
NUM_PRIMITIVE = args.num_primitive
HISTORY_LENGTH = args.history_length
FUTURE_LENGTH = args.future_length

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)

w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

if args.dataname == 'kit' :
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 21

elif args.dataname == 'hhi':
    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55
elif args.dataname == 'hhi_shape':
    dataset_opt_path = 'checkpoints/hhi_shape/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55

else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
train_file = 'dataset/interx/train.txt'
train_motion = 'dataset/interx/motions/train.h5'
train_loader = dataset_interx.DATALoader(args.dataname,
                                        args.batch_size,
                                        train_file,
                                        train_motion,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)
# train_loader = dataset_interx.DATALoaderPrimitive(args.dataname,
#                                          args.batch_size,
#                                          train_file,
#                                          train_motion,
#                                         8,
#                                          NUM_PRIMITIVE,
#                                          HISTORY_LENGTH,
#                                          FUTURE_LENGTH,
#                                          unit_length=2 ** args.down_t)

train_loader_iter = dataset_interx.cycle(train_loader)


val_file = 'dataset/interx/splits/val.txt'
val_motion = 'dataset/interx/motions/val.h5'
val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                    32,
                                    val_file,
                                    w_vectorizer,
                                    val_motion,
                                    unit_length=2**args.down_t,
                                    shuffle=False)

##### ---- Network ---- #####
mld = False
args.sep_uplow = True
if args.sep_uplow:
    net = VQVAE_SEP3_shape(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm,)
elif mld:
    net = VAE_SHAPE_CNN(args,  ## use args to define different parameters in different quantizers
                  args.code_dim,
                  args.down_t,
                  args.stride_t,
                  args.width,
                  args.depth,
                  args.dilation_growth_rate,
                  args.vq_act,
                  args.vq_norm, )

else:
    net = HumanVQVAE(args, ## use args to define different parameters in different quantizers
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

args.resume_pth = 'output/vq-sep/2026-02-06-16-21-52_vqvae_shape_sep_single_token/net_300000.pth'
if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')

    state_dict = ckpt['net']

    # 1. 提取旧的权重
    key_p1 = "fk_layers.p1_positions_lookup.weight"
    key_p2 = "fk_layers.p2_positions_lookup.weight"
    new_key = "fk_layers.unified_positions_lookup.weight"

    if key_p1 in state_dict and key_p2 in state_dict:
        logger.info("Merging p1 and p2 lookup weights into unified_positions_lookup...")

        # 2. 拼接权重 (与你在 __init__ 中 torch.cat 的顺序一致)
        p1_weight = state_dict.pop(key_p1)  # pop 会删除并返回该值
        p2_weight = state_dict.pop(key_p2)
        combined_weight = torch.cat([p1_weight, p2_weight], dim=0)

        # 3. 放入新 key
        state_dict[new_key] = combined_weight

    # 4. 加载修改后的 state_dict
    net.load_state_dict(state_dict, True)
net.eval()
net.to(device)
# fk_layer = net.fk_layers
# Loss = losses.ReConsLossInter(args, args.recons_loss, args.nb_joints, fk_layer)
# gt_motion, betas, gender, group = next(train_loader_iter)
# gt_motion = gt_motion.to(device).float()  # (bs, 64, dim)
# betas = betas.to(device).float()
# gender = gender.to(device)
# group = group.to(device)


# next(iter(train_loader))
# gt_motion, betas, gender, group, text = next(iter(train_loader))
# all_mean = []
# all_std = []
# with torch.no_grad():
#     for batch in tqdm(train_loader):
#         gt_motion, betas, gender, group = batch
#         gt_motion = gt_motion.to(device).float()  # (bs, 64, dim)
#         betas = betas.to(device)
#         gender = gender.to(device)
#         latent, dist = net.encode(gt_motion, betas, gender)
#
#         all_mean.append(latent.mean())
#         all_std.append((latent - latent.mean()).pow(2).mean().sqrt())
#
#     all_mean = torch.stack(all_mean).mean()
#     all_std = torch.stack(all_std).mean()
#     net.register_buffer("latent_mean", all_mean)
#     net.register_buffer("latent_std", all_std)
#     print(f"latent mean: {all_mean}, latent std: {all_std}")
#
# torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, f'vae_shape_100000_std_scale.pth'))
save_folder = 'results/vqvae_sep_single'
os.makedirs(save_folder, exist_ok=True)
fids = []
for i in tqdm(range(1)):
    fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(save_folder, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, savenpy=True, primitive=False, continuous=False, shape_change=False, single=True)
    fids.append(fid)

print(np.array(fids).mean())

