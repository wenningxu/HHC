import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
import math

import options.option_transformer as option_trans
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP3_shape
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize, dataset_interx, dataset_interx_TM, dataset_interx_tokenize
import models.t2m_trans as trans
from models.t2i_trans import Text2Motion_Transformer
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
import wandb
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
device = args.device
args.out_dir = os.path.join(args.out_dir, f't2i_trans')
# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
init_save_folder(args)

# [TODO] make the 'output/' folder as arg
args.vq_dir = f'./checkpoints/vq/' #os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
codebook_dir = f'{args.vq_dir}/codebook/'
args.resume_pth = f'{args.vq_dir}/net_best_fid.pth'
os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
# wandb.init(
#     project="NewInter",  # <-- 替换成你的项目名称
#     config=args,  # 自动记录所有超参数
#     group='vae_baseline_primitive',
#     name='vae_shape_primitive_0',
#     id='vae_shape_primitive_0'  # (可选) 为这次运行命名，可以基于时间戳或配置
# )

w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')
val_file = 'dataset/interx/val.txt'
val_motion = 'dataset/interx/motions/val.h5'
val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                          32,
                                          val_file,
                                          w_vectorizer,
                                          val_motion,
                                          unit_length=2 ** args.down_t)

dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
args.nb_joints = 55

wrapper_opt = get_opt(dataset_opt_path, device)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

net = VQVAE_SEP3_shape(args,  ## use args to define different parameters in different quantizers
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


trans_encoder = Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


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
    net.load_state_dict(state_dict, strict=True)
print ('loading checkpoint from {}'.format(args.resume_pth))
net.eval()
net.to(device)

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.to(device)
# trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####
train_file = 'dataset/interx/train.txt'
train_motion = 'dataset/interx/motions/train.h5'
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_interx_tokenize.DATALoader(args.dataname,
                                        1,
                                        train_file,
                                        w_vectorizer,
                                        train_motion,
                                        unit_length=2 ** args.down_t)
    for batch in tqdm(train_loader_token):
        pose, m_length, name, betas, gender, group = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.to(device).float() # bs, nb_joints, joints_dim, seq_len
        betas = betas.to(device).float()
        gender = gender.to(device)
        target = net(pose, betas, gender, type='encode')
        target = target.cpu().numpy()
        np.save(pjoin(codebook_dir, name[0] +'.npy'), target)


train_loader = dataset_interx_TM.DATALoader_Train(args.dataname, args.batch_size, train_file, args.nb_code, codebook_dir, unit_length=2**args.down_t)
train_loader_iter = dataset_interx_TM.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid=1000 
best_iter=0 
best_div=100 
best_top1=0 
best_top2=0 
best_top3=0 
best_matching=100
pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(
    args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid, best_iter, best_div, best_top1,
    best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper)


@torch.no_grad()  # 1. 关键：切断梯度计算，确保显存即时释放
def get_acc(cls_pred, target, mask):
    """
    内存优化版准确率计算
    cls_pred: (B, T, 3, V)
    target:   (B, T, 3)
    mask:     (B, T, 3)
    """
    # 2. 关键：先在完整 tensor 上做 Argmax
    # 这一步直接消掉了巨大的 V 维度 (如 512/1024)
    # 结果 shape: (B, T, 3)，只是一个轻量级的整数 Tensor
    pred_indices = cls_pred.argmax(dim=-1)

    # 3. 此时再进行 Mask 选择
    # 我们只在 (B, T, 3) 的整数上做操作，几乎不占显存
    # pred_selected: (N_masked,)
    pred_selected = pred_indices[mask]
    target_selected = target[mask]

    # 4. 计算准确率
    right_num = (pred_selected == target_selected).sum()
    total_num = mask.sum().clamp(min=1)

    # 5. 返回 Python float，彻底断开与 Tensor 的联系
    return (right_num * 100 / total_num).item()


# ---------------------------------------------------------
# 2. 训练循环
# ---------------------------------------------------------

# 假设 train_loader_iter, optimizer, scheduler, writer, logger 等已定义
# net 是包含 vqvae 的模型，trans_encoder 是你的 Text2Motion_Transformer

for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    # ------------------
    # A. 数据加载
    # ------------------
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, _, _ = batch

    # m_tokens shape: (B, T, 3) -> 包含 Upper, Lower, Root
    m_tokens, m_tokens_len = m_tokens.to(device), m_tokens_len.to(device)

    target = m_tokens.clone()  # GT
    batch_size, max_len, num_parts = target.shape  # num_parts 应该为 3

    # CLIP Text Encoding
    text = clip.tokenize(clip_text, truncate=True).to(device)
    feat_clip_text, word_emb = clip_model(text)

    # ------------------
    # B. 伯努利噪声 Mask (Token Replacement)
    # ------------------
    # 生成基础概率 Mask (B, T, 3)
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask_bernoulli = torch.bernoulli(proba * torch.ones(target.shape, device=target.device))
    else:
        mask_bernoulli = torch.bernoulli(args.pkeep * torch.ones(target.shape, device=target.device))

    # 生成 Padding Mask (B, T) -> 扩展到 (B, T, 3)
    # generate_src_mask 返回 True 表示由数据，False 表示 Pad
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # (B, T)
    seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)  # (B, T, 3)

    # 保护 Pad 不被替换 (Mask = 1 表示保留原值，0 表示替换)
    # 我们希望 Pad 的位置 mask 始终为 1 (不替换)，或者 MaskGIT逻辑中 Pad 始终不被选为 [MASK]
    # 这里的 mask 逻辑通常是：1=Keep, 0=Replace with Noise
    # mask_bernoulli = torch.logical_or(mask_bernoulli, ~seq_mask_no_end_3c).int()
    #
    # # 随机替换 Token
    # r_indices = torch.randint_like(target, args.nb_code)
    # # input_indices: 包含部分原始 token 和部分随机噪声 token
    # input_indices = mask_bernoulli * target + (1 - mask_bernoulli) * r_indices
    input_indices = target.clone()
    # ------------------
    # C. MaskGIT 时间步 Masking (核心修改)
    # ------------------
    mask_token_id = get_model(net).num_code + 2  # Mask Token ID

    # 1. 随机生成 Mask 比例 (每个样本不同)
    rand_time = torch.rand(batch_size, device=m_tokens_len.device)  # t ~ [0, 1]
    # 简单的 cosine schedule 模拟
    rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)
    rand_mask_probs = rand_mask_probs.clamp(min=0.1, max=1.0)

    # 2. 计算需要 Mask 掉多少个 Token
    # [关键] 总有效长度是 m_tokens_len * 3
    flat_valid_len = m_tokens_len * 3
    num_token_masked = (flat_valid_len * rand_mask_probs).round().clamp(min=1)  # (B,)

    # 3. 生成随机排列 (argsort) 以决定 Mask 哪些位置
    # 我们在 (B, T, 3) 的维度上进行随机，因此需要生成 (B, T, 3) 的随机分值
    batch_rand_score = torch.rand((batch_size, max_len, 3), device=target.device)
    mask_bias = torch.tensor([0.2, 1.0, 0.0], device=target.device)  # [Upper, Lower, Root]

    # 广播加法: (B, T, 3) + (3,)
    batch_rand_score = batch_rand_score + mask_bias

    # [关键] 增加 Pad 位置的分数，使其排序排在最后 (不被选中为 Mask)
    # ~seq_mask_no_end_3c 为 True 的地方是 Pad
    batch_rand_score = batch_rand_score + (~seq_mask_no_end_3c).float() * 1e9

    # 展平为 (B, T*3) 进行排序，分数小的排前面（将被 Mask）
    batch_randperm = batch_rand_score.view(batch_size, -1).argsort(dim=-1)  # (B, T*3)

    # 生成 Mask 矩阵 (前 k 个为 True)
    # num_token_masked: (B,) -> (B, 1)
    mask_token_flat = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    # 变回 (B, T, 3)
    mask_token = mask_token_flat.view(batch_size, max_len, 3)

    # 4. 应用 Mask
    # 被选中的位置变为 mask_id，没选中的保持 input_indices (原值或噪声)
    prob_random = 0.1
    mask_random_bool = torch.bernoulli(torch.full(mask_token.shape, prob_random, device=target.device)).bool()
    # 只有在 (MaskGIT选中) 且 (随机概率选中) 的地方，才替换为随机词
    final_mask_random = mask_token & mask_random_bool
    # 生成随机 Token (注意范围是 0 ~ num_vq-1)
    rand_tokens = torch.randint_like(target, high=args.nb_code)
    # 应用随机替换
    input_indices = torch.where(final_mask_random, rand_tokens, input_indices)
    # Step 2: 90% * 88.8% ≈ 80% Replace with [MASK] Token
    # 剩下的位置中 (即 mask_token 为 True 且 没被随机替换的)，有 ~89% 概率变为 [MASK]
    # 剩下的 ~11% (即总体的 10%) 保持原样 (Keep Original)
    prob_mask_token = 0.88
    mask_mask_bool = torch.bernoulli(torch.full(mask_token.shape, prob_mask_token, device=target.device)).bool()
    # 只有在 (MaskGIT选中) 且 (没被随机替换) 且 (Mask概率选中) 的地方，才替换为 [MASK] ID
    final_mask_mask = mask_token & (~final_mask_random) & mask_mask_bool

    # 应用 Mask Token 替换
    masked_input_indices = torch.where(final_mask_mask, mask_token_id, input_indices)

    # masked_input_indices = torch.where(mask_token, mask_token_id, input_indices)

    # ------------------
    # D. 模型前向传播
    # ------------------
    att_txt = None

    # Attention Mask 只需要 (B, T)，因为 Transformer 内部是对 Time 维度做 Attention
    # (B, T, 3) 的特征已经在模型内部拼接成 (B, T, C_emb) 了
    seq_mask_attn = generate_src_mask(max_len, m_tokens_len + 1)  # +1 for condition token

    # cls_pred: (B, T, 3, Num_VQ) -> 注意这里去掉了第一个 condition token 的输出
    cls_pred = trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                             word_emb=word_emb)
    # print('cls_pred', cls_pred.shape, 'target', target.shape)
    # 假设你的 trans_encoder 返回的是包含 condition 的，所以切片 [:, 1:]
    # 如果 trans_encoder 内部已经切了，请去掉这步
    # cls_pred = cls_pred[:, 1:]

    # ------------------
    # E. Loss 计算
    # ------------------
    # 1. 计算权重：只计算非 Pad 区域的 Loss
    # seq_mask_no_end_3c: (B, T, 3)
    weights = seq_mask_no_end_3c.float()

    # 归一化权重 (可选，MaskGIT 原文是 sum 后除以 batch size 或 token sum)
    # 这里我们按照 token 数量归一化，避免长短样本 Loss 差异过大
    # 每个样本内部归一化：
    sample_valid_tokens = seq_mask_no_end_3c.sum(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    weights = weights / sample_valid_tokens.clamp(min=1)

    # 2. 展平用于 CrossEntropy
    # Pred:   (B, T, 3, V) -> (-1, V)
    # Target: (B, T, 3)    -> (-1)
    # Weight: (B, T, 3)    -> (-1)
    cls_pred_flat = cls_pred[seq_mask_no_end_3c].reshape(-1, cls_pred.shape[-1])
    target_flat = target[seq_mask_no_end_3c].reshape(-1)
    weights_flat = weights[seq_mask_no_end_3c].reshape(-1)

    # 3. 计算 Loss
    # reduction='none' 以便应用权重
    loss_cls = F.cross_entropy(cls_pred_flat, target_flat, reduction='none')
    loss_cls = (loss_cls * weights_flat).sum()
    # 因为 weights 已经在样本内归一化了，sum 得到的是 batch 内平均 loss * B，所以除以 BS
    loss_cls = loss_cls / batch_size

    # ------------------
    # F. 反向传播
    # ------------------
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    # ------------------
    # G. 日志记录
    # ------------------
    if nb_iter % args.print_iter == 0:
        # 1. 整体准确率 (只看非 Pad 部分)
        probs = torch.softmax(cls_pred, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1)  # (B, T, 3)

        right_token = (cls_pred_index == target) & seq_mask_no_end_3c
        # 计算总准确率
        acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()

        writer.add_scalar('./Loss/all', loss_cls.item(), nb_iter)
        writer.add_scalar('./ACC/every_token', acc_all.item(), nb_iter)

        # 2. Masked 部分准确率 (模型“填空”的能力)
        # real_mask_token: 被 MaskGIT 掩盖 且 不是 Pad 的区域
        real_mask_token = mask_token & seq_mask_no_end_3c
        acc_masked = get_acc(cls_pred, target, real_mask_token)
        writer.add_scalar('./ACC/masked', acc_masked, nb_iter)

        # 3. Unmasked 部分准确率 (模型“重构/保持”的能力)
        # no_mask_token: 没有被 MaskGIT 掩盖 且 不是 Pad 的区域
        no_mask_token = (~mask_token) & seq_mask_no_end_3c
        acc_nomask = get_acc(cls_pred, target, no_mask_token)
        writer.add_scalar('./ACC/no_masked', acc_nomask, nb_iter)

        msg = f"Train. Iter {nb_iter} : Loss. {loss_cls:.5f}, ACC. {acc_all:.4f}"
        logger.info(msg)

    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            # val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)
        pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos)
        # for i in range(4):
        #     x = pose[i].detach().cpu().numpy()
        #     y = pred_pose_eval[i].detach().cpu().numpy()
        #     l = m_length[i]
        #     caption = clip_text[i]
        #     cleaned_name = '-'.join(caption[:200].split('/'))
        #     visualize_2motions(x, val_loader.dataset.std, val_loader.dataset.mean, args.dataname, l, y, save_path=f'{args.out_dir}/html/{str(nb_iter)}_{cleaned_name}_{l}.html')

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            