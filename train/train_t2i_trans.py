import os
import sys

# Allow running as `python train/train_t2i_trans.py` from repo root by putting
# the repo root on sys.path before importing project packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
import pytorch_lightning as pl
from os.path import join as pjoin
import clip
import math
import time

import options.option_transformer as option_trans
from models.model_tools import build_model
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_interx_TM, dataset_interx_tokenize
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import wandb
from utils.runtime import generate_src_mask, init_save_folder
from einops import rearrange
import torch.nn.functional as F
from utils.word_vectorizer import WordVectorizer
from lightning.pytorch.loggers import WandbLogger
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx


def merge_interx_lookup_weights(state_dict):
    key_p1 = "fk_layers.p1_positions_lookup.weight"
    key_p2 = "fk_layers.p2_positions_lookup.weight"
    new_key = "fk_layers.unified_positions_lookup.weight"
    if key_p1 in state_dict and key_p2 in state_dict:
        p1_weight = state_dict.pop(key_p1)
        p2_weight = state_dict.pop(key_p2)
        state_dict[new_key] = torch.cat([p1_weight, p2_weight], dim=0)
    return state_dict


def load_transformer_checkpoint(module, ckpt_path):
    print(f'loading transformer checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    candidates = []
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            candidates.append(ckpt['state_dict'])
        if 'trans_encoder' in ckpt:
            candidates.append(ckpt['trans_encoder'])
        if 'net' in ckpt:
            candidates.append(ckpt['net'])
        candidates.append(ckpt)
    else:
        candidates.append(ckpt)

    prefixes = ('trans_encoder.', 'model.trans_encoder.', 'module.trans_encoder.', 'model.')
    last_error = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        filtered = {}
        for key, value in candidate.items():
            stripped = None
            for prefix in prefixes:
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    break
            if stripped is not None:
                filtered[stripped] = value
        if not filtered:
            filtered = candidate
        try:
            missing, unexpected = module.load_state_dict(filtered, strict=False)
            print(f'transformer checkpoint loaded with {len(missing)} missing and {len(unexpected)} unexpected keys')
            return missing, unexpected
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Unable to load transformer checkpoint: {ckpt_path}')


def build_split_loader(args, split_name, w_vectorizer):
    split_file = f'dataset/interx/{split_name}.txt'
    split_motion = f'dataset/interx/motions/{split_name}.h5'
    return dataset_interx_TM.DATALoader(
        args.dataname,
        32,
        split_file,
        w_vectorizer,
        split_motion,
        unit_length=2 ** args.down_t,
        shuffle=False,
    )


def run_generation_export(args, model, val_loader):
    class LoggerWrapper:
        def info(self, msg):
            print(msg)

    output_dir = args.save_dir
    if output_dir is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.gen_ckpt))[0].replace('/', '_')
        output_dir = os.path.join(args.out_dir, f'mmodality_{args.gen_split}_{ckpt_stem}_temp1_repeat{args.num_repeat}_nocontact')
    os.makedirs(output_dir, exist_ok=True)

    logger = LoggerWrapper()
    evaluation_args = dict(
        out_dir=output_dir,
        val_loader=val_loader,
        net=model.net,
        trans=model.trans_encoder,
        logger=logger,
        writer=None,
        nb_iter=0,
        best_fid=1000,
        best_iter=0,
        best_div=100,
        best_top1=0,
        best_top2=0,
        best_top3=0,
        best_matching=100,
        clip_model=model.clip_model,
        eval_wrapper=model.eval_wrapper,
        dataname=args.dataname,
        save=False,
        savegif=False,
        num_repeat=args.num_repeat,
        rand_pos=True,
        CFG=3.5,
        savenpy=True,
        contact=False,
        shape_change=None,
        save_embeddings=args.save_embeddings,
    )
    eval_trans.evaluation_transformer(**evaluation_args)
    print(f'generation export finished: {output_dir}')


# 辅助函数：计算准确率（保持原有逻辑）
@torch.no_grad()
def get_acc(cls_pred, target, mask):
    pred_indices = cls_pred.argmax(dim=-1)
    pred_selected = pred_indices[mask]
    target_selected = target[mask]
    right_num = (pred_selected == target_selected).sum()
    total_num = mask.sum().clamp(min=1)
    return (right_num * 100 / total_num).item()


# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb


class MaskGITLightning(pl.LightningModule):
    def __init__(self, args, eval_wrapper, val_loader):
        super().__init__()
        self.save_hyperparameters(ignore=['eval_wrapper', 'val_loader'])
        self.automatic_optimization = False
        self.args = args
        self.eval_wrapper = eval_wrapper
        self.val_loader = val_loader  # 保存用于自定义 evaluation

        self.contact = args.contact
        self.shape = args.shape

        # ------------------
        # 1. 初始化模型
        # ------------------
        # CLIP
        clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = TextCLIP(clip_model)

        # VQVAE
        self.net = build_model(args, "interx_vq")

        # Transformer
        self.trans_encoder = build_model(args, "interx_transformer", vqvae=self.net)

        # ------------------
        # 2. 加载权重 (Resume Logic)
        # ------------------
        if args.resume_pth:
            print(f'loading checkpoint from {args.resume_pth}')
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            state_dict = ckpt['net']

            # 合并 P1 P2 权重逻辑
            key_p1 = "fk_layers.p1_positions_lookup.weight"
            key_p2 = "fk_layers.p2_positions_lookup.weight"
            new_key = "fk_layers.unified_positions_lookup.weight"
            if key_p1 in state_dict and key_p2 in state_dict:
                print("Merging p1 and p2 lookup weights into unified_positions_lookup...")
                p1_weight = state_dict.pop(key_p1)
                p2_weight = state_dict.pop(key_p2)
                combined_weight = torch.cat([p1_weight, p2_weight], dim=0)
                state_dict[new_key] = combined_weight

            self.net.load_state_dict(state_dict, strict=True)
            self.net.eval()  # VQVAE 始终 Eval

        if args.resume_trans:
            load_transformer_checkpoint(self.trans_encoder, args.resume_trans)
        self.trans_encoder.eval()

        # ------------------
        # 3. 评估指标状态记录
        # ------------------
        self.best_fid = 1000
        self.best_iter = 0
        self.best_div = 100
        self.best_top1 = 0
        self.best_top2 = 0
        self.best_top3 = 0
        self.best_matching = 100

        # 这里的 writer 会在 fit 开始后由 Logger Connector 接管，
        # 但 eval_trans 需要原始 writer，我们会在 on_train_start 里处理

    def configure_optimizers(self):
        optimizer = utils_model.initial_optim(self.args.decay_option, self.args.lr, self.args.weight_decay,
                                              self.trans_encoder, self.args.optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # ------------------
        # A. 数据准备
        # ------------------
        clip_text, m_tokens, m_tokens_len, betas, gender, contact = batch
        # Lightning 会自动处理 device，但如果 eval_trans 需要特定类型，注意保持
        target = m_tokens.clone()
        batch_size, max_len, num_parts = target.shape

        # CLIP Encode

        text = clip.tokenize(clip_text, truncate=True).to(self.device)
        feat_clip_text, word_emb = self.clip_model(text)
        # ------------------
        # B. 初始化 (纯净 GT)
        # ------------------
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # (B, T)
        seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)  # (B, T, 3)

        if self.args.pkeep == -1:
            proba = np.random.rand(1)[0]
            mask_bernoulli = torch.bernoulli(proba * torch.ones(target.shape, device=target.device))
        else:
            mask_bernoulli = torch.bernoulli(args.pkeep * torch.ones(target.shape, device=target.device))

        mask_bernoulli = torch.logical_or(mask_bernoulli, ~seq_mask_no_end_3c).int()

        # 随机替换 Token
        r_indices = torch.randint_like(target, args.nb_code)
        # input_indices: 包含部分原始 token 和部分随机噪声 token
        input_indices = mask_bernoulli * target + (1 - mask_bernoulli) * r_indices
        # ------------------
        # C. MaskGIT + BERT 策略
        # ------------------
        mask_token_id = self.args.nb_code + 2

        # 1. 随机生成 Mask 比例
        rand_time = torch.rand(batch_size, device=self.device)
        rand_mask_probs = torch.cos(rand_time * math.pi * 0.5).clamp(min=0.1, max=1.0)

        # 2. 计算 Mask 数量
        flat_valid_len = m_tokens_len * 3
        num_token_masked = (flat_valid_len.float() * rand_mask_probs).round().clamp(min=1).long()

        # 3. 结构化 Mask Bias (Root > Lower > Upper)
        batch_rand_score = torch.rand((batch_size, max_len, 3), device=self.device)
        mask_bias = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # Upper易Mask, Root难Mask
        batch_rand_score = batch_rand_score + mask_bias

        # 保护 Pad
        batch_rand_score = batch_rand_score + (~seq_mask_no_end_3c).float() * 1e9

        # 生成 Mask
        batch_randperm = batch_rand_score.view(batch_size, -1).argsort(dim=-1)
        mask_token_flat = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        mask_token = mask_token_flat.view(batch_size, max_len, 3)
        # 4. 应用 Mask
        masked_input_indices = torch.where(mask_token, mask_token_id, input_indices)

        # ------------------
        # D. Forward
        # ------------------
        att_txt = None

        seq_mask_attn = generate_src_mask(max_len, m_tokens_len + 1)

        if self.contact and not self.shape:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, contact_mask =contact)
        elif self.contact and self.shape:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, contact_mask =contact, betas=betas, gender=gender)
        else:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, betas=betas, gender=gender)


        # ------------------
        # E. Loss
        # ------------------
        weights = seq_mask_no_end_3c.float()
        sample_valid_tokens = seq_mask_no_end_3c.sum(dim=(1, 2), keepdim=True)
        weights = weights / sample_valid_tokens.clamp(min=1)

        cls_pred_flat = cls_pred[seq_mask_no_end_3c].reshape(-1, cls_pred.shape[-1])
        target_flat = target[seq_mask_no_end_3c].reshape(-1)
        weights_flat = weights[seq_mask_no_end_3c].reshape(-1)

        loss_cls = F.cross_entropy(cls_pred_flat, target_flat, reduction='none')
        loss_cls = (loss_cls * weights_flat).sum()
        loss_cls = loss_cls / batch_size

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss_cls)
        opt.step()
        self.lr_schedulers().step()
        # ------------------
        # F. Logging
        # ------------------
        # 这里的 global_step 是 Lightning 自动管理的
        probs = torch.softmax(cls_pred, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1)
        right_token = (cls_pred_index == target) & seq_mask_no_end_3c
        acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()
        real_mask_token = mask_token & seq_mask_no_end_3c
        acc_masked = get_acc(cls_pred, target, real_mask_token)
        no_mask_token = (~mask_token) & seq_mask_no_end_3c
        acc_nomask = get_acc(cls_pred, target, no_mask_token)
        # 使用 Lightning 的 logger
        self.log('Train/all', loss_cls, prog_bar=True)
        self.log('Train/every_token', acc_all, prog_bar=True)
        self.log('Train/masked', acc_masked, prog_bar=True)
        self.log('Train/no_masked', acc_nomask, prog_bar=True)
        # print(f"Iter {self.global_step} : Loss {loss_cls.item():.5f}, ACC {acc_all:.4f}")

        return loss_cls

    def validation_step(self, batch, batch_idx):
        pass
        # Lightning 需要这个方法，但我们的自定义 eval 是在 epoch start 统一跑的
        # clip_text, m_tokens, m_tokens_len, _, _, contact = batch
        # # Lightning 会自动处理 device，但如果 eval_trans 需要特定类型，注意保持
        # target = m_tokens.clone()
        # batch_size, max_len, num_parts = target.shape
        #
        # # CLIP Encode
        # text = clip.tokenize(clip_text, truncate=True).to(self.device)
        # feat_clip_text, word_emb = self.clip_model(text)
        # # ------------------
        # # B. 初始化 (纯净 GT)
        # # ------------------
        # seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # (B, T)
        # seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)  # (B, T, 3)
        #
        # if self.args.pkeep == -1:
        #     proba = np.random.rand(1)[0]
        #     mask_bernoulli = torch.bernoulli(proba * torch.ones(target.shape, device=target.device))
        # else:
        #     mask_bernoulli = torch.bernoulli(args.pkeep * torch.ones(target.shape, device=target.device))
        #
        # mask_bernoulli = torch.logical_or(mask_bernoulli, ~seq_mask_no_end_3c).int()
        #
        # # 随机替换 Token
        # r_indices = torch.randint_like(target, args.nb_code)
        # # input_indices: 包含部分原始 token 和部分随机噪声 token
        # input_indices = mask_bernoulli * target + (1 - mask_bernoulli) * r_indices
        # # ------------------
        # # C. MaskGIT + BERT 策略
        # # ------------------
        # mask_token_id = self.args.nb_code + 2
        #
        # # 1. 随机生成 Mask 比例
        # rand_time = torch.rand(batch_size, device=self.device)
        # rand_mask_probs = torch.cos(rand_time * math.pi * 0.5).clamp(min=0.1, max=1.0)
        #
        # # 2. 计算 Mask 数量
        # flat_valid_len = m_tokens_len * 3
        # num_token_masked = (flat_valid_len.float() * rand_mask_probs).round().clamp(min=1).long()
        #
        # # 3. 结构化 Mask Bias (Root > Lower > Upper)
        # batch_rand_score = torch.rand((batch_size, max_len, 3), device=self.device)
        # mask_bias = torch.tensor([0.5, 1.0, 0.0], device=self.device)  # Upper易Mask, Root难Mask
        # batch_rand_score = batch_rand_score + mask_bias
        #
        # # 保护 Pad
        # batch_rand_score = batch_rand_score + (~seq_mask_no_end_3c).float() * 1e9
        #
        # # 生成 Mask
        # batch_randperm = batch_rand_score.view(batch_size, -1).argsort(dim=-1)
        # mask_token_flat = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        # mask_token = mask_token_flat.view(batch_size, max_len, 3)
        # # 4. 应用 Mask
        # masked_input_indices = torch.where(mask_token, mask_token_id, input_indices)
        #
        # # ------------------
        # # D. Forward
        # # ------------------
        # att_txt = None
        # seq_mask_attn = generate_src_mask(max_len, m_tokens_len + 1)
        #
        # if self.contact:
        #     cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
        #                                   word_emb=word_emb, contact_mask=contact)
        # else:
        #     cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
        #                                   word_emb=word_emb)
        #
        # # ------------------
        # # E. Loss
        # # ------------------
        # weights = seq_mask_no_end_3c.float()
        # sample_valid_tokens = seq_mask_no_end_3c.sum(dim=(1, 2), keepdim=True)
        # weights = weights / sample_valid_tokens.clamp(min=1)
        #
        # cls_pred_flat = cls_pred[seq_mask_no_end_3c].reshape(-1, cls_pred.shape[-1])
        # target_flat = target[seq_mask_no_end_3c].reshape(-1)
        # weights_flat = weights[seq_mask_no_end_3c].reshape(-1)
        #
        # loss_cls = F.cross_entropy(cls_pred_flat, target_flat, reduction='none')
        # loss_cls = (loss_cls * weights_flat).sum()
        # loss_cls = loss_cls / batch_size
        # # ------------------
        # # F. Logging
        # # ------------------
        # # 这里的 global_step 是 Lightning 自动管理的
        # probs = torch.softmax(cls_pred, dim=-1)
        # _, cls_pred_index = torch.max(probs, dim=-1)
        # right_token = (cls_pred_index == target) & seq_mask_no_end_3c
        # acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()
        # real_mask_token = mask_token & seq_mask_no_end_3c
        # acc_masked = get_acc(cls_pred, target, real_mask_token)
        # no_mask_token = (~mask_token) & seq_mask_no_end_3c
        # acc_nomask = get_acc(cls_pred, target, no_mask_token)
        # # 使用 Lightning 的 logger
        # self.log('Validation/all', loss_cls, prog_bar=True)
        # self.log('Validation/every_token', acc_all, prog_bar=True)
        # self.log('Validation/masked', acc_masked, prog_bar=True)
        # self.log('Validation/no_masked', acc_nomask, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Lightning 需要这个方法，但我们的自定义 eval 是在 epoch start 统一跑的
        pass

    def generate(self, texts, betas, gender, length, out_dir, name = None, contact_mask = None):
        normalizer = MotionNormalizerTorchInterX('raw')
        device = self.device
        max_length = self.trans_encoder.block_size - 1
        vocab_contact_map = np.load('checkpoints/vq/code_contact_probs.npy')
        vocab_contact_map = torch.from_numpy(vocab_contact_map).float().to(device)
        if name is None:
            name = [t[:10] for t in texts]

        betas = betas.to(device).float()
        gender = gender.to(device)
        tick = time.time()
        text = clip.tokenize(texts, truncate=True).to(device)
        length = torch.tensor(length).to(device)
        m_tokens_len = torch.ceil((length) / 4)

        bs = text.shape[0]

        feat_clip_text, word_emb = self.clip_model(text)
        if contact_mask is not None:
            contact_mask = contact_mask.view(bs, -1, 4)
            contact_mask = (torch.sum(contact_mask, dim=-1) > 0).long()
            if contact_mask.shape[1] < max_length:
                contact_mask = torch.cat([
                    contact_mask,
                    torch.zeros((bs, max_length - contact_mask.shape[1]), dtype=contact_mask.dtype),
                ], dim=1).to(device)

        index_motion = self.trans_encoder(feat_clip_text, word_emb, type="sample", m_length=m_tokens_len,
                             rand_pos=False, betas=betas, gender=gender, CFG=3.5)

        for k in range(bs):
            pred_pose = self.net(index_motion[k:k + 1, :int(m_tokens_len[k].item())], betas[k:k + 1], gender[k:k + 1],
                            type='decode')
            pred_pose = normalizer.backward(pred_pose)
            print('time taken for generation: ', time.time() - tick)
            motion_length = int(m_tokens_len[k].item()) * 4
            trans1, pose1, trans2, pose2 = recover_interx(pred_pose[:motion_length].squeeze())
            gender_word = ['male' if g == 0 else 'female' for g in gender[k]]

            np.savez(os.path.join(out_dir, name[k] + '_pred_p1.npz'),
                     poses=pose1, trans=trans1, betas=betas[k, 0:10].cpu().detach().numpy(), gender=gender_word[0],
                     mocap_framerate=30)
            np.savez(os.path.join(out_dir, name[k] + '_pred_p2.npz'),
                     poses=pose2, trans=trans2, betas=betas[k, 10:].cpu().detach().numpy(), gender=gender_word[1],
                     mocap_framerate=30)

    def on_test_epoch_start(self):
        wan_writer = None

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        multi = []

        class LoggerWrapper:
            def __init__(self, log_path, mode='a'):
                """
                log_path: log文件的路径 (例如 'output/train.log')
                mode: 'w' 表示覆盖写入, 'a' 表示追加写入
                """
                # 自动创建父目录（如果不存在）
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                self.file = open(log_path, mode)
                self.path = log_path

            def info(self, msg):
                # 1. 打印到控制台
                print(msg)

                # 2. 写入文件 (注意要手动加换行符 \n)
                self.file.write(msg + '\n')

                # 3. 立即刷新缓冲区，确保程序意外中断时日志已被写入
                self.file.flush()

            def close(self):
                # 关闭文件句柄
                if not self.file.closed:
                    self.file.close()

            def __del__(self):
                # 对象销毁时尝试关闭文件
                self.close()

        custom_logger = LoggerWrapper(self.args.out_dir + '/test_log.txt', mode='a')
        repeat_time = 1
        save_folder = 'final_demo/shape_0_rbh'
        os.makedirs(save_folder, exist_ok=True)
        for i in tqdm(range(repeat_time)):
            _, pose, m_length, clip_text, \
                best_fid, best_iter, best_div, \
                best_top1, best_top2, best_top3, \
                best_matching, best_multi, \
                _, _ = eval_trans.evaluation_transformer(
                save_folder,
                self.trainer.test_dataloaders,
                self.net,
                self.trans_encoder,
                custom_logger,
                wan_writer,
                i,
                1000,
                0,
                100,
                0,
                0,
                0,
                100,
                num_repeat=30,
                clip_model=self.clip_model,
                eval_wrapper=self.eval_wrapper,
                dataname=self.args.dataname,
                save=False,
                savenpy=False,
                rand_pos=False,
                contact=True,
                shape_change=None,
                CFG=3.5
            )
            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
            multi.append(best_multi)
        print('final result:')
        print('fid: ', sum(fid) / repeat_time)
        print('div: ', sum(div) / repeat_time)
        print('top1: ', sum(top1) / repeat_time)
        print('top2: ', sum(top2) / repeat_time)
        print('top3: ', sum(top3) / repeat_time)
        print('matching: ', sum(matching) / repeat_time)
        print('multi: ', sum(multi) / repeat_time)
        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        multi = np.array(multi)
        msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi) * 1.96 / np.sqrt(repeat_time):.3f}"
        custom_logger.info(msg_final)

    def on_validation_epoch_start(self):
        # ------------------
        # G. 自定义评估逻辑
        # ------------------
        # 注意：这里调用的是 eval_trans.evaluation_transformer
        # 这个函数需要 logger 和 writer 对象。
        # 我们使用 self.logger.experiment (TensorboardWriter)

        # 只有主进程跑评估
        if self.trainer.is_global_zero:
            current_iter = self.global_step
            # 获取原始 TensorBoard writer
            # wandb writer
            wan_writer = None

            # 使用 Python logging 的 logger (utils_model.get_logger 创建的)
            # 或者简单传入 self.print
            # 这里为了兼容 eval_trans 接口，我们需要一个类似 logger 的对象
            # 我们可以直接传 None，或者一个简单的 Wrapper，因为 eval_trans 里主要是 logger.info

            # 兼容性处理
            class LoggerWrapper:
                def info(self, msg):
                    print(msg)

            custom_logger = LoggerWrapper()

            print(f"\n[Validation] Starting evaluation at iter {current_iter}...")

            # 调用原始评估函数
            # 注意：eval_trans 内部会循环 val_loader，所以这里不需要 batch
            save_val_dir = 'results/vq_mask_gen_4_contact_ablation_adalnshape_new'
            os.makedirs(save_val_dir, exist_ok=True)
            fid, pose, m_length, clip_text, \
                self.best_fid, self.best_iter, self.best_div, \
                self.best_top1, self.best_top2, self.best_top3, \
                self.best_matching, self.best_multi, \
                _, _ = eval_trans.evaluation_transformer(
                save_val_dir,
                self.val_loader,
                self.net,
                self.trans_encoder,
                custom_logger,
                wan_writer,
                current_iter,
                self.best_fid,
                self.best_iter,
                self.best_div,
                self.best_top1,
                self.best_top2,
                self.best_top3,
                self.best_matching,
                clip_model=self.clip_model,
                eval_wrapper=self.eval_wrapper,
                dataname=self.args.dataname,
                save=False,
                savenpy=False,
                contact = self.contact,
                CFG=3.5
            )

            self.log('val/FID', fid)
            self.log('val/Diversity', self.best_div)
            self.log('val/TOP1', self.best_top1)
            self.log('val/TOP2', self.best_top2)
            self.log('val/TOP3', self.best_top3)
            self.log('val/Matching', self.best_matching)

            # np.save(os.path.join(self.args.out_dir, f'pred{self.global_step}.bpy'), pose.cpu().detach().numpy())

if __name__ == '__main__':
    # ------------------
    # 0. 配置与目录准备
    # ------------------
    args = option_trans.get_args_parser()
    pl.seed_everything(args.seed)
    if args.generate_only:
        args.contact = False
        args.device = 'cuda:0'
        if args.gen_ckpt is None:
            args.gen_ckpt = 'output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt'
        args.resume_trans = args.gen_ckpt

    args.out_dir = os.path.join(args.out_dir, f't2i_trans')
    init_save_folder(args)

    # 路径设置
    args.vq_dir = f'./checkpoints/vq/'
    codebook_dir = f'{args.vq_dir}/codebook/'
    args.resume_pth = f'{args.vq_dir}/net_175000.pth'
    os.makedirs(args.vq_dir, exist_ok=True)
    os.makedirs(codebook_dir, exist_ok=True)
    os.makedirs(f'{args.vq_dir}/contact_token_label', exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Logger
    # logger = utils_model.get_logger(args.out_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ------------------
    # 1. 数据准备
    # ------------------
    w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

    # Val Loader
    val_file = 'dataset/interx/val.txt'
    val_motion = 'dataset/interx/motions/val.h5'
    val_loader = build_split_loader(args, 'val', w_vectorizer)

    # Eval Wrapper
    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55
    # 注意：Lightning 会处理 device，这里 device 只是用来初始化 wrapper
    eval_wrapper = EvaluatorModelWrapper(get_opt(dataset_opt_path, args.device))

    # Codebook 生成逻辑 (只在 codebook 目录为空时运行)
    train_file = 'dataset/interx/train.txt'
    train_motion = 'dataset/interx/motions/train.h5'
    all_file = 'dataset/interx/all.txt'
    all_motion = 'dataset/interx/motions/all.h5'

    if len(os.listdir(codebook_dir)) == 0:
        print("Generating codebook cache...")
        # 这里的 net 初始化只是为了生成 codebook，可以用 cpu 或 gpu
        # 为了简单，我们手动跑一下这部分逻辑，或者把它也封装进 LightningModule (略显复杂)
        # 这里保持原样，手动跑
        net = build_model(args, "interx_vq")
        if args.resume_pth:
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            state_dict = ckpt['net']
            # ... (合并权重逻辑同上) ...
            key_p1 = "fk_layers.p1_positions_lookup.weight"
            key_p2 = "fk_layers.p2_positions_lookup.weight"
            new_key = "fk_layers.unified_positions_lookup.weight"
            if key_p1 in state_dict and key_p2 in state_dict:
                p1_weight = state_dict.pop(key_p1);
                p2_weight = state_dict.pop(key_p2)
                state_dict[new_key] = torch.cat([p1_weight, p2_weight], dim=0)
            net.load_state_dict(state_dict, strict=True)
            net.to(args.device)
            net.eval()

        train_loader_token = dataset_interx_tokenize.DATALoader(args.dataname, 1, all_file, w_vectorizer,
                                                                all_motion, unit_length=2 ** args.down_t)
        for batch in tqdm(train_loader_token):
            pose, m_length, name, betas, gender, group, contact = batch
            pose = pose.to(args.device).float()
            betas = betas.to(args.device).float()
            gender = gender.to(args.device)
            target = net(pose, betas, gender, type='encode')
            np.save(pjoin('checkpoints/vq/contact_token_label', name[0] + '.npy'), contact.cpu().numpy())
            np.save(pjoin(codebook_dir, name[0] + '.npy'), target.cpu().numpy())
        del net

    # ------------------
    # 2. 训练设置
    # ------------------
    model = MaskGITLightning(args, eval_wrapper, val_loader)
    if args.generate_only:
        export_loader = build_split_loader(args, args.gen_split, w_vectorizer)
        model = model.to(args.device)
        model.eval()
        run_generation_export(args, model, export_loader)
        raise SystemExit(0)

    # Train Loader
    train_loader = dataset_interx_TM.DATALoader_Train(args.dataname, args.batch_size, train_file, args.nb_code,
                                                      codebook_dir, unit_length=2 ** args.down_t)
    val_loss_loader = dataset_interx_TM.DATALoader_Train(args.dataname, args.batch_size, val_file, args.nb_code,
                                                         codebook_dir, unit_length=2 ** args.down_t)

    # Checkpoint 回调 (保存最佳 FID)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.out_dir,
        monitor='val/FID',  # 我们在 eval_trans 里手动保存了，这里只是占位或者保存 last
        filename='model-{step:02d}-{val/FID:.4f}',
        save_top_k=10,
        mode='min',
    )

    # Logger
    wandb_logger = WandbLogger(
        project="NewInter",  # <-- 替换成你的项目名称
        config=args,  # 自动记录所有超参数
        group='mask_gen',
        name='vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas',
        id='vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas'  # (可选) 为这次运行命名，可以基于时间戳或配置
    )

    # Trainer
    # 注意：limit_val_batches=0 会跳过 Lightning 的标准 validate loop，但会触发 on_validation_epoch_start
    # 但为了让 eval_trans 运行，我们需要让 validate 至少跑一次。
    # 这里我们用一个小技巧：val_check_interval 控制频率，
    # 传入 val_loader 给 trainer，但在 validation_step 里什么都不做。

    trainer = pl.Trainer(
        max_steps=args.total_iter,
        accelerator='gpu',
        devices=[0] if args.device.endswith(':0') else [1],  # 根据你的 GPU 设置
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=args.eval_iter,  # 替代 nb_iter % args.eval_iter == 0
        check_val_every_n_epoch=None,
        precision="bf16-mixed"# 只需要触发一次 validation start 即可，不需要跑完整个 val loader (因为我们有自定义 eval)
    )

    # ------------------
    # 3. 开始训练
    # ------------------
    # 这里的 val_dataloaders 只是为了触发 validation hook
    # 真实的 evaluation 数据在 MaskGITLightning.val_loader 里
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loss_loader)

    # -------------------------
    # 3. 直接运行验证（跳过训练）
    # test_file = 'dataset/interx/test.txt'
    # test_motion = 'dataset/interx/motions/test.h5'
    # test_loader = dataset_interx_TM.DATALoader(args.dataname, 32, test_file, w_vectorizer, test_motion, unit_length=2 ** args.down_t, shuffle=False)
    # trainer.test(model, dataloaders=val_loader, ckpt_path='output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt')

    # trainer.validate(model, dataloaders=val_loader, ckpt_path='output/t2i_trans/2026-02-02-11-22-01_vq_mask_gen_4/model-step=190000-val/FID=0.2891.ckpt')

    # generate
