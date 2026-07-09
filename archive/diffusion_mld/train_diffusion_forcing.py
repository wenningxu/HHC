from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler
from models.vqvae_sep import VAE_CNN
from models.mld_denoiser import DiffusionForcingTransformer, DenoiserTransformerFull
import options.option_diffusion as option_diffusion
from utils.losses import ReConsLossInter
from utils import eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM
from utils.word_vectorizer import WordVectorizer
from exit.utils import init_save_folder
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from exit.utils import generate_src_mask, lengths_to_mask

import torch.optim as optim
import lightning as L
import torch
from torch.cuda import amp
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import functools
import os
from dataclasses import dataclass, asdict
import copy
import clip

def create_gaussian_diffusion(args, enable_ddim=True):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.respacing if enable_ddim else ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

class LitInteraction(L.LightningModule):
    def __init__(self, args, vae, eval_wrapper):
        super().__init__()

        self.automatic_optimization = False
        self.args = args
        self.save_hyperparameters(args)
        self.test_time= int(0)
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        self.denoiser_model = DenoiserTransformerFull(
            **asdict(args.denoiser_args)
        )
        if args.ema_decay > 0:
            self.denoiser_model_avg = copy.deepcopy(self.denoiser_model)
            self.denoiser_model_avg.eval()
            for p in self.denoiser_model_avg.parameters():
                p.requires_grad = False


        self.diffusion = create_gaussian_diffusion(args, enable_ddim=False)
        self.ddim_diffusion = create_gaussian_diffusion(args, enable_ddim=True)
        self.clip_model = self.load_and_freeze_clip(args.clip_version, device=self.device)
        self.use_ddp = False
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        # fk_layer = vae.fk_layers
        self.Loss = torch.nn.HuberLoss(reduction='none', delta=1.0)
        self.normalizer = MotionNormalizerTorchInterX()
        # self.Loss_Inter = ReConsLossInter(args, args.recons_loss, args.nb_joints, fk_layer)

        self.eval_wrapper = eval_wrapper
         # tracking best eval metrics
        self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2, self.best_top3, self.best_matching = (
            1000,
            0,
            100,
            0,
            0,
            0,
            100,
        )

    def prepare_targets(self, motion):
        """
        将原始动作序列编码为 Latent 序列
        Input: motion [B, Total_Frames, D]
        Output: latent_gt [B, Seq_Len, D_latent]
        """
        # 1. 调整维度以适应 VAE (通常 VAE 期望 [B, D, T] 或 [B*L, D, Window])
        # 这里假设你的 VAE 可以处理变长序列，或者你需要在此处进行 reshape
        # 假设 VAE 将 4 帧压成 1 帧: Total_Frames -> Seq_Len

        # 这里的实现取决于你的 VAE 具体 forward 签名
        # 假设 encode 接受 [B, T, D] 输出 [B, L, D_latent]
        # 注意：不再传入 history_motion
        with torch.no_grad():
            latent_dist, _ = self.vae.encode(motion, scale_latent=self.args.rescale_latent)
            # 或者是 latent_dist.sample()，取决于你是否需要确定性目标
            latent_gt = latent_dist  # 假设返回就是 tensor [B, L, D_latent]

        return latent_gt

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        opt = self.optimizers()
        self.vae.eval()

        # 1. Unpack Batch
        # 数据集现在应该给出一整段长序列，而不是切好的 primitives
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch
        text_embedding = self.encode_text(caption)
        max_length = int(motion.shape[1] // 4)  # assuming VAE downsamples by factor of 4

        gt_motion = self.normalizer.backward(motion)

        # 3. VAE Encode -> 得到整个序列的“干净”Latent
        # Shape: [B, L, D_latent]
        # x_start = self.prepare_targets(motion)
        x_start = motion
        B, L, D_latent = x_start.shape

        m_tokens_len = torch.ceil((m_length)/4)
        contact_mask = contact.view(B, max_length, 4)
        contact_mask = (torch.sum(contact_mask, dim=-1) > 0).long()
        mask = generate_src_mask(L, m_tokens_len).to(x_start.device).unsqueeze(-1)

        # 2. 全局 Condition (Text/Style 等，不包含 pixel history)
        cond = {
            'betas': betas,
            'gender': gender,
            'text_embedding': text_embedding,
            'm_tokens_len': m_tokens_len,
            'contact_mask': contact_mask
        }
        # -----------------------------------------------------------
        # [Diffusion Forcing Core Logic]
        # -----------------------------------------------------------

        # 4. 采样独立噪声 (Independent Noise Sampling)
        # 关键修改：我们需要为序列中的 [每一个 Token] 生成一个独立的 timestep
        # Shape: [B, L]
        # t = torch.randint(
        #     0, self.diffusion.num_timesteps, (B, L),
        #     device=self.device
        # ).long()

        # 5. 前向加噪 (Forward Process / q_sample)

        # 注意：标准的 q_sample 通常接受 [B] 的 t。
        # 这里我们需要支持 [B, L] 的 t。
        # 如果你的 self.diffusion.q_sample 不支持广播，你需要手动写一下或修改 diffusion 库
        # 这里假设 q_sample 已经支持广播 或者我们手动实现简化版：
        t, weights = self.schedule_sampler.sample(self.args.batch_size, device=self.device)
        # 获取 alpha_bar [B, L, 1]
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))

        # 6. 模型预测 (Model Prediction)
        # 输入:
        #   - x_t: [B, L, D] (含有不同程度噪声的序列)
        #   - t:   [B, L] (告诉模型每一帧有多脏)
        #   - cond: global conditions
        #
        # [CRITICAL]: 你的 self.denoiser_model 必须是 Sequence Model (Transformer)
        # 并且必须在内部使用 Causal Mask (Triangular Mask)，确保第 i 帧只能看到 0...i 帧

        # 这里的 timesteps 需要由模型正确处理 (通常作为 Positional Embedding 的一部分加入)
        model_output = self.denoiser_model(x_t, timesteps=t, y=cond)

        # 7. 计算损失 (Loss Calculation)
        # 通常预测 noise 或者 x_start
        target = x_start

        # 基础 MSE Loss [B, L, D] -> [B, L]
        # 这里是对 Latent 进行 Loss 计算，符合 Diffusion Forcing 原文
        # print(model_output.shape, target.shape, mask.shape)
        raw_loss = self.seq_masked_mse(model_output, target, mask)

        # 8. Loss Reweighting (Optional but Recommended)
        # Diffusion Forcing 论文中提到对不同噪声水平加权
        # 如果没有特殊需求，可以使用标准的 Simple Diffusion Loss Weighting (通常全是1)
        loss_weights = 1.0
        # 如果需要类似 Min-SNR weighting，需根据 t 计算 weights

        loss = (raw_loss * loss_weights).mean()

        # 9. Logging & Backward
        opt.zero_grad()
        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.denoiser_model.parameters(), self.args.grad_clip)
        opt.step()

        # EMA & Scheduler (Standard)
        if args.ema_decay > 0:
            for param, avg_param in zip(self.denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                avg_param.data.mul_(args.ema_decay).add_(
                    param.data, alpha=1 - args.ema_decay)
        if sch is not None:
            sch.step()

        self.log("Training/loss", loss, prog_bar=True)
        return loss



    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        pass

    def test_step(self, batch, batch_idx):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch

        motion = motion.float()
        betas = betas.float()
        gender = gender
        motion_denorm = self.normalizer.backward(motion)
        m_tokens_len = torch.ceil((m_length) / 4)
        pred_len = m_length
        pred_tok_len = m_tokens_len

        num_joints = 21 if motion.shape[-1] == 251 else 22
        bs, seq = motion.shape[0], motion.shape[1]
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1]))
        texts_embedding = self.encode_text(caption).float()
        scale = torch.ones(1, device=self.device) * 3.5
        for i in range(bs):
            cond = {
                'text_embedding': texts_embedding[i:i + 1],
                'm_tokens_len': m_length[i:i + 1],
                'scale': scale,
            }
            pred_pose = self.ddim_diffusion.ddim_sample_loop(
                self.denoiser_model,
                (1, m_length[i], 56 * 12),
                clip_denoised=False,
                progress=False,
                model_kwargs={'y': cond},
            )
            pred_pose = self.normalizer.backward(pred_pose)
            gt_trans1, gt_pose1, gt_trans2, gt_pose2 = recover_interx(pred_pose[0, :m_length[0]])
            np.savez(f'test_{i}_p1',
                     poses=gt_pose1, trans=gt_trans1, betas=np.zeros(10), gender='male',
                     mocap_framerate=30)
            np.savez(f'test_{i}_p2',
                     poses=gt_pose2, trans=gt_trans2, betas=np.zeros(10), gender='male',
                     mocap_framerate=30)

    def on_validation_epoch_end(self) -> None:
        sample_fn = self.ddim_diffusion.ddim_sample_loop
        inference_model = self.denoiser_model_avg if self.args.ema_decay > 0 else self.denoiser_model
        self.fid, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2, self.best_top3, self.best_matching = eval_diffusion.evaluation_diffusion(
            self.args.out_dir, self.trainer.val_dataloaders, inference_model, self.vae, sample_fn, self.clip_model, self.global_step, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2,
            self.best_top3, self.best_matching, eval_wrapper=self.eval_wrapper, forcing=False, primitive=False, savenpy=False)
        self.log('Validation/FID', self.fid, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_FID', self.best_fid, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Iteration', self.best_iter, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Diversity', self.best_div, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Top1', self.best_top1, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Top2', self.best_top2, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Top3', self.best_top3, prog_bar=True, batch_size=self.args.batch_size)
        self.log('Validation/Best_Matching', self.best_matching, prog_bar=True, batch_size=self.args.batch_size)


    class CosineWarmupScheduler(optim.lr_scheduler.LRScheduler):
        def __init__(self, optimizer, warmup, max_iters):
            self.warmup = warmup
            self.max_num_iters = max_iters
            super().__init__(optimizer)

        def get_lr(self):
            lr_factor = self.get_lr_factor(epoch=self.last_epoch)
            return [base_lr * lr_factor for base_lr in self.base_lrs]

        def get_lr_factor(self, epoch):
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
            if epoch <= self.warmup:
                lr_factor *= epoch * 1.0 / self.warmup
            return lr_factor

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=list(self.denoiser_model.parameters()),
                                      lr=self.args.lr,
                                      weight_decay=self.args.weight_decay)
        lr_scheduler = self.CosineWarmupScheduler(optimizer=optimizer, warmup=self.args.warm_up_iter, max_iters=self.args.total_iter)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def load_and_freeze_clip(self, clip_version, device='cpu'):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = self.device
        texts = clip.tokenize(raw_text, truncate=True).to(
            device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"

    cond_mask_prob: float = 0.1
    """probability of masking the conditioning input"""

    clip_dim: int = 512
    noise_shape: tuple = (1, 56*12)


if __name__ == "__main__":
    args = option_diffusion.get_args_parser()
    denoiser_args = DenoiserTransformerArgs()
    args.denoiser_args = denoiser_args
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'test')  # /{args.exp_name}
    init_save_folder(args)
    # args.warm_up_iter = 5
    # args.total_iter = 1

    avae = VAE_CNN(args,  ## use args to define different parameters in different quantizers
                  args.output_emb_width,
                  args.down_t,
                  args.stride_t,
                  args.width,
                  args.depth,
                  args.dilation_growth_rate,
                  args.vq_act,
                  args.vq_norm, )
    args.resume_pth = 'output/vae-sep/net_100000_std_scale.pth'
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    state_dict = ckpt['net']
    if 'latent_mean' not in state_dict:
        state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in state_dict:
        state_dict['latent_std'] = torch.tensor(1)
    avae.load_state_dict(state_dict, strict=True)
    avae.latent_mean = state_dict['latent_mean']  # register buffer seems to be not loaded by load_state_dict
    avae.latent_std = state_dict['latent_std']
    avae.eval()

    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda:0'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

    train_file = 'dataset/interx/splits/overfit.txt'
    train_motion = 'dataset/interx/motions/all.h5'
    train_loader = dataset_interx_TM.DATALoader(args.dataname,
                                    args.batch_size,
                                    train_file,
                                    w_vectorizer,
                                    train_motion,
                                    unit_length=2**args.down_t)

    # train_loader_iter = dataset_interx.cycle(train_loader)

    val_file = 'dataset/interx/val.txt'
    val_motion = 'dataset/interx/motions/val.h5'
    val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                              32,
                                              train_file,
                                              w_vectorizer,
                                              train_motion,
                                              unit_length=4,
                                              shuffle=False)

    test_file = 'dataset/interx/test.txt'
    test_motion = 'dataset/interx/motions/test.h5'
    test_loader = dataset_interx_TM.DATALoader(args.dataname,
                                              32,
                                              test_file,
                                              w_vectorizer,
                                              test_motion,
                                              unit_length=4,
                                              shuffle=False)

    model = LitInteraction(args, avae, eval_wrapper)
    # model.denoiser_model = torch.compile(model.denoiser_model)

    wandb_logger = WandbLogger(
    project="NewInter",  # <-- 替换成你的项目名称
    config=args,  # 自动记录所有超参数
    group='mld_diffusion_forcing',
    name='mdm_3_test_overfit4data',
    id='mdm_3_test_overfit4data'  # (可选) 为这次运行命名，可以基于时间戳或配置
)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        filename='model-{step:02d}-{Training/loss:.2f}',
        save_top_k=-1,
        every_n_epochs=500,
        save_last=True,
    )

    trainer = L.Trainer(
        max_steps=args.total_iter,
        logger=wandb_logger,
        val_check_interval=args.eval_iter,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback],
        devices=[1],
        accelerator='gpu',
        precision="bf16-mixed",
    )

    # ckpt = torch.load('output/primitive_mld/2026-01-03-15-52-58_mld_primitive_1/net_last.pth')
    # model.denoiser_model_avg.load_state_dict(ckpt['net'], strict=True)

    # trainer.fit(model, train_dataloaders=train_loader)
    # trainer.validate(model, val_loader, ckpt_path='output/primitive_mld/2026-01-23-16-45-24_mld_diffusion_forcing_0/model-step=180000-Validation/FID=3.12.ckpt')
    trainer.test(model, train_loader, ckpt_path='output/test/2026-01-25-12-11-36_mdm_2_test_overfit4data/last.ckpt')