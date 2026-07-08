from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler
from models.vqtrans import AutoMldVae
from models.vqvae_sep import VAE_CNN, VAE_SHAPE_CNN
from models.mld_denoiser import DenoiserTransformer, DenoiserTransformerFull, DiffusionForcingTransformer
import options.option_diffusion as option_diffusion
from utils.losses import ReConsLossInter
from utils import eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM
from utils.word_vectorizer import WordVectorizer
from exit.utils import init_save_folder
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from utils.rotation_conversions import *
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
        self.vae.eval()
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

    def seq_masked_mse(self, prediction, target, mask, weights=None):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        batch_size, num_frames, *_ = xs.shape
        noise_levels = torch.randint(0, self.diffusion.num_timesteps, (batch_size, num_frames), device=xs.device)

        if masks is not None:
            # for frames that are not available, treat as full noise
            noise_levels = torch.where(masks, noise_levels, torch.full_like(noise_levels, self.diffusion.num_timesteps - 1))

        # first frame is almost always known, this reflect that
        if np.random.rand() < 0.5:
            noise_levels[:, 0] = torch.randint(0, self.diffusion.num_timesteps // 4, (batch_size,), device=xs.device)

        return noise_levels

    def calc_loss(self, motion_pred, motion_gt, latent_gt, latent_pred, mask, seq_mask, weights):
        terms = {}
        loss_latent = self.seq_masked_mse(latent_pred, latent_gt, mask.unsqueeze(-1), weights)
        loss_recons = self.seq_masked_mse(motion_pred, motion_gt, seq_mask.unsqueeze(-1))

        # complete_motion_gt = torch.cat([history_motion[:, -1:, :], future_motion_gt], dim=1)
        # terms = self.Loss_Inter.forward_denoiser(future_motion_pred, future_motion_gt)
        terms['loss_latent'] = loss_latent
        terms['loss_recons'] = loss_recons

        pred_motion_tensor = self.normalizer.backward(motion_pred)
        pred_rotation = pred_motion_tensor[:, :, :55*12]
        pred_trans = pred_motion_tensor[:, :, 55*12:]

        gt_motion_tensor = self.normalizer.backward(motion_gt)
        gt_rotation = gt_motion_tensor[:, :, :55*12]
        gt_trans = gt_motion_tensor[:, :, 55*12:]

        pred_velocity = pred_trans[:, 1:] - pred_trans[:, :-1]
        gt_velocity = gt_trans[:, 1:] - gt_trans[:, :-1]

        B, T, D = pred_rotation.shape
        pred_rotation_matrix = rotation_6d_to_matrix(
            pred_rotation.view(B, T, 55, 2, 6)).view(B, T, 55, 2, 3, 3)
        gt_rotation_matrix = rotation_6d_to_matrix(
            gt_rotation.view(B, T, 55, 2, 6)).view(B, T, 55, 2, 3, 3)
        pred_rotation_delta = torch.matmul(
            pred_rotation_matrix[:, 1:], pred_rotation_matrix[:, :-1].transpose(-1, -2)
        )  # [B, T-1, J, 2, 3, 3]
        gt_rotation_delta = torch.matmul(
            gt_rotation_matrix[:, 1:], gt_rotation_matrix[:, :-1].transpose(-1, -2)
        )  # [B, T-1, J, 2, 3, 3]
        pred_rotation_delta_6d = matrix_to_rotation_6d(
            pred_rotation_delta.contiguous()
        ).view(pred_rotation_delta.shape[0], pred_rotation_delta.shape[1], 55*12)
        gt_rotation_delta_6d = matrix_to_rotation_6d(
            gt_rotation_delta.contiguous()
        ).view(gt_rotation_delta.shape[0], gt_rotation_delta.shape[1], 55*12)

        velocity_loss = self.seq_masked_mse(pred_velocity, gt_velocity, seq_mask[:, 1:].unsqueeze(-1))
        rotation_loss = self.seq_masked_mse(pred_rotation_delta_6d, gt_rotation_delta_6d, seq_mask[:, 1:].unsqueeze(-1))
        terms['velocity_loss'] = velocity_loss
        terms['rotation_loss'] = rotation_loss

        loss = args.weight_latent_rec * loss_latent + args.weight_feature_rec * loss_recons + \
               args.weight_velocity * velocity_loss + args.weight_rotation * rotation_loss

        terms['loss'] = loss

        return terms

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        sch = self.lr_schedulers()

        # gt_motion, betas, gender, group, text = batch
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch
        latent_dist, _ = self.vae.encode(motion, betas, gender, scale_latent=self.args.rescale_latent)
        # 或者是 latent_dist.sample()，取决于你是否需要确定性目标
        latent_gt = latent_dist  # 假设返回就是 tensor [B, L, D_latent]
        B, L, D_latent = latent_gt.shape

        m_tokens_len = torch.ceil((m_length) / 4)
        contact = contact.reshape(B, -1, 4)
        contact = (torch.sum(contact, dim=-1) > 0).long()

        text_embedding = self.encode_text(caption)
        cond = {
            'contact' : contact,
            'text_embedding': text_embedding,
        }

        opt = self.optimizers()

        _, L_motion, D_motion = motion.shape

        mask = generate_src_mask(L, m_tokens_len).to(latent_gt.device)
        mask_motion = generate_src_mask(L_motion, m_length).to(latent_gt.device)
        t = self._generate_noise_levels(latent_gt, mask).long()  # [B,]
        weights = self.diffusion.compute_loss_weights(t)

        # t, weights = self.schedule_sampler.sample(self.args.batch_size, device=self.device)  # weights always 1
        # print('t:', t, 'weights:', weights)

        # forward diffusion
        x_start = latent_gt  # [B, T=1, D]
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))
        x_start_pred = self.denoiser_model(x_t=x_t, timesteps=self.diffusion._scale_timesteps(t), y=cond, is_causal=True)
        # print('x_start_pred:', x_start_pred.shape, 'x_start:', x_start.shape)

        motion_pred = self.vae.decode(x_start_pred, beta=betas, gender=gender,
                                          scale_latent=True)  # [B, F, D], normalized
        loss_terms = self.calc_loss(motion_pred, motion, latent_gt, x_start_pred, mask, mask_motion, weights)

        opt.zero_grad()
        self.manual_backward(loss_terms['loss'])

        torch.nn.utils.clip_grad_norm_(self.denoiser_model.parameters(), args.grad_clip)
        opt.step()

        if args.ema_decay > 0:
            for param, avg_param in zip(self.denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                avg_param.data.mul_(args.ema_decay).add_(
                    param.data, alpha=1 - args.ema_decay)


        if sch is not None:
            sch.step()

        self.log("Training/loss", loss_terms['loss'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/loss_latent', loss_terms['loss_latent'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/loss_recons', loss_terms['loss_recons'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/velocity_loss', loss_terms['velocity_loss'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/rotation_loss', loss_terms['rotation_loss'], prog_bar=True, batch_size=self.args.batch_size)

        return loss_terms['loss']

    def test_step(self, batch, batch_idx):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch

        motion = motion.float()
        betas = betas.float()
        gender = gender
        motion_denorm = self.normalizer.backward(motion)
        m_tokens_len = torch.ceil((m_length) / 4)
        pred_len = m_length
        pred_tok_len = m_tokens_len

        bs, seq = motion.shape[0], motion.shape[1]
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1]))
        texts_embedding = self.encode_text(caption).float()
        scale = torch.ones(1, device=self.device) * 3.5
        for i in range(bs):
            cond = {
                'text_embedding': texts_embedding[i:i + 1],
                'm_tokens_len': m_tokens_len[i:i + 1],
                'scale': scale,
            }

            pred_latent_future = self.ddim_diffusion.diffusion_forcing_ddim_sample_loop(
                self.denoiser_model,
                (1, m_tokens_len[i], 512),
                future_length=int(m_tokens_len[i]),
                clip_denoised=False,
                progress=False,
                model_kwargs={'y': cond},
                uncertainty_scale=2.0,
            )
            pred_pose = self.vae.decode(pred_latent_future, beta=betas[i:i + 1], gender=gender[i:i + 1],
                                          lengths=m_tokens_len[i:i + 1],
                                          scale_latent=True)  # [B, F, D], normalized
            pred_pose = self.normalizer.backward(pred_pose)
            gt_trans1, gt_pose1, gt_trans2, gt_pose2 = recover_interx(pred_pose[:, :m_length[i]])
            print(gt_pose1.shape)
            np.savez(f'test_{i}_p1',
                     poses=gt_pose1, trans=gt_trans1, betas=np.zeros(10), gender='male',
                     mocap_framerate=30)
            np.savez(f'test_{i}_p2',
                     poses=gt_pose2, trans=gt_trans2, betas=np.zeros(10), gender='male',
                     mocap_framerate=30)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch
        latent_dist, _ = self.vae.encode(motion, betas, gender, scale_latent=self.args.rescale_latent)
        # 或者是 latent_dist.sample()，取决于你是否需要确定性目标
        latent_gt = latent_dist  # 假设返回就是 tensor [B, L, D_latent]
        B, L, D_latent = latent_gt.shape

        m_tokens_len = torch.ceil((m_length) / 4)
        contact = contact.reshape(B, -1, 4)
        contact = (torch.sum(contact, dim=-1) > 0).long()

        text_embedding = self.encode_text(caption)
        cond = {
            'contact' : contact,
            'text_embedding': text_embedding,
        }

        _, L_motion, D_motion = motion.shape

        mask = generate_src_mask(L, m_tokens_len).to(latent_gt.device)
        mask_motion = generate_src_mask(L_motion, m_length).to(latent_gt.device)
        t = self._generate_noise_levels(latent_gt, mask).long()  # [B,]
        weights = self.diffusion.compute_loss_weights(t)

        # forward diffusion
        x_start = latent_gt  # [B, T=1, D]
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))
        x_start_pred = self.denoiser_model(x_t=x_t, timesteps=self.diffusion._scale_timesteps(t), y=cond, is_causal=True)

        motion_pred = self.vae.decode(x_start_pred, beta=betas, gender=gender,
                                          scale_latent=True)  # [B, F, D], normalized
        loss_terms = self.calc_loss(motion_pred, motion, latent_gt, x_start_pred, mask, mask_motion, weights)

        self.log("Training/loss", loss_terms['loss'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/loss_latent', loss_terms['loss_latent'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/loss_recons', loss_terms['loss_recons'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/velocity_loss', loss_terms['velocity_loss'], prog_bar=True, batch_size=self.args.batch_size)
        self.log('Training/rotation_loss', loss_terms['rotation_loss'], prog_bar=True, batch_size=self.args.batch_size)

    def on_validation_epoch_end(self) -> None:
        sample_fn = self.ddim_diffusion.diffusion_forcing_ddim_sample_loop
        self.fid, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2, self.best_top3, self.best_matching = eval_trans.evaluation_diffusion(
            'results/training_df', self.trainer.val_dataloaders, self.denoiser_model_avg, self.vae, sample_fn, self.clip_model, self.global_step, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2,
            self.best_top3, self.best_matching, eval_wrapper=self.eval_wrapper, primitive=False, forcing=True, savenpy=True)
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
    h_dim: int = 768
    ff_size: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"

    cond_mask_prob: float = 0.1
    contact: bool = False
    """probability of masking the conditioning input"""

    clip_dim: int = 512
    noise_shape: tuple = (1, 256)


if __name__ == "__main__":
    args = option_diffusion.get_args_parser()
    denoiser_args = DenoiserTransformerArgs()
    args.denoiser_args = denoiser_args
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'diffusion_forcing')  # /{args.exp_name}
    init_save_folder(args)
    # args.warm_up_iter = 5
    # args.total_iter = 1

    avae = VAE_SHAPE_CNN(args,  ## use args to define different parameters in different quantizers
                  args.code_dim,
                  args.down_t,
                  args.stride_t,
                  args.width,
                  args.depth,
                  args.dilation_growth_rate,
                  args.vq_act,
                  args.vq_norm, )
    args.resume_pth = 'output/vae-sep/vae_shape_100000_std_scale.pth'
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

    overfit_file = 'dataset/interx/splits/overfit.txt'

    train_file = 'dataset/interx/splits/train.txt'
    train_motion = 'dataset/interx/motions/train.h5'

    train_loader = dataset_interx_TM.DATALoader(args.dataname,
                                              args.batch_size,
                                              train_file,
                                              w_vectorizer,
                                              train_motion,
                                              unit_length=4,
                                              shuffle=True)

    val_file = 'dataset/interx/val.txt'
    val_motion = 'dataset/interx/motions/val.h5'
    val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                              32,
                                              overfit_file,
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
    group='mld_primitive',
    name='df_shape_1_snr',
    id='df_shape_1_snr'  # (可选) 为这次运行命名，可以基于时间戳或配置
)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        monitor='Validation/FID',
        filename='model-{step:02d}-{Validation/FID:.2f}',
        save_top_k=10,
        mode='min',
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.out_dir,
    #     save_top_k=-1,
    #     save_last=True,
    # )

    trainer = L.Trainer(
        max_steps=args.total_iter,
        logger=wandb_logger,
        val_check_interval=args.eval_iter,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback],
        devices=[0],
        accelerator='gpu',
        precision="bf16-mixed"
    )

    # ckpt = torch.load('output/primitive_mld/2026-01-03-15-52-58_mld_primitive_1/net_last.pth')
    # model.denoiser_model_avg.load_state_dict(ckpt['net'], strict=True)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    # trainer.validate(model, test_loader, ckpt_path='output/diffusion_forcing/2026-01-26-19-15-19_diffusion_forcing_2/model-step=140000-Validation/FID=1.60.ckpt')
    # trainer.test(model, train_loader, ckpt_path='output/diffusion_forcing/2026-01-26-16-27-05_diffusion_forcing_test/last.ckpt')