from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler
from models.vqtrans import AutoMldVae
from models.mld_denoiser import DenoiserTransformer, DenoiserTransformerFull
import options.option_diffusion as option_diffusion
from utils.losses import ReConsLossInter
from utils import eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_VQ, dataset_TM_eval, dataset_interx, dataset_interx_TM
from utils.word_vectorizer import WordVectorizer
from exit.utils import init_save_folder
from utils.motion_process import MotionNormalizerTorchInterX
from utils.rotation_conversions import *

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
        self.NUM_PRIMITIVE = args.num_primitive
        self.HISTORY_LENGTH = args.history_length
        self.FUTURE_LENGTH = args.future_length
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        self.denoiser_model = DenoiserTransformer(
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
        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
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

    def forward_backward(self, motion, cond, last_primitive, full_rollout=False):
        future_length = self.args.future_length
        history_length = self.args.history_length
        num_primitive = self.args.num_primitive

        future_motion_gt = motion[:, -future_length:, :]
        history_motion_gt = motion[:, :history_length, :]
        future_contact = cond['contact'][:, -future_length:]  # [B, F]
        cond['contact'] = future_contact  # only use future contact
        if last_primitive is not None:
            rollout_history = last_primitive[:, -history_length:, :]
            history_motion = rollout_history  # [B, H, D]
        else:
            history_motion = history_motion_gt
        latent_gt, _ = self.vae.encode(future_motion=future_motion_gt,
                                             history_motion=history_motion,
                                             scale_latent=args.rescale_latent)  # [T=1, B, D]
        # print('latent_gt:', latent_gt)
        # pdb.set_trace()
        # print(future_motion_gt.shape, history_motion.shape, latent_gt.shape)
        t, weights = self.schedule_sampler.sample(self.args.batch_size, device=self.device)  # weights always 1
        # print('t:', t, 'weights:', weights)

        # forward diffusion
        x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))

        cond['history_motion_normalized'] = history_motion
        x_start_pred = self.denoiser_model(x_t=x_t, timesteps=self.diffusion._scale_timesteps(t), y=cond)  # [B, T=1, D]
        # print('x_start_pred:', x_start_pred.shape)
        latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

        future_motion_pred = self.vae.decode(latent_pred, history_motion, nfuture=future_length,
                                                   beta=cond['betas'], gender=cond['gender'],
                                                   scale_latent=args.rescale_latent)  # [B, F, D], normalized

        loss_dict = self.calc_loss(motion, cond, history_motion, future_motion_gt, future_motion_pred, latent_gt,
                                   latent_pred, weights)

        if full_rollout:  # sample with full ddpm loop to get rollout history
            sample_fn = self.diffusion.p_sample_loop
            with torch.no_grad():
                x_start_pred = sample_fn(
                    self.denoiser_model,
                    x_start.shape,
                    clip_denoised=False,
                    model_kwargs={'y': cond},
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=x_start,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                # x_start_pred = sample_fn(
                #     self.denoiser_model,
                #     x_start.shape,
                #     clip_denoised=False,
                #     progress=False,
                #     model_kwargs={'y': cond},
                # )
                latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
                # if torch.isnan(latent_pred).any() or torch.isinf(latent_pred).any():
                #     print('latent_pred numerical error')
                #     pdb.set_trace()
                future_motion_pred = self.vae.decode(latent_pred, history_motion, nfuture=future_length,
                                                            beta=cond['betas'], gender=cond['gender'],
                                                               scale_latent=args.rescale_latent)  # [B, F, D], normalized

        return loss_dict, future_motion_pred

    def calc_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent_gt, latent_pred,
                  weights):
        terms = {}
        latent_rec_loss = self.rec_criterion(latent_pred, latent_gt)
        feature_rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)

        # complete_motion_gt = torch.cat([history_motion[:, -1:, :], future_motion_gt], dim=1)
        # terms = self.Loss_Inter.forward_denoiser(future_motion_pred, future_motion_gt)
        terms['latent_rec_loss'] = latent_rec_loss
        terms['feature_rec_loss'] = feature_rec_loss

        pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
        pred_motion_tensor = self.normalizer.backward(pred_motion_tensor)
        pred_rotation = pred_motion_tensor[:, :, :55*12]
        pred_trans = pred_motion_tensor[:, :, 55*12:]

        gt_motion_tensor = motion[:, -self.FUTURE_LENGTH-1:]
        gt_motion_tensor = self.normalizer.backward(gt_motion_tensor)
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

        velocity_loss = self.rec_criterion(pred_velocity, gt_velocity)
        rotation_loss = self.rec_criterion(pred_rotation_delta_6d, gt_rotation_delta_6d)
        terms['velocity_loss'] = velocity_loss
        terms['rotation_loss'] = rotation_loss

        loss = args.weight_latent_rec * latent_rec_loss + args.weight_feature_rec * feature_rec_loss + \
               args.weight_velocity * velocity_loss + args.weight_rotation * rotation_loss

        terms['loss'] = loss

        return terms


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        sch = self.lr_schedulers()

        gt_motion, betas, gender, group, text, contact = batch
        # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch
        text_embedding = self.encode_text(text)
        cond = {
            'betas': betas,
            'gender': gender,
            'text_embedding': text_embedding,
        }
        last_primitive = None
        avg_loss, avg_latent_loss, avg_feature_loss, avg_vel_loss, avg_rotation_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        opt = self.optimizers()
        opt.zero_grad()

        for primitive_idx in range(self.NUM_PRIMITIVE):
            # 从大张量中提取出当前基元的批次
            # motion_tensor shape: [B, primitive_length, D_motion]
            motion_tensor = gt_motion[:, primitive_idx, :, :]
            contact_tensor = contact[:, primitive_idx].long()
            # [B, F, D]
            cond['contact'] = contact_tensor

            if self.global_step > args.stage1_steps:
                rollout_prob = min(0.25, (self.global_step - args.stage1_steps) / max(
                    float(args.stage2_steps), 1e-6))
                if torch.rand(1).item() < rollout_prob:
                    loss_dict, future_motion_pred = self.forward_backward(motion_tensor, cond, last_primitive, full_rollout=True)
                    last_primitive = future_motion_pred.detach()  # assume future length >= history length
                else:
                    loss_dict, future_motion_pred = self.forward_backward(motion_tensor, cond, last_primitive)
                    last_primitive = None
            else:
                loss_dict, future_motion_pred = self.forward_backward(motion_tensor, cond, last_primitive)
                last_primitive = None
            loss = loss_dict['loss']

            self.manual_backward(loss / self.NUM_PRIMITIVE)

            avg_loss += loss.item()
            avg_latent_loss += loss_dict.get('latent_rec_loss', 0).item()
            avg_feature_loss += loss_dict.get('feature_rec_loss', 0).item()
            avg_vel_loss += loss_dict.get('velocity_loss', 0).item()
            avg_rotation_loss += loss_dict.get('rotation_loss', 0).item()

        torch.nn.utils.clip_grad_norm_(self.denoiser_model.parameters(), args.grad_clip)
        opt.step()

        if args.ema_decay > 0:
            for param, avg_param in zip(self.denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                avg_param.data.mul_(args.ema_decay).add_(
                    param.data, alpha=1 - args.ema_decay)

        avg_loss /= self.NUM_PRIMITIVE
        avg_latent_loss /= self.NUM_PRIMITIVE
        avg_feature_loss /= self.NUM_PRIMITIVE
        avg_vel_loss /= self.NUM_PRIMITIVE
        avg_rotation_loss /= self.NUM_PRIMITIVE

        if sch is not None:
            sch.step()

        self.log("Training/loss", avg_loss, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Training/latent_rec_loss", avg_latent_loss, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Training/feature_rec_loss", avg_feature_loss, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Training/velocity_loss", avg_vel_loss, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Training/rotation_loss", avg_rotation_loss, prog_bar=True, batch_size=self.args.batch_size)

        return torch.tensor(avg_loss)  # 返回整个序列的平均 loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        pass

    def on_validation_epoch_end(self) -> None:
        sample_fn = self.ddim_diffusion.ddim_sample_loop
        self.fid, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2, self.best_top3, self.best_matching = eval_trans.evaluation_diffusion(
            'results/mld_contact_0', self.trainer.val_dataloaders, self.denoiser_model_avg, self.vae, sample_fn, self.clip_model, self.global_step, self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2,
            self.best_top3, self.best_matching, eval_wrapper=self.eval_wrapper, primitive=True, savenpy=True)
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
    history_shape: tuple = (2, 56*12)
    noise_shape: tuple = (1, 256)


if __name__ == "__main__":
    args = option_diffusion.get_args_parser()
    denoiser_args = DenoiserTransformerArgs()
    args.denoiser_args = denoiser_args
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'test')  # /{args.exp_name}
    init_save_folder(args)
    # args.warm_up_iter = 5
    # args.total_iter = 1

    avae = AutoMldVae(56 * 12)
    args.resume_pth = 'output/primitive_vae/net_100000_std_scale.pth'
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

    train_file = 'dataset/interx/splits/train.txt'
    train_motion = 'dataset/interx/motions/train.h5'
    train_loader = dataset_interx.DATALoaderPrimitive(args.dataname,
                                                      args.batch_size,
                                                      train_file,
                                                      train_motion,
                                                      8,
                                                      args.num_primitive,
                                                      args.history_length,
                                                      args.future_length,
                                                      unit_length=4)

    # train_loader = dataset_interx_TM.DATALoader(args.dataname,
    #                                           args.batch_size,
    #                                           train_file,
    #                                           w_vectorizer,
    #                                           train_motion,
    #                                           unit_length=4,
    #                                           shuffle=False)

    val_file = 'dataset/interx/val.txt'
    val_motion = 'dataset/interx/motions/val.h5'
    val_loader = dataset_interx_TM.DATALoader(args.dataname,
                                              32,
                                              val_file,
                                              w_vectorizer,
                                              val_motion,
                                              unit_length=4,
                                              shuffle=False)

    test_file = 'dataset/interx/splits/test.txt'
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
    name='mld_contact_0',
    id='mld_contact_0'  # (可选) 为这次运行命名，可以基于时间戳或配置
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
    #     filename='model-{step:02d}-{Training/loss:.2f}',
    #     save_top_k=-1,
    #     every_n_epochs=500,
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
    # trainer.validate(model, test_loader, ckpt_path='output/primitive_mld/2026-01-13-04-13-25_mld_primitive_7_ddpm10/model-step=220000-Validation/FID=0.27.ckpt')