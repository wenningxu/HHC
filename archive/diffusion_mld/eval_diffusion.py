import os

import clip
import numpy as np
import torch
from tqdm import tqdm

from archive.diffusion_mld.cfg_sampler import ClassifierFreeSampleModel
from utils.eval_trans import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_R_precision,
)
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx

@torch.no_grad()
def evaluation_diffusion(out_dir, val_loader, net, vae, sample_func, clip_model, nb_iter, best_fid, best_iter, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, forcing=False, save=True, primitive=False,
                     savenpy=False):
    net.eval()
    device = next(vae.parameters()).device
    cfg_net = ClassifierFreeSampleModel(net)
    normalizer = MotionNormalizerTorchInterX()
    history_len = 3
    future_len = 12
    chunk_size = 8
    primitive_len = history_len + future_len

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    # print('len' , len(val_loader))
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, contact = batch

        motion = motion.to(device).float()
        betas = betas.to(device).float()
        gender = gender.to(device)
        contact = contact.to(device).long()
        motion_denorm = normalizer.backward(motion)
        m_tokens_len = torch.ceil((m_length) / 4)
        bs, seq = motion.shape[0], motion.shape[1]
        contact = contact.reshape(bs, -1, 4)
        contact = (torch.sum(contact, dim=-1) > 0).long()

        num_joints = 21 if motion.shape[-1] == 251 else 22

        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(device)
        texts = clip.tokenize(caption, truncate=True).to(
            device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        texts_embedding = clip_model.encode_text(texts).float()

        for i in range(bs):
            if primitive:
                frame_idx = 0
                pred_pose_full = []
                current_history = motion[i:i + 1, :history_len]
                pred_pose_full.append(current_history)

                scale = torch.ones(1, device=device) * 3.5
                cond = {
                    'text_embedding': texts_embedding[i:i + 1],
                    'scale': scale,
                }
                while frame_idx <= m_length[i] - 15 - 1:
                    cond['history_motion_normalized'] = current_history
                    cond['contact'] = contact[i:i + 1, frame_idx:frame_idx + future_len]
                    pred_latent_future = sample_func(
                        cfg_net,
                        (1, 1, 256),
                        clip_denoised=False,
                        progress=False,
                        model_kwargs={'y': cond},
                    )
                    pred_latent_future = pred_latent_future.permute(1, 0, 2)
                    pred_pose_future = vae.decode(pred_latent_future, current_history, nfuture=future_len,
                                                            beta=betas[i:i + 1], gender=gender[i:i + 1],
                                                               scale_latent=True)  # [B, F, D], normalized
                    pred_pose_full.append(pred_pose_future)
                    current_history = pred_pose_future[:, -history_len:]
                    frame_idx += future_len
                pred_pose_full = torch.cat(pred_pose_full, dim=1)
                pred_pose_full = normalizer.backward(pred_pose_full)
                pred_pose_eval[i:i + 1, :pred_pose_full.shape[1], :] = pred_pose_full
                m_length[i] = pred_pose_full.shape[1]
                pred_pose = pred_pose_full
            elif forcing:
                texts = clip.tokenize(caption, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
                texts_embedding = clip_model.encode_text(texts).float()
                scale = torch.ones(1, device=device) * 3.5

                cond = {
                    'text_embedding': texts_embedding[i:i + 1],
                    'contact': contact[i:i+1, :int(m_tokens_len[i])],
                    'scale': scale,
                }

                pred_latent_future = sample_func(
                    cfg_net,
                    (1, m_tokens_len[i], 256),
                    future_length=int(m_tokens_len[i]),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs={'y': cond},
                    uncertainty_scale=2.0,
                )
                pred_pose_future = vae.decode(pred_latent_future, beta=betas[i:i + 1], gender=gender[i:i + 1],
                                              scale_latent=True)  # [B, F, D], normalized
                pred_pose_future = normalizer.backward(pred_pose_future)
                pred_pose_eval[i:i + 1, :m_length[i], :] = pred_pose_future[:, :m_length[i]]
                pred_pose = pred_pose_future
            else:
                texts = clip.tokenize(caption, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
                texts_embedding = clip_model.encode_text(texts).float()
                scale = torch.ones(1, device=device) * 3.5
                cond = {
                    'text_embedding': texts_embedding[i:i + 1],
                    'm_tokens_len': m_length[i:i+1],
                    'scale': scale,
                }
                pred_pose = sample_func(
                    cfg_net,
                    (1, m_length[i], 56*12),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs={'y': cond},

                )
                # pred_pose_future = vae.decode(pred_pose, beta=betas[i:i + 1], gender=gender[i:i + 1],
                #                               lengths=m_tokens_len[i:i + 1],
                #                               scale_latent=True)  # [B, F, D], normalized
                pred_pose_future = normalizer.backward(pred_pose)
                pred_pose_eval[i:i + 1, :m_length[i], :] = pred_pose_future[:, :m_length[i]]
                pred_pose = pred_pose_future

            if savenpy:
                gt_trans1, gt_pose1, gt_trans2, gt_pose2 = recover_interx(motion_denorm[i, :m_length[i]])
                trans1, pose1, trans2, pose2 = recover_interx(pred_pose.squeeze())
                gender_word = ['male' if g == 0 else 'female' for g in gender[i]]
                np.savez(os.path.join(out_dir, name[i] + '_gt_p1.npz'),
                         poses=gt_pose1, trans=gt_trans1, betas=betas[i, 0:10].cpu().detach().numpy(),
                         gender=gender_word[0], mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i] + '_gt_p2.npz'),
                         poses=gt_pose2, trans=gt_trans2, betas=betas[i, 10:].cpu().detach().numpy(),
                         gender=gender_word[1], mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i] + '_pred_p1.npz'),
                         poses=pose1, trans=trans1, betas=betas[i, 0:10].cpu().detach().numpy(), gender=gender_word[0],
                         mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i] + '_pred_p2.npz'),
                         poses=pose2, trans=trans2, betas=betas[i, 10:].cpu().detach().numpy(), gender=gender_word[1],
                         mocap_framerate=30)
                # np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                # np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())
        motion_denorm[..., -9:-6] = 0
        motion_denorm[..., -3:] = 0
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_denorm, m_length)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 30)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 30)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

        # if nb_iter % 5000 == 0 :
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)

        # if nb_iter % 5000 == 0 :
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching
