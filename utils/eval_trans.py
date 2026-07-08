import os
import json

import clip
import numpy as np
import torch
from scipy import linalg

# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric, MotionNormalizerTorchInterX, recover_interx, MotionNormalizerTorchInterXSingle, recover_interx_from_features, prepare_interx
from exit.utils import get_model, visualize_2motions, generate_src_mask
from tqdm import tqdm
from models.cfg_sampler import ClassifierFreeSampleModel

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = False, save = False, primitive=False, savenpy=False, continuous=False, shape_change=False, single=False, vel=True) :
    net.eval()
    device = next(net.parameters()).device
    nb_sample = 0
    normalizer = MotionNormalizerTorchInterXSingle() if single else MotionNormalizerTorchInterX()
    gt_normalizer = MotionNormalizerTorchInterX('raw')
    draw_org = []
    draw_pred = []
    draw_text = []
    history_len = 3
    future_len = 12
    primitive_len = history_len + future_len

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    # print('len' , len(val_loader))
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, betas, gender, _ = batch

        B, T, D = motion.shape
        motion = motion.to(device).float()
        betas = betas.to(device).float()
        gender = gender.to(device)
        if shape_change:
            betas = betas.view(betas.shape[0], 2, -1)
            # 把他们的beta， gender变成相同的，随机选择一个人的beta
            random_idx = torch.randint(0,2,(1,)).item()
            betas = betas[:, random_idx, :].unsqueeze(1).repeat(1,2,1).view(betas.shape[0], -1)
            gender[:, 1-random_idx] = gender[:, random_idx]


        motion_denorm = gt_normalizer.backward(motion)
        if single:
            motion_denorm = motion_denorm.reshape(B, T, 56, 2, 6)
            motion_p1 = normalizer.forward(motion_denorm[..., 0, :].reshape(B, T, 56*6))
            motion_p2 = normalizer.forward(motion_denorm[..., 1, :].reshape(B, T, 56*6))
            motion_p1_single_denorm = normalizer.backward(motion_p1).reshape(B, T, 56, 6)
            motion_p2_single_denorm = normalizer.backward(motion_p2).reshape(B, T, 56, 6)
            motion_single_denorm = torch.cat([motion_p1_single_denorm, motion_p2_single_denorm], dim=-1).reshape(B, T, -1)

        motion_denorm = motion_denorm.reshape(B, T, -1)

        num_joints = 21 if motion.shape[-1] == 251 else 22
        bs, seq = motion.shape[0], motion.shape[1]
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(device)

        for i in range(bs):
            if primitive:
                frame_idx = 0
                pred_pose_full = []
                current_history = motion[i:i + 1, :history_len]
                pred_pose_full.append(current_history)
                while frame_idx <= m_length[i] - 15 - 1:
                    motion_primitive = motion[i:i + 1, frame_idx:frame_idx + primitive_len]
                    current_future = motion_primitive[:, -future_len:]
                    latent, dist = net.encode(current_future, current_history)
                    pred_pose_future = net.decode(latent, current_history, future_len, betas[i:i + 1], gender[i:i + 1])
                    pred_pose_full.append(pred_pose_future)
                    current_history = pred_pose_future[:, -history_len:]
                    frame_idx += future_len
                pred_pose_full = torch.cat(pred_pose_full, dim=1)
                pred_pose_full = normalizer.backward(pred_pose_full)
                pred_pose_eval[i:i + 1, :pred_pose_full.shape[1], :] = pred_pose_full
                m_length[i] = pred_pose_full.shape[1]
                pred_pose = pred_pose_full
            elif continuous:
                pred_pose, z_parts, dists = net(motion[i:i+1, :m_length[i]], betas[i:i+1], gender[i:i+1])
                pred_pose = normalizer.backward(pred_pose)
                # pred_pose[..., -9:-6] = 0
                # pred_pose[..., -3:] = 0
                pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose
            elif single:
                pred_pose_p1, _, _ = net(motion_p1[i:i + 1, :m_length[i]], betas[i:i + 1, :10], gender[i:i + 1, 0])
                pred_pose_p2, _, _ = net(motion_p2[i:i + 1, :m_length[i]], betas[i:i + 1, 10:], gender[i:i + 1, 1])
                pred_pose_p1 = normalizer.backward(pred_pose_p1).reshape(1, m_length[i], 56, 6)
                pred_pose_p2 = normalizer.backward(pred_pose_p2).reshape(1, m_length[i], 56, 6)
                pred_pose = torch.cat([pred_pose_p1, pred_pose_p2], dim=-1).reshape(1, m_length[i], -1)
                pred_pose_eval[i:i + 1, :m_length[i], :] = pred_pose[:, :m_length[i]]
            else:
                if vel:
                    motion_input = prepare_interx(motion_denorm[i, :m_length[i]]).unsqueeze(0)
                    motion_input = normalizer.forward(motion_input)
                else:
                    motion_input = motion[i:i+1, :m_length[i]]
                pred_pose, loss_commit, perplexity = net(motion_input, betas[i:i+1], gender[i:i+1])
                pred_pose = normalizer.backward(pred_pose)
                if vel:
                    pred_pose = recover_interx_from_features(pred_pose.squeeze()).unsqueeze(0)
                pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose[:, :m_length[i]]

            if savenpy:
                if single:
                    gt_trans1, gt_pose1, gt_trans2, gt_pose2 = recover_interx(motion_single_denorm[i, :m_length[i]])
                else:
                    gt_trans1, gt_pose1, gt_trans2, gt_pose2 = recover_interx(motion_denorm[i, :m_length[i]])
                trans1, pose1, trans2, pose2 = recover_interx(pred_pose.squeeze())
                gender_word = ['male' if g == 0 else'female' for g in gender[i]]
                np.savez(os.path.join(out_dir, name[i]+'_gt_p1.npz'),
                             poses=gt_pose1, trans=gt_trans1, betas=betas[i,0:10].cpu().detach().numpy(), gender=gender_word[0], mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i]+'_gt_p2.npz'),
                             poses=gt_pose2, trans=gt_trans2, betas=betas[i,10:].cpu().detach().numpy(), gender=gender_word[1], mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i] + '_pred_p1.npz'),
                         poses=pose1, trans=trans1, betas=betas[i, 0:10].cpu().detach().numpy(), gender=gender_word[0], mocap_framerate=30)
                np.savez(os.path.join(out_dir, name[i]+'_pred_p2.npz'),
                             poses=pose2, trans=trans2, betas=betas[i,10:].cpu().detach().numpy(), gender=gender_word[1], mocap_framerate=30)
                # np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                # np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_denorm, m_length)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

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
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 30)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 30)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


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


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, dataname='t2m', draw = False, save = False, savegif=False, num_repeat=1, rand_pos=False, CFG=-1, savenpy=False, contact=False, shape_change=None, save_embeddings=False) :
    if num_repeat < 0:
        is_avg_all = True
        num_repeat = -num_repeat
    else:
        is_avg_all = False
    normalizer = MotionNormalizerTorchInterX('raw')
    device = next(net.parameters()).device
    max_length = trans.block_size - 1

    vocab_contact_map = np.load('checkpoints/vq/code_contact_probs.npy')
    vocab_contact_map = torch.from_numpy(vocab_contact_map).float().to(device)
    motions_dir = os.path.join(out_dir, 'motions')
    embeddings_dir = os.path.join(out_dir, 'embeddings')
    manifest_path = os.path.join(out_dir, 'manifest.jsonl')
    manifest_file = None
    if savenpy:
        os.makedirs(motions_dir, exist_ok=True)
        if save_embeddings:
            os.makedirs(embeddings_dir, exist_ok=True)
        manifest_file = open(manifest_path, 'a', encoding='utf-8')

    if shape_change is not None:
        betas_group = np.load(f'dataset/interx/betas_group/{shape_change}.npz', allow_pickle=True)
        gender_group = betas_group['gender']
        betas_group = torch.from_numpy(betas_group['betas']).float().to(device).reshape(-1)
        gender_group = [0 if m == 'male' else 1 for m in gender_group]
        gender_group = torch.tensor(gender_group).long().to(device)

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    fixed_orders = []

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, betas, gender, contact_mask = batch

        bs, seq = pose.shape[:2]
        pose = pose.to(device).float()
        betas = betas.to(device).float()
        gender = gender.to(device)
        contact_mask_raw = contact_mask.clone()

        if shape_change is not None:
            betas[:] = betas_group
            gender[:] = gender_group

        
        text = clip.tokenize(clip_text, truncate=True).to(device)

        feat_clip_text, word_emb = clip_model(text)
        
        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length)/4)
        # contact_mask = torch.cummax(contact_mask, dim=1)[0]
        if contact:
            contact_mask = contact_mask.view(bs, -1, 4)
            contact_mask = (torch.sum(contact_mask, dim=-1) > 0).long()
            if contact_mask.shape[1] < max_length:
                contact_mask = torch.cat([
                    contact_mask,
                    torch.zeros((bs, max_length-contact_mask.shape[1]), dtype=contact_mask.dtype),
                ], dim=1).to(device)
            else:
                contact_mask = contact_mask[:, :max_length]

         
        pred_len = m_length.to(device)
        pred_tok_len = m_tokens_len
        new_motion_lengths = torch.zeros_like(m_length)

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(device)
            # pred_len = torch.ones(bs).long()
            #print(next(trans.parameters()).device)
            #print(feat_clip_text.device, word_emb.device, pred_len.device)
            if contact:
                index_motion = trans(feat_clip_text, word_emb, type="sample", m_length=pred_len,
                                     rand_pos=rand_pos, betas=betas, gender=gender, CFG=CFG, contact_mask=contact_mask, vocab_contact_map=vocab_contact_map)
            else:
                index_motion = trans(feat_clip_text, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos, betas=betas, gender=gender, CFG=CFG)
            # last_attn = trans.trans_base.cross_att[-1].attn.last_attn
            # np.save('test_cross_attn.npy', last_attn.cpu().numpy())
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()[:, 0]
            # fixed_orders.append(fixed_order.cpu().numpy())
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):

            ######### [INFO] Eval only the predicted length
                # print(pred_length)
                #if pred_length[k] == 0:
                #    pred_len[k] = seq
                #    continue
                #pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], betas[k:k+1], gender[k:k+1], type='decode')
                #pred_pose = normalizer.backward(pred_pose)
                #cur_len = pred_pose.shape[1]

                #pred_len[k] = min(cur_len, seq)
                #pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            ######################################################
            
            ######### [INFO] Eval by m_length
                pred_pose = net(index_motion[k:k+1, :int(pred_tok_len[k].item())], betas[k:k+1], gender[k:k+1], type='decode')
                pred_pose = normalizer.backward(pred_pose)
                motion_length = int(pred_len[k].item())
                pred_pose_eval[k:k+1, :motion_length] = pred_pose[:, :motion_length]
                new_motion_lengths[k] = motion_length
                if savenpy:
                    trans1, pose1, trans2, pose2 = recover_interx(pred_pose.squeeze())
                    gender_word = ['male' if g == 0 else'female' for g in gender[k]]
                    if shape_change is not None:
                        endfix = f'_G{shape_change}'
                    else:
                        endfix = ''
                    sample_key = f"{name[k]}{endfix}_r{i:03d}"
                    motion_p1_path = os.path.join(motions_dir, sample_key + '_pred_p1.npz')
                    motion_p2_path = os.path.join(motions_dir, sample_key + '_pred_p2.npz')
                    np.savez(motion_p1_path,
                             poses=pose1, trans=trans1, betas=betas[k, 0:10].cpu().detach().numpy(), gender=gender_word[0], mocap_framerate=30)
                    np.savez(motion_p2_path,
                             poses=pose2, trans=trans2, betas=betas[k,10:].cpu().detach().numpy(), gender=gender_word[1], mocap_framerate=30)

            #############################################################
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, new_motion_lengths)
            ######################################################
            align_idx = np.argsort(new_motion_lengths.cpu().numpy().tolist())[::-1].copy()
            inv_align_idx = np.empty_like(align_idx)
            inv_align_idx[align_idx] = np.arange(len(align_idx))
            em_pred_original_order = em_pred[inv_align_idx]
            if savenpy:
                for k in range(bs):
                    if shape_change is not None:
                        endfix = f'_G{shape_change}'
                    else:
                        endfix = ''
                    sample_key = f"{name[k]}{endfix}_r{i:03d}"
                    embedding_path = ''
                    if save_embeddings:
                        embedding_path = os.path.join(embeddings_dir, sample_key + '_emb.npy')
                        np.save(embedding_path, em_pred_original_order[k].detach().cpu().numpy())
                    manifest_record = {
                        'sample_name': name[k],
                        'repeat_id': i,
                        'text': clip_text[k],
                        'motion_p1_path': os.path.join('motions', sample_key + '_pred_p1.npz'),
                        'motion_p2_path': os.path.join('motions', sample_key + '_pred_p2.npz'),
                        'embedding_path': os.path.join('embeddings', sample_key + '_emb.npy') if save_embeddings else '',
                    }
                    manifest_file.write(json.dumps(manifest_record, ensure_ascii=True) + '\n')



            ######################################################
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0 or is_avg_all:
                pose = pose.to(device).float()
                pose = normalizer.backward(pose)
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, new_motion_lengths)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 50)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 50)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    # motion_multimodality = motion_multimodality.reshape(-1, motion_multimodality.shape[1] * 4, motion_multimodality.shape[-1])
    # np.save(os.path.join(out_dir, f'{nb_iter}'), motion_multimodality)
    if num_repeat > 1:
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # np.save(os.path.join(out_dir, 'fixed_orders.npy'), np.concatenate(fixed_orders))

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    logger.info(msg)
    if manifest_file is not None:
        manifest_file.close()
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('./Test/multimodality', multimodality, nb_iter)

        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return fid, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger

def evaluation_transformer_uplow(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, dataname, draw = True, save = True, savegif=False, num_repeat=1, rand_pos=False, CFG=-1) : 
    from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        pose = pose.cuda(1).float()
        pose_lower = pose[..., HML_LOWER_BODY_MASK]
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda(1)

        feat_clip_text, word_emb = clip_model(text)
        
        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length)/4)

         
        pred_len = m_length.cuda(1)
        pred_tok_len = m_tokens_len

        max_motion_length = int(seq/4) + 1
        mot_end_idx = get_model(net).vqvae.num_code
        mot_pad_idx = get_model(net).vqvae.num_code + 1
        target_lower = []
        for k in range(bs):
            target = net(pose[k:k+1, :m_length[k]], type='encode')
            if m_tokens_len[k]+1 < max_motion_length:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx, 
                                    torch.ones((1, max_motion_length-1-m_tokens_len[k].int().item(), 2), dtype=int, device=target.device) * mot_pad_idx], axis=1)
            else:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx], axis=1)
            target_lower.append(target[..., 1])
        target_lower = torch.cat(target_lower, axis=0)

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda(1)
            # pred_len = torch.ones(bs).long()

            index_motion = trans(feat_clip_text, target_lower, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos, CFG=CFG)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion[..., 0] >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
            ######### [INFO] Eval only the predicted length
            #     if pred_length[k] == 0:
            #         pred_len[k] = seq
            #         continue
            #     pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], type='decode')
            #     cur_len = pred_pose.shape[1]

            #     pred_len[k] = min(cur_len, seq)
            #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            ######################################################
            
            ######### [INFO] Eval by m_length
                all_tokens = torch.cat([
                    index_motion[k:k+1, :int(pred_tok_len[k].item()), None],
                    target_lower[k:k+1, :int(pred_tok_len[k].item()), None]
                ], axis=-1)
                pred_pose = net(all_tokens, type='decode')
                pred_pose_eval[k:k+1, :int(pred_len[k].item())] = pred_pose
            pred_pose_eval[..., HML_LOWER_BODY_MASK] = pose_lower
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            ######################################################

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if num_repeat > 1:
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('./Test/multimodality', multimodality, nb_iter)

        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger

@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda(1)

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda(1)
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).cuda(1).long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(1), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda(1).float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(1), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
