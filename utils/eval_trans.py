import os
import json

import clip
import numpy as np
import torch
from scipy import linalg

from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from exit.utils import get_model
from tqdm import tqdm

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

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, betas, gender, contact_mask = batch

        bs, seq = pose.shape[:2]
        pose = pose.to(device).float()
        betas = betas.to(device).float()
        gender = gender.to(device)

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
            for k in range(bs):
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

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, new_motion_lengths)
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


            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0 or is_avg_all:
                pose = pose.to(device).float()
                pose = normalizer.backward(pose)
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, new_motion_lengths)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

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
