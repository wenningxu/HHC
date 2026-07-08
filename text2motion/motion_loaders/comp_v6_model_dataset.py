import torch
from networks.modules import *
from networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
# from data.dataset import collate_fn
from utils.motion_process import MotionNormalizerTorchInterX
import clip

def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator


class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                # print(tokens)
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                # print(cap_lens)
                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                # if is_mm:
                #     print(mm_num_now, i, mm_idxs[mm_num_now])
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                # print(m_lens[0].item(), cap_lens[0].item())
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

                    # if len(mm_generated_motions) < mm_num_samples:
                    #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # print(len(generated_motion))
        # print(len(mm_generated_motions))
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        # tokens = text_data['tokens']
        # if len(tokens) < self.opt.max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print(tokens)
        # print(caption)
        # print(m_length)
        # print(self.opt.max_motion_length)
        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


# class MaskVQGeneratedDataset(Dataset):
#
#     def __init__(self, opt, dataset, net, trans, clip_model, w_vectorizer, mm_num_samples, mm_num_repeats):
#         assert mm_num_samples < len(dataset)
#         print(opt.model_dir)
#         normalizer = MotionNormalizerTorchInterX()
#         device = next(net.parameters()).device
#         dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
#         generated_motion = []
#         mm_generated_motions = []
#         mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
#         mm_idxs = np.sort(mm_idxs)
#
#         with torch.no_grad():
#             for i, data in tqdm(enumerate(dataloader)):
#                 word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, betas, gender = data
#
#                 bs, seq = pose.shape[:2]
#                 pose = pose.to(device).float()
#                 betas = betas.to(device).float()
#                 gender = gender.to(device)
#
#                 text = clip.tokenize(clip_text, truncate=True).to(device)
#
#                 feat_clip_text, word_emb = clip_model(text)
#
#                 motion_multimodality_batch = []
#                 m_tokens_len = torch.ceil((m_length) / 4)
#
#                 pred_len = m_length.to(device)
#                 pred_tok_len = m_tokens_len
#                 new_motion_lengths = torch.zeros_like(m_length)
#                 mm_num_now = len(mm_generated_motions)
#                 is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
#                 # if is_mm:
#                 #     print(mm_num_now, i, mm_idxs[mm_num_now])
#                 repeat_times = mm_num_repeats if is_mm else 1
#                 mm_motions = []
#                 blank_id = trans.num_vq
#                 for t in range(repeat_times):
#                     pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(device)
#                     index_motion = trans(feat_clip_text, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos)
#                     pred_length = (index_motion >= blank_id).int()
#                     pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()[:, 0]
#
#                     for k in range(bs):
#
#                         pred_pose = net(index_motion[k:k + 1, :int(pred_tok_len[k].item())], betas[k:k + 1],
#                                         gender[k:k + 1], type='decode')
#                         pred_pose = normalizer.backward(pred_pose)
#                         motion_length = int(pred_len[k].item() * 3 / 4)
#                         pred_pose_eval[k:k + 1, :motion_length] = pred_pose[:, :motion_length]
#                         new_motion_lengths[k] = motion_length
#                     if t == 0:
#                         # print(m_lens)
#                         # print(text_data)
#                         sub_dict = {'motion': pred_motions[0].cpu().numpy(),
#                                     'length': m_lens[0].item(),
#                                     'cap_len': cap_lens[0].item(),
#                                     'caption': caption[0],
#                                     'tokens': tokens}
#                         generated_motion.append(sub_dict)
#
#                     if is_mm:
#                         mm_motions.append({
#                             'motion': pred_motions[0].cpu().numpy(),
#                             'length': m_lens[0].item()
#                         })
#                 if is_mm:
#                     mm_generated_motions.append({'caption': caption[0],
#                                                  'tokens': tokens,
#                                                  'cap_len': cap_lens[0].item(),
#                                                  'mm_motions': mm_motions})
#
#                     # if len(mm_generated_motions) < mm_num_samples:
#                     #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
#         self.generated_motion = generated_motion
#         self.mm_generated_motion = mm_generated_motions
#         # print(len(generated_motion))
#         # print(len(mm_generated_motions))
#         self.opt = opt
#         self.w_vectorizer = w_vectorizer
#
#
#     def __len__(self):
#         return len(self.generated_motion)
#
#
#     def __getitem__(self, item):
#         data = self.generated_motion[item]
#         motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
#         sent_len = data['cap_len']
#         # tokens = text_data['tokens']
#         # if len(tokens) < self.opt.max_text_len:
#         #     # pad with "unk"
#         #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#         #     sent_len = len(tokens)
#         #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
#         # else:
#         #     # crop
#         #     tokens = tokens[:self.opt.max_text_len]
#         #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#         #     sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             try:
#                 word_emb, pos_oh = self.w_vectorizer[token]
#             except:
#                 word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)
#
#         # print(tokens)
#         # print(caption)
#         # print(m_length)
#         # print(self.opt.max_motion_length)
#         if m_length < self.opt.max_motion_length:
#             motion = np.concatenate([motion,
#                                      np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
#                                      ], axis=0)
#         return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)