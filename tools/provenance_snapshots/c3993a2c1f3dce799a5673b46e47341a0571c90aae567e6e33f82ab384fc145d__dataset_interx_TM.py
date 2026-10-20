import numpy as np
import torch
import random
import codecs as cs
import h5py
import re
import os

from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from utils.misc import to_torch
import utils.rotation_conversions as geometry
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from torch.utils.data._utils.collate import default_collate
from utils.motion_process import swap_left_right_hhi, MotionNormalizerTorchInterX

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class Text2MotionDatasetV2HHI(data.Dataset):
    def __init__(self, split_file, w_vectorizer, motion_file, unit_length=4, text_dir='./dataset/interx/texts_processed'):
        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        # Both the official Inter-X evaluator and InterMask evaluate 150-frame
        # clips. Keep this independent of the MGT block size so the GT protocol
        # cannot silently change with a model configuration.
        self.max_motion_length = 150
        self.num_person = 2
        self.unit_length = unit_length
        self.normalizer = MotionNormalizerTorchInterX('raw')
        self.max_text_len = 35
        self.beta_dir = 'dataset/interx/betas'
        self.contact_dir = 'dataset/interx/contact'
        # Official Inter-X/InterMask evaluation keeps clips with >=24 frames.
        min_motion_len = 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    beta_data = np.load(pjoin(self.beta_dir, name + '.npz'), allow_pickle=True)
                    betas = beta_data['betas'].reshape(-1,)  # 形状应为 (2, 10)
                    gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
                    gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]
                    contact_path = pjoin(self.contact_dir, name + '.npy')
                    if os.path.exists(contact_path):
                        contact = np.load(contact_path)
                    else:
                        contact = np.zeros((motion.shape[0],), dtype=np.float32)

                    motion_aug = swap_left_right_hhi(motion)
                    text_data = []
                    flag = False
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data,
                                           'betas': betas,
                                           'gender': gender_codes,
                                           'contact': contact}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                        # data_dict[name+'mirror'] = {'motion': motion_aug,
                        #                    'length': len(motion),
                        #                    'text': text_data,
                        #                    'betas': betas,
                        #                    'gender': gender_codes}
                        # new_name_list.append(name+'mirror')
                        # length_list.append(len(motion))

                except Exception as e:
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list, betas, gender, contact = data['motion'], data['length'], data['text'], data['betas'], data['gender'], data['contact']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
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
        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        contact = contact[idx:idx + m_length]
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
            contact = np.concatenate([contact,
                                        np.zeros((self.max_motion_length - m_length, ))
                                        ], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            contact = contact[:self.max_motion_length]
            m_length = self.max_motion_length
        motion = self.to_rot_6d(motion).float()
        # vmotion = self.trans2vel(motion.reshape(-1, 56, 12)).reshape(motion.shape[0], -1)
        motion = self.normalizer.forward(motion)

        betas = torch.from_numpy(betas).float()  # -> 形状 (2, 10)
        gender = torch.LongTensor(gender)  # -> 形状 (2,)
        contact = torch.from_numpy(contact).long()
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, betas, gender, contact

    def to_rot_6d(self, data):
        joints3D = np.expand_dims(data[:, -1, :], axis=1)
        joints3D = to_torch(joints3D)
        ret_tr = joints3D[:, 0, :]

        pose = data[:, :-1, :]

        pose = to_torch(pose)
        pose_all = []
        for ii in range(self.num_person):
            pose_all.append(
                geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:, :, 3 * ii:3 * ii + 3])))
        ret = torch.cat(pose_all, dim=2)
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        for ii in range(self.num_person):
            padded_tr[:, 6 * ii:6 * ii + 3] = ret_tr[:, 3 * ii:3 * ii + 3]
        ret = torch.cat((ret, padded_tr[:, None]), 1)
        T = ret.shape[0]
        ret = torch.reshape(ret, (T, -1))
        return ret

    def trans2vel(self, data):
        root_trans_p1 = data[:, -1, :3]
        root_trans_p2 = data[:, -1, 6:9]

        root_vel_p1 = root_trans_p1[1:] - root_trans_p1[:-1]
        root_relative_p2 = root_trans_p2 - root_trans_p1
        data_new = data.clone()
        data_new[1:, -1, 3:6] = root_vel_p1
        data_new[:, -1, 9:] = root_relative_p2

        return data_new


def DATALoader(dataset_name,
               batch_size,
               split_file,
               w_vectorizer,
               motion_file,
               num_workers=8,
               unit_length=8,
               shuffle=True):
    trainSet = Text2MotionDatasetV2HHI(split_file, w_vectorizer, motion_file,unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=shuffle,
                                               # sampler=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               drop_last=True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x



class Text2MotionDatasetV2HHI_Token(data.Dataset):
    def __init__(self, split_file, tokenizer_name, unit_length=4,
                 codebook_size=1024, text_dir='./dataset/interx/texts_processed',
                 max_motion_length=50, contact_dir='checkpoints/vq/contact_token_label'):

        self.tokenizer_name = tokenizer_name
        self.unit_length = unit_length
        self.codebook_size = codebook_size

        # Token 特殊标记
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        # 参数设置
        self.max_motion_length = max_motion_length
        self.max_text_len = 35
        self.beta_dir = 'dataset/interx/betas'
        self.contact_dir = contact_dir

        # 训练时的 Bucket Sampling 指针
        self.pointer = 0
        self.max_length = 20  # 初始的一个bucket长度

        # 最小 Token 长度过滤 (原 min_motion_len=30帧, 假设下采样4倍, 约等于 7-8个token)
        min_motion_token_len = 6

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        # 遍历数据
        for name in tqdm(id_list):
            try:
                # [修改点 1] 加载 Token 文件 (.npy) 而不是 H5
                token_path = pjoin(self.tokenizer_name, f'{name}.npy')
                if not os.path.exists(token_path):
                    continue

                # 假设 .npy 存储的是 (T,) 或 (1, T) 的 token 序列
                m_token_list = np.load(token_path)
                contact = np.load(pjoin(self.contact_dir, name + '.npy'))
                contact = contact.reshape(-1, )
                if len(m_token_list.shape) > 1:
                    m_token_list = m_token_list.reshape(-1, 3)  # 展平为 1D 数组


                # 长度过滤
                if len(m_token_list) < min_motion_token_len:
                    continue

                # [保留] 加载 Beta 和 Gender 数据 (原逻辑)
                beta_data = np.load(pjoin(self.beta_dir, name + '.npz'), allow_pickle=True)
                betas = beta_data['betas'].reshape(-1, )  # (2, 10) -> (20,) 或保持原样
                gender_list_str = beta_data['gender']
                gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]

                # [保留] 加载文本数据 (原逻辑)
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            # 你的原逻辑如果不是整段且不需要切分token，这里直接退出或continue
                            # 如果需要支持切分，需要在这里切分 m_token_list
                            pass

                if flag:
                    data_dict[name] = {
                        'm_token_list': m_token_list,  # 存储 token
                        'length': len(m_token_list),  # 存储 token 长度
                        'text': text_data,
                        'betas': betas,
                        'gender': gender_codes,
                        'contact': contact
                    }
                    new_name_list.append(name)
                    length_list.append(len(m_token_list))

            except Exception as e:
                # print(e)
                pass

        # [保留] 排序用于 Bucket Sampling
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        # 确保 length 不超过我们定义的最大 token 长度
        length = min(length, self.max_motion_length)
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]

        m_tokens = data['m_token_list']
        text_list = data['text']
        betas = data['betas']
        gender = data['gender']
        contact = data['contact']

        # [保留] 随机选择 Caption 并处理 Text Embedding (原逻辑)
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        # [修改点 2] Token 的 Augmentation (参考数据集逻辑)
        # 随机丢弃头部或尾部的一个 token，增加鲁棒性
        coin = np.random.choice([False, False, True])
        if coin and len(m_tokens) > 1:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
                contact = contact[:-1]
            else:
                m_tokens = m_tokens[1:]
                contact = contact[1:]

        # [修改点 3] Token 的 Padding 和 Truncating
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len + 1 < self.max_motion_length:
            # 长度不够：加上 <End> token，然后补 <Pad> token
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1, 3), dtype=int) * self.mot_end_idx,
                np.ones((self.max_motion_length - 1 - m_tokens_len, 3), dtype=int) * self.mot_pad_idx
            ], axis=0)
            contact = np.concatenate([
                contact,
                (np.zeros((self.max_motion_length - m_tokens_len, ), dtype=contact.dtype) + 2)
            ], axis=0)
        else:
            # 长度过长：截断，最后一位强制设为 <End>
            # 注意：这里我们取前 max_len-1 个，留一个位置给 End
            m_tokens = m_tokens[:self.max_motion_length - 1]
            contact = contact[:self.max_motion_length - 1]
            m_tokens = np.concatenate([
                m_tokens,
                np.ones((1, 3), dtype=int) * self.mot_end_idx
            ], axis=0)
            contact = np.concatenate([
                contact,
                (np.zeros((1, ), dtype=contact.dtype)+2)
            ], axis=0)
            m_tokens_len = self.max_motion_length - 1  # 更新有效长度

        # 转换格式
        betas = torch.from_numpy(betas).float()
        gender = torch.LongTensor(gender)
        m_tokens = torch.from_numpy(m_tokens).long()  # Token 应该是 LongTensor
        contact = torch.from_numpy(contact).long()

        # 返回值保持你原有的格式，将 motion 替换为 m_tokens
        return caption, m_tokens, m_tokens_len, betas, gender, contact

def DATALoader_Train(dataset_name,
               batch_size,
               split_file,
               codebook_size,
               motion_file,
               num_workers=8,
               unit_length=4,
               shuffle=True,
               contact_dir='checkpoints/vq/contact_token_label',
               max_motion_length=50):
    trainSet = Text2MotionDatasetV2HHI_Token(
        split_file, motion_file, unit_length, codebook_size,
        contact_dir=contact_dir, max_motion_length=max_motion_length,
    )
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=shuffle,
                                               # sampler=shuffle,
                                               num_workers=num_workers,
                                               drop_last=True)

    return train_loader
