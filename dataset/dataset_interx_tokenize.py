import numpy as np
import torch
import random
import hashlib
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


CACHE_CROP_POLICY = 'source_single_single_double_per_sample_sha256_v1'


def deterministic_cache_crop(name, motion_length, unit_length, cache_seed):
    """Return the source crop policy without depending on loader/RNG state.

    The released MMM cache builder samples ``single/single/double`` and a
    uniform crop offset from process-global RNGs.  Keying those same choices by
    (seed, sample name) makes interrupted cache construction byte-reproducible
    while retaining the source distribution.
    """
    digest = hashlib.sha256(
        f'{int(cache_seed)}\0{name}'.encode('utf-8')
    ).digest()
    sample_seed = int.from_bytes(digest[:4], byteorder='little', signed=False)
    numpy_rng = np.random.RandomState(sample_seed)
    python_rng = random.Random(sample_seed)
    crop_mode = numpy_rng.choice(['single', 'single', 'double'])
    token_units = motion_length // unit_length
    if crop_mode == 'double':
        token_units -= 1
    cropped_length = token_units * unit_length
    crop_start = python_rng.randint(0, motion_length - cropped_length)
    return cropped_length, crop_start

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class Text2MotionDatasetV2HHI(data.Dataset):
    def __init__(self, split_file, w_vectorizer, motion_file, unit_length=4,
                 fill=False, cache_seed=None, include_names=None,
                 beta_dir='dataset/interx/betas',
                 contact_dir='dataset/interx/contact',
                 fallback_beta_dir=None, fallback_contact_dir=None,
                 strict_assets=False):
        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 152
        self.num_person = 2
        self.unit_length = unit_length
        # MGT token caches must use the same raw-translation normalization as
        # the paper VQ tokenizer they are encoded with.
        self.normalizer = MotionNormalizerTorchInterX(motion_rep='raw')
        self.max_text_len = 35
        self.beta_dir = beta_dir
        self.contact_dir = contact_dir
        self.fallback_beta_dir = fallback_beta_dir
        self.fallback_contact_dir = fallback_contact_dir
        self.strict_assets = strict_assets
        self.fill = fill
        self.cache_seed = cache_seed
        min_motion_len = 30

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        if include_names is not None:
            include_names = set(include_names)
            id_list = [name for name in id_list if name in include_names]
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    match = re.search(r'G(\d+)', name)
                    group_str = match.group(1)  # e.g., '031'
                    group = int(group_str) - 1 # 转为整数：'031' -> 31
                    group = np.array([0, group, 1, group])
                    beta_path = pjoin(self.beta_dir, name + '.npz')
                    beta_from_primary = os.path.isfile(beta_path)
                    if not os.path.isfile(beta_path) and self.fallback_beta_dir:
                        beta_path = pjoin(self.fallback_beta_dir, name + '.npz')
                    beta_data = np.load(beta_path, allow_pickle=True)
                    betas = beta_data['betas'].reshape(-1,)  # 形状应为 (2, 10)
                    contact_path = pjoin(self.contact_dir, name + '.npy')
                    if not os.path.isfile(contact_path) and self.fallback_contact_dir:
                        contact_path = pjoin(self.fallback_contact_dir, name + '.npy')
                    if os.path.exists(contact_path):
                        contact = np.load(contact_path)
                    else:
                        if self.strict_assets and beta_from_primary:
                            raise FileNotFoundError(
                                f'augmented sample lacks contact timing: {contact_path}'
                            )
                        contact = np.zeros((motion.shape[0],), dtype=np.float32)
                    gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
                    gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]

                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'betas': betas,
                                       'gender': gender_codes,
                                       'name': name,
                                       'group': group,
                                       'contact': contact}
                    new_name_list.append(name)
                    length_list.append(len(motion))

                except Exception as e:
                    if self.strict_assets:
                        raise RuntimeError(
                            f'failed to load tokenizer asset for {name}: {e}'
                        ) from e
                    print(e)
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
        motion, m_length, betas, gender, group, contact = data['motion'], data['length'], data['betas'], data['gender'], data['group'], data['contact']

        # Preserve MMM's single/single/double crop distribution.  Cache builds
        # key the random choices by sample so an interrupted/resumed export is
        # independent of shuffle order, worker scheduling, and existing files.
        if self.cache_seed is not None and self.unit_length < 10:
            m_length, idx = deterministic_cache_crop(
                name, m_length, self.unit_length, self.cache_seed
            )
        else:
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
        if self.fill:
            if m_length < self.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                         ], axis=0)
            else:
                motion = motion[:self.max_motion_length]
                m_length = self.max_motion_length
        motion = self.to_rot_6d(motion).float()
        # vmotion = self.trans2vel(motion.reshape(-1, 56, 12)).reshape(motion.shape[0], -1)
        motion = self.normalizer.forward(motion)

        target_len = m_length // self.unit_length
        # print(contact.shape, name)
        contact = contact.reshape(target_len, self.unit_length)
        contact = (np.sum(contact, axis=1) > 0).astype(np.float32)

        betas = torch.from_numpy(betas).float()  # -> 形状 (2, 10)
        gender = torch.LongTensor(gender)  # -> 形状 (2,)
        group = torch.LongTensor(group)
        return motion, m_length, name, betas, gender, group, contact

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
               shuffle=True,
               cache_seed=None,
               include_names=None,
               beta_dir='dataset/interx/betas',
               contact_dir='dataset/interx/contact',
               fallback_beta_dir=None,
               fallback_contact_dir=None,
               strict_assets=False):
    trainSet = Text2MotionDatasetV2HHI(
        split_file, w_vectorizer, motion_file, unit_length,
        cache_seed=cache_seed, include_names=include_names,
        beta_dir=beta_dir, contact_dir=contact_dir,
        fallback_beta_dir=fallback_beta_dir,
        fallback_contact_dir=fallback_contact_dir,
        strict_assets=strict_assets,
    )
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=shuffle,
                                               # sampler=shuffle,
                                               num_workers=num_workers,
                                               drop_last=True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
