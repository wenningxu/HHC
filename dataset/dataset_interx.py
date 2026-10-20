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
from utils.motion_process import MotionNormalizerTorchInterX, MotionNormalizerTorchInterXSingle, prepare_interx, swap_left_right_hhi


class MotionDatasetV2HHI(data.Dataset):
    def __init__(self, split_file, motion_file, window_size = 64, motion_rep='raw'):

        self.window_size = window_size
        self.data = []
        self.lengths = []
        self.betas = []
        self.genders = []
        self.groups = []
        self.num_person = 2
        # The representation selector must be identical on normalization and
        # every loss/export de-normalization path.  The legacy default is the
        # velocity/root-feature statistics and is invalid for raw translation.
        self.normalizer = MotionNormalizerTorchInterX(motion_rep=motion_rep)
        self.beta_dir = 'dataset/interx/betas'
        self.split = split_file
        self.motion_rep = motion_rep
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if motion.shape[0] < self.window_size:
                        continue
                    match = re.search(r'G(\d+)', name)
                    group_str = match.group(1)  # e.g., '031'
                    group = int(group_str) - 1 # 转为整数：'031' -> 31
                    group = np.array([0, group, 1, group])
                    beta_data = np.load(pjoin(self.beta_dir, name + '.npz'), allow_pickle=True)
                    betas = beta_data['betas'].reshape(-1,)
                    gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
                    gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.data.append(motion)
                    self.betas.append(betas)
                    self.genders.append(gender_codes)
                    self.groups.append(group)

                    # offset_xz = np.random.uniform(low=-0.2, high=0.2, size=(2,))
                    # offset = np.array([offset_xz[0], 0.0, offset_xz[1]])  # shape (3,)
                    # motion_aug = motion.copy()
                    # motion_aug[:, -1, :3] += offset
                    # motion_aug[:, -1, 3:] += offset
                    # if 'train' in split_file.lower():
                    #     motion_aug = swap_left_right_hhi(motion)
                    #
                    #     self.lengths.append(motion_aug.shape[0] - self.window_size)
                    #     self.data.append(motion_aug)
                    #     self.betas.append(betas)
                    #     self.genders.append(gender_codes)
                    #     self.groups.append(group)


                except Exception as e:
                    print(e)
                    pass

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        betas = self.betas[motion_id]
        gender = self.genders[motion_id]
        group = self.groups[motion_id]

        motion = self.to_rot_6d(motion).float()
        if self.motion_rep == 'vel':
            if np.random.rand() > 0.5:
                motion = prepare_interx(motion, swap=True)
            else:
                motion = prepare_interx(motion)
        motion = self.normalizer.forward(motion)

        betas = torch.from_numpy(betas).float()  # -> 形状 (2, 10)
        gender = torch.tensor(gender, dtype=torch.long)
        group = torch.tensor(group, dtype=torch.long)

        if ('kkkk' in self.split.lower()) and (np.random.rand() > 5):
            rand_item = np.random.randint(0, self.cumsum[-1])

            # 2. 【关键】完全复用你的索引定位逻辑
            # 这样能保证取出的 rand_idx + window_size 绝对不会越界
            if rand_item != 0:
                rand_motion_id = np.searchsorted(self.cumsum, rand_item) - 1
                rand_idx = rand_item - self.cumsum[rand_motion_id] - 1
            else:
                rand_motion_id = 0
                rand_idx = 0

            motion_rand = self.data[rand_motion_id][rand_idx: rand_idx + self.window_size]
            betas_rand = self.betas[rand_motion_id]
            gender_rand = self.genders[rand_motion_id]
            group_rand = self.groups[rand_motion_id]

            motion_rand = self.to_rot_6d(motion_rand).float()
            motion_rand = self.normalizer.forward(motion_rand)
            betas_rand = torch.from_numpy(betas_rand).float()
            gender_rand = torch.LongTensor(gender_rand)
            group_rand = torch.LongTensor(group_rand)

            motion = motion.reshape(-1, 56, 12)
            motion_rand = motion_rand.reshape(-1, 56, 12)
            motion_list = [motion[..., :6], motion[..., 6:]]
            motion_rand_list = [motion_rand[..., :6], motion_rand[..., 6:]]
            p1_random = random.choice([0, 1])
            p2_random = random.choice([0, 1])
            motion = torch.cat([motion_list[p1_random], motion_rand_list[p2_random]], dim=-1)
            motion = motion.reshape(self.window_size, -1)
            betas = betas.reshape(2,10)
            betas_rand = betas_rand.reshape(2,10)
            betas = torch.stack([betas[p1_random], betas_rand[p2_random]], dim=0)
            betas = betas.reshape(-1,)
            gender = torch.LongTensor([gender[p1_random], gender_rand[p2_random]])
            group = torch.LongTensor([p1_random, group[1], p2_random, group_rand[3]])

        # swap person order randomly
        if ('kkkk' in self.split.lower()) and (np.random.rand() > 5):
            motion = motion.reshape(self.window_size, 56, 2,6)
            motion[:, : , [0, 1]] = motion[:, :, [1, 0]]
            motion = motion.reshape(self.window_size, -1)
            betas = betas.reshape(2,10)
            betas[[0, 1], :] = betas[[1, 0], :]
            betas = betas.reshape(-1,)
            gender[[0, 1]] = gender[[1, 0]]
            group = torch.LongTensor([group[2], group[3], group[0], group[1]])

        # to do, random choice
        # group = torch.LongTensor(group)
        return motion, betas, gender, group

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


class MotionDatasetPrimitiveHHI(data.Dataset):
    """
    修改后的 HHI 数据集：
    - 保留了 h5py, betas, genders, groups 和 cumsum 加载逻辑。
    - __init__ 被修改为计算 "基元序列" (sample) 的总长度。
    - __getitem__ 被修改为返回一整串 `num_primitive` 个基元。
    """

    def __init__(self, split_file, motion_file, num_primitive,
                 history_length,
                 future_length,
                 beta_dir='dataset/interx/betas',
                 text_dir='./dataset/interx/texts_processed'):

        # 1. 设置新的基元和样本长度参数
        self.num_primitive = num_primitive
        self.history_length = history_length
        self.future_length = future_length

        # 单个基元的长度
        self.primitive_length = self.history_length + self.future_length
        # 一个完整训练样本（一串基元）所需的总帧数
        self.sample_length = self.history_length + self.future_length * self.num_primitive

        print(f"--- HHI 基元序列数据集 ---")
        print(f"  基元长度 (H+F): {self.primitive_length}")
        print(f"  样本总长度 (H + F*N): {self.sample_length}")
        print(f"  基元数量 (N): {self.num_primitive}")

        # 2. 保留您原有的加载逻辑
        self.data = []
        self.lengths = []
        self.betas = []
        self.genders = []
        self.groups = []
        self.texts_data = []
        self.contacts = []
        self.num_person = 2
        self.normalizer = MotionNormalizerTorchInterX()  # 使用您自己的 Normalizer
        self.beta_dir = beta_dir
        self.contact_dir = 'dataset/interx/contact'

        id_list = []
        # with cs.open(split_file, 'r') as f: # 替换为您自己的文件打开方式
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')

                    # 3. !! 核心修改点 (A) !!
                    # 检查序列长度是否足够长，以提取至少一个完整的 "样本"
                    if motion.shape[0] < self.sample_length:
                        continue

                    match = re.search(r'G(\d+)', name)
                    group_str = match.group(1)  # e.g., '031'
                    group = int(group_str) - 1  # 转为整数：'031' -> 31
                    group = np.array([0, group, 1, group])
                    # 假设 'name' 中不包含 '.npz' 后缀
                    beta_path = pjoin(self.beta_dir, name + '.npz')

                    beta_data = np.load(beta_path, allow_pickle=True)
                    betas = beta_data['betas']  # 假设betas已经是 (2, 10) 或 (20,)
                    betas = betas.reshape(-1,)  # 确保是 (2, 10)

                    gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
                    gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]  # [0, 1]
                    contact_path = pjoin(self.contact_dir, name + '.npy')
                    if os.path.exists(contact_path):
                        contact = np.load(contact_path)
                    else:
                        contact = np.zeros((motion.shape[0],), dtype=np.float32)

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
                        self.texts_data.append(text_data)

                    # 4. !! 核心修改点 (B) !!
                    # 计算这个序列能产生多少个合法的 "起始点"
                    # (原: motion.shape[0] - self.window_size)
                    num_valid_starts = motion.shape[0] - self.sample_length + 1
                    self.lengths.append(num_valid_starts)

                    # 保留完整数据
                    self.data.append(motion)
                    self.betas.append(betas)
                    self.genders.append(gender_codes)
                    self.groups.append(group)
                    self.contacts.append(contact)

                except Exception as e:
                    print(f"加载 {name} 时出错: {e}")
                    pass

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, Total samples (primitive sequences) {}".format(len(self.data),
                                                                                          self.cumsum[-1]))

    def __len__(self):
        # 这部分逻辑不变，它现在返回 "样本" 的总数
        return self.cumsum[-1]

    def __getitem__(self, item):
        # 1. 索引查找逻辑 (不变)
        # 找到 'item' 对应的长序列 (motion_id) 和在该序列中的起始帧 (seq_start_frame)
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            seq_start_frame = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            seq_start_frame = 0

        # 2. !! 核心修改点 (C): 数据切片 !!

        # 获取完整的长序列数据
        betas = self.betas[motion_id]
        gender = self.genders[motion_id]
        group = self.groups[motion_id]
        contact = self.contacts[motion_id]
        full_motion_data = self.data[motion_id]
        text_list = self.texts_data[motion_id]
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # 遍历并切片所有 num_primitive 个基元
        primitive_list_np = []
        contact_list = []
        for i in range(self.num_primitive):
            # 计算当前基元的起始帧
            prim_start = seq_start_frame + i * self.future_length
            prim_end = prim_start + self.primitive_length

            # 切片
            motion_primitive = full_motion_data[prim_start: prim_end]
            contact_primitive = contact[prim_start: prim_end]
            primitive_list_np.append(motion_primitive)
            contact_list.append(contact_primitive)

        # 将基元列表堆叠成一个单独的Numpy数组
        # Shape: [num_primitive, primitive_length, D_motion]
        all_primitives_np = np.stack(primitive_list_np, axis=0)
        all_contact = np.stack(contact_list, axis=0)

        # 3. !! 核心修改点 (D): 后处理 !!
        # 我们需要对 [N, L, D_full] 张量应用 to_rot_6d 和 normalize
        # 它们可能期望 [T, D_full] 形状，所以我们先 reshape
        N, L, _, _ = all_primitives_np.shape

        # 转换为Torch张量
        motion_tensor_flat = to_torch(all_primitives_np.reshape(N * L, 56, 6))

        # 应用 6D 转换 (假设 to_rot_6d 返回 torch tensor)
        motion_6d_flat = self.to_rot_6d(motion_tensor_flat).float()

        # 应用归一化
        motion_normalized_flat = self.normalizer.forward(motion_6d_flat)

        # Reshape 回 [N, L, D_processed]
        D_processed = motion_normalized_flat.shape[-1]
        motion_processed = motion_normalized_flat.reshape(N, L, D_processed)
        all_contact = torch.from_numpy(all_contact).reshape(N, L).long()

        # 4. 加载其他数据
        betas = torch.from_numpy(betas).float()  # -> 形状 (2, 10)
        gender = torch.LongTensor(gender)  # -> 形状 (2,)
        group = torch.LongTensor(group)

        # 5. 返回字典 (用于 collate_fn)
        return motion_processed, betas, gender, group, caption, all_contact

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

class MotionDatasetV2HHISingle(data.Dataset):
    def __init__(self, split_file, motion_file, window_size = 64):

        self.window_size = window_size
        self.data = []
        self.lengths = []
        self.betas= []
        self.genders = []
        self.groups = []
        self.num_person = 1
        self.normalizer = MotionNormalizerTorchInterXSingle()
        self.beta_dir = 'dataset/interx/betas'
        self.split = split_file
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if motion.shape[0] < self.window_size:
                        continue
                    match = re.search(r'G(\d+)', name)
                    group_str = match.group(1)  # e.g., '031'
                    group = int(group_str) - 1 # 转为整数：'031' -> 31
                    group_p1 = np.array([0, group])
                    group_p2 = np.array([1, group])
                    beta_data = np.load(pjoin(self.beta_dir, name + '.npz'), allow_pickle=True)
                    betas = beta_data['betas'].reshape(2,10)
                    gender_list_str = beta_data['gender']  # 这是一个列表, 例如 ['male', 'female']
                    gender_codes = [0 if g == 'male' else 1 for g in gender_list_str]
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.lengths.append(motion.shape[0] - self.window_size)
                    motion = motion.reshape(motion.shape[0], 56, 6)
                    self.data.append(motion[..., :3])
                    self.data.append(motion[..., 3:])
                    self.betas.append(betas[0])
                    self.betas.append(betas[1])
                    self.genders.append(gender_codes[0])
                    self.genders.append(gender_codes[1])
                    self.groups.append(group_p1)
                    self.groups.append(group_p2)


                except Exception as e:
                    print(e)
                    pass

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        betas = self.betas[motion_id]
        gender = self.genders[motion_id]
        group = self.groups[motion_id]

        motion = self.to_rot_6d(motion).float()
        motion = self.normalizer.forward(motion)

        betas = torch.from_numpy(betas).float()  # -> 形状 (, 10)
        gender = torch.tensor(gender, dtype=torch.long)
        group = torch.tensor(group, dtype=torch.long)


        return motion, betas, gender, group

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
               motion_file,
               num_workers=8,
               window_size=64,
               unit_length=4,
               motion_rep='raw'):
    trainSet = MotionDatasetV2HHI(split_file, motion_file, window_size=window_size, motion_rep=motion_rep)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               # collate_fn=collate_fn,
                                               drop_last=True)

    return train_loader

def DATALoaderPrimitive(dataset_name,
               batch_size,
               split_file,
               motion_file,
               num_workers=8,
               num_primitive=4,
               history_length=3,
               future_length=12,
               unit_length=4):
    trainSet = MotionDatasetPrimitiveHHI(split_file, motion_file, num_primitive=num_primitive,
                                        history_length=history_length,
                                        future_length=future_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    return train_loader

def DATALoaderSingle(dataset_name,
               batch_size,
               split_file,
               motion_file,
               num_workers=8,
               window_size=64,
               unit_length=4):
    trainSet = MotionDatasetV2HHISingle(split_file, motion_file, window_size=window_size)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               # collate_fn=collate_fn,
                                               drop_last=True)

    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x




# class MotionNormalizerTorchInterX:
#     def __init__(self, mean_path="./dataset/interx/data_shape/Interx_Mean_shape.npy", std_path="./dataset/interx/data_shape/Interx_Std_shape.npy"):
#         mean = np.load(mean_path).astype(np.float32)
#         std = np.load(std_path).astype(np.float32)
#         self.motion_mean = torch.from_numpy(mean).float()
#         self.motion_std = torch.from_numpy(std).float()
#
#     def normalize(self, x: torch.Tensor) -> torch.Tensor:
#         if not isinstance(x, torch.Tensor):
#             raise TypeError("输入必须是 PyTorch Tensor")
#         if x.dim() != 2:
#             raise ValueError(f"输入张量应为2维 (T, C)，但收到了 {x.dim()} 维")
#
#         device = x.device
#
#         x_normalized = (x.clone() - self.motion_mean.to(device)) / self.motion_std.to(device)
#
#         return x_normalized
#
#     def denormalize(self, x: torch.Tensor) -> torch.Tensor:
#         if not isinstance(x, torch.Tensor):
#             raise TypeError("输入必须是 PyTorch Tensor")
#         if x.dim() != 2:
#             raise ValueError(f"输入张量应为2维 (T, C)，但收到了 {x.dim()} 维")
#
#         device = x.device
#
#         x_denormalized = x.clone() * self.motion_std.to(device) + self.motion_mean.to(device)
#
#         return x_denormalized
