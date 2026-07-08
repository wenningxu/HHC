import codecs as cs
import os
import re
from os.path import join as pjoin

import h5py
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

import utils.rotation_conversions as geometry
from utils.misc import to_torch
from utils.motion_process import MotionNormalizerTorchChi3D


def _load_ids(split_file):
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            name = line.strip()
            if name:
                id_list.append(name)
    return id_list


def _parse_name(name):
    match = re.match(r"G(\d+)_([^_]+)_(\d+)", name)
    if match is None:
        raise ValueError(f"Invalid chi3d sample name: {name}")
    group = int(match.group(1))
    action = match.group(2)
    sample_idx = int(match.group(3))
    return group, action, sample_idx


def _build_action_map(id_list):
    actions = sorted({_parse_name(name)[1] for name in id_list})
    action_to_id = {action: idx for idx, action in enumerate(actions)}
    return actions, action_to_id


def _make_onehot(action_id, num_actions):
    onehot = np.zeros(num_actions, dtype=np.float32)
    onehot[action_id] = 1.0
    return onehot


class _Chi3DMixin:
    num_person = 2

    def to_rot_6d(self, data):
        joints3d = np.expand_dims(data[:, -1, :], axis=1)
        joints3d = to_torch(joints3d)
        ret_tr = joints3d[:, 0, :]

        pose = data[:, :-1, :]
        pose = to_torch(pose)

        pose_all = []
        for ii in range(self.num_person):
            pose_all.append(
                geometry.matrix_to_rotation_6d(
                    geometry.axis_angle_to_matrix(pose[:, :, 3 * ii:3 * ii + 3])
                )
            )

        ret = torch.cat(pose_all, dim=2)
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        for ii in range(self.num_person):
            padded_tr[:, 6 * ii:6 * ii + 3] = ret_tr[:, 3 * ii:3 * ii + 3]
        ret = torch.cat((ret, padded_tr[:, None]), 1)
        return torch.reshape(ret, (ret.shape[0], -1))


class MotionDatasetChi3D(data.Dataset, _Chi3DMixin):
    def __init__(self, split_file, motion_file, window_size=64):
        self.window_size = window_size
        self.motion_file = motion_file
        self.normalizer = MotionNormalizerTorchChi3D()

        self.data = []
        self.lengths = []
        self.groups = []
        self.names = []

        id_list = _load_ids(split_file)
        self.actions, self.action_to_id = _build_action_map(id_list)

        with h5py.File(motion_file, "r") as mf:
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype("float32")
                    if motion.shape[0] < self.window_size:
                        continue

                    group, _, _ = _parse_name(name)
                    num_valid_starts = motion.shape[0] - self.window_size + 1

                    self.data.append(motion)
                    self.lengths.append(num_valid_starts)
                    self.groups.append(group)
                    self.names.append(name)
                except Exception as exc:
                    print(exc)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(f"Total number of chi3d motions {len(self.data)}, snippets {self.cumsum[-1]}")

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        motion_id = np.searchsorted(self.cumsum, item, side="right") - 1
        idx = item - self.cumsum[motion_id]

        motion = self.data[motion_id][idx:idx + self.window_size]
        motion = self.to_rot_6d(motion).float()
        motion = self.normalizer.forward(motion)

        group_idx = self.groups[motion_id] - 2
        group_ids = torch.tensor([0, group_idx, 1, group_idx], dtype=torch.long)
        return motion, group_ids


class Chi3DTokenDataset(data.Dataset):
    def __init__(self, split_file, tokenizer_name, unit_length=4,
                 codebook_size=1024, max_motion_length=38):
        self.tokenizer_name = tokenizer_name
        self.unit_length = unit_length
        self.codebook_size = codebook_size
        self.max_motion_length = max_motion_length

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        id_list = _load_ids(split_file)
        self.actions, self.action_to_id = _build_action_map(id_list)
        self.num_actions = len(self.actions)

        self.data = []
        min_motion_token_len = 1

        for name in tqdm(id_list):
            token_path = pjoin(self.tokenizer_name, f"{name}.npy")
            if not os.path.exists(token_path):
                continue
            try:
                m_token_list = np.load(token_path)
                if len(m_token_list.shape) > 1:
                    m_token_list = m_token_list.reshape(-1, m_token_list.shape[-1])

                if len(m_token_list) < min_motion_token_len:
                    continue

                group, action, _ = _parse_name(name)
                action_id = self.action_to_id[action]
                action_onehot = _make_onehot(action_id, self.num_actions)

                self.data.append(
                    {
                        "name": name,
                        "group": group,
                        "action_name": action,
                        "action_id": action_id,
                        "action_onehot": action_onehot,
                        "m_token_list": m_token_list,
                    }
                )
            except Exception as exc:
                print(exc)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_dict = self.data[item]
        m_tokens = data_dict["m_token_list"].copy()

        coin = np.random.choice([False, False, True])
        if coin and len(m_tokens) > 1:
            if np.random.choice([True, False]):
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        token_width = 1 if m_tokens.ndim == 1 else m_tokens.shape[1]

        if m_tokens_len + 1 < self.max_motion_length:
            end_token = np.ones((1, token_width), dtype=np.int64) * self.mot_end_idx
            pad_tokens = np.ones(
                (self.max_motion_length - 1 - m_tokens_len, token_width), dtype=np.int64
            ) * self.mot_pad_idx
            if token_width == 1:
                end_token = end_token.reshape(-1)
                pad_tokens = pad_tokens.reshape(-1)
            m_tokens = np.concatenate([m_tokens, end_token, pad_tokens], axis=0)
        else:
            m_tokens = m_tokens[:self.max_motion_length - 1]
            end_token = np.ones((1, token_width), dtype=np.int64) * self.mot_end_idx
            if token_width == 1:
                end_token = end_token.reshape(-1)
            m_tokens = np.concatenate([m_tokens, end_token], axis=0)
            m_tokens_len = self.max_motion_length - 1

        return (
            data_dict["action_name"],
            torch.tensor(data_dict["action_id"], dtype=torch.long),
            torch.from_numpy(data_dict["action_onehot"]).float(),
            torch.from_numpy(m_tokens).long(),
            m_tokens_len,
            torch.tensor(data_dict["group"], dtype=torch.long),
        )


class Chi3DActionDataset(data.Dataset, _Chi3DMixin):
    def __init__(self, split_file, motion_file=None, include_motion=False, max_motion_length=196):
        self.include_motion = include_motion
        self.motion_file = motion_file
        self.max_motion_length = max_motion_length
        self.normalizer = MotionNormalizerTorchChi3D() if include_motion else None

        id_list = _load_ids(split_file)
        self.actions, self.action_to_id = _build_action_map(id_list)
        self.num_actions = len(self.actions)

        motion_dict = {}
        if include_motion:
            if motion_file is None:
                raise ValueError("motion_file is required when include_motion=True")
            with h5py.File(motion_file, "r") as mf:
                for name in id_list:
                    if name in mf:
                        motion_dict[name] = mf[name][:].astype("float32")

        self.data = []
        for name in id_list:
            group, action, _ = _parse_name(name)
            action_id = self.action_to_id[action]
            sample = {
                "name": name,
                "group": group,
                "action_name": action,
                "action_id": action_id,
                "action_onehot": _make_onehot(action_id, self.num_actions),
            }
            if include_motion and name in motion_dict:
                sample["motion"] = motion_dict[name]
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        batch = [
            sample["name"],
            sample["action_name"],
            torch.tensor(sample["action_id"], dtype=torch.long),
            torch.from_numpy(sample["action_onehot"]).float(),
            torch.tensor(sample["group"], dtype=torch.long),
        ]

        if self.include_motion and "motion" in sample:
            motion = sample["motion"]
            motion_len = min(len(motion), self.max_motion_length)
            motion = motion[:motion_len]
            motion = self.to_rot_6d(motion).float()
            motion = self.normalizer.forward(motion)
            if motion_len < self.max_motion_length:
                pad = torch.zeros((self.max_motion_length - motion_len, motion.shape[1]), dtype=motion.dtype)
                motion = torch.cat([motion, pad], dim=0)
            batch.append(motion)
        return tuple(batch)


def DATALoader(dataset_name,
               batch_size,
               split_file,
               motion_file,
               num_workers=8,
               window_size=64,
               shuffle=True):
    train_set = MotionDatasetChi3D(split_file, motion_file, window_size=window_size)
    return torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def DATALoader_Train(dataset_name,
                     batch_size,
                     split_file,
                     codebook_size,
                     motion_file,
                     num_workers=8,
                     unit_length=4,
                     shuffle=True,
                     max_motion_length=38):
    train_set = Chi3DTokenDataset(
        split_file,
        motion_file,
        unit_length=unit_length,
        codebook_size=codebook_size,
        max_motion_length=max_motion_length,
    )
    return torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def DATALoader_Action(dataset_name,
                      batch_size,
                      split_file,
                      motion_file=None,
                      num_workers=8,
                      shuffle=False,
                      include_motion=False,
                      max_motion_length=196):
    train_set = Chi3DActionDataset(
        split_file,
        motion_file=motion_file,
        include_motion=include_motion,
        max_motion_length=max_motion_length,
    )
    return torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
