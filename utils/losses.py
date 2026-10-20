import torch
import numpy as np
import torch.nn as nn
from utils.motion_process import (
    recover_from_ric,
    MotionNormalizerTorchInterX,
    MotionNormalizerTorchInterXSingle,
    recover_interx_from_features_batch,
)
from utils.rotation_conversions import *
class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()

        # ReConsLossInter and native SMPL RawLoss use their supplied FK layer.
        # Load the full SMPL-X assets only for this legacy loss class.
        from utils.rotation2xyz import Rotation2xyz_x
        from utils.paramUtil import SMPLX_MODEL_MALE_PATH, SMPLX_MODEL_FEMALE_PATH
        self.rot2xyz_male = Rotation2xyz_x('cpu', model_path=SMPLX_MODEL_MALE_PATH)
        self.rot2xyz_female = Rotation2xyz_x('cpu', model_path=SMPLX_MODEL_FEMALE_PATH)
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        self.motion_dim = nb_joints * 12 + 2 * 6
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_joint(self, motion_pred, motion_gt) :
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss


class ReConsLossInter(nn.Module):
    def __init__(self, args, recons_loss, nb_joints, fk_layers, normalizer=None):
        super(ReConsLossInter, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        self.args = args
        self.motion_rep = args.motion_rep

        self.normalizer = normalizer if normalizer is not None else MotionNormalizerTorchInterX()
        self.fk_pair_processors = fk_layers
        self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        self.motion_dim = nb_joints * 12 + 2 * 6
        self.fids = [7, 10, 8, 11]
        self.losses = {}

    def forward(self, motion_pred, motion_gt, beta, gender, group):

        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])

        self.losses['main'] = loss

        self.motion_pred_de = self.normalizer.backward(motion_pred).reshape(motion_pred.shape[0], -1, 56, 12)
        self.motion_gt_de = self.normalizer.backward(motion_gt).reshape(motion_pred.shape[0], -1, 56, 12)
        if self.motion_rep == 'vel':
            self.motion_pred_de_interx = recover_interx_from_features_batch(self.motion_pred_de).reshape(motion_pred.shape[0], -1, 56, 12)
            self.motion_gt_de_interx = recover_interx_from_features_batch(self.motion_gt_de).reshape(motion_pred.shape[0], -1, 56, 12)
            motion_pred_de_joints_p1, motion_pred_de_joints_p2 = self.fk_batch(self.motion_pred_de_interx, group)
            motion_gt_de_joints_p1, motion_gt_de_joints_p2 = self.fk_batch(self.motion_gt_de_interx, group)
        else:
            motion_pred_de_joints_p1, motion_pred_de_joints_p2 = self.fk_batch(self.motion_pred_de, group)
            motion_gt_de_joints_p1, motion_gt_de_joints_p2 = self.fk_batch(self.motion_gt_de, group)

        self.motion_gt_de_joints_p1 = self.normalize_floor(motion_gt_de_joints_p1)
        self.motion_gt_de_joints_p2 = self.normalize_floor(motion_gt_de_joints_p2)
        self.motion_pred_de_joints_p1 = self.normalize_floor(motion_pred_de_joints_p1)
        self.motion_pred_de_joints_p2 = self.normalize_floor(motion_pred_de_joints_p2)

        loss_position = self.forward_joint()  # This now populates self.losses['Joint']

        # and sets self.motion_* variables

        # Calculate other losses (they will use the self.motion_* variables set above)
        loss_root = self.forward_root()
        loss_vel = self.forward_vel()
        loss_rotation = self.forward_rotation()
        loss_fc = self.forward_foot_contact()
        # self.forward_relative_oriental()
        loss_dp = self.forward_distance_map()

        self.losses['Joint'] = self.args.loss_explicit * loss_position
        self.losses['Root'] = self.args.loss_root * loss_root
        self.losses['Velocity'] = self.args.loss_vel * loss_vel
        self.losses['Rotation'] = self.args.loss_geo * loss_rotation
        self.losses['FootContact'] = self.args.loss_fc * loss_fc
        self.losses['DistanceMap'] = self.args.loss_distance_map * loss_dp

        return self.losses

    def fk_batch(self, motion_batch, group_ids_batch):

        p1_final_coords, p2_final_coords = self.fk_pair_processors(motion_batch, group_ids_batch)
        return p1_final_coords, p2_final_coords

    def forward_joint(self):



        # save for test

        # np.save('debug_motion_gt_de_joints_p1.npy', self.motion_gt_de_joints_p1.detach().cpu().numpy())

        loss = (self.Loss(self.motion_pred_de_joints_p1,

                          self.motion_gt_de_joints_p1) +

                self.Loss(self.motion_pred_de_joints_p2,

                          self.motion_gt_de_joints_p2)) / 2

        return loss

    def forward_root(self):

        loss_orientation = self.Loss(self.motion_pred_de[..., 0, :], self.motion_gt_de[..., 0, :])
        trans_p1 = self.motion_gt_de_joints_p1[..., 0, [0,2]]
        trans_p2 = self.motion_gt_de_joints_p2[..., 0, [0,2]]
        trans_pred_p1 = self.motion_pred_de_joints_p1[..., 0, [0,2]]
        trans_pred_p2 = self.motion_pred_de_joints_p2[..., 0, [0, 2]]
        loss_trans_p1 = self.Loss(trans_pred_p1, trans_p1)
        loss_trans_p2 = self.Loss(trans_pred_p2, trans_p2)
        loss_trans_vel_p1 = self.Loss(trans_pred_p1[:, 1:] - trans_pred_p1[:, :-1], trans_p1[:, 1:] - trans_p1[:, :-1])
        loss_trans_vel_p2 = self.Loss(trans_pred_p2[:, 1:] - trans_pred_p2[:, :-1], trans_p2[:, 1:] - trans_p2[:, :-1])
        loss_relative = self.Loss(trans_pred_p2-trans_pred_p1, trans_p2-trans_p1)
        loss_trans = (loss_trans_p1 + loss_trans_p2 + loss_trans_vel_p1 + loss_trans_vel_p2) / 2 + loss_relative
        if self.motion_rep == 'vel':
            # 提取 12 维特征: [0:3]v_a, [3:6]v_b, [6:9]delta_p, [9:12]delta_v
            feat_pred = self.motion_pred_de[..., -1, :]  # (Batch, Frames, 12)
            v_a_pred = feat_pred[..., 0:3]
            v_b_pred = feat_pred[..., 3:6]
            r_p_pred = feat_pred[..., 6:9]
            r_v_pred = feat_pred[..., 9:12]

            # A. 速度一致性 (Speed Consistency): \dot{g}_b - \dot{g}_a = \Delta\dot{p}_{ba}
            # 确保预测的个体速度与预测的相对速度在数学上闭环
            loss_speed_consistency = self.Loss(v_b_pred - v_a_pred, r_v_pred)

            # B. 导数一致性 (Derivative Consistency): \Delta p_t - \Delta p_{t-1} = \Delta\dot{p}_t
            # 确保预测的相对位置随时间的变化量等于预测的相对速度
            rel_pos_diff = r_p_pred[:, 1:] - r_p_pred[:, :-1]
            loss_derivative_consistency = self.Loss(rel_pos_diff, r_v_pred[:, 1:])
            # 汇总一致性项
            loss_trans += (loss_speed_consistency + loss_derivative_consistency)


        return loss_orientation + loss_trans

    def forward_vel(self):

        vel_pred_p1 = self.motion_pred_de_joints_p1[:, 1:] - self.motion_pred_de_joints_p1[:, :-1]

        vel_pred_p2 = self.motion_pred_de_joints_p2[:, 1:] - self.motion_pred_de_joints_p2[:, :-1]

        vel_gt_p1 = self.motion_gt_de_joints_p1[:, 1:] - self.motion_gt_de_joints_p1[:, :-1]

        vel_gt_p2 = self.motion_gt_de_joints_p2[:, 1:] - self.motion_gt_de_joints_p2[:, :-1]

        loss = (self.Loss(vel_pred_p1, vel_gt_p1) +

                self.Loss(vel_pred_p2, vel_gt_p2)) / 2

        return loss

    def forward_rotation(self, eps=1e-7):

        pred_m = rotation_6d_to_matrix(self.motion_pred_de.reshape(self.motion_pred_de.shape[0], -1, 56, 2, 6))[:, :,
                 :-1].reshape(-1, 3, 3)

        gt_m = rotation_6d_to_matrix(self.motion_gt_de.reshape(self.motion_pred_de.shape[0], -1, 56, 2, 6))[:, :,
               :-1].reshape(-1, 3, 3)

        # bf16 rounds the acos clamp's 1-eps to 1, producing infinite gradients.
        with torch.autocast(device_type=pred_m.device.type, enabled=False):
            m = torch.bmm(gt_m.float(), pred_m.float().transpose(1, 2))

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

        theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))

        return torch.mean(theta)

    def normalize_floor(self, positions: torch.Tensor) -> torch.Tensor:

        # Find floor height for each batch element (min y across seqlen & jointsnum)

        floor_heights = positions[..., 1].view(positions.shape[0], -1).min(dim=1).values

        # Subtract floor height from y-coordinates using broadcasting

        normalized_positions = positions.clone()

        normalized_positions[..., 1] -= floor_heights.view(-1, 1, 1)

        return positions  # Returns modified tensor (in-place)

    def cal_foot_contact(self, feet_vel, feet_h):

        thres = 0.001

        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(

            [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

        feet_x = (feet_vel[..., 0]) ** 2

        feet_y = (feet_vel[..., 1]) ** 2

        feet_z = (feet_vel[..., 2]) ** 2

        contact = ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)

        return contact

    def forward_foot_contact(self):

        motion_p1 = self.normalize_floor(self.motion_gt_de_joints_p1)
        motion_p2 = self.normalize_floor(self.motion_gt_de_joints_p2)
        pred_motion_p1 = self.normalize_floor(self.motion_pred_de_joints_p1)
        pred_motion_p2 = self.normalize_floor(self.motion_pred_de_joints_p2)
        feet_vel_p1 = motion_p1[:, 1:, self.fids, :] - motion_p1[:, :-1, self.fids, :]
        pred_feet_vel_p1 = pred_motion_p1[:, 1:, self.fids, :] - pred_motion_p1[:, :-1, self.fids, :]

        feet_h = motion_p1[:, :-1, self.fids, 1]
        contact_p1 = self.cal_foot_contact(feet_vel_p1, feet_h)
        feet_vel_p2 = motion_p2[:, 1:, self.fids, :] - motion_p2[:, :-1, self.fids, :]
        pred_feet_vel_p2 = pred_motion_p2[:, 1:, self.fids, :] - pred_motion_p2[:, :-1, self.fids, :]
        feet_h_p2 = motion_p2[:, :-1, self.fids, 1]
        contact_p2 = self.cal_foot_contact(feet_vel_p2, feet_h_p2)
        fc_loss_p1 = self.Loss(pred_feet_vel_p1[contact_p1], torch.zeros_like(pred_feet_vel_p1)[contact_p1])
        fc_loss_p2 = self.Loss(pred_feet_vel_p2[contact_p2], torch.zeros_like(pred_feet_vel_p2)[contact_p2])
        fc_loss = (fc_loss_p1 + fc_loss_p2) / 2

        feet_h_pred_p1 = pred_motion_p1[:, :-1, self.fids, 1]
        contact_pred_p1 = self.cal_foot_contact(pred_feet_vel_p1, feet_h_pred_p1)
        feet_h_pred_p2 = pred_motion_p2[:, :-1, self.fids, 1]
        contact_pred_p2 = self.cal_foot_contact(pred_feet_vel_p2, feet_h_pred_p2)
        fc_loss_p1_pred = self.Loss(feet_vel_p1[contact_pred_p1], torch.zeros_like(feet_vel_p1)[contact_pred_p1])
        fc_loss_p2_pred = self.Loss(feet_vel_p2[contact_pred_p2], torch.zeros_like(feet_vel_p2)[contact_pred_p2])
        fc_loss += (fc_loss_p1_pred + fc_loss_p2_pred) / 2

        # Handle NaN values by replacing them with zero

        if torch.isnan(fc_loss_p1) or torch.isnan(fc_loss_p2):
            fc_loss = torch.tensor(0).to(motion_p1.device)

        return fc_loss

    def forward_relative_oriental(self):

        # recover global translation from root velocity

        pred_oriental = rotation_6d_to_matrix(self.motion_pred_de.reshape(self.motion_pred_de.shape[0], -1, 56, 2, 6))[
                        :, :, 0].reshape(-1, 2, 3, 3)

        gt_oriental = rotation_6d_to_matrix(self.motion_gt_de.reshape(self.motion_pred_de.shape[0], -1, 56, 2, 6))[:, :,
                      0].reshape(-1, 2, 3, 3)

        with torch.autocast(device_type=pred_oriental.device.type, enabled=False):
            pred_oriental, gt_oriental = pred_oriental.float(), gt_oriental.float()
            pred_rel_oriental = torch.bmm(pred_oriental[:, 0].transpose(1, 2), pred_oriental[:, 1])
            gt_rel_oriental = torch.bmm(gt_oriental[:, 0].transpose(1, 2), gt_oriental[:, 1])
            m = torch.bmm(gt_rel_oriental, pred_rel_oriental.transpose(1, 2))

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

        theta = torch.acos(torch.clamp(cos, -1 + 1e-7, 1 - 1e-7))

        return torch.mean(theta)

    def forward_distance_map(self):

        pred_distance_matrix = torch.cdist(self.motion_pred_de_joints_p1.contiguous(), self.motion_pred_de_joints_p2.contiguous())
        tgt_distance_matrix = torch.cdist(self.motion_gt_de_joints_p1.contiguous(), self.motion_gt_de_joints_p2.contiguous())
        distance_matrix_mask = (pred_distance_matrix < 1).float()
        loss = self.Loss(pred_distance_matrix * distance_matrix_mask, tgt_distance_matrix * distance_matrix_mask)

        ja_matrix_mask = (tgt_distance_matrix < 0.1).float()
        loss += self.Loss(pred_distance_matrix * ja_matrix_mask, tgt_distance_matrix * ja_matrix_mask)

        return loss

class ReConsLossSingle(nn.Module):
    def __init__(self, args, recons_loss, nb_joints, fk_layers):
        super(ReConsLossSingle, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        self.args = args

        self.normalizer = MotionNormalizerTorchInterXSingle()
        self.fk_pair_processors = fk_layers
        self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        self.motion_dim = nb_joints * 12 + 2 * 6
        self.fids = [7, 10, 8, 11]
        self.losses = {}

    def forward(self, motion_pred, motion_gt, beta, gender, group):

        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])

        self.losses['main'] = loss

        self.motion_pred_de = self.normalizer.backward(motion_pred).reshape(motion_pred.shape[0], -1, 56, 6)
        self.motion_gt_de = self.normalizer.backward(motion_gt).reshape(motion_pred.shape[0], -1, 56, 6)
        motion_pred_de_joints = self.fk_batch(self.motion_pred_de, group)
        motion_gt_de_joints = self.fk_batch(self.motion_gt_de, group)

        self.motion_gt_de_joints = self.normalize_floor(motion_gt_de_joints)
        self.motion_pred_de_joints = self.normalize_floor(motion_pred_de_joints)

        loss_position = self.forward_joint()  # This now populates self.losses['Joint']

        # and sets self.motion_* variables

        # Calculate other losses (they will use the self.motion_* variables set above)

        loss_root = self.forward_root()
        loss_vel = self.forward_vel()
        loss_rotation = self.forward_rotation()
        loss_fc = self.forward_foot_contact()

        self.losses['Joint'] = self.args.loss_explicit * loss_position
        self.losses['Root'] = self.args.loss_root * loss_root
        self.losses['Velocity'] = self.args.loss_vel * loss_vel
        self.losses['Rotation'] = self.args.loss_geo * loss_rotation
        self.losses['FootContact'] = self.args.loss_fc * loss_fc
        self.losses['DistanceMap'] = torch.tensor(0)

        return self.losses

    def fk_batch(self, motion_batch, group_ids_batch):

        final_coords = self.fk_pair_processors(motion_batch, group_ids_batch)
        return final_coords

    def forward_joint(self):

        loss = self.Loss(self.motion_pred_de_joints,
                          self.motion_gt_de_joints)
        return loss

    def forward_root(self):

        loss_orientation = self.Loss(self.motion_pred_de[..., 0, :], self.motion_gt_de[..., 0, :])
        trans = self.motion_gt_de_joints[..., 0, [0,2]]
        trans_pred = self.motion_pred_de_joints[..., 0, [0,2]]
        loss_trans = self.Loss(trans_pred, trans)
        loss_trans_vel = self.Loss(trans_pred[:, 1:] - trans_pred[:, :-1], trans[:, 1:] - trans[:, :-1])

        loss_trans = loss_trans + loss_trans_vel

        return loss_orientation + loss_trans

    def forward_vel(self):
        vel_pred = self.motion_pred_de_joints[:, 1:] - self.motion_pred_de_joints[:, :-1]
        vel_gt = self.motion_gt_de_joints[:, 1:] - self.motion_gt_de_joints[:, :-1]
        loss = self.Loss(vel_pred, vel_gt)
        return loss

    def forward_rotation(self, eps=1e-7):

        pred_m = rotation_6d_to_matrix(self.motion_pred_de.reshape(self.motion_pred_de.shape[0], -1, 56, 6))[:, :,
                 :-1].reshape(-1, 3, 3)
        gt_m = rotation_6d_to_matrix(self.motion_gt_de.reshape(self.motion_pred_de.shape[0], -1, 56, 6))[:, :,
               :-1].reshape(-1, 3, 3)
        with torch.autocast(device_type=pred_m.device.type, enabled=False):
            m = torch.bmm(gt_m.float(), pred_m.float().transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))
        return torch.mean(theta)

    def normalize_floor(self, positions: torch.Tensor) -> torch.Tensor:

        floor_heights = positions[..., 1].view(positions.shape[0], -1).min(dim=1).values
        normalized_positions = positions.clone()
        normalized_positions[..., 1] -= floor_heights.view(-1, 1, 1)
        return positions  # Returns modified tensor (in-place)

    def cal_foot_contact(self, feet_vel, feet_h):

        thres = 0.001
        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
            [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)
        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2
        contact = ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)
        return contact

    def forward_foot_contact(self):

        motion_p1 = self.normalize_floor(self.motion_gt_de_joints)
        pred_motion_p1 = self.normalize_floor(self.motion_pred_de_joints)
        feet_vel_p1 = motion_p1[:, 1:, self.fids, :] - motion_p1[:, :-1, self.fids, :]
        pred_feet_vel_p1 = pred_motion_p1[:, 1:, self.fids, :] - pred_motion_p1[:, :-1, self.fids, :]

        feet_h = motion_p1[:, :-1, self.fids, 1]
        contact_p1 = self.cal_foot_contact(feet_vel_p1, feet_h)

        fc_loss_p1 = self.Loss(pred_feet_vel_p1[contact_p1], torch.zeros_like(pred_feet_vel_p1)[contact_p1])
        fc_loss = fc_loss_p1

        feet_h_pred_p1 = pred_motion_p1[:, :-1, self.fids, 1]
        contact_pred_p1 = self.cal_foot_contact(pred_feet_vel_p1, feet_h_pred_p1)
        fc_loss_p1_pred = self.Loss(feet_vel_p1[contact_pred_p1], torch.zeros_like(feet_vel_p1)[contact_pred_p1])
        fc_loss += fc_loss_p1_pred

        # Handle NaN values by replacing them with zero

        if torch.isnan(fc_loss_p1):
            fc_loss = torch.tensor(0).to(motion_p1.device)

        return fc_loss
