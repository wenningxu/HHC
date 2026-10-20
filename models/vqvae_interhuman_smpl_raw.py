"""Inter-X raw contract with hands removed: 22 paired rotations + root row.

Root is pelvis XYZ in Y-up coordinates; unused velocity slots are zero.
Local rotations remain SMPL local rotations. Only global orientation changes
world basis. No per-person floor/heading alignment occurs here.
"""
import numpy as np
import torch
from torch import nn
from smplx.lbs import batch_rigid_transform

from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset
from models.vqvae_sep3_shape import VQVAE_SEP3_shape
from tools.reconstruct_interhuman_raw_body import load_model
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d
from utils.losses import ReConsLossInter


class NativeSMPLFK(nn.Module):
    def __init__(self):
        super().__init__()
        model = load_model()
        self.register_buffer('rest0', torch.einsum('jv,vc->jc', model.J_regressor, model.v_template))
        self.register_buffer('rest_shape', torch.einsum('jv,vck->jck', model.J_regressor, model.shapedirs))
        self.register_buffer('parents', model.parents)
        self.register_buffer('basis', torch.tensor([[1.,0,0],[0,0,1],[0,-1,0]]))

    def rest(self, beta):
        return self.rest0 + torch.einsum('...k,jck->...jc', beta, self.rest_shape)

    def from_archive(self, motion, beta):
        """[T,23,12] native translation → Inter-X raw pelvis, both actors."""
        result = motion.clone()
        root = rotation_6d_to_matrix(motion[:, 0].reshape(-1, 2, 6))
        result[:, 0] = matrix_to_rotation_6d(self.basis @ root).reshape(-1,12)
        rest = self.rest(beta.reshape(2,10))[:,0]
        for p in range(2):
            result[:,22,p*6:p*6+3] = (motion[:,22,p*6:p*6+3]+rest[p]) @ self.basis.T
            result[:,22,p*6+3:p*6+6] = 0
        return result

    def to_archive(self, motion, beta):
        result = motion.clone()
        root = rotation_6d_to_matrix(motion[:,0].reshape(-1,2,6))
        result[:,0] = matrix_to_rotation_6d(self.basis.T @ root).reshape(-1,12)
        rest = self.rest(beta.reshape(2,10))[:,0]
        for p in range(2):
            result[:,22,p*6:p*6+3] = motion[:,22,p*6:p*6+3] @ self.basis-rest[p]
            result[:,22,p*6+3:p*6+6] = 0
        return result

    def forward(self, motion, beta):
        batch, frames = motion.shape[:2]
        rotations = rotation_6d_to_matrix(motion[:,:,:22].reshape(batch,frames,22,2,6))
        rotations = rotations.permute(0,1,3,2,4,5).reshape(-1,22,3,3)
        eye = torch.eye(3, device=motion.device, dtype=motion.dtype)
        rotations = torch.cat([rotations, eye.expand(len(rotations),2,3,3)], dim=1)
        rest = self.rest(beta.reshape(batch,2,10))
        rest = (rest-rest[:,:,:1])[:,None].expand(-1,frames,-1,-1,-1).reshape(-1,24,3)
        joints, _ = batch_rigid_transform(rotations, rest, self.parents)
        joints = joints[:,:22].reshape(batch,frames,2,22,3)
        trans = torch.stack([motion[:,:,22,:3], motion[:,:,22,6:9]], dim=2)
        joints = joints+trans[:,:,:,None]
        return joints[:,:,0], joints[:,:,1]


class VQSMPLRaw(nn.Module):
    """Original modules and shape conditioning; hand channel removed, not merged.

    Original code_dim=256 allocated body96/hand96/root64. Keep body96/root64;
    decoder has160+8 channels. Do not silently double body capacity.
    """
    channel_names = ('body', 'root')
    mask_shape = VQVAE_SEP3_shape.mask_shape

    def __init__(self, args):
        super().__init__()
        self.num_code = args.nb_code
        self.temporal_downsample = 4
        self.shape_mask_prob = .1
        self.gender_embedding = nn.Embedding(2,4)
        self.shape_projection = nn.Linear(20,40)
        self.condition_mlp = nn.Sequential(nn.Linear(48,64),nn.ReLU(),nn.Linear(64,8))
        kw = dict(down_t=2,stride_t=2,width=512,depth=3,dilation_growth_rate=3,activation='relu',norm=None)
        self.encoder_upper = Encoder(264,96,**kw)
        self.encoder_root = Encoder(6,64,**kw)
        self.quantizer_upper = QuantizeEMAReset(args.nb_code,96,args)
        self.quantizer_root = QuantizeEMAReset(args.nb_code,64,args)
        self.decoder = Decoder(270,168,**kw)

    def forward(self, x, beta, gender, type='full'):
        if type == 'decode':
            latents = [q.dequantize(x[...,i]).transpose(1,2) for i,q in enumerate((self.quantizer_upper,self.quantizer_root))]
        else:
            assert x.ndim == 3 and x.shape[-1] == 276 and x.shape[1]%4 == 0
            latents = [self.encoder_upper(x[...,:264].transpose(1,2)),
                       self.encoder_root(torch.cat([x[...,264:267],x[...,270:273]],dim=-1).transpose(1,2))]
            if type == 'encode':
                return torch.stack([q.quantize(z.transpose(1,2).reshape(-1,z.shape[1])).reshape(x.shape[0],-1)
                                    for z,q in zip(latents,(self.quantizer_upper,self.quantizer_root))],dim=-1)
            if type != 'full':
                raise ValueError(type)
            values = [q(z) for z,q in zip(latents,(self.quantizer_upper,self.quantizer_root))]
            latents = [v[0] for v in values]
        condition = self.condition_mlp(torch.cat([self.gender_embedding(gender).reshape(-1,8),
                                                   self.shape_projection(self.mask_shape(beta))],dim=-1))
        decoded = self.decoder(torch.cat([*latents,condition[:,:,None].expand(-1,-1,latents[0].shape[-1])],dim=1)).transpose(1,2)
        zero = torch.zeros_like(decoded[...,:3])
        output = torch.cat([decoded[...,:267],zero,decoded[...,267:],zero],dim=-1)
        if type == 'decode':
            return output
        return output, sum(v[1] for v in values), torch.stack([v[2] for v in values])


class RawNormalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)

    def backward(self, x):
        return x*self.std+self.mean


class RawLoss(ReConsLossInter):
    """Reuse original loss equations; replace only 55-joint reshape and FK.

    Empty contact selections return zero, not NaN; nonempty reductions are
    identical to the original. No floor normalization 'fix' is applied.
    """
    def __init__(self, args, fk, normalizer):
        super().__init__(args,'l1_smooth',22,fk,normalizer)

    def forward(self, pred, target, beta, gender=None, group=None):
        self.motion_pred_de = self.normalizer.backward(pred).reshape(*pred.shape[:2],23,12)
        self.motion_gt_de = self.normalizer.backward(target).reshape(*target.shape[:2],23,12)
        self.motion_pred_de_joints_p1,self.motion_pred_de_joints_p2 = self.fk_pair_processors(self.motion_pred_de,beta)
        self.motion_gt_de_joints_p1,self.motion_gt_de_joints_p2 = self.fk_pair_processors(self.motion_gt_de,beta)
        return {'main': self.Loss(pred,target), 'Joint':3*self.forward_joint(),
                'Root':.1*self.forward_root(), 'Velocity':100*self.forward_vel(),
                'Rotation':.01*self.forward_rotation(), 'FootContact':10*self.forward_foot_contact(),
                'DistanceMap':.1*self.forward_distance_map()}

    def forward_rotation(self, eps=1e-7):
        pred = rotation_6d_to_matrix(self.motion_pred_de[:,:,:22].reshape(-1,6))
        gt = rotation_6d_to_matrix(self.motion_gt_de[:,:,:22].reshape(-1,6))
        product = gt @ pred.transpose(-1,-2)
        cosine = (product.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        return torch.acos(cosine.clamp(-1+eps,1-eps)).mean()

    def forward_foot_contact(self):
        total = self.motion_pred_de.sum()*0
        for gt,pred in ((self.motion_gt_de_joints_p1,self.motion_pred_de_joints_p1),
                        (self.motion_gt_de_joints_p2,self.motion_pred_de_joints_p2)):
            vg = gt[:,1:,self.fids]-gt[:,:-1,self.fids]
            vp = pred[:,1:,self.fids]-pred[:,:-1,self.fids]
            cg = self.cal_foot_contact(vg,gt[:,:-1,self.fids,1])
            cp = self.cal_foot_contact(vp,pred[:,:-1,self.fids,1])
            for velocity,mask in ((vp,cg),(vg,cp)):
                if mask.any():
                    total = total+.5*self.Loss(velocity[mask],torch.zeros_like(velocity[mask]))
        return total
