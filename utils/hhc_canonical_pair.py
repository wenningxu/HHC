"""Differentiable joint-only equivalent of official InterHuman pair framing.

Input is the native FK's Y-up paired world joints. This is a training loss
coordinate transform, never a replacement for the released evaluator.
Official facing uses hips [2,1] only; listed shoulders are unused in source.
"""
import torch


def canonical_pair_joints(joints,anchor_person=1,output_person_swap=False):
    if joints.ndim!=5 or joints.shape[2:]!=(2,22,3) or joints.shape[1]<2:
        raise ValueError('Expected [B,T>=2,2,22,3] Y-up world joints')
    if anchor_person not in (1,2):raise ValueError('Anchor must be 1 or 2')
    anchor=anchor_person-1
    ordered=joints[:,:,[anchor,1-anchor]]
    floor=ordered[...,1].amin(dim=(1,3))
    shift=torch.stack([torch.zeros_like(floor),floor,torch.zeros_like(floor)],dim=-1)
    grounded=ordered-shift[:,None,:,None]
    center=grounded[:,0,0,0]*joints.new_tensor([1.,0.,1.])
    relative=grounded-center[:,None,None,None]
    across=ordered[:,0,0,2]-ordered[:,0,0,1]
    horizontal_norm=torch.linalg.vector_norm(across[:,[0,2]],dim=-1)
    if torch.any(horizontal_norm<=1e-8):raise ValueError('Degenerate horizontal hip facing')
    fx=across[:,2]/horizontal_norm;fz=-across[:,0]/horizontal_norm
    x,y,z=relative.unbind(-1)
    fx=fx[:,None,None,None];fz=fz[:,None,None,None]
    result=torch.stack([fz*x-fx*z,y,fx*x+fz*z],dim=-1)
    if output_person_swap:result=result[:,:,[1,0]]
    # Official features expose position at frames [0,T-2] and velocity [0,T-2].
    return result[:,:-1]


def canonical_pair_loss(predicted,target):
    """Equal mean of both possible anchors, in metres, SmoothL1 like RawLoss."""
    return sum(torch.nn.functional.smooth_l1_loss(canonical_pair_joints(predicted,anchor),
               canonical_pair_joints(target,anchor)) for anchor in (1,2))/2
