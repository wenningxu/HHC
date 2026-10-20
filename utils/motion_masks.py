import math

import numpy as np
import torch
import torch.nn.functional as F

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK


ALL_JOINT_FALSE = np.full(*HML_ROOT_BINARY.shape, False)
HML_UPPER_BODY_JOINTS_BINARY = np.array([i in SMPL_UPPER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])

UPPER_JOINT_Y_TRUE = np.array([ALL_JOINT_FALSE[1:], HML_UPPER_BODY_JOINTS_BINARY[1:], ALL_JOINT_FALSE[1:]])
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.T
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.reshape(ALL_JOINT_FALSE[1:].shape[0]*3)

UPPER_JOINT_Y_MASK = np.concatenate(([False]*(1+2+1),
                                UPPER_JOINT_Y_TRUE,
                                ALL_JOINT_FALSE[1:].repeat(6),
                                ALL_JOINT_FALSE.repeat(3),
                                [False] * 4))

HHI_ROOT_ONLY_MASK = (np.array([True]).repeat(12))
HHI_ROOT_ONLY_MASK_OLD = np.concatenate((np.array([True]).repeat(3), np.array([False]).repeat(3), np.array([True]).repeat(3), np.array([False]).repeat(3)))
HHI_BODY_ROOT_MASK = np.concatenate((np.array([True]).repeat(22*12), np.array([False]).repeat(33*12), HHI_ROOT_ONLY_MASK))
HHI_HAND_ROOT_MASK = np.concatenate((np.array([False]).repeat(25*12), np.array([True]).repeat(30*12), HHI_ROOT_ONLY_MASK))
HHI_BODY_MASK = np.concatenate((np.array([True]).repeat(22*12), np.array([False]).repeat(33*12), np.array([False]).repeat(12)))
HHI_HAND_MASK = np.concatenate((np.array([False]).repeat(25*12), np.array([True]).repeat(30*12), np.array([False]).repeat(12)))
HHI_ROOT_MASK = np.concatenate((np.array([False]).repeat(55*12), HHI_ROOT_ONLY_MASK))
HHI_ROOT_MASK_OLD = np.concatenate((np.array([False]).repeat(55*12), HHI_ROOT_ONLY_MASK_OLD))
HHI_WHOLE_MASK = np.concatenate((np.array([True]).repeat(22*12), np.array([False]).repeat(3*12), np.array([True]).repeat(30*12), HHI_ROOT_ONLY_MASK_OLD))

SMPLX_BODY_MASK = np.concatenate((np.array([True]).repeat(22*6), np.array([False]).repeat(33*6), np.array([False]).repeat(6)))
SMPLX_HAND_MASK = np.concatenate((np.array([False]).repeat(25*6), np.array([True]).repeat(30*6), np.array([False]).repeat(6)))
SMPLX_ROOT_MASK = np.concatenate((np.array([False]).repeat(55*6), np.array([True]).repeat(3), np.array([False]).repeat(3)))


# Token channels used by VQVAE_SEP3_shape and Text2Motion_Transformer.
BODY_TOKEN_CHANNEL = 0
HAND_TOKEN_CHANNEL = 1
ROOT_TOKEN_CHANNEL = 2


def sample_kinematic_cascade_mask(valid_mask, generator=None):
    """Sample the asymmetric Root -> Body -> Hand mask from the HHC paper.

    ``valid_mask`` is a boolean ``[B, T, 3]`` tensor in the tokenizer's
    ``[body, hand, root]`` channel order. Three independent schedule variables
    are sorted so that tau_root > tau_body > tau_hand and transformed by
    gamma(tau) = cos(pi*tau/2). Since gamma is decreasing, roots are visible
    most often and hands are masked most often.

    InterHuman uses two independent variables in ``[body, root]`` order,
    with tau_root >= tau_body (Root -> Body). The three-channel RNG order
    and equations remain unchanged.

    The paper specifies the continuous ratios but not how a fractional token
    count is discretized.  We retain the authors' HHC/MMM and InterMask source
    convention: round ``valid_length * gamma`` and keep at least one token
    masked for every non-empty part.
    """
    if valid_mask.ndim != 3 or valid_mask.shape[-1] not in (2, 3):
        raise ValueError(f"valid_mask must have shape [B, T, 2 or 3], got {tuple(valid_mask.shape)}")
    valid_mask = valid_mask.bool()
    batch_size, max_len, parts = valid_mask.shape

    raw_taus = torch.rand(batch_size, parts, device=valid_mask.device, generator=generator)
    if parts == 3:
        tau_hand, tau_body, tau_root = raw_taus.sort(dim=-1).values.unbind(dim=-1)
        taus = torch.stack((tau_body, tau_hand, tau_root), dim=-1)
    else:
        taus = raw_taus.sort(dim=-1).values  # body then root
    ratios = torch.cos(taus * (math.pi * 0.5))

    valid_lengths = valid_mask.sum(dim=1)
    num_masked = torch.round(valid_lengths.float() * ratios).long()
    num_masked = torch.minimum(num_masked, valid_lengths)
    num_masked = torch.where(valid_lengths > 0, num_masked.clamp(min=1), num_masked)

    scores = torch.rand(batch_size, max_len, parts, device=valid_mask.device, generator=generator)
    scores = scores.masked_fill(~valid_mask, float("inf"))
    ranks = scores.argsort(dim=1).argsort(dim=1)
    mask = ranks < num_masked.unsqueeze(1)
    return mask & valid_mask, ratios, taus


def masked_token_nll(logits, target, mask):
    """Mean per-sample NLL over masked codebook tokens only.

    Token datasets retain out-of-vocabulary END/PAD IDs outside ``mask``.
    Replace those ignored targets before calling cross entropy, because a zero
    loss weight cannot prevent PyTorch from validating an invalid class ID.
    """
    if logits.shape[:-1] != target.shape or target.shape != mask.shape:
        raise ValueError(
            f'incompatible masked NLL shapes: logits={tuple(logits.shape)}, '
            f'target={tuple(target.shape)}, mask={tuple(mask.shape)}'
        )
    mask = mask.bool()
    safe_target = target.masked_fill(~mask, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), safe_target.reshape(-1),
        reduction='none',
    ).view_as(target)
    return (
        token_loss.masked_fill(~mask, 0.0).sum(dim=tuple(range(1, target.ndim)))
        / mask.sum(dim=tuple(range(1, mask.ndim))).clamp(min=1)
    ).mean()


def global_masked_token_nll(logits, target, mask):
    """Global mean NLL over masked tokens, matching released InterMask.

    This differs from :func:`masked_token_nll` only when samples have unequal
    masked-token counts. It remains safe for out-of-vocabulary END/PAD targets
    because targets outside ``mask`` are replaced before cross entropy.
    """
    if logits.shape[:-1] != target.shape or target.shape != mask.shape:
        raise ValueError(
            f'incompatible masked NLL shapes: logits={tuple(logits.shape)}, '
            f'target={tuple(target.shape)}, mask={tuple(mask.shape)}'
        )
    mask = mask.bool()
    safe_target = target.masked_fill(~mask, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), safe_target.reshape(-1),
        reduction='none',
    ).view_as(target)
    return token_loss.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)


def sample_historical_mmm_corruption(
        target, valid_mask, num_codes, pkeep=0.5, generator=None):
    """Reproduce the pre-cleanup MMM training corruption in a diagnostic arm.

    This is deliberately not the paper mask. It retains the historical random
    token replacement, single cosine ratio, channel bias, and one-sided
    ``argsort < count`` expression so the known-good training contract can be
    isolated before its settings are migrated one at a time.
    """
    if target.shape != valid_mask.shape or target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(
            f'target/valid_mask must both be [B,T,3], got '
            f'{tuple(target.shape)} and {tuple(valid_mask.shape)}'
        )
    valid_mask = valid_mask.bool()
    keep_probability = (
        float(np.random.rand(1)[0]) if pkeep == -1 else float(pkeep)
    )
    keep = torch.bernoulli(
        torch.full(target.shape, keep_probability, device=target.device),
        generator=generator,
    ).bool()
    keep = keep | ~valid_mask
    random_indices = torch.randint(
        num_codes, target.shape, dtype=target.dtype, device=target.device,
        generator=generator,
    )
    corrupted = torch.where(keep, target, random_indices)

    batch_size, max_len, _ = target.shape
    rand_time = torch.rand(batch_size, device=target.device, generator=generator)
    ratios = torch.cos(rand_time * math.pi * 0.5).clamp(min=0.1, max=1.0)
    valid_lengths = valid_mask.sum(dim=(1, 2))
    num_masked = (valid_lengths.float() * ratios).round().clamp(min=1).long()
    scores = torch.rand(
        batch_size, max_len, 3, device=target.device, generator=generator,
    )
    scores = scores + torch.tensor(
        [0.5, 1.0, 0.0], device=target.device, dtype=scores.dtype,
    )
    scores = scores + (~valid_mask).to(scores.dtype) * 1e9
    historical_argsort = scores.view(batch_size, -1).argsort(dim=-1)
    mask = (historical_argsort < num_masked[:, None]).view_as(target)
    masked_input = torch.where(mask, int(num_codes) + 2, corrupted)
    return masked_input, mask, ratios


def sample_historical_mmm_corruption_with_kinematic_ratios(
        target, valid_mask, num_codes, pkeep=0.5, generator=None):
    """Change only the historical mask-ratio rule to the HHC paper rule.

    This is the causal performance contract requested for the HHC
    reproduction.  It deliberately preserves the historical MMM/HHC
    BERT-style random-token corruption before masking and changes only the
    mask schedule/count allocation: one sorted cosine ratio per kinematic
    channel instead of one shared flat ratio plus fixed channel-score biases.

    Everything outside this function (all-valid-token loss, model topology,
    optimizer, token window, conditioning dropout and validation/RNG
    behavior) remains under the historical training contract.
    """
    if target.shape != valid_mask.shape or target.ndim != 3 or target.shape[-1] not in (2, 3):
        raise ValueError(
            f'target/valid_mask must both be [B,T,2 or 3], got '
            f'{tuple(target.shape)} and {tuple(valid_mask.shape)}'
        )
    valid_mask = valid_mask.bool()
    keep_probability = (
        float(np.random.rand(1)[0]) if pkeep == -1 else float(pkeep)
    )
    keep = torch.bernoulli(
        torch.full(target.shape, keep_probability, device=target.device),
        generator=generator,
    ).bool()
    keep = keep | ~valid_mask
    random_indices = torch.randint(
        num_codes, target.shape, dtype=target.dtype, device=target.device,
        generator=generator,
    )
    corrupted = torch.where(keep, target, random_indices)

    mask, ratios, taus = sample_kinematic_cascade_mask(
        valid_mask, generator=generator
    )
    masked_input = torch.where(mask, int(num_codes) + 2, corrupted)
    return masked_input, mask, ratios, taus


def all_valid_token_nll(logits, target, valid_mask):
    """Historical MMM's exact all-valid-token NLL reduction.

    Keep the recovered operation order, including weighting each selected token
    before a single batch-wide sum.  Rewriting this as per-sample sums followed
    by ``mean`` is algebraically equivalent but changes floating-point reduction
    and gradients for unequal sequence lengths, which drifts an exact replay.
    """
    if logits.shape[:-1] != target.shape or target.shape != valid_mask.shape:
        raise ValueError(
            f'incompatible all-valid NLL shapes: logits={tuple(logits.shape)}, '
            f'target={tuple(target.shape)}, valid={tuple(valid_mask.shape)}'
        )
    valid_mask = valid_mask.bool()
    batch_size = target.shape[0]
    weights = valid_mask.float()
    sample_valid_tokens = valid_mask.sum(dim=(1, 2), keepdim=True)
    weights = weights / sample_valid_tokens.clamp(min=1)

    selected_logits = logits[valid_mask].reshape(-1, logits.shape[-1])
    selected_target = target[valid_mask].reshape(-1)
    selected_weights = weights[valid_mask].reshape(-1)
    token_loss = F.cross_entropy(
        selected_logits, selected_target, reduction='none'
    )
    return (token_loss * selected_weights).sum() / batch_size
