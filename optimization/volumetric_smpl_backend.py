"""Differentiable SMPL-X signed distances using the paper's VolumetricSMPL."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import smplx
from VolumetricSMPL import attach_volume


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class VolumetricSMPLBackend(nn.Module):
    """Own frozen male/female SMPL-X volume models and expose SDF queries."""

    # Inter-X stores the complete 55-joint axis-angle pose around a flat-hand
    # zero pose.  human_body_prior.BodyModel, which supplies the contact and
    # auxiliary-loss meshes, uses the same convention.  smplx.create defaults
    # to adding the MANO hand mean; leaving that default enabled makes the SDF
    # mesh differ from the contact mesh by up to centimetres even for identical
    # parameters.
    FLAT_HAND_MEAN = True

    def __init__(
        self,
        model_root='body_models',
        device='cuda',
        genders=('male', 'female'),
        max_query_points_per_call=1024,
        body_encoding_samples=None,
    ):
        super().__init__()
        if max_query_points_per_call < 1:
            raise ValueError('max_query_points_per_call must be positive')
        self.device = torch.device(device)
        self.max_query_points_per_call = int(max_query_points_per_call)
        self.models = nn.ModuleDict()
        self.checkpoint_records = {}
        checkpoint_root = Path(torch.hub.get_dir()) / 'checkpoints'
        for gender in genders:
            if gender not in ('male', 'female'):
                raise ValueError(
                    'released VolumetricSMPL weights are required per gender; '
                    f'unsupported gender {gender!r}'
                )
            model = smplx.create(
                model_root,
                model_type='smplx',
                gender=gender,
                use_pca=False,
                flat_hand_mean=self.FLAT_HAND_MEAN,
                num_betas=10,
                batch_size=1,
            )
            model = attach_volume(
                model, pretrained=True, device=self.device
            ).eval()
            if body_encoding_samples is not None:
                if int(body_encoding_samples) < 1:
                    raise ValueError('body_encoding_samples must be positive')
                model.volume.n_samples = int(body_encoding_samples)
            if not bool(model.flat_hand_mean):
                raise RuntimeError(
                    'VolumetricSMPL body must use the Inter-X flat-hand pose '
                    'convention'
                )
            model.requires_grad_(False)
            self.models[gender] = model
            checkpoint = checkpoint_root / (
                f'VolumetricSMPL_smplx_{gender}.ckpt'
            )
            self.checkpoint_records[gender] = {
                'path': str(checkpoint.resolve()),
                'bytes': checkpoint.stat().st_size,
                'sha256': sha256_file(checkpoint),
            }
        self.representation_contract = {
            'dataset': 'Inter-X',
            'pose_layout': '55-joint axis-angle',
            'hand_pose_convention': 'flat_hand_mean',
            'smplx_flat_hand_mean': True,
            'volumetric_body_encoding_samples': int(
                next(iter(self.models.values())).volume.n_samples
            ),
            'contact_mesh_backend': 'human_body_prior.BodyModel',
            'required_geometry_relation': (
                'VolumetricSMPL and contact-loss meshes must be identical for '
                'the same pose, shape, gender and translation'
            ),
        }

    @staticmethod
    def normalize_gender(gender):
        if isinstance(gender, torch.Tensor):
            if gender.numel() != 1:
                raise ValueError('optimization requires one gender per person')
            gender = int(gender.detach().cpu().item())
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8')
        if gender in (0, 'male'):
            return 'male'
        if gender in (1, 'female'):
            return 'female'
        raise ValueError(
            'paper VolumetricSMPL optimization requires binary male/female '
            f'gender weights, got {gender!r}'
        )

    def begin_iteration(self, seed=None):
        """Start a reproducible iteration and discard stale target encodings."""
        if seed is not None:
            torch.manual_seed(int(seed))
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(int(seed))
        for model in self.models.values():
            model.volume.impl_code = None

    @staticmethod
    def detached_output(output):
        """Detach the three fields consumed by VolumetricSMPL's body encoder."""
        return SimpleNamespace(
            vertices=output.vertices.detach(),
            joints=output.joints.detach(),
            full_pose=output.full_pose.detach(),
        )

    def body(self, params):
        gender = self.normalize_gender(params['gender'])
        model = self.models[gender]
        frame_count = params['pose_body'].shape[0]
        betas = params['betas']
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        if betas.shape[0] == 1:
            betas = betas.expand(frame_count, -1)
        if betas.shape != (frame_count, 10):
            raise ValueError(
                f'expected betas [{frame_count},10], got {tuple(betas.shape)}'
            )
        # smplx.create(batch_size=1) registers default face/expression tensors
        # with a singleton batch.  Passing all non-optimized components
        # explicitly is required when a complete motion is evaluated at once.
        zeros3 = torch.zeros(
            frame_count,
            3,
            device=params['pose_body'].device,
            dtype=params['pose_body'].dtype,
        )
        expression = torch.zeros(
            frame_count,
            model.num_expression_coeffs,
            device=zeros3.device,
            dtype=zeros3.dtype,
        )
        output = model(
            body_pose=params['pose_body'].reshape(frame_count, 63),
            left_hand_pose=params['pose_lhand'].reshape(frame_count, 45),
            right_hand_pose=params['pose_rhand'].reshape(frame_count, 45),
            global_orient=params['root_orient'].reshape(frame_count, 3),
            transl=params['trans'].reshape(frame_count, 3),
            betas=betas,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=expression,
            return_full_pose=True,
        )
        return output, gender

    def signed_distance(self, query_points, target_output, target_gender):
        """Query target SDF in point chunks while retaining all gradients."""
        if query_points.ndim != 3 or query_points.shape[-1] != 3:
            raise ValueError(
                f'query points must be [B,N,3], got {tuple(query_points.shape)}'
            )
        if query_points.shape[0] != target_output.vertices.shape[0]:
            raise ValueError('query and target frame batches must match')
        model = self.models[self.normalize_gender(target_gender)]
        target_tensors = (
            target_output.vertices,
            target_output.joints,
            target_output.full_pose,
        )
        source_only_grad = (
            torch.is_grad_enabled()
            and query_points.requires_grad
            and not any(tensor.requires_grad for tensor in target_tensors)
        )
        cached_code = None
        if source_only_grad:
            # Eq. 4 detaches the paired character.  Its body encoding can be
            # frozen once while checkpointing only the source-query network.
            model.volume.impl_code = None
            with torch.no_grad():
                model.volume.encode_body(target_output)
            cached_code = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in model.volume.impl_code.items()
            }

        def query_chunk(points):
            if cached_code is not None:
                model.volume.impl_code = cached_code
                return model.volume.query(points)
            return model.volume.query(points, target_output)

        chunks = []
        for start in range(0, query_points.shape[1], self.max_query_points_per_call):
            end = min(
                start + self.max_query_points_per_call,
                query_points.shape[1],
            )
            points = query_points[:, start:end]
            if source_only_grad:
                chunks.append(checkpoint(
                    query_chunk,
                    points,
                    use_reentrant=True,
                    preserve_rng_state=True,
                ))
            else:
                chunks.append(query_chunk(points))
        result = torch.cat(chunks, dim=1)
        if result.shape != query_points.shape[:2]:
            raise RuntimeError(
                f'unexpected VolumetricSMPL SDF shape: {tuple(result.shape)}'
            )
        return result

    @staticmethod
    def _slice_output(output, frame_slice):
        return SimpleNamespace(
            vertices=output.vertices[frame_slice],
            joints=output.joints[frame_slice],
            full_pose=output.full_pose[frame_slice],
        )

    def _query_frame(self, query_points, target_output, target_gender):
        """Query one frame, checkpointing the neural SDF during optimization."""
        model = self.models[self.normalize_gender(target_gender)]

        def query_function(points, vertices, joints, full_pose):
            # The volume object owns a mutable encoding cache.  Rebuilding it
            # here makes activation checkpoint recomputation independent of
            # whichever body/frame was queried most recently.
            model.volume.impl_code = None
            output = SimpleNamespace(
                vertices=vertices,
                joints=joints,
                full_pose=full_pose,
            )
            chunks = []
            for start in range(
                0, points.shape[1], self.max_query_points_per_call
            ):
                end = min(
                    start + self.max_query_points_per_call,
                    points.shape[1],
                )
                chunks.append(model.volume.query(points[:, start:end], output))
            return torch.cat(chunks, dim=1)

        inputs = (
            query_points,
            target_output.vertices,
            target_output.joints,
            target_output.full_pose,
        )
        needs_grad = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in inputs
        )
        if needs_grad:
            return checkpoint(
                query_function,
                *inputs,
                use_reentrant=True,
                preserve_rng_state=True,
            )
        return query_function(*inputs)

    def thresholded_penetration_squared(
        self,
        query_vertices,
        target_output,
        target_gender,
        temporal_mask,
        shallow_threshold,
        interaction_threshold,
        weight,
        frame_batch_size=1,
    ):
        """Return the current MMM penetration numerator and active count.

        This changes only the signed-distance evaluator. Thresholding, the
        per-direction active-vertex count, and the caller's reduction remain
        the current MMM implementation's operations.
        """
        frame_batch_size = int(frame_batch_size)
        if frame_batch_size > 1:
            from optimization.volumetric_smpl_batching import (
                penetration_squared_sum_frame_batched,
            )
            total, active_count = penetration_squared_sum_frame_batched(
                self,
                query_vertices,
                target_output,
                target_gender,
                temporal_mask,
                shallow_threshold=shallow_threshold,
                interaction_threshold=interaction_threshold,
                frame_batch_size=frame_batch_size,
                return_active_count=True,
            )
            return float(weight) * total, active_count
        if frame_batch_size < 1:
            raise ValueError('frame_batch_size must be positive')

        frame_count = query_vertices.shape[0]
        if temporal_mask is None:
            temporal_mask = torch.zeros(
                frame_count,
                device=query_vertices.device,
                dtype=query_vertices.dtype,
            )
        else:
            temporal_mask = temporal_mask.reshape(-1)
        if temporal_mask.shape[0] != frame_count:
            raise ValueError('temporal mask and body frame counts must match')

        total = query_vertices.sum() * 0.0
        active_count = torch.zeros((), dtype=torch.long, device=query_vertices.device)
        model = self.models[self.normalize_gender(target_gender)]
        padding_margin_ratio = (float(model.volume.bbox_padding) - 1.0) * 0.5
        for frame_index in range(frame_count):
            query_frame = query_vertices[frame_index]
            target_vertices = target_output.vertices[frame_index].detach()
            raw_min = target_vertices.amin(dim=0)
            raw_max = target_vertices.amax(dim=0)
            margin = (raw_max - raw_min) * padding_margin_ratio
            candidate_mask = (
                (query_frame.detach() >= raw_min - margin)
                & (query_frame.detach() <= raw_max + margin)
            ).all(dim=-1)
            if not bool(candidate_mask.any()):
                continue
            candidates = query_frame[candidate_mask].unsqueeze(0)
            target_frame = self._slice_output(
                target_output, slice(frame_index, frame_index + 1)
            )
            signed_distance = self._query_frame(
                candidates, target_frame, target_gender
            ).squeeze(0)
            threshold = torch.where(
                temporal_mask[frame_index] > 0.1,
                signed_distance.new_tensor(interaction_threshold),
                signed_distance.new_tensor(shallow_threshold),
            )
            active = (
                (signed_distance < 0)
                & (signed_distance.abs() < threshold)
            )
            total = total + float(weight) * (signed_distance[active] ** 2).sum()
            active_count = active_count + active.sum()
        return total, active_count

    def penetration_squared_sum(
        self,
        query_vertices,
        target_output,
        target_gender,
        temporal_mask,
        shallow_threshold=0.05,
        interaction_threshold=10.0,
        frame_batch_size=1,
    ):
        """Compute paper Eq. 6 exactly while bounding activation memory.

        Points outside the target's global vertex AABB cannot be inside any
        VolumetricSMPL part box, so excluding them does not change the active
        penetration set.  Each remaining frame is activation-checkpointed.
        """
        frame_batch_size = int(frame_batch_size)
        if frame_batch_size < 1:
            raise ValueError('frame_batch_size must be positive')
        if frame_batch_size > 1:
            from optimization.volumetric_smpl_batching import (
                penetration_squared_sum_frame_batched,
            )
            return penetration_squared_sum_frame_batched(
                self,
                query_vertices,
                target_output,
                target_gender,
                temporal_mask,
                shallow_threshold=shallow_threshold,
                interaction_threshold=interaction_threshold,
                frame_batch_size=frame_batch_size,
            )
        if query_vertices.ndim != 3 or query_vertices.shape[-1] != 3:
            raise ValueError('query vertices must have shape [T,V,3]')
        frame_count = query_vertices.shape[0]
        if target_output.vertices.shape[0] != frame_count:
            raise ValueError('query and target frame counts must match')
        if temporal_mask is None:
            temporal_mask = torch.zeros(
                frame_count,
                device=query_vertices.device,
                dtype=query_vertices.dtype,
            )
        else:
            temporal_mask = temporal_mask.reshape(-1)
            if temporal_mask.shape[0] != frame_count:
                raise ValueError('temporal mask and body frame counts must match')

        total = query_vertices.sum() * 0.0
        model = self.models[self.normalize_gender(target_gender)]
        padding_margin_ratio = (float(model.volume.bbox_padding) - 1.0) * 0.5
        for frame_index in range(frame_count):
            query_frame = query_vertices[frame_index]
            target_vertices = target_output.vertices[frame_index].detach()
            raw_min = target_vertices.amin(dim=0)
            raw_max = target_vertices.amax(dim=0)
            # Every part box is formed from a subset of these vertices and is
            # enlarged by bbox_padding around its own center.  Expanding the
            # global box by the same worst-case margin therefore contains all
            # padded part boxes; outside it query_fast is analytically positive.
            margin = (raw_max - raw_min) * padding_margin_ratio
            bbox_min = raw_min - margin
            bbox_max = raw_max + margin
            candidate_mask = (
                (query_frame.detach() >= bbox_min)
                & (query_frame.detach() <= bbox_max)
            ).all(dim=-1)
            if not bool(candidate_mask.any()):
                continue
            candidates = query_frame[candidate_mask].unsqueeze(0)
            target_frame = self._slice_output(
                target_output, slice(frame_index, frame_index + 1)
            )
            signed_distance = self._query_frame(
                candidates, target_frame, target_gender
            ).squeeze(0)
            threshold = torch.where(
                temporal_mask[frame_index] > 0.1,
                signed_distance.new_tensor(interaction_threshold),
                signed_distance.new_tensor(shallow_threshold),
            )
            active = (
                (signed_distance < 0)
                & (signed_distance.abs() < threshold)
            )
            total = total + (signed_distance[active] ** 2).sum()
        return total
