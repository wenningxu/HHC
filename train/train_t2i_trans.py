import os
import sys

# Allow running as `python train/train_t2i_trans.py` from repo root by putting
# the repo root on sys.path before importing project packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
import h5py
import pytorch_lightning as pl
from os.path import join as pjoin
import clip
import math
import time
import json
import random
import hashlib
import datetime
import subprocess
import shutil
import re
from contextlib import contextmanager, nullcontext

import options.option_transformer as option_trans
from models.model_tools import build_model
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_interx_TM, dataset_interx_tokenize
from dataset import (
    dataset_interx_tokenize_historical_root,
)
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import wandb
from utils.runtime import generate_src_mask, init_save_folder, preserving_rng_state
from utils.training_telemetry import (
    ThroughputMeter,
    gpu_peak_memory_gib,
    parameter_and_gradient_telemetry,
)
from utils.motion_masks import (
    all_valid_token_nll,
    global_masked_token_nll,
    masked_token_nll,
    sample_historical_mmm_corruption,
    sample_historical_mmm_corruption_with_kinematic_ratios,
    sample_kinematic_cascade_mask,
)
from utils.contact_map import build_code_contact_map
from utils.interx_shape_augmentation import (
    validate_materialized_interx_shape_augmentation,
)
from utils.hybrid_shape_token_cache import (
    TOKEN_CACHE_CONTRACT as HYBRID_SHAPE_TOKEN_CACHE_CONTRACT,
    audit_hybrid_cache_subsets,
    import_original_pairs_byte_exact,
    partition_expected_names,
)
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from utils.word_vectorizer import WordVectorizer
from lightning.pytorch.loggers import WandbLogger
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from utils.eval_trans import downsample_contact_mask


_PROCESS_START_TIME = datetime.datetime.now().astimezone().isoformat(timespec='seconds')


@contextmanager
def temporary_inference_conditioning_contract(transformer, contract):
    """Use final CFG semantics during observational validation, then restore."""
    targets = [transformer]
    if hasattr(transformer, 'trans_base'):
        targets.append(transformer.trans_base)
    previous = []
    for target in targets:
        if hasattr(target, 'conditioning_contract'):
            previous.append((target, target.conditioning_contract))
            target.conditioning_contract = contract
    try:
        yield
    finally:
        for target, value in previous:
            target.conditioning_contract = value

MGT_TRAINING_SOURCE_PATHS = (
    'train/train_t2i_trans.py',
    'options/option_transformer.py',
    'dataset/dataset_interx_TM.py',
    'dataset/dataset_interx_tokenize.py',
    'dataset/dataset_interx_tokenize_historical_root.py',
    'models/model_tools.py',
    'models/motion_transformer.py',
    'models/two_person_transformer.py',
    'models/vqvae_sep3_shape.py',
    'utils/contact_map.py',
    'utils/interx_shape_augmentation.py',
    'utils/hybrid_shape_token_cache.py',
    'utils/eval_trans.py',
    'utils/motion_masks.py',
    'utils/motion_process.py',
    'utils/runtime.py',
    'utils/training_telemetry.py',
)
MGT_RESUME_IDENTITY_KEYS = (
    'exp_name', 'vq_checkpoint', 'cache_root', 'codebook_dir',
    'train_asset_root', 'augmentation_manifest', 'train_split_file',
    'val_split_file', 'all_split_file', 'train_motion_file',
    'val_motion_file', 'all_motion_file', 'stats_dir', 'stats_prefix',
    'nb_code', 'code_dim', 'shape_condition_dim', 'width',
    'embed_dim_gpt', 'n_head_gpt', 'num_layers', 'num_local_layer',
    'batch_size', 'total_iter', 'warm_up_iter', 'lr', 'weight_decay',
    'contact', 'contact_conditioning', 'shape', 'paper_architecture',
    'training_mask_contract', 'training_loss_contract',
    'optimizer_contract', 'conditioning_contract', 'text_drop_prob',
    'training_token_length_contract', 'training_validation_contract',
    'sampler', 'temperature', 'sampling_steps', 'cfg_scale',
    'contact_guidance_scale', 'seed', 'wandb_mode',
    'token_cache_contract', 'validation_mm_repeats',
)
MGT_RESUME_PATH_KEYS = {
    'vq_checkpoint', 'cache_root', 'codebook_dir', 'train_asset_root',
    'augmentation_manifest', 'train_split_file', 'val_split_file',
    'all_split_file', 'train_motion_file', 'val_motion_file',
    'all_motion_file', 'stats_dir',
}


class FixedStepValidationPairCheckpoint(pl.Callback):
    """Persist one validation pair that rolling/top-k cleanup cannot delete."""

    def __init__(self, directory, step=20000):
        super().__init__()
        self.directory = os.path.abspath(directory)
        self.step = int(step)
        self.pre_path = os.path.join(
            self.directory, f'prevalidation-{self.step:09d}.ckpt'
        )
        self.post_path = os.path.join(
            self.directory, f'postvalidation-{self.step:09d}.ckpt'
        )
        self.pre_saved = False
        self.post_saved = False

    def state_dict(self):
        return {
            'step': self.step,
            'pre_saved': self.pre_saved,
            'post_saved': self.post_saved,
        }

    def load_state_dict(self, state_dict):
        if int(state_dict.get('step', self.step)) != self.step:
            raise ValueError('fixed validation-pair checkpoint step changed')
        self.pre_saved = bool(state_dict.get('pre_saved', False))
        self.post_saved = bool(state_dict.get('post_saved', False))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step != self.step or os.path.isfile(self.pre_path):
            return
        if not trainer.is_global_zero:
            return
        rolling = os.path.join(
            self.directory, f'latest-{self.step:09d}.ckpt'
        )
        if os.path.isfile(rolling):
            try:
                os.link(rolling, self.pre_path)
                self.pre_saved = True
                print(
                    'MGT fixed pre-validation audit checkpoint hard-linked: '
                    f'{self.pre_path}'
                )
                return
            except OSError:
                pass
        self.pre_saved = True
        temporary = self.pre_path + f'.tmp.{os.getpid()}'
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, self.pre_path)
        print(f'MGT fixed pre-validation audit checkpoint saved: {self.pre_path}')

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step != self.step or os.path.isfile(self.post_path):
            return
        if not trainer.is_global_zero:
            return
        self.post_saved = True
        temporary = self.post_path + f'.tmp.{os.getpid()}'
        trainer.save_checkpoint(temporary, weights_only=False)
        os.replace(temporary, self.post_path)
        print(f'MGT fixed post-validation audit checkpoint saved: {self.post_path}')


class DiagnosticStopAfterValidation(pl.Callback):
    """Stop cleanly after a configured milestone validation.

    The optimizer and LR scheduler retain the full ``total_iter`` horizon, so
    an early loss-reduction control is directly comparable to a formal run at
    the same step and can later be resumed without changing its schedule.
    """

    def __init__(self, step):
        super().__init__()
        self.step = int(step)

    def state_dict(self):
        return {'step': self.step}

    def load_state_dict(self, state_dict):
        if int(state_dict.get('step', self.step)) != self.step:
            raise ValueError('diagnostic stop step changed inside checkpoint')

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step >= self.step:
            trainer.should_stop = True
            print(f'MGT diagnostic clean stop after validation at step {trainer.global_step}')


def merge_interx_lookup_weights(state_dict):
    key_p1 = "fk_layers.p1_positions_lookup.weight"
    key_p2 = "fk_layers.p2_positions_lookup.weight"
    new_key = "fk_layers.unified_positions_lookup.weight"
    if key_p1 in state_dict and key_p2 in state_dict:
        p1_weight = state_dict.pop(key_p1)
        p2_weight = state_dict.pop(key_p2)
        state_dict[new_key] = torch.cat([p1_weight, p2_weight], dim=0)
    return state_dict


def load_transformer_checkpoint(module, ckpt_path, *, preserve_vqvae=False):
    print(f'loading transformer checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    candidates = []
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            candidates.append(ckpt['state_dict'])
        if 'trans_encoder' in ckpt:
            candidates.append(ckpt['trans_encoder'])
        if 'net' in ckpt:
            candidates.append(ckpt['net'])
        candidates.append(ckpt)
    else:
        candidates.append(ckpt)

    prefixes = ('trans_encoder.', 'model.trans_encoder.', 'module.trans_encoder.', 'model.')
    last_error = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        filtered = {}
        for key, value in candidate.items():
            stripped = None
            for prefix in prefixes:
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    break
            if stripped is not None:
                filtered[stripped] = value
        if not filtered:
            filtered = candidate
        try:
            if preserve_vqvae:
                vq_keys = {
                    key for key in module.state_dict()
                    if key.startswith('trans_base.vqvae.')
                }
                allowed_missing = set(vq_keys)
                filtered = {
                    key: value for key, value in filtered.items()
                    if key not in vq_keys
                }
                missing, unexpected = module.load_state_dict(filtered, strict=False)
                if set(missing) != allowed_missing or unexpected:
                    raise RuntimeError(
                        'transformer-only checkpoint mismatch while preserving VQ: '
                        f'missing={sorted(set(missing) - allowed_missing)}, '
                        f'unexpected={unexpected}'
                    )
                print(
                    'transformer checkpoint loaded exactly for its active architecture; '
                    'requested VQ was preserved'
                )
                return missing, unexpected
            module.load_state_dict(filtered, strict=True)
            print('transformer checkpoint loaded with exact key match')
            return [], []
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Unable to load transformer checkpoint: {ckpt_path}')


def build_split_loader(args, split_name, w_vectorizer, *, shuffle=True):
    split_file = f'dataset/interx/splits/{split_name}.txt'
    split_motion = f'dataset/interx/motions/{split_name}.h5'
    return dataset_interx_TM.DATALoader(
        args.dataname,
        32,
        split_file,
        w_vectorizer,
        split_motion,
        unit_length=2 ** args.down_t,
        shuffle=shuffle,
    )


def build_multimodality_loader(eval_loader, num_samples, seed):
    """Select the official 100-text multimodality subset reproducibly."""
    if num_samples >= len(eval_loader.dataset):
        raise ValueError(
            f'mm_num_samples ({num_samples}) must be smaller than the test dataset '
            f'({len(eval_loader.dataset)}), matching the official evaluator contract.'
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(eval_loader.dataset), generator=generator)[:num_samples].tolist()
    return DataLoader(
        Subset(eval_loader.dataset, indices),
        batch_size=eval_loader.batch_size,
        shuffle=False,
        num_workers=eval_loader.num_workers,
        collate_fn=dataset_interx_TM.collate_fn,
        drop_last=False,
    )


def seed_evaluation(seed):
    """Seed randomness used by captions, crops, shuffling and sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save_npy(path, array):
    """Write a cache entry atomically so interruption cannot leave a valid-looking partial file."""
    temporary = path + '.tmp.npy'
    np.save(temporary, array)
    os.replace(temporary, path)


def import_historical_token_pairs(code_source, contact_source, code_destination,
                                  contact_destination, names):
    """Import an exact, preflighted subset of the historical MMM cache.

    The source cache also contains test tokens.  Callers pass the official
    eligible train+validation names explicitly so no test token can enter the
    isolated training cache.  All source pairs are checked before the first
    write, preventing an incomplete import when one side is missing.
    """
    names = sorted(set(names))
    source_pairs = []
    missing = []
    for name in names:
        code_path = os.path.join(code_source, name + '.npy')
        contact_path = os.path.join(contact_source, name + '.npy')
        if not os.path.isfile(code_path):
            missing.append(('code', name))
        if not os.path.isfile(contact_path):
            missing.append(('contact', name))
        source_pairs.append((name, code_path, contact_path))
    if missing:
        raise FileNotFoundError(
            'historical MMM cache is incomplete: '
            + ', '.join(f'{kind}:{name}' for kind, name in missing[:10])
        )

    os.makedirs(code_destination, exist_ok=True)
    os.makedirs(contact_destination, exist_ok=True)
    for name, code_path, contact_path in source_pairs:
        for source, destination_directory in (
            (code_path, code_destination),
            (contact_path, contact_destination),
        ):
            destination = os.path.join(destination_directory, name + '.npy')
            temporary = destination + f'.tmp.{os.getpid()}'
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
    return len(source_pairs)


def assert_exact_historical_cache_scope(code_names, contact_names, expected_names):
    """Reject train-cache contamination by historical test/unknown tokens."""
    expected_names = set(expected_names)
    unexpected_code = sorted(set(code_names) - expected_names)
    unexpected_contact = sorted(set(contact_names) - expected_names)
    if unexpected_code or unexpected_contact:
        raise RuntimeError(
            'historical isolated cache contains tokens outside official '
            'eligible train+val: '
            f'code={unexpected_code[:5]}, contact={unexpected_contact[:5]}'
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_named_files(directory, names):
    """Hash a named cache set, including filenames and file contents."""
    digest = hashlib.sha256()
    for name in sorted(names):
        encoded_name = name.encode('utf-8')
        digest.update(len(encoded_name).to_bytes(8, 'big'))
        digest.update(encoded_name)
        path = os.path.join(directory, name + '.npy')
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(directory, suffix):
    """Hash a symlink-backed dataset directory by relative name and content."""
    digest = hashlib.sha256()
    matched = 0
    for root, directories, filenames in os.walk(directory, followlinks=True):
        directories.sort()
        for filename in sorted(filenames):
            if not filename.endswith(suffix):
                continue
            path = os.path.join(root, filename)
            relative = os.path.relpath(path, directory).replace(os.sep, '/')
            encoded_name = relative.encode('utf-8')
            digest.update(len(encoded_name).to_bytes(8, 'big'))
            digest.update(encoded_name)
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            matched += 1
    if matched == 0:
        raise RuntimeError(f'no {suffix} files found under {directory}')
    return {'files': matched, 'sha256': digest.hexdigest()}


def repository_provenance():
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=_REPO_ROOT
    ).decode('utf-8').strip()
    tracked_diff = subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD', '--'], cwd=_REPO_ROOT
    )
    return {
        'revision': revision,
        'tracked_diff_sha256': hashlib.sha256(tracked_diff).hexdigest(),
    }


def prepare_mgt_output_directory(args):
    """Create a new run directory or strictly resume one in place."""
    if args.resume_in_place and not args.resume_training_ckpt:
        raise ValueError('--resume-in-place requires --resume-training-ckpt')
    if not args.resume_in_place:
        init_save_folder(args)
        return {'mode': 'new_directory'}

    checkpoint = os.path.abspath(args.resume_training_ckpt)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    run_directory = os.path.dirname(checkpoint)
    expected_parent = os.path.abspath(args.out_dir)
    if os.path.dirname(run_directory) != expected_parent:
        raise ValueError(
            'in-place resume checkpoint is outside the configured MGT output '
            f'root: {checkpoint} not under {expected_parent}'
        )
    if not os.path.basename(run_directory).endswith('_' + args.exp_name):
        raise ValueError(
            'in-place resume run directory does not match --exp-name: '
            f'{run_directory} vs {args.exp_name}'
        )
    checkpoint_match = re.fullmatch(
        r'latest-(\d+)\.ckpt', os.path.basename(checkpoint)
    )
    if checkpoint_match is None:
        raise ValueError(
            'in-place resume requires an atomic latest-<step>.ckpt checkpoint'
        )
    checkpoint_step = int(checkpoint_match.group(1))
    if not 0 < checkpoint_step < int(args.total_iter):
        raise ValueError(
            f'in-place resume step must be within (0,{args.total_iter}), got '
            f'{checkpoint_step}'
        )

    manifest_path = os.path.join(run_directory, 'training_manifest.json')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding='utf-8') as handle:
        previous = json.load(handle)
    previous_args = previous.get('resolved_arguments', {})
    mismatches = {}
    for key in MGT_RESUME_IDENTITY_KEYS:
        current = getattr(args, key)
        prior = previous_args.get(key)
        if (key in MGT_RESUME_PATH_KEYS
                and current is not None and prior is not None):
            current = os.path.abspath(current)
            prior = os.path.abspath(prior)
        if current != prior:
            mismatches[key] = {'previous': prior, 'current': current}
    if mismatches:
        raise ValueError(
            'in-place MGT resume contract differs from the original run: '
            f'{mismatches}'
        )

    previous_hashes = previous.get('sha256', {})
    missing_source_hashes = sorted(
        set(MGT_TRAINING_SOURCE_PATHS) - set(previous_hashes)
    )
    if missing_source_hashes:
        raise ValueError(
            'in-place MGT resume manifest lacks launch source hashes: '
            f'{missing_source_hashes}'
        )
    source_mismatches = {}
    for relative in MGT_TRAINING_SOURCE_PATHS:
        prior_hash = previous_hashes[relative]
        current_hash = sha256_file(os.path.join(_REPO_ROOT, relative))
        if current_hash != prior_hash:
            source_mismatches[relative] = {
                'previous': prior_hash, 'current': current_hash,
            }
    if source_mismatches:
        raise ValueError(
            'in-place MGT resume source differs from launch source: '
            f'{source_mismatches}'
        )
    args.out_dir = run_directory
    print(
        f'MGT strict in-place resume: step={checkpoint_step}, '
        f'run_directory={run_directory}'
    )
    return {
        'mode': 'strict_in_place_resume',
        'checkpoint': checkpoint,
        'checkpoint_step': checkpoint_step,
        'prior_manifest': manifest_path,
    }


def write_mgt_training_manifest(args, cache_manifest_path, contact_map_path,
                                loader_contract):
    """Bind an MGT run to its final VQ, cache, data, code, and recipe."""
    with open(cache_manifest_path, 'r', encoding='utf-8') as handle:
        cache_manifest = json.load(handle)
    if cache_manifest.get('status') != 'complete':
        raise RuntimeError('MGT training requires a complete token-cache manifest')

    file_hashes = {
        path: sha256_file(path) for path in MGT_TRAINING_SOURCE_PATHS
    }
    artifact_paths = {
        'vq_checkpoint': args.vq_checkpoint,
        'vq_training_manifest': os.path.join(args.vq_dir, 'training_manifest.json'),
        'token_cache_manifest': cache_manifest_path,
        'contact_guidance_map': contact_map_path,
        'train_split': args.train_split_file,
        'validation_split': args.val_split_file,
        'raw_mean': os.path.join(
            args.stats_dir, f'{args.stats_prefix}_Mean.npy'
        ),
        'raw_std': os.path.join(
            args.stats_dir, f'{args.stats_prefix}_Std.npy'
        ),
        'evaluator_options': 'checkpoints/hhi/Comp_v6_KLD01/opt.txt',
        'evaluator_weights': 'checkpoints/hhi/text_mot_match/model/finest.tar',
        'clip_weights': os.path.expanduser('~/.cache/clip/ViT-B-32.pt'),
    }
    if args.token_cache_contract == 'deterministic_historical_root_v1':
        artifact_paths.update({
            'root_tokenizer_implementation': (
                'dataset/dataset_interx_tokenize_historical_root.py'
            ),
            'tokenizer_root_mean': os.path.join(
                args.stats_dir, f'{args.stats_prefix}_Mean_root.npy'
            ),
            'tokenizer_root_std': os.path.join(
                args.stats_dir, f'{args.stats_prefix}_Std_root.npy'
            ),
        })
    if args.training_validation_contract == 'historical_test_side_effect':
        artifact_paths.update({
            'historical_validation_test_split': 'dataset/interx/splits/test.txt',
            'historical_validation_test_motion': 'dataset/interx/motions/test.h5',
        })
    if args.augmentation_manifest:
        artifact_paths['augmentation_manifest'] = args.augmentation_manifest
        artifact_paths['materialized_train_motion'] = args.train_motion_file
        artifact_paths['materialized_all_motion'] = args.all_motion_file
    if args.resume_training_ckpt:
        artifact_paths['resume_training_checkpoint'] = args.resume_training_ckpt
    for label, path in artifact_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'MGT provenance artifact is missing ({label}): {path}')
        file_hashes[label] = sha256_file(path)

    paper_training_contract = (
        args.paper_architecture
        and args.training_mask_contract == 'paper_kinematic'
        and args.training_loss_contract == 'paper_masked_only'
        and args.optimizer_contract == 'paper_warmup_cosine'
        and args.conditioning_contract == 'paper_text_cfg'
        and option_trans.resolve_text_drop_probability(args) == 0.1
        and args.training_token_length_contract in {
            'original_mmm_38', 'paper_50'
        }
        and args.training_validation_contract == 'paper_isolated_common'
    )
    historical_performance_paper_mask_contract = (
        not args.paper_architecture
        and args.training_mask_contract
        == 'historical_pipeline_paper_kinematic_ratio'
        and args.training_loss_contract == 'historical_all_valid'
        and args.optimizer_contract == 'historical_multistep'
        and args.conditioning_contract == 'historical_mmm_shape_dropout'
        and args.training_token_length_contract == 'historical_38'
        and args.training_validation_contract == 'historical_test_side_effect'
        and args.save_all_validation_checkpoints
    )
    manifest = {
        'artifact_role': (
            'HHC paper-config kinematic-cascade MGT retraining provenance'
            if paper_training_contract
            else (
                'HHC historical-performance pipeline with paper kinematic '
                'mask-ratio retraining; eligible for held-out validation selection'
                if historical_performance_paper_mask_contract
                else 'MGT repair diagnostic bridge; not eligible for final reporting'
            )
        ),
        'command': ['python'] + sys.argv,
        'cwd': os.getcwd(),
        'launch_time': _PROCESS_START_TIME,
        'observability': {
            'wandb_mode': args.wandb_mode,
            'wandb_project': 'NewInter',
            'wandb_run_id': args.wandb_run_id,
            'periodic_validation_mm_samples': args.mm_num_samples,
            'periodic_validation_mm_repeats': args.validation_mm_repeats,
            'formal_mm_repeats': args.mm_num_repeats,
            'periodic_validation_inference_conditioning_contract': (
                'paper_text_cfg'
                if args.training_validation_contract == 'paper_isolated_common'
                else args.conditioning_contract
            ),
            'metric_profile': 'masked_token_classification',
            'telemetry_interval_optimizer_steps': 50,
        },
        'repository': repository_provenance(),
        'resolved_arguments': vars(args),
        'paper_configuration': {
            'architecture_contract': (
                'paper_reconstructed_cross_self_adaln_9_blocks'
                if args.paper_architecture
                else 'legacy_mmm_topology_control_not_for_final_reporting'
            ),
            'paper_architecture': bool(args.paper_architecture),
            'complete_paper_training_contract': bool(paper_training_contract),
            'total_iterations': int(args.total_iter),
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.lr),
            'warmup_iterations': int(args.warm_up_iter),
            'schedule': (
                'linear warm-up then cosine decay'
                if args.optimizer_contract == 'paper_warmup_cosine'
                else f'MultiStepLR milestones={list(args.lr_scheduler)} gamma={args.gamma}'
            ),
            'optimizer': 'AdamW',
            'weight_decay': float(args.weight_decay),
            'codebook_size_per_part': int(args.nb_code),
            'blocks': int(args.num_layers),
            'attention_heads': int(args.n_head_gpt),
            'latent_dimension': int(args.embed_dim_gpt),
            'mask_schedule': args.training_mask_contract,
            'mask_count_discretization': {
                'rule': 'round(valid_part_length * mask_ratio), minimum 1 for non-empty parts',
                'provenance': 'source-derived (HHC/MMM and InterMask); not specified in the paper text',
            },
            'objective': args.training_loss_contract,
            'conditioning': args.conditioning_contract,
            'token_length_contract': args.training_token_length_contract,
            'training_validation_contract': args.training_validation_contract,
            'effective_validation_sampler': (
                'sample_intra_cas_historical'
                if args.training_validation_contract == 'historical_test_side_effect'
                else args.sampler
            ),
            'sampling_order': 'Root->Body->Hands with immediate within-step updates',
            'sampling_steps': int(args.sampling_steps),
            'temperature': float(args.temperature),
            'cfg_scale': float(args.cfg_scale),
            'contact_guidance_scale': float(args.contact_guidance_scale),
            'contact_as_training_input': bool(args.contact_conditioning),
            'shape_conditioning': bool(args.shape),
        },
        'source_derived_configuration': {
            'precision': 'bf16-mixed',
            'minimum_training_tokens': 20,
            'conditioning_dropout': (
                {
                    'text_probability': option_trans.resolve_text_drop_probability(args),
                    'shape_probability': 0.0,
                    'word_tokens_dropped_with_global_text': True,
                }
                if args.conditioning_contract == 'paper_text_cfg'
                else {
                    'global_clip_probability': 0.2,
                    'shape_probability': 0.2,
                    'word_tokens_dropped': False,
                }
            ),
            'contact_guidance_log_epsilon': 1e-6,
            'clip_backbone': 'ViT-B/32',
            'optimizer_betas': (
                [0.9, 0.99]
                if args.optimizer_contract == 'paper_warmup_cosine'
                else [0.5, 0.9]
            ),
            'optimizer_parameter_membership': (
                'all_transformer_parameters_including_detached_vq'
                if args.training_validation_contract == 'historical_test_side_effect'
                else 'requires_grad_transformer_parameters_only'
            ),
            'training_module_mode_contract': (
                'historical_transformer_train_vq_eval_after_sanity_restore'
                if args.training_validation_contract == 'historical_test_side_effect'
                else 'formal_transformer_train_frozen_clip_vq_eval'
            ),
            'feedforward_expansion': int(args.ff_rate),
            'transformer_dropout': float(args.drop_out_rate),
            'maximum_motion_tokens': (
                38 if args.training_token_length_contract in {
                    'original_mmm_38', 'historical_38'
                }
                else int(args.block_size) - 1
            ),
            'cache_crop_policy': cache_manifest['cache_crop_policy'],
            'validation_seed': (
                None if args.training_validation_contract == 'historical_test_side_effect'
                else int(args.seed) + 500_000
            ),
            'validation_rng_is_observational': (
                args.training_validation_contract == 'paper_isolated_common'
            ),
        },
        'loader_contract': loader_contract,
        'checkpoint_policy': {
            'latest_interval_updates': int(args.checkpoint_iter),
            'validation_interval_updates': int(args.eval_iter),
            'validation_checkpoint_storage': 'weights_only',
            'rolling_recovery_checkpoint_storage': 'full_training_state',
            'retain_all_validation_checkpoints': bool(
                args.save_all_validation_checkpoints
            ),
            'primary_selection': (
                'posthoc minimum held-out official validation FID over every '
                'retained validation checkpoint'
                if args.save_all_validation_checkpoints
                else 'minimum fixed-protocol validation FID'
            ),
            'test_metrics_used_for_selection': (
                args.training_validation_contract == 'historical_test_side_effect'
                and not args.save_all_validation_checkpoints
            ),
            'historical_test_metric_controls_retention': (
                args.training_validation_contract == 'historical_test_side_effect'
                and not args.save_all_validation_checkpoints
            ),
        },
        'cache_manifest': cache_manifest,
        'dataset_aggregates': {
            'betas': sha256_tree('dataset/interx/betas', '.npz'),
            'processed_text': sha256_tree('dataset/interx/texts_processed', '.txt'),
            'augmentation_betas': (
                sha256_tree(os.path.join(args.train_asset_root, 'betas'), '.npz')
                if args.train_asset_root else None
            ),
            'augmentation_processed_text': (
                sha256_tree(
                    os.path.join(args.train_asset_root, 'texts_processed'), '.txt'
                ) if args.train_asset_root else None
            ),
        },
        'paths': {label: os.path.abspath(path) for label, path in artifact_paths.items()},
        'sha256': file_hashes,
    }
    manifest_path = os.path.join(args.out_dir, 'training_manifest.json')
    atomic_write_json(manifest_path, manifest)
    print(f'MGT training provenance: {manifest_path}')
    return manifest_path


def eligible_motion_names(names, motion_path, min_length=30, max_length=1000):
    """Mirror the tokenizing dataset's availability and length filters."""
    with h5py.File(motion_path, 'r') as motions:
        return {
            name for name in names
            if name in motions and min_length <= motions[name].shape[0] < max_length
        }


def atomic_write_json(path, record):
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_generation_export(args, model, val_loader):
    class LoggerWrapper:
        def info(self, msg):
            print(msg)

    output_dir = args.save_dir
    if output_dir is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.gen_ckpt))[0].replace('/', '_')
        guidance_tag = 'contact' if args.contact else 'nocontact'
        output_dir = os.path.join(
            args.out_dir,
            f'mmodality_{args.gen_split}_{ckpt_stem}_temp{args.temperature}_repeat{args.num_repeat}_{guidance_tag}',
        )
    os.makedirs(output_dir, exist_ok=True)

    logger = LoggerWrapper()
    evaluation_args = dict(
        out_dir=output_dir,
        val_loader=val_loader,
        net=model.net,
        trans=model.trans_encoder,
        logger=logger,
        writer=None,
        nb_iter=0,
        best_fid=1000,
        best_iter=0,
        best_div=100,
        best_top1=0,
        best_top2=0,
        best_top3=0,
        best_matching=100,
        clip_model=model.clip_model,
        eval_wrapper=model.eval_wrapper,
        dataname=args.dataname,
        save=False,
        savegif=False,
        num_repeat=args.num_repeat,
        rand_pos=True,
        CFG=args.cfg_scale,
        savenpy=True,
        contact=args.contact,
        shape_change=None,
        save_embeddings=args.save_embeddings,
        sampler=args.sampler,
        temperature=args.temperature,
        sampling_steps=args.sampling_steps,
        contact_guidance_scale=args.contact_guidance_scale,
        contact_map_path=args.contact_map_path,
    )
    eval_trans.evaluation_transformer(**evaluation_args)
    print(f'generation export finished: {output_dir}')


def run_repeated_evaluation(args, model, eval_loader):
    """Run the official Inter-X/InterMask 20-replication test protocol.

    Each replication evaluates one generation for every retained test item and
    MModality on 100 randomly selected texts with 30 generations per text.
    Explicit seeds let competing methods use identical GT crops, captions and
    multimodality subsets.
    """
    class FileLogger:
        def __init__(self, path):
            self.handle = open(path, 'w', encoding='utf-8')

        def info(self, message):
            print(message)
            self.handle.write(message + '\n')
            self.handle.flush()

        def close(self):
            self.handle.close()

    ckpt_stem = os.path.splitext(os.path.basename(args.gen_ckpt))[0]
    output_dir = args.save_dir or os.path.join(args.out_dir, f'official_eval_{args.eval_split}_{ckpt_stem}')
    os.makedirs(output_dir, exist_ok=True)
    logger = FileLogger(os.path.join(output_dir, 'evaluation.log'))
    records = []
    try:
        for replication in range(args.replication_times):
            replication_seed = args.eval_seed + replication
            seed_evaluation(replication_seed)
            result = eval_trans.evaluation_transformer(
                output_dir, eval_loader, model.net, model.trans_encoder, logger, None,
                replication, 1000, 0, 100, 0, 0, 0, 100,
                clip_model=model.clip_model, eval_wrapper=model.eval_wrapper,
                dataname=args.dataname, num_repeat=1, rand_pos=True,
                CFG=args.cfg_scale, contact=args.contact,
                sampler=args.sampler, temperature=args.temperature,
                sampling_steps=args.sampling_steps,
                contact_guidance_scale=args.contact_guidance_scale,
                contact_map_path=args.contact_map_path,
            )
            mm_loader = build_multimodality_loader(
                eval_loader, args.mm_num_samples, replication_seed
            )
            seed_evaluation(args.eval_seed + 10_000 + replication)
            mm_result = eval_trans.evaluation_transformer(
                output_dir, mm_loader, model.net, model.trans_encoder, logger, None,
                10_000 + replication, 1000, 0, 100, 0, 0, 0, 100,
                clip_model=model.clip_model, eval_wrapper=model.eval_wrapper,
                dataname=args.dataname, num_repeat=args.mm_num_repeats,
                rand_pos=True, CFG=args.cfg_scale, contact=args.contact,
                sampler=args.sampler, temperature=args.temperature,
                sampling_steps=args.sampling_steps,
                contact_guidance_scale=args.contact_guidance_scale,
                contact_map_path=args.contact_map_path,
            )
            records.append({
                'fid': float(result[0]),
                'diversity': float(result[6]),
                'top1': float(result[7]),
                'top2': float(result[8]),
                'top3': float(result[9]),
                'matching': float(result[10]),
                'multimodality': float(mm_result[11]),
                'seed': replication_seed,
            })
    finally:
        logger.close()

    summary = {}
    for metric in ('fid', 'diversity', 'top1', 'top2', 'top3', 'matching', 'multimodality'):
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        summary[metric] = {
            'mean': float(values.mean()),
            'ci95': float(1.96 * values.std() / np.sqrt(len(values))),
        }
    report = {
        'protocol': {
            'split': args.eval_split,
            'replication_times': args.replication_times,
            'batch_size': eval_loader.batch_size,
            'num_samples_after_drop_last': len(eval_loader) * eval_loader.batch_size,
            'dataset_samples_before_drop_last': len(eval_loader.dataset),
            'mm_num_samples': args.mm_num_samples,
            'mm_num_repeats': args.mm_num_repeats,
            'eval_seed': args.eval_seed,
            'sampler': args.sampler,
            'temperature': args.temperature,
            'sampling_steps': args.sampling_steps,
            'cfg_scale': args.cfg_scale,
        },
        'checkpoint': args.gen_ckpt,
        'records': records,
        'summary': summary,
    }
    report_path = os.path.join(output_dir, 'metrics.json')
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'official evaluation report: {report_path}')


# 辅助函数：计算准确率（保持原有逻辑）
@torch.no_grad()
def get_acc(cls_pred, target, mask):
    pred_indices = cls_pred.argmax(dim=-1)
    pred_selected = pred_indices[mask]
    target_selected = target[mask]
    right_num = (pred_selected == target_selected).sum()
    total_num = mask.sum().clamp(min=1)
    # Keep a detached scalar tensor. Sparse telemetry calls ``detach`` on
    # every task metric, including batches with an empty selected mask.
    return right_num * 100.0 / total_num


# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb


class MaskGITLightning(pl.LightningModule):
    def __init__(self, args, eval_wrapper, val_loader):
        super().__init__()
        self.save_hyperparameters(ignore=['eval_wrapper', 'val_loader'])
        self.automatic_optimization = False
        self.args = args
        self.eval_wrapper = eval_wrapper
        self.val_loader = val_loader  # 保存用于自定义 evaluation
        self.historical_replay = (
            args.training_validation_contract == 'historical_test_side_effect'
        )
        self.telemetry_interval_optimizer_steps = 50
        self.training_throughput = ThroughputMeter()

        # Paper Sec. 3.2 keeps contact schedules out of the model input and
        # applies them only as a sampling-time logit bias.
        self.contact = args.contact_conditioning
        self.shape = args.shape

        # ------------------
        # 1. 初始化模型
        # ------------------
        # CLIP
        clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = TextCLIP(clip_model)

        # VQVAE
        self.net = build_model(args, "interx_vq")

        # Transformer
        self.trans_encoder = build_model(args, "interx_transformer", vqvae=self.net)
        self.trans_encoder.trans_base.text_drop_prob = option_trans.resolve_text_drop_probability(args)

        # ------------------
        # 2. 加载权重 (Resume Logic)
        # ------------------
        if args.resume_pth:
            print(f'loading checkpoint from {args.resume_pth}')
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            state_dict = ckpt['net']

            # 合并 P1 P2 权重逻辑
            key_p1 = "fk_layers.p1_positions_lookup.weight"
            key_p2 = "fk_layers.p2_positions_lookup.weight"
            new_key = "fk_layers.unified_positions_lookup.weight"
            if key_p1 in state_dict and key_p2 in state_dict:
                print("Merging p1 and p2 lookup weights into unified_positions_lookup...")
                p1_weight = state_dict.pop(key_p1)
                p2_weight = state_dict.pop(key_p2)
                combined_weight = torch.cat([p1_weight, p2_weight], dim=0)
                state_dict[new_key] = combined_weight

            self.net.load_state_dict(state_dict, strict=True)
            self.net.eval()  # VQVAE 始终 Eval

        # Formal MGT training freezes the tokenizer.  The isolated historical
        # replay deliberately preserves the recovered runtime contract instead:
        # its AdamW group contained all 482 transformer parameters (including
        # 138 detached VQ parameters), although only 344 ever acquired state.
        if not self.historical_replay:
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)

        if args.resume_trans:
            load_transformer_checkpoint(
                self.trans_encoder, args.resume_trans, preserve_vqvae=True
            )

        # ------------------
        # 3. 评估指标状态记录
        # ------------------
        self.best_fid = 1000
        self.best_iter = 0
        self.best_div = 100
        self.best_top1 = 0
        self.best_top2 = 0
        self.best_top3 = 0
        self.best_matching = 100

        # 这里的 writer 会在 fit 开始后由 Logger Connector 接管，
        # 但 eval_trans 需要原始 writer，我们会在 on_train_start 里处理

    def train(self, mode=True):
        super().train(mode)
        # The recovered Lightning module did not override train(), so its
        # frozen CLIP and detached VQ were recursively put into train mode. Keep
        # that diagnostic behavior only for the exact historical replay.
        if not self.historical_replay:
            self.net.eval()
            self.clip_model.eval()
        return self

    def on_train_start(self):
        # Lightning's sanity loop captures every submodule's pre-validation
        # mode and restores it verbatim.  Initializing the transformer in eval
        # mode therefore keeps dropout disabled for the entire fit; the fit
        # loop does not subsequently call model.train() to repair that state.
        # The recovered MMM run entered sanity validation with the transformer
        # in train mode and the loaded VQ in eval mode.
        if not self.trans_encoder.training:
            raise RuntimeError(
                'MGT transformer entered training in eval mode; dropout would '
                'be disabled and the training/FID trajectory would be invalid'
            )
        if self.net.training:
            raise RuntimeError(
                'MGT VQ tokenizer entered training in train mode; expected the '
                'loaded tokenizer to remain in eval mode'
            )
        print(
            'MGT module-mode contract: transformer=train, VQ=eval, '
            f'historical_replay={self.historical_replay}'
        )

    def configure_optimizers(self):
        if self.args.optimizer_contract == 'historical_multistep':
            historical_parameters = list(self.trans_encoder.parameters())
            if self.historical_replay and len(historical_parameters) != 482:
                raise RuntimeError(
                    'historical optimizer membership drift: expected 482 '
                    f'parameter tensors, got {len(historical_parameters)}'
                )
            if self.historical_replay:
                print(
                    'Historical optimizer contract: 482 parameter tensors; '
                    'the recovered checkpoint acquires Adam state for 344.'
                )
            optimizer = torch.optim.AdamW(
                historical_parameters, lr=self.args.lr,
                betas=(0.5, 0.9), weight_decay=self.args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args.lr_scheduler,
                gamma=self.args.gamma,
            )
            return [optimizer], [scheduler]

        trainable_parameters = [
            parameter for parameter in self.trans_encoder.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError('MGT has no trainable transformer parameters')
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=self.args.lr,
            betas=(0.9, 0.99), weight_decay=self.args.weight_decay,
        )

        def warmup_cosine(step):
            if step < self.args.warm_up_iter:
                return float(step + 1) / float(max(1, self.args.warm_up_iter))
            progress = (step - self.args.warm_up_iter) / max(1, self.args.total_iter - self.args.warm_up_iter)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # ------------------
        # A. 数据准备
        # ------------------
        clip_text, m_tokens, m_tokens_len, betas, gender, contact = batch
        # Lightning 会自动处理 device，但如果 eval_trans 需要特定类型，注意保持
        target = m_tokens.clone()
        batch_size, max_len, num_parts = target.shape

        # CLIP Encode

        text = clip.tokenize(clip_text, truncate=True).to(self.device)
        feat_clip_text, word_emb = self.clip_model(text)
        # ------------------
        # B. 初始化 (纯净 GT)
        # ------------------
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # (B, T)
        seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)  # (B, T, 3)

        if self.args.training_mask_contract == 'historical_flat_random_replace':
            masked_input_indices, mask_token, flat_mask_ratio = (
                sample_historical_mmm_corruption(
                    target, seq_mask_no_end_3c, self.args.nb_code,
                    pkeep=self.args.pkeep,
                )
            )
            mask_ratios = flat_mask_ratio[:, None].expand(-1, 3)
        elif self.args.training_mask_contract == (
            'historical_pipeline_paper_kinematic_ratio'
        ):
            masked_input_indices, mask_token, mask_ratios, _ = (
                sample_historical_mmm_corruption_with_kinematic_ratios(
                    target, seq_mask_no_end_3c, self.args.nb_code,
                    pkeep=self.args.pkeep,
                )
            )
        else:
            # Paper training uses three sorted cosine schedules, not a shared
            # mask probability or BERT-style random token corruption.
            mask_token_id = self.args.nb_code + 2
            mask_token, mask_ratios, _ = sample_kinematic_cascade_mask(
                seq_mask_no_end_3c
            )
            masked_input_indices = torch.where(mask_token, mask_token_id, target)

        # ------------------
        # D. Forward
        # ------------------
        att_txt = None

        attention_lengths = (
            m_tokens_len if self.args.paper_architecture else m_tokens_len + 1
        )
        seq_mask_attn = generate_src_mask(max_len, attention_lengths)

        if self.contact and not self.shape:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, contact_mask =contact)
        elif self.contact and self.shape:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, contact_mask =contact, betas=betas, gender=gender)
        else:
            cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
                                      word_emb=word_emb, betas=betas, gender=gender)


        # ------------------
        # E. Loss
        # ------------------
        if self.args.training_loss_contract == 'historical_all_valid':
            loss_cls = all_valid_token_nll(
                cls_pred, target, seq_mask_no_end_3c
            )
        elif self.args.training_loss_contract == 'paper_masked_only':
            loss_cls = masked_token_nll(cls_pred, target, mask_token)
        elif self.args.training_loss_contract == 'paper_masked_global_token_mean':
            loss_cls = global_masked_token_nll(
                cls_pred, target, mask_token
            )
        else:
            raise ValueError(self.args.training_loss_contract)

        # Accuracy is required by the sparse telemetry event below. Compute it
        # without extending the autograd graph, before the optimizer step and
        # before the first possible event at step 50.
        with torch.no_grad():
            cls_pred_index = cls_pred.detach().argmax(dim=-1)
            right_token = (cls_pred_index == target) & seq_mask_no_end_3c
            acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()
            real_mask_token = mask_token & seq_mask_no_end_3c
            acc_masked = get_acc(cls_pred.detach(), target, real_mask_token)
            no_mask_token = (~mask_token) & seq_mask_no_end_3c
            acc_nomask = get_acc(cls_pred.detach(), target, no_mask_token)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss_cls)
        self.training_throughput.update(batch_size)
        next_step = int(self.global_step) + 1
        if next_step % self.telemetry_interval_optimizer_steps == 0:
            telemetry = parameter_and_gradient_telemetry(
                self.trans_encoder, fail_on_nonfinite=False
            )
            telemetry.update({
                'train/optimizer_step': float(next_step),
                'train/loss': loss_cls.detach(),
                'train/accuracy_all': acc_all.detach(),
                'train/accuracy_masked': acc_masked.detach(),
                'train/accuracy_visible': acc_nomask.detach(),
                'train/mask_ratio_body': mask_ratios[:, 0].mean(),
                'train/mask_ratio_hand': mask_ratios[:, 1].mean(),
                'train/mask_ratio_root': mask_ratios[:, 2].mean(),
                'train/learning_rate': float(opt.param_groups[0]['lr']),
                'train/samples_per_second': (
                    self.training_throughput.measure_and_reset()
                ),
                'train/gpu_peak_memory_gib': gpu_peak_memory_gib(
                    self.device
                ),
            })
            self.log_dict(telemetry, on_step=True, logger=True)
            if (telemetry['train/nonfinite_gradients']
                    or telemetry['train/nonfinite_weights']):
                raise FloatingPointError(
                    f'nonfinite training telemetry: {telemetry}'
                )
        opt.step()
        self.lr_schedulers().step()
        # ------------------
        # F. Logging
        # ------------------
        # 使用 Lightning 的 logger
        self.log('Train/all', loss_cls, prog_bar=True)
        self.log('Train/every_token', acc_all, prog_bar=True)
        self.log('Train/masked', acc_masked, prog_bar=True)
        self.log('Train/no_masked', acc_nomask, prog_bar=True)
        self.log('Train/mask_ratio_body', mask_ratios[:, 0].mean())
        self.log('Train/mask_ratio_hand', mask_ratios[:, 1].mean())
        self.log('Train/mask_ratio_root', mask_ratios[:, 2].mean())
        self.log('train/mask_ratio_body', mask_ratios[:, 0].mean(), on_step=True, logger=True)
        self.log('train/mask_ratio_hand', mask_ratios[:, 1].mean(), on_step=True, logger=True)
        self.log('train/mask_ratio_root', mask_ratios[:, 2].mean(), on_step=True, logger=True)
        self.log('train/loss', loss_cls, on_step=True, logger=True)
        self.log('train/accuracy_all', acc_all, on_step=True, logger=True)
        self.log('train/accuracy_masked', acc_masked, on_step=True, logger=True)
        self.log('train/accuracy_visible', acc_nomask, on_step=True, logger=True)
        self.log(
            'train/learning_rate', opt.param_groups[0]['lr'],
            on_step=True, logger=True,
        )
        # print(f"Iter {self.global_step} : Loss {loss_cls.item():.5f}, ACC {acc_all:.4f}")

        return loss_cls

    def validation_step(self, batch, batch_idx):
        clip_text, m_tokens, m_tokens_len, betas, gender, contact = batch
        target = m_tokens.clone()
        batch_size, max_len, _ = target.shape
        valid = generate_src_mask(max_len, m_tokens_len)
        valid_3c = valid.unsqueeze(-1).expand(-1, -1, 3)
        generator = torch.Generator(device=target.device)
        generator.manual_seed(
            self.args.seed + 700_000 + int(self.global_step) * 10_000 + batch_idx
        )
        if self.args.training_mask_contract == 'historical_flat_random_replace':
            masked_input, mask_token, _ = sample_historical_mmm_corruption(
                target, valid_3c, self.args.nb_code,
                pkeep=self.args.pkeep, generator=generator,
            )
        elif self.args.training_mask_contract == (
            'historical_pipeline_paper_kinematic_ratio'
        ):
            masked_input, mask_token, _, _ = (
                sample_historical_mmm_corruption_with_kinematic_ratios(
                    target, valid_3c, self.args.nb_code,
                    pkeep=self.args.pkeep, generator=generator,
                )
            )
        else:
            mask_token, _, _ = sample_kinematic_cascade_mask(
                valid_3c, generator=generator
            )
            masked_input = torch.where(
                mask_token, self.args.nb_code + 2, target
            )
        text = clip.tokenize(clip_text, truncate=True).to(self.device)
        feat_clip_text, word_emb = self.clip_model(text)
        attention_lengths = (
            m_tokens_len if self.args.paper_architecture else m_tokens_len + 1
        )
        attention_mask = generate_src_mask(max_len, attention_lengths)
        forward_kwargs = {
            'src_mask': attention_mask,
            'att_txt': None,
            'word_emb': word_emb,
            'betas': betas,
            'gender': gender,
        }
        if self.contact:
            forward_kwargs['contact_mask'] = contact
        logits = self.trans_encoder(
            masked_input, feat_clip_text, **forward_kwargs
        )
        if self.args.training_loss_contract == 'historical_all_valid':
            loss = all_valid_token_nll(logits, target, valid_3c)
        elif self.args.training_loss_contract == 'paper_masked_global_token_mean':
            from utils.motion_masks import global_masked_token_nll
            loss = global_masked_token_nll(logits, target, mask_token)
        else:
            loss = masked_token_nll(logits, target, mask_token)
        prediction = logits.argmax(dim=-1)
        accuracy_all = ((prediction == target) & valid_3c).sum() * 100.0 / valid_3c.sum()
        accuracy_masked = get_acc(logits, target, mask_token & valid_3c)
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True,
            logger=True, batch_size=batch_size,
        )
        self.log(
            'val/accuracy_all', accuracy_all, on_step=False, on_epoch=True,
            logger=True, batch_size=batch_size,
        )
        self.log(
            'val/accuracy_masked', accuracy_masked, on_step=False,
            on_epoch=True, logger=True, batch_size=batch_size,
        )
        # Lightning 需要这个方法，但我们的自定义 eval 是在 epoch start 统一跑的
        # clip_text, m_tokens, m_tokens_len, _, _, contact = batch
        # # Lightning 会自动处理 device，但如果 eval_trans 需要特定类型，注意保持
        # target = m_tokens.clone()
        # batch_size, max_len, num_parts = target.shape
        #
        # # CLIP Encode
        # text = clip.tokenize(clip_text, truncate=True).to(self.device)
        # feat_clip_text, word_emb = self.clip_model(text)
        # # ------------------
        # # B. 初始化 (纯净 GT)
        # # ------------------
        # seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # (B, T)
        # seq_mask_no_end_3c = seq_mask_no_end.unsqueeze(-1).expand(-1, -1, 3)  # (B, T, 3)
        #
        # if self.args.pkeep == -1:
        #     proba = np.random.rand(1)[0]
        #     mask_bernoulli = torch.bernoulli(proba * torch.ones(target.shape, device=target.device))
        # else:
        #     mask_bernoulli = torch.bernoulli(args.pkeep * torch.ones(target.shape, device=target.device))
        #
        # mask_bernoulli = torch.logical_or(mask_bernoulli, ~seq_mask_no_end_3c).int()
        #
        # # 随机替换 Token
        # r_indices = torch.randint_like(target, args.nb_code)
        # # input_indices: 包含部分原始 token 和部分随机噪声 token
        # input_indices = mask_bernoulli * target + (1 - mask_bernoulli) * r_indices
        # # ------------------
        # # C. MaskGIT + BERT 策略
        # # ------------------
        # mask_token_id = self.args.nb_code + 2
        #
        # # 1. 随机生成 Mask 比例
        # rand_time = torch.rand(batch_size, device=self.device)
        # rand_mask_probs = torch.cos(rand_time * math.pi * 0.5).clamp(min=0.1, max=1.0)
        #
        # # 2. 计算 Mask 数量
        # flat_valid_len = m_tokens_len * 3
        # num_token_masked = (flat_valid_len.float() * rand_mask_probs).round().clamp(min=1).long()
        #
        # # 3. 结构化 Mask Bias (Root > Lower > Upper)
        # batch_rand_score = torch.rand((batch_size, max_len, 3), device=self.device)
        # mask_bias = torch.tensor([0.5, 1.0, 0.0], device=self.device)  # Upper易Mask, Root难Mask
        # batch_rand_score = batch_rand_score + mask_bias
        #
        # # 保护 Pad
        # batch_rand_score = batch_rand_score + (~seq_mask_no_end_3c).float() * 1e9
        #
        # # 生成 Mask
        # batch_randperm = batch_rand_score.view(batch_size, -1).argsort(dim=-1)
        # mask_token_flat = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        # mask_token = mask_token_flat.view(batch_size, max_len, 3)
        # # 4. 应用 Mask
        # masked_input_indices = torch.where(mask_token, mask_token_id, input_indices)
        #
        # # ------------------
        # # D. Forward
        # # ------------------
        # att_txt = None
        # seq_mask_attn = generate_src_mask(max_len, m_tokens_len + 1)
        #
        # if self.contact:
        #     cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
        #                                   word_emb=word_emb, contact_mask=contact)
        # else:
        #     cls_pred = self.trans_encoder(masked_input_indices, feat_clip_text, src_mask=seq_mask_attn, att_txt=att_txt,
        #                                   word_emb=word_emb)
        #
        # # ------------------
        # # E. Loss
        # # ------------------
        # weights = seq_mask_no_end_3c.float()
        # sample_valid_tokens = seq_mask_no_end_3c.sum(dim=(1, 2), keepdim=True)
        # weights = weights / sample_valid_tokens.clamp(min=1)
        #
        # cls_pred_flat = cls_pred[seq_mask_no_end_3c].reshape(-1, cls_pred.shape[-1])
        # target_flat = target[seq_mask_no_end_3c].reshape(-1)
        # weights_flat = weights[seq_mask_no_end_3c].reshape(-1)
        #
        # loss_cls = F.cross_entropy(cls_pred_flat, target_flat, reduction='none')
        # loss_cls = (loss_cls * weights_flat).sum()
        # loss_cls = loss_cls / batch_size
        # # ------------------
        # # F. Logging
        # # ------------------
        # # 这里的 global_step 是 Lightning 自动管理的
        # probs = torch.softmax(cls_pred, dim=-1)
        # _, cls_pred_index = torch.max(probs, dim=-1)
        # right_token = (cls_pred_index == target) & seq_mask_no_end_3c
        # acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()
        # real_mask_token = mask_token & seq_mask_no_end_3c
        # acc_masked = get_acc(cls_pred, target, real_mask_token)
        # no_mask_token = (~mask_token) & seq_mask_no_end_3c
        # acc_nomask = get_acc(cls_pred, target, no_mask_token)
        # # 使用 Lightning 的 logger
        # self.log('Validation/all', loss_cls, prog_bar=True)
        # self.log('Validation/every_token', acc_all, prog_bar=True)
        # self.log('Validation/masked', acc_masked, prog_bar=True)
        # self.log('Validation/no_masked', acc_nomask, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Lightning 需要这个方法，但我们的自定义 eval 是在 epoch start 统一跑的
        pass

    def generate(self, texts, betas, gender, length, out_dir, name = None, contact_mask = None):
        normalizer = MotionNormalizerTorchInterX('raw')
        device = self.device
        max_length = self.trans_encoder.block_size - 1
        vocab_contact_map = None
        contact_map_path = self.args.contact_map_path
        if contact_mask is not None:
            vocab_contact_map = torch.from_numpy(np.load(contact_map_path)).float().to(device)
        if name is None:
            name = [t[:10] for t in texts]

        betas = betas.to(device).float()
        gender = gender.to(device)
        tick = time.time()
        text = clip.tokenize(texts, truncate=True).to(device)
        length = torch.tensor(length).to(device)
        m_tokens_len = torch.ceil((length) / 4)

        bs = text.shape[0]

        feat_clip_text, word_emb = self.clip_model(text)
        if contact_mask is not None:
            contact_mask = downsample_contact_mask(contact_mask, factor=4)
            if contact_mask.shape[1] < max_length:
                contact_mask = torch.cat([
                    contact_mask,
                    torch.zeros((bs, max_length - contact_mask.shape[1]), dtype=contact_mask.dtype),
                ], dim=1).to(device)

        sample_kwargs = dict(
            m_length=length, rand_pos=True, betas=betas, gender=gender,
            CFG=self.args.cfg_scale, contact_mask=contact_mask,
            max_steps=self.args.sampling_steps,
        )
        if self.args.sampler == 'sample_intra_cas':
            sample_kwargs.update(
                vocab_contact_map=vocab_contact_map,
                contact_guidance_scale=self.args.contact_guidance_scale,
                temperature=self.args.temperature,
            )
        index_motion = self.trans_encoder(
            feat_clip_text, word_emb, type=self.args.sampler, **sample_kwargs
        )

        for k in range(bs):
            pred_pose = self.net(index_motion[k:k + 1, :int(m_tokens_len[k].item())], betas[k:k + 1], gender[k:k + 1],
                            type='decode')
            pred_pose = normalizer.backward(pred_pose)
            print('time taken for generation: ', time.time() - tick)
            motion_length = int(m_tokens_len[k].item()) * 4
            trans1, pose1, trans2, pose2 = recover_interx(pred_pose[:motion_length].squeeze())
            gender_word = ['male' if g == 0 else 'female' for g in gender[k]]

            np.savez(os.path.join(out_dir, name[k] + '_pred_p1.npz'),
                     poses=pose1, trans=trans1, betas=betas[k, 0:10].cpu().detach().numpy(), gender=gender_word[0],
                     mocap_framerate=30)
            np.savez(os.path.join(out_dir, name[k] + '_pred_p2.npz'),
                     poses=pose2, trans=trans2, betas=betas[k, 10:].cpu().detach().numpy(), gender=gender_word[1],
                     mocap_framerate=30)

    def on_test_epoch_start(self):
        wan_writer = None

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        multi = []

        class LoggerWrapper:
            def __init__(self, log_path, mode='a'):
                """
                log_path: log文件的路径 (例如 'output/train.log')
                mode: 'w' 表示覆盖写入, 'a' 表示追加写入
                """
                # 自动创建父目录（如果不存在）
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                self.file = open(log_path, mode)
                self.path = log_path

            def info(self, msg):
                # 1. 打印到控制台
                print(msg)

                # 2. 写入文件 (注意要手动加换行符 \n)
                self.file.write(msg + '\n')

                # 3. 立即刷新缓冲区，确保程序意外中断时日志已被写入
                self.file.flush()

            def close(self):
                # 关闭文件句柄
                if not self.file.closed:
                    self.file.close()

            def __del__(self):
                # 对象销毁时尝试关闭文件
                self.close()

        custom_logger = LoggerWrapper(self.args.out_dir + '/test_log.txt', mode='a')
        repeat_time = 1
        save_folder = 'final_demo/shape_0_rbh'
        os.makedirs(save_folder, exist_ok=True)
        for i in tqdm(range(repeat_time)):
            _, pose, m_length, clip_text, \
                best_fid, best_iter, best_div, \
                best_top1, best_top2, best_top3, \
                best_matching, best_multi, \
                _, _ = eval_trans.evaluation_transformer(
                save_folder,
                self.trainer.test_dataloaders,
                self.net,
                self.trans_encoder,
                custom_logger,
                wan_writer,
                i,
                1000,
                0,
                100,
                0,
                0,
                0,
                100,
                num_repeat=30,
                clip_model=self.clip_model,
                eval_wrapper=self.eval_wrapper,
                dataname=self.args.dataname,
                save=False,
                savenpy=False,
                rand_pos=True,
                contact=self.args.contact,
                shape_change=None,
                CFG=self.args.cfg_scale,
                sampler=self.args.sampler,
                temperature=self.args.temperature,
                sampling_steps=self.args.sampling_steps,
                contact_guidance_scale=self.args.contact_guidance_scale,
                contact_map_path=self.args.contact_map_path,
            )
            fid.append(best_fid)
            div.append(best_div)
            top1.append(best_top1)
            top2.append(best_top2)
            top3.append(best_top3)
            matching.append(best_matching)
            multi.append(best_multi)
        print('final result:')
        print('fid: ', sum(fid) / repeat_time)
        print('div: ', sum(div) / repeat_time)
        print('top1: ', sum(top1) / repeat_time)
        print('top2: ', sum(top2) / repeat_time)
        print('top3: ', sum(top3) / repeat_time)
        print('matching: ', sum(matching) / repeat_time)
        print('multi: ', sum(multi) / repeat_time)
        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        multi = np.array(multi)
        msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi) * 1.96 / np.sqrt(repeat_time):.3f}"
        custom_logger.info(msg_final)

    def on_validation_epoch_start(self):
        # ------------------
        # G. 自定义评估逻辑
        # ------------------
        # 注意：这里调用的是 eval_trans.evaluation_transformer
        # 这个函数需要 logger 和 writer 对象。
        # 我们使用 self.logger.experiment (TensorboardWriter)

        # 只有主进程跑评估
        if self.trainer.is_global_zero:
            current_iter = self.global_step
            # 获取原始 TensorBoard writer
            # wandb writer
            wan_writer = None

            # 使用 Python logging 的 logger (utils_model.get_logger 创建的)
            # 或者简单传入 self.print
            # 这里为了兼容 eval_trans 接口，我们需要一个类似 logger 的对象
            # 我们可以直接传 None，或者一个简单的 Wrapper，因为 eval_trans 里主要是 logger.info

            # 兼容性处理
            class LoggerWrapper:
                def info(self, msg):
                    print(msg)

            custom_logger = LoggerWrapper()

            print(f"\n[Validation] Starting evaluation at iter {current_iter}...")

            # 调用原始评估函数
            # 注意：eval_trans 内部会循环 val_loader，所以这里不需要 batch
            save_val_dir = os.path.join(self.args.out_dir, 'validation')
            os.makedirs(save_val_dir, exist_ok=True)
            # Formal validation is observational: it uses a fixed seed and
            # restores every RNG stream.  The explicit historical diagnostic
            # instead reproduces the 2026-02-13 sanity/milestone side effects,
            # because those random draws changed all subsequent training masks,
            # dropout and data-loader order in the original run.
            isolated_validation = (
                self.args.training_validation_contract == 'paper_isolated_common'
            )
            validation_sampler = (
                self.args.sampler if isolated_validation
                else 'sample_intra_cas_historical'
            )
            validation_context = (
                preserving_rng_state() if isolated_validation else nullcontext()
            )
            with validation_context:
                if isolated_validation:
                    seed_evaluation(self.args.seed + 500_000)
                inference_context = (
                    temporary_inference_conditioning_contract(
                        self.trans_encoder, 'paper_text_cfg'
                    )
                    if isolated_validation else nullcontext()
                )
                with inference_context:
                    fid, pose, m_length, clip_text, \
                        self.best_fid, self.best_iter, self.best_div, \
                        self.best_top1, self.best_top2, self.best_top3, \
                        self.best_matching, self.best_multi, \
                        _, _, current_metrics = eval_trans.evaluation_transformer(
                        save_val_dir,
                        self.val_loader,
                        self.net,
                        self.trans_encoder,
                        custom_logger,
                        wan_writer,
                        current_iter,
                        self.best_fid,
                        self.best_iter,
                        self.best_div,
                        self.best_top1,
                        self.best_top2,
                        self.best_top3,
                        self.best_matching,
                        clip_model=self.clip_model,
                        eval_wrapper=self.eval_wrapper,
                        dataname=self.args.dataname,
                        save=False,
                        savenpy=False,
                        rand_pos=isolated_validation,
                        contact=self.args.contact,
                        CFG=self.args.cfg_scale,
                        sampler=validation_sampler,
                        temperature=self.args.temperature,
                        sampling_steps=self.args.sampling_steps,
                        contact_guidance_scale=(
                            self.args.contact_guidance_scale
                        ),
                        contact_map_path=self.args.contact_map_path,
                        return_current_metrics=True,
                    )

                    # A one-generation validation pass cannot measure
                    # multimodality and previously produced a misleading zero.
                    # Use a fixed, cheap subset during training; the final metric
                    # still uses the official 100 prompts x 30 generations.
                    validation_mm = None
                    if isolated_validation and self.args.validation_mm_repeats > 1:
                        mm_loader = build_multimodality_loader(
                            self.val_loader,
                            self.args.mm_num_samples,
                            self.args.seed + 510_000,
                        )
                        seed_evaluation(self.args.seed + 520_000)
                        mm_result = eval_trans.evaluation_transformer(
                            save_val_dir,
                            mm_loader,
                            self.net,
                            self.trans_encoder,
                            custom_logger,
                            None,
                            current_iter,
                            1000,
                            0,
                            100,
                            0,
                            0,
                            0,
                            100,
                            clip_model=self.clip_model,
                            eval_wrapper=self.eval_wrapper,
                            dataname=self.args.dataname,
                            num_repeat=self.args.validation_mm_repeats,
                            save=False,
                            savenpy=False,
                            rand_pos=True,
                            contact=self.args.contact,
                            CFG=self.args.cfg_scale,
                            sampler=self.args.sampler,
                            temperature=self.args.temperature,
                            sampling_steps=self.args.sampling_steps,
                            contact_guidance_scale=(
                                self.args.contact_guidance_scale
                            ),
                            contact_map_path=self.args.contact_map_path,
                        )
                        validation_mm = float(mm_result[11])

            self.log('val/FID', current_metrics['FID'])
            self.log('val/Diversity', current_metrics['Diversity'])
            self.log('val/TOP1', current_metrics['R_precision_top1'])
            self.log('val/TOP2', current_metrics['R_precision_top2'])
            self.log('val/TOP3', current_metrics['R_precision_top3'])
            self.log('val/Matching', current_metrics['Matching'])
            self.log('val/R_precision_top1', current_metrics['R_precision_top1'])
            self.log('val/R_precision_top2', current_metrics['R_precision_top2'])
            self.log('val/R_precision_top3', current_metrics['R_precision_top3'])
            if validation_mm is not None:
                self.log(
                    'val/Multimodality_diagnostic', validation_mm,
                )

            # np.save(os.path.join(self.args.out_dir, f'pred{self.global_step}.bpy'), pose.cpu().detach().numpy())

if __name__ == '__main__':
    # ------------------
    # 0. 配置与目录准备
    # ------------------
    args = option_trans.get_args_parser()
    non_training_mode = (
        args.cache_only or args.generate_only or args.evaluate_only
    )
    if not non_training_mode and args.wandb_mode != 'online':
        raise ValueError(
            'all newly launched MGT training requires --wandb-mode online'
        )
    if (
        args.training_validation_contract == 'paper_isolated_common'
        and 1 < args.validation_mm_repeats <= 10
    ):
        raise ValueError(
            '--validation-mm-repeats must be 1 (disabled) or at least 11: '
            'the official evaluator samples 10 repeat pairs and requires '
            'strictly more generated repeats'
        )
    os.environ['HHC_INTERX_STATS_DIR'] = os.path.abspath(args.stats_dir)
    os.environ['HHC_INTERX_STATS_PREFIX'] = args.stats_prefix
    pl.seed_everything(args.seed)
    historical_validation = (
        args.training_validation_contract == 'historical_test_side_effect'
    )
    if historical_validation:
        required_historical_validation = {
            'sampler': (args.sampler, 'sample_intra_cas'),
            'sampling_steps': (args.sampling_steps, 10),
            'temperature': (args.temperature, 0.0),
            'cfg_scale': (args.cfg_scale, 3.5),
            'contact': (args.contact, False),
        }
        mismatches = {
            key: {'actual': actual, 'required': required}
            for key, (actual, required) in required_historical_validation.items()
            if actual != required
        }
        if mismatches:
            raise ValueError(
                'historical_test_side_effect requires the recovered 2026-02-13 '
                f'validation runtime: {mismatches}'
            )
    if args.generate_only or args.evaluate_only:
        if args.gen_ckpt is None:
            raise ValueError('--gen-ckpt is required for generation or evaluation')
        args.resume_trans = args.gen_ckpt

    args.out_dir = os.path.join(args.out_dir, f't2i_trans')

    # 路径设置
    args.vq_dir = os.path.dirname(args.vq_checkpoint)
    args.cache_root = os.path.abspath(args.cache_root or args.vq_dir)
    args.contact_map_path = os.path.join(args.cache_root, 'code_contact_probs.npy')
    codebook_dir = args.codebook_dir
    contact_token_dir = os.path.join(args.cache_root, 'contact_token_label')
    args.resume_pth = args.vq_checkpoint
    if not os.path.isfile(args.resume_pth):
        raise FileNotFoundError(
            f'VQ checkpoint not found: {args.resume_pth}. Train it with train/train_vq_interx.py first.'
        )
    os.makedirs(args.vq_dir, exist_ok=True)
    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(codebook_dir, exist_ok=True)
    os.makedirs(contact_token_dir, exist_ok=True)
    run_state = prepare_mgt_output_directory(args)
    os.makedirs(args.out_dir, exist_ok=True)
    args.wandb_run_id = None
    if args.wandb_mode == 'online':
        if run_state.get('mode') == 'strict_in_place_resume':
            with open(run_state['prior_manifest'], encoding='utf-8') as handle:
                prior_observability = json.load(handle).get('observability', {})
            args.wandb_run_id = prior_observability.get('wandb_run_id')
            if not args.wandb_run_id:
                raise ValueError('online MGT resume manifest lacks wandb_run_id')
        else:
            args.wandb_run_id = wandb.util.generate_id()

    # Logger
    # logger = utils_model.get_logger(args.out_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ------------------
    # 1. 数据准备
    # ------------------
    w_vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')

    # Val Loader
    val_file = (
        'dataset/interx/splits/test.txt'
        if historical_validation else args.val_split_file
    )
    val_motion = (
        'dataset/interx/motions/test.h5'
        if historical_validation else args.val_motion_file
    )
    val_loader = dataset_interx_TM.DATALoader(
        args.dataname, 32, val_file, w_vectorizer, val_motion,
        unit_length=2 ** args.down_t,
        shuffle=not historical_validation,
        max_motion_length=152 if historical_validation else 150,
        min_motion_len=30 if historical_validation else 24,
    )

    # Eval Wrapper
    dataset_opt_path = 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
    args.nb_joints = 55
    # 注意：Lightning 会处理 device，这里 device 只是用来初始化 wrapper
    eval_wrapper = EvaluatorModelWrapper(get_opt(dataset_opt_path, args.device))

    # Generate/resume the token cache per sample.  Checking only whether the
    # directory is empty makes an interrupted 11k-sample export look complete.
    train_file = args.train_split_file
    train_motion = args.train_motion_file
    # Training/cache identity remains train+official-val even for the historical
    # diagnostic.  The test split above is used only by its recovered validation
    # hook and must never enter token-cache construction or contact-map fitting.
    val_file = args.val_split_file
    all_file = args.all_split_file
    all_motion = args.all_motion_file
    if args.train_asset_root:
        if not args.augmentation_manifest:
            raise ValueError(
                '--augmentation-manifest is required with --train-asset-root'
            )
        augmentation_manifest_path = os.path.abspath(args.augmentation_manifest)
        train_asset_root = os.path.abspath(args.train_asset_root)
        augmentation_manifest = validate_materialized_interx_shape_augmentation(
            train_asset_root, augmentation_manifest_path,
        )
        expected_training_paths = {
            'train_split_file': os.path.join(train_asset_root, 'splits', 'train.txt'),
            'all_split_file': os.path.join(train_asset_root, 'splits', 'all.txt'),
            'train_motion_file': os.path.join(train_asset_root, 'motions', 'train.h5'),
            'all_motion_file': os.path.join(train_asset_root, 'motions', 'all.h5'),
            'stats_dir': train_asset_root,
        }
        path_mismatches = {
            key: (os.path.abspath(getattr(args, key)), expected)
            for key, expected in expected_training_paths.items()
            if os.path.abspath(getattr(args, key)) != expected
        }
        if path_mismatches:
            raise ValueError(
                'shape augmentation requires the materialized train/all split, '
                f'motion, and unchanged copied statistics: {path_mismatches}'
            )
        if args.stats_prefix != 'Interx':
            raise ValueError('shape augmentation requires --stats-prefix Interx')
        args.augmentation_mode = augmentation_manifest['augmentation_mode']
        train_beta_dir = os.path.join(train_asset_root, 'betas')
        train_contact_dir = os.path.join(train_asset_root, 'contact')
        train_text_dir = os.path.join(train_asset_root, 'texts_processed')
        for path in (train_beta_dir, train_contact_dir, train_text_dir):
            if not os.path.isdir(path):
                raise FileNotFoundError(path)
        augmentation_manifest_sha256 = sha256_file(augmentation_manifest_path)
    else:
        if args.augmentation_manifest:
            raise ValueError(
                '--augmentation-manifest requires --train-asset-root'
            )
        train_beta_dir = 'dataset/interx/betas'
        train_contact_dir = 'dataset/interx/contact'
        train_text_dir = 'dataset/interx/texts_processed'
        augmentation_manifest_sha256 = None
        augmentation_manifest = None
        args.augmentation_mode = None

    def split_names(path):
        with open(path, 'r', encoding='utf-8') as handle:
            return {line.strip() for line in handle if line.strip()}

    raw_train_names = split_names(train_file)
    raw_val_names = split_names(val_file)
    train_cache_names = eligible_motion_names(raw_train_names, all_motion)
    val_cache_names = eligible_motion_names(raw_val_names, all_motion)
    expected_cache_names = train_cache_names | val_cache_names
    if raw_train_names & raw_val_names:
        raise RuntimeError('Inter-X train and validation splits overlap')
    print(
        'MGT cache contract: '
        f'{len(train_cache_names)}/{len(raw_train_names)} train + '
        f'{len(val_cache_names)}/{len(raw_val_names)} validation samples are eligible'
    )
    cached_code_names = {
        os.path.splitext(filename)[0] for filename in os.listdir(codebook_dir)
        if filename.endswith('.npy')
    }
    cached_contact_names = {
        os.path.splitext(filename)[0] for filename in os.listdir(contact_token_dir)
        if filename.endswith('.npy')
    }
    contact_map_path = args.contact_map_path
    cache_manifest_path = os.path.join(codebook_dir, '_cache_manifest.json')
    cache_identity = {
        'vq_checkpoint_sha256': sha256_file(args.resume_pth),
        'nb_code': int(args.nb_code),
        'code_dim': int(args.code_dim),
        'down_t': int(args.down_t),
        'token_cache_contract': args.token_cache_contract,
        'contact_map_scope': 'train_only',
        'contact_map_estimator': 'empirical_frequency_unseen_neutral_0.5',
        'eligibility': 'present_in_h5_and_30<=frames<1000',
        'train_split_sha256': sha256_file(train_file),
        'val_split_sha256': sha256_file(val_file),
        'eligible_train_samples': int(len(train_cache_names)),
        'eligible_val_samples': int(len(val_cache_names)),
        'expected_samples': int(len(expected_cache_names)),
    }
    hybrid_original_names = set()
    hybrid_augmented_names = set()
    if args.token_cache_contract in (
            'historical_mmm_exact', HYBRID_SHAPE_TOKEN_CACHE_CONTRACT):
        is_hybrid_cache = (
            args.token_cache_contract == HYBRID_SHAPE_TOKEN_CACHE_CONTRACT
        )
        if args.train_asset_root and not is_hybrid_cache:
            raise ValueError(
                'historical_mmm_exact is a frozen baseline cache and cannot '
                'tokenize shape-augmentation overlays'
            )
        if is_hybrid_cache and not args.train_asset_root:
            raise ValueError(
                'hybrid shape cache requires --train-asset-root and a bound '
                '--augmentation-manifest'
            )
        if not args.historical_codebook_source or not args.historical_contact_source:
            raise ValueError(
                f'{args.token_cache_contract} requires '
                '--historical-codebook-source and --historical-contact-source'
            )
        historical_codebook_source = os.path.abspath(args.historical_codebook_source)
        historical_contact_source = os.path.abspath(args.historical_contact_source)
        if not os.path.isdir(historical_codebook_source):
            raise FileNotFoundError(historical_codebook_source)
        if not os.path.isdir(historical_contact_source):
            raise FileNotFoundError(historical_contact_source)
        if is_hybrid_cache:
            hybrid_original_names, hybrid_augmented_names, _ = (
                partition_expected_names(
                    expected_cache_names, augmentation_manifest_path,
                )
            )
            historical_expected_names = hybrid_original_names
        else:
            historical_expected_names = expected_cache_names
        missing_historical_code = sorted(
            name for name in historical_expected_names
            if not os.path.isfile(pjoin(historical_codebook_source, name + '.npy'))
        )
        missing_historical_contact = sorted(
            name for name in historical_expected_names
            if not os.path.isfile(pjoin(historical_contact_source, name + '.npy'))
        )
        if missing_historical_code or missing_historical_contact:
            raise FileNotFoundError(
                'historical MMM cache is incomplete for official train+val: '
                f'code={missing_historical_code[:5]}, '
                f'contact={missing_historical_contact[:5]}'
            )
        cache_identity.update({
            'cache_seed': int(args.seed) if is_hybrid_cache else None,
            'cache_crop_policy': (
                'historical_exact_original_plus_deterministic_historical_root_augmented_v1'
                if is_hybrid_cache else
                'historical_mmm_process_rng_single_single_double_v1'
            ),
            'cache_scope': (
                'shape_augmented_train+official_val_hybrid'
                if is_hybrid_cache else
                'official_train+val_imported_from_full_historical_cache'
            ),
            'cache_input_filter': 'official_eligible_train_val_only',
            'normalization_mean_sha256': sha256_file(os.path.join(
                args.stats_dir, f'{args.stats_prefix}_Mean_root.npy'
            )),
            'normalization_std_sha256': sha256_file(os.path.join(
                args.stats_dir, f'{args.stats_prefix}_Std_root.npy'
            )),
            'historical_codebook_source': historical_codebook_source,
            'historical_contact_source': historical_contact_source,
            'historical_codebook_expected_names_sha256': sha256_named_files(
                historical_codebook_source, historical_expected_names
            ),
            'historical_contact_expected_names_sha256': sha256_named_files(
                historical_contact_source, historical_expected_names
            ),
        })
        if is_hybrid_cache:
            cache_identity.update({
                'cache_input_filter': (
                    'historical_original_import_plus_missing_augmented_only'
                ),
                'hybrid_original_samples': len(hybrid_original_names),
                'hybrid_augmented_samples': len(hybrid_augmented_names),
                'tokenizer_implementation_sha256': sha256_file(
                    'dataset/dataset_interx_tokenize_historical_root.py'
                ),
            })
            tokenizer_dataset = dataset_interx_tokenize_historical_root
        else:
            assert_exact_historical_cache_scope(
                cached_code_names, cached_contact_names, expected_cache_names
            )
    else:
        if args.token_cache_contract == 'deterministic_historical_root_v1':
            tokenizer_dataset = dataset_interx_tokenize_historical_root
            normalization_suffix = '_root'
        else:
            tokenizer_dataset = dataset_interx_tokenize
            normalization_suffix = ''
        normalization_mean_path = os.path.join(
            args.stats_dir,
            f'{args.stats_prefix}_Mean{normalization_suffix}.npy',
        )
        normalization_std_path = os.path.join(
            args.stats_dir,
            f'{args.stats_prefix}_Std{normalization_suffix}.npy',
        )
        historical_codebook_source = None
        historical_contact_source = None
        cache_identity.update({
            'cache_seed': int(args.seed),
            'cache_crop_policy': tokenizer_dataset.CACHE_CROP_POLICY,
            'cache_scope': 'train+val',
            'cache_input_filter': 'missing_eligible_train_val_before_tokenization',
            'normalization_mean_sha256': sha256_file(normalization_mean_path),
            'normalization_std_sha256': sha256_file(normalization_std_path),
        })
        if args.token_cache_contract == 'deterministic_historical_root_v1':
            cache_identity['tokenizer_implementation_sha256'] = sha256_file(
                'dataset/dataset_interx_tokenize_historical_root.py'
            )
    # Preserve byte-level compatibility with already completed baseline cache
    # manifests.  Augmentation identity is an extension of that contract, not
    # a pair of nullable fields that retroactively invalidates every baseline.
    if augmentation_manifest_sha256 is not None:
        cache_identity.update({
            'augmentation_manifest_sha256': augmentation_manifest_sha256,
            'train_asset_root': os.path.abspath(args.train_asset_root),
            'augmentation_mode': augmentation_manifest['augmentation_mode'],
            'augmentation_train_motion_sha256': (
                augmentation_manifest['motions']['train_sha256']
            ),
            'augmentation_all_motion_sha256': (
                augmentation_manifest['motions']['all_sha256']
            ),
        })
    cache_artifacts_exist = bool(
        cached_code_names
        or cached_contact_names
        or os.path.isfile(contact_map_path)
    )
    cache_build_mode = not (args.generate_only or args.evaluate_only)
    if os.path.isfile(cache_manifest_path):
        with open(cache_manifest_path, 'r', encoding='utf-8') as handle:
            cache_manifest = json.load(handle)
        mismatches = {
            key: (cache_manifest.get(key), expected)
            for key, expected in cache_identity.items()
            if cache_manifest.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                'token cache belongs to a different VQ/configuration: '
                f'{mismatches}. Use empty cache/contact paths before training.'
            )
        if cache_manifest.get('status') == 'complete':
            recorded_contact_hash = cache_manifest.get('contact_map_sha256')
            if not os.path.isfile(contact_map_path):
                raise FileNotFoundError(
                    f'complete cache manifest references a missing contact map: {contact_map_path}'
                )
            actual_contact_hash = sha256_file(contact_map_path)
            if recorded_contact_hash != actual_contact_hash:
                raise RuntimeError(
                    'contact-guidance map hash does not match its cache manifest: '
                    f'{recorded_contact_hash} != {actual_contact_hash}'
                )
            if cache_build_mode:
                complete_code_names = cached_code_names & expected_cache_names
                complete_contact_names = cached_contact_names & expected_cache_names
                if (complete_code_names != expected_cache_names or
                        complete_contact_names != expected_cache_names):
                    raise RuntimeError('complete manifest has an incomplete token/contact cache')
                actual_code_hash = sha256_named_files(codebook_dir, expected_cache_names)
                actual_cached_contact_hash = sha256_named_files(
                    contact_token_dir, expected_cache_names
                )
                if cache_manifest.get('code_cache_sha256') != actual_code_hash:
                    raise RuntimeError('code-token cache hash does not match its manifest')
                if cache_manifest.get('contact_cache_sha256') != actual_cached_contact_hash:
                    raise RuntimeError('contact-token cache hash does not match its manifest')
    elif cache_build_mode:
        if cache_artifacts_exist:
            raise RuntimeError(
                'token/contact cache artifacts exist without a provenance manifest. '
                'They cannot be safely associated with this VQ checkpoint.'
            )
        cache_manifest = dict(cache_identity, status='building')
        atomic_write_json(cache_manifest_path, cache_manifest)
    else:
        cache_manifest = None

    cache_was_complete = bool(
        cache_manifest is not None and cache_manifest.get('status') == 'complete'
    )
    if not cache_build_mode and args.contact and not cache_was_complete:
        raise RuntimeError(
            'contact-guided generation requires a complete, validated token-cache manifest'
        )

    missing_cache_names = expected_cache_names - (cached_code_names & cached_contact_names)
    if (cache_build_mode and missing_cache_names
            and args.token_cache_contract in (
                'historical_mmm_exact', HYBRID_SHAPE_TOKEN_CACHE_CONTRACT)):
        import_names = (
            missing_cache_names & hybrid_original_names
            if args.token_cache_contract == HYBRID_SHAPE_TOKEN_CACHE_CONTRACT
            else missing_cache_names
        )
        print(
            f'Importing {len(import_names)} byte-exact historical MMM '
            'token/contact pairs into the isolated cache...'
        )
        if args.token_cache_contract == HYBRID_SHAPE_TOKEN_CACHE_CONTRACT:
            import_original_pairs_byte_exact(
                import_names, historical_codebook_source,
                historical_contact_source, codebook_dir, contact_token_dir,
            )
        else:
            import_historical_token_pairs(
                historical_codebook_source,
                historical_contact_source,
                codebook_dir,
                contact_token_dir,
                import_names,
            )
        cached_code_names = {
            os.path.splitext(filename)[0] for filename in os.listdir(codebook_dir)
            if filename.endswith('.npy')
        }
        cached_contact_names = {
            os.path.splitext(filename)[0] for filename in os.listdir(contact_token_dir)
            if filename.endswith('.npy')
        }
        missing_cache_names = expected_cache_names - (
            cached_code_names & cached_contact_names
        )
    if not (args.generate_only or args.evaluate_only) and missing_cache_names:
        print(f"Generating/resuming codebook cache for {len(missing_cache_names)} samples...")
        # 这里的 net 初始化只是为了生成 codebook，可以用 cpu 或 gpu
        # 为了简单，我们手动跑一下这部分逻辑，或者把它也封装进 LightningModule (略显复杂)
        # 这里保持原样，手动跑
        net = build_model(args, "interx_vq")
        if args.resume_pth:
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            state_dict = ckpt['net']
            # ... (合并权重逻辑同上) ...
            key_p1 = "fk_layers.p1_positions_lookup.weight"
            key_p2 = "fk_layers.p2_positions_lookup.weight"
            new_key = "fk_layers.unified_positions_lookup.weight"
            if key_p1 in state_dict and key_p2 in state_dict:
                p1_weight = state_dict.pop(key_p1);
                p2_weight = state_dict.pop(key_p2)
                state_dict[new_key] = torch.cat([p1_weight, p2_weight], dim=0)
            net.load_state_dict(state_dict, strict=True)
            net.to(args.device)
            net.eval()

        train_loader_token = tokenizer_dataset.DATALoader(
            args.dataname, 1, all_file, w_vectorizer, all_motion,
            unit_length=2 ** args.down_t,
            shuffle=False,
            cache_seed=args.seed,
            include_names=missing_cache_names,
            beta_dir=train_beta_dir,
            contact_dir=train_contact_dir,
            fallback_beta_dir='dataset/interx/betas',
            fallback_contact_dir='dataset/interx/contact',
            strict_assets=True,
            stats_dir=args.stats_dir,
            stats_prefix=args.stats_prefix,
        )
        for batch in tqdm(train_loader_token):
            pose, m_length, name, betas, gender, group, contact = batch
            if name[0] not in missing_cache_names:
                continue
            pose = pose.to(args.device).float()
            betas = betas.to(args.device).float()
            gender = gender.to(args.device)
            target = net(pose, betas, gender, type='encode')
            atomic_save_npy(pjoin(contact_token_dir, name[0] + '.npy'), contact.cpu().numpy())
            atomic_save_npy(pjoin(codebook_dir, name[0] + '.npy'), target.cpu().numpy())
        del net
        cached_code_names = {
            os.path.splitext(filename)[0] for filename in os.listdir(codebook_dir)
            if filename.endswith('.npy')
        }
        cached_contact_names = {
            os.path.splitext(filename)[0] for filename in os.listdir(contact_token_dir)
            if filename.endswith('.npy')
        }
        still_missing = expected_cache_names - (cached_code_names & cached_contact_names)
        if still_missing:
            raise RuntimeError(
                f'code/contact cache incomplete after export: {len(still_missing)} missing; '
                f'first={sorted(still_missing)[:5]}'
            )
    if cache_build_mode:
        complete_code_names = cached_code_names & expected_cache_names
        complete_contact_names = cached_contact_names & expected_cache_names
        if complete_code_names != expected_cache_names or complete_contact_names != expected_cache_names:
            raise RuntimeError('refusing to finalize an incomplete MGT token/contact cache')
        cache_manifest = dict(
            cache_identity,
            status='tokens_complete',
            cached_code_samples=len(complete_code_names),
            cached_contact_samples=len(complete_contact_names),
            code_cache_sha256=sha256_named_files(codebook_dir, expected_cache_names),
            contact_cache_sha256=sha256_named_files(contact_token_dir, expected_cache_names),
        )
        if args.token_cache_contract == HYBRID_SHAPE_TOKEN_CACHE_CONTRACT:
            hybrid_audit = audit_hybrid_cache_subsets(
                hybrid_original_names, hybrid_augmented_names,
                historical_codebook_source, historical_contact_source,
                codebook_dir, contact_token_dir,
            )
            if hybrid_audit['status'] != 'pass':
                raise RuntimeError('hybrid shape token-cache subset audit failed')
            cache_manifest['hybrid_subset_audit'] = hybrid_audit
        atomic_write_json(cache_manifest_path, cache_manifest)

    if cache_build_mode and (
            not cache_was_complete or not os.path.isfile(contact_map_path)):
        probabilities, _, used_files = build_code_contact_map(
            codebook_dir, contact_token_dir, args.nb_code,
            included_names=train_cache_names,
        )
        atomic_save_npy(contact_map_path, probabilities)
        print(f'Generated contact-guidance map from {used_files} token files: {contact_map_path}')
    if cache_build_mode:
        cache_manifest.update({
            'status': 'complete',
            'contact_map_path': os.path.abspath(contact_map_path),
            'contact_map_sha256': sha256_file(contact_map_path),
            'contact_map_samples': int(len(train_cache_names)),
        })
        atomic_write_json(cache_manifest_path, cache_manifest)
    elif args.contact and not os.path.isfile(contact_map_path):
        raise FileNotFoundError(
            f'contact guidance map not found: {contact_map_path}. Build the VQ '
            'token cache with the training command first.'
        )

    if args.cache_only:
        if args.generate_only or args.evaluate_only:
            raise ValueError('--cache-only cannot be combined with generation/evaluation modes')
        print(
            f'VQ token/contact cache is complete and provenance-validated: '
            f'{cache_manifest_path}'
        )
        raise SystemExit(0)

    # ------------------
    # 2. 训练设置
    # ------------------
    model = MaskGITLightning(args, eval_wrapper, val_loader)
    if args.generate_only or args.evaluate_only:
        export_split = args.eval_split if args.evaluate_only else args.gen_split
        export_loader = build_split_loader(args, export_split, w_vectorizer)
        model = model.to(args.device)
        model.eval()
        if args.generate_only:
            run_generation_export(args, model, export_loader)
        else:
            run_repeated_evaluation(args, model, export_loader)
        raise SystemExit(0)

    # Train Loader
    training_token_max = (
        38 if args.training_token_length_contract in {
                    'original_mmm_38', 'historical_38'
                }
        else args.block_size - 1
    )
    train_loader = dataset_interx_TM.DATALoader_Train(args.dataname, args.batch_size, train_file, args.nb_code,
                                                      codebook_dir, unit_length=2 ** args.down_t,
                                                      num_workers=args.num_workers, contact_dir=contact_token_dir,
                                                      max_motion_length=training_token_max,
                                                      text_dir=train_text_dir,
                                                      beta_dir=train_beta_dir,
                                                      fallback_beta_dir='dataset/interx/betas',
                                                      fallback_text_dir='dataset/interx/texts_processed',
                                                      strict_assets=True)
    val_loss_file = val_file
    val_loss_codebook_dir = codebook_dir
    val_loss_contact_dir = contact_token_dir
    val_loss_expected_names = val_cache_names
    val_loss_label = 'validation'
    if historical_validation:
        val_loss_file = 'dataset/interx/splits/test.txt'
        val_loss_codebook_dir = args.historical_codebook_source
        val_loss_contact_dir = args.historical_contact_source
        if not val_loss_codebook_dir or not val_loss_contact_dir:
            raise ValueError(
                'historical validation requires the recovered full code/contact cache'
            )
        test_names = split_names(val_loss_file)
        val_loss_expected_names = eligible_motion_names(
            test_names, 'dataset/interx/motions/test.h5'
        )
        val_loss_label = 'historical_test_validation_hook'
    val_loss_loader = dataset_interx_TM.DATALoader_Train(args.dataname, args.batch_size, val_loss_file, args.nb_code,
                                                         val_loss_codebook_dir, unit_length=2 ** args.down_t,
                                                         num_workers=args.num_workers, contact_dir=val_loss_contact_dir,
                                                         max_motion_length=training_token_max,
                                                         text_dir=train_text_dir,
                                                         beta_dir=train_beta_dir,
                                                         fallback_beta_dir='dataset/interx/betas',
                                                         fallback_text_dir='dataset/interx/texts_processed',
                                                         strict_assets=True)
    loader_contract = {}
    for split_name, loader, expected_names in (
        ('train', train_loader, train_cache_names),
        (val_loss_label, val_loss_loader, val_loss_expected_names),
    ):
        loaded_names = set(loader.dataset.data_dict)
        if loaded_names != expected_names:
            raise RuntimeError(
                f'{split_name} MGT dataset silently lost or gained samples: '
                f'missing={sorted(expected_names - loaded_names)[:5]}, '
                f'extra={sorted(loaded_names - expected_names)[:5]}'
            )
        active_names = set(loader.dataset.name_list[loader.dataset.pointer:])
        loader_contract[split_name] = {
            'eligible_cached_samples': len(loaded_names),
            'active_samples_at_minimum_20_tokens': len(active_names),
            'drop_last_batches': len(loader),
            'batch_size': loader.batch_size,
        }
        print(
            f'MGT {split_name} loader contract: {len(loaded_names)} cached samples, '
            f'{len(active_names)} with at least {loader.dataset.max_length} tokens, '
            f'{len(loader)} drop-last batches of {loader.batch_size}'
        )
    write_mgt_training_manifest(
        args, cache_manifest_path, contact_map_path, loader_contract
    )

    # Checkpoint 回调 (保存最佳 FID)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.out_dir,
        monitor='val/FID',  # 我们在 eval_trans 里手动保存了，这里只是占位或者保存 last
        filename='model-{step:09d}-{val/FID:.4f}',
        save_top_k=(-1 if args.save_all_validation_checkpoints else 10),
        # Validation candidates are used only for held-out validation replay
        # and final weight selection. Keep optimizer/scheduler state in the
        # independent rolling latest/last checkpoints below, rather than
        # duplicating Adam moments in every 10k candidate.
        save_weights_only=True,
        mode='min',
        auto_insert_metric_name=False,
    )
    latest_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.out_dir,
        filename='latest-{step:09d}',
        every_n_train_steps=args.checkpoint_iter,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    validation_pair_audit_callback = FixedStepValidationPairCheckpoint(
        args.out_dir, step=20000
    )
    callbacks = [
        checkpoint_callback,
        latest_checkpoint_callback,
        validation_pair_audit_callback,
    ]
    if args.diagnostic_stop_iter:
        if not (0 < args.diagnostic_stop_iter < args.total_iter):
            raise ValueError(
                'diagnostic-stop-iter must be within (0,total-iter)'
            )
        if args.diagnostic_stop_iter % args.eval_iter:
            raise ValueError('diagnostic-stop-iter must align with eval-iter')
        callbacks.append(DiagnosticStopAfterValidation(args.diagnostic_stop_iter))

    # Logger
    wandb_logger = False if args.wandb_mode == 'disabled' else WandbLogger(
        project="NewInter", config=args, group='mask_gen',
        name=args.exp_name, offline=(args.wandb_mode == 'offline'),
        id=args.wandb_run_id,
        resume='allow' if args.wandb_mode == 'online' else None,
    )

    # Trainer
    # 注意：limit_val_batches=0 会跳过 Lightning 的标准 validate loop，但会触发 on_validation_epoch_start
    # 但为了让 eval_trans 运行，我们需要让 validate 至少跑一次。
    # 这里我们用一个小技巧：val_check_interval 控制频率，
    # 传入 val_loader 给 trainer，但在 validation_step 里什么都不做。

    trainer = pl.Trainer(
        max_steps=args.total_iter,
        accelerator='gpu',
        devices=[int(args.device.rsplit(':', 1)[1])],
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=args.eval_iter,  # 替代 nb_iter % args.eval_iter == 0
        check_val_every_n_epoch=None,
        precision="bf16-mixed",
        num_sanity_val_steps=2 if historical_validation else 0,
    )

    # ------------------
    # 3. 开始训练
    # ------------------
    # 这里的 val_dataloaders 只是为了触发 validation hook
    # 真实的 evaluation 数据在 MaskGITLightning.val_loader 里
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loss_loader,
        ckpt_path=args.resume_training_ckpt,
    )

    # -------------------------
    # 3. 直接运行验证（跳过训练）
    # test_file = 'dataset/interx/test.txt'
    # test_motion = 'dataset/interx/motions/test.h5'
    # test_loader = dataset_interx_TM.DATALoader(args.dataname, 32, test_file, w_vectorizer, test_motion, unit_length=2 ** args.down_t, shuffle=False)
    # trainer.test(model, dataloaders=val_loader, ckpt_path='output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt')

    # trainer.validate(model, dataloaders=val_loader, ckpt_path='output/t2i_trans/2026-02-02-11-22-01_vq_mask_gen_4/model-step=190000-val/FID=0.2891.ckpt')

    # generate
