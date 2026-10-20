#!/usr/bin/env python
"""Measure MGT token prediction under training-like and cascade inference masks.

This diagnostic does not decode motion or use the test split.  It compares
fixed validation tokens under the masks seen during training with the fully
masked and Root -> Body -> Hands contexts used at generation time.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
import numpy as np
import torch
import torch.nn.functional as F

from dataset.dataset_interx_TM import Text2MotionDatasetV2HHI_Token
from train.train_t2i_trans import MaskGITLightning
from utils.motion_masks import sample_kinematic_cascade_mask
from utils.runtime import generate_src_mask


PARTS = ('body', 'hand', 'root')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_batch(dataset, names, max_length):
    if type(max_length) is not int or max_length < 2:
        raise ValueError('token window must leave room for payload and end token')
    targets, lengths, captions, betas, genders = [], [], [], [], []
    end_id = dataset.codebook_size
    pad_id = dataset.codebook_size + 1
    for name in names:
        record = dataset.data_dict[name]
        tokens = np.asarray(record['m_token_list'], dtype=np.int64)
        length = min(len(tokens), max_length - 1)
        target = np.full((max_length, 3), pad_id, dtype=np.int64)
        target[:length] = tokens[:length]
        target[length] = end_id
        targets.append(target)
        lengths.append(length)
        captions.append(record['text'][0]['caption'])
        betas.append(np.asarray(record['betas'], dtype=np.float32))
        genders.append(np.asarray(record['gender'], dtype=np.int64))
    return (
        captions,
        torch.from_numpy(np.stack(targets)),
        torch.tensor(lengths, dtype=torch.long),
        torch.from_numpy(np.stack(betas)),
        torch.from_numpy(np.stack(genders)),
    )


def accumulator(calibration_temperatures=()):
    return {
        condition: {
            part: {
                'loss_sum': 0.0,
                'correct': 0,
                'tokens': 0,
                'entropy_sum': 0.0,
                'top1_probability_sum': 0.0,
                'top1_top2_margin_sum': 0.0,
                'ranking_saturated_tokens': 0,
                'ranking_logit_margin_sum': 0.0,
                'target_probability_sum': 0.0,
                'class_count': None,
                'temperature_loss_sums': {
                    str(float(value)): 0.0
                    for value in calibration_temperatures
                },
            }
            for part in PARTS
        }
        for condition in (
            'paper_scheduled_mask',
            'all_parts_fully_masked',
            'body_after_gt_root',
            'hand_after_gt_root_body',
            'single_part_masked_other_parts_gt',
        )
    }


def add_metrics(store, condition, logits, target, evaluation_mask,
                calibration_temperatures=(), sharpness_temperature=0.5):
    if sharpness_temperature <= 0:
        raise ValueError('sharpness temperature must be positive')
    prediction = logits.argmax(dim=-1)
    for channel, part in enumerate(PARTS):
        mask = evaluation_mask[..., channel]
        count = int(mask.sum().item())
        if not count:
            continue
        selected_logits = logits[..., channel, :][mask].float()
        selected_target = target[..., channel][mask]
        class_count = int(selected_logits.shape[-1])
        previous_class_count = store[condition][part]['class_count']
        if previous_class_count not in (None, class_count):
            raise ValueError('logit class count changed within one diagnostic')
        store[condition][part]['class_count'] = class_count
        # Commitment ranking uses untempered probabilities, unlike proposals.
        ranking_probability = selected_logits.softmax(dim=-1).amax(dim=-1)
        logit_top2 = selected_logits.topk(2, dim=-1).values
        store[condition][part]['ranking_saturated_tokens'] += int(
            (ranking_probability == 1.0).sum().item()
        )
        store[condition][part]['ranking_logit_margin_sum'] += float(
            (logit_top2[:, 0] - logit_top2[:, 1]).sum().item()
        )
        log_probability = F.log_softmax(
            selected_logits / float(sharpness_temperature), dim=-1
        )
        probability = log_probability.exp()
        top2 = probability.topk(2, dim=-1).values
        store[condition][part]['entropy_sum'] += float(
            (-(probability * log_probability).sum(dim=-1)).sum().item()
        )
        store[condition][part]['top1_probability_sum'] += float(
            top2[:, 0].sum().item()
        )
        store[condition][part]['top1_top2_margin_sum'] += float(
            (top2[:, 0] - top2[:, 1]).sum().item()
        )
        store[condition][part]['target_probability_sum'] += float(
            probability.gather(1, selected_target[:, None]).sum().item()
        )
        store[condition][part]['loss_sum'] += float(
            F.cross_entropy(selected_logits, selected_target, reduction='sum').item()
        )
        for value in calibration_temperatures:
            if value <= 0:
                raise ValueError('calibration temperatures must be positive')
            store[condition][part]['temperature_loss_sums'][
                str(float(value))
            ] += float(F.cross_entropy(
                selected_logits / float(value), selected_target,
                reduction='sum',
            ).item())
        store[condition][part]['correct'] += int(
            (prediction[..., channel][mask] == selected_target).sum().item()
        )
        store[condition][part]['tokens'] += count


@torch.inference_mode()
def audit_checkpoint(args, manifest_args, dataset, names, checkpoint, device,
                     contract='paper_current'):
    model_args = Namespace(**manifest_args)
    if contract == 'historical_mmm':
        model_args.paper_architecture = False
        model_args.embed_dim_gpt = 1024
        model_args.n_head_gpt = 16
        model_args.num_layers = 9
        model_args.num_local_layer = 2
    elif contract != 'paper_current':
        raise ValueError(f'unknown MGT contract: {contract}')
    model_args.resume_trans = str(checkpoint)
    model_args.resume_training_ckpt = None
    model_args.device = str(device)
    model = MaskGITLightning(model_args, None, None).to(device).eval()
    if torch.device(device).type == 'cpu':
        model.clip_model.float()
    trans = model.trans_encoder
    inference_contract = getattr(args, 'inference_conditioning_contract', None)
    if inference_contract is not None:
        trans.trans_base.conditioning_contract = inference_contract
    mask_id = int(model_args.nb_code) + 2
    max_length = int(model_args.block_size) - 1
    requested_window = getattr(args, 'token_window', None)
    if requested_window is not None:
        if not 2 <= requested_window <= max_length:
            raise ValueError('token window exceeds model capacity or has no payload')
        max_length = requested_window
    totals = accumulator(args.calibration_temperatures)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    selection_generator = torch.Generator(device=device).manual_seed(args.seed + 700)
    selection_totals = {part: {} for part in PARTS}

    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start + args.batch_size]
        captions, target, lengths, betas, genders = make_batch(
            dataset, batch_names, max_length
        )
        target = target.to(device)
        lengths = lengths.to(device)
        betas = betas.to(device)
        genders = genders.to(device)
        valid = generate_src_mask(max_length, lengths)
        valid3 = valid.unsqueeze(-1).expand(-1, -1, 3)
        attention_mask = (
            valid if model_args.paper_architecture
            else generate_src_mask(max_length, lengths + 1)
        )
        text = clip.tokenize(captions, truncate=True).to(device)
        clip_global, clip_words = model.clip_model(text)

        scheduled, _, _ = sample_kinematic_cascade_mask(
            valid3, generator=generator
        )
        scheduled_input = torch.where(scheduled, mask_id, target)
        scheduled_logits = trans(
            scheduled_input, clip_global, src_mask=attention_mask, word_emb=clip_words,
            betas=betas, gender=genders,
        )
        add_metrics(
            totals, 'paper_scheduled_mask', scheduled_logits, target, scheduled,
            args.calibration_temperatures, args.sharpness_temperature,
        )

        fully_masked = torch.where(valid3, mask_id, target)
        fully_masked_logits = trans(
            fully_masked, clip_global, src_mask=attention_mask, word_emb=clip_words,
            betas=betas, gender=genders,
        )
        add_metrics(
            totals, 'all_parts_fully_masked', fully_masked_logits, target, valid3,
            args.calibration_temperatures, args.sharpness_temperature,
        )
        if getattr(args, 'cfg_scale', None) is not None:
            # Diagnose the actual first sampling distribution, including its
            # text-free reference. This does not change generation settings.
            unconditional = trans(
                fully_masked, clip_global, src_mask=attention_mask,
                word_emb=clip_words, betas=betas, gender=genders, force_mask=True,
            )
            for label, values in (
                ('all_parts_fully_masked_unconditional', unconditional),
                ('all_parts_fully_masked_guided', unconditional + args.cfg_scale
                 * (fully_masked_logits - unconditional)),
            ):
                if label not in totals:
                    totals[label] = accumulator(args.calibration_temperatures)[
                        'all_parts_fully_masked'
                    ]
                add_metrics(
                    totals, label, values, target, valid3,
                    args.calibration_temperatures, args.sharpness_temperature,
                )
                if label.endswith('_guided') and getattr(args, 'selection_draws', 0):
                    from tools.audit_sampling_selection import selection_counts
                    batch_counts = selection_counts(
                        values, valid, temperature=args.sharpness_temperature,
                        draws=args.selection_draws, generator=selection_generator,
                    )
                    for part, counts in batch_counts.items():
                        for key, value in counts.items():
                            selection_totals[part][key] = selection_totals[part].get(key, 0) + value

        body_input = fully_masked.clone()
        body_input[..., 2] = torch.where(valid, target[..., 2], body_input[..., 2])
        body_logits = trans(
            body_input, clip_global, src_mask=attention_mask, word_emb=clip_words,
            betas=betas, gender=genders,
        )
        body_mask = torch.zeros_like(valid3)
        body_mask[..., 0] = valid
        add_metrics(
            totals, 'body_after_gt_root', body_logits, target, body_mask,
            args.calibration_temperatures, args.sharpness_temperature,
        )

        hand_input = body_input.clone()
        hand_input[..., 0] = torch.where(valid, target[..., 0], hand_input[..., 0])
        hand_logits = trans(
            hand_input, clip_global, src_mask=attention_mask, word_emb=clip_words,
            betas=betas, gender=genders,
        )
        hand_mask = torch.zeros_like(valid3)
        hand_mask[..., 1] = valid
        add_metrics(
            totals, 'hand_after_gt_root_body', hand_logits, target, hand_mask,
            args.calibration_temperatures, args.sharpness_temperature,
        )

        for channel in range(3):
            single_input = target.clone()
            single_input[..., channel] = torch.where(
                valid, mask_id, single_input[..., channel]
            )
            single_logits = trans(
                single_input, clip_global, src_mask=attention_mask, word_emb=clip_words,
                betas=betas, gender=genders,
            )
            single_mask = torch.zeros_like(valid3)
            single_mask[..., channel] = valid
            add_metrics(
                totals, 'single_part_masked_other_parts_gt', single_logits,
                target, single_mask, args.calibration_temperatures,
                args.sharpness_temperature,
            )

    for condition in totals.values():
        for record in condition.values():
            count = record['tokens']
            record['cross_entropy'] = record.pop('loss_sum') / max(count, 1)
            record['temperature_cross_entropy'] = {
                key: value / max(count, 1)
                for key, value in record.pop(
                    'temperature_loss_sums'
                ).items()
            }
            record['accuracy_percent'] = 100.0 * record.pop('correct') / max(count, 1)
            record['untempered_top1_probability_exactly_one_fraction'] = (
                record.pop('ranking_saturated_tokens') / max(count, 1)
            )
            record['mean_untempered_top1_top2_logit_margin'] = (
                record.pop('ranking_logit_margin_sum') / max(count, 1)
            )
            mean_entropy = record.pop('entropy_sum') / max(count, 1)
            record['sharpness_temperature'] = float(args.sharpness_temperature)
            record['predictive_entropy_nats'] = mean_entropy
            class_count = record.pop('class_count')
            record['logit_class_count'] = class_count
            record['normalized_predictive_entropy'] = (
                mean_entropy / math.log(class_count) if class_count else 0.0
            )
            record['entropy_effective_code_count'] = math.exp(mean_entropy)
            record['mean_top1_probability'] = (
                record.pop('top1_probability_sum') / max(count, 1)
            )
            record['mean_top1_top2_probability_margin'] = (
                record.pop('top1_top2_margin_sum') / max(count, 1)
            )
            record['mean_target_probability'] = (
                record.pop('target_probability_sum') / max(count, 1)
            )
    cascade_records = (
        totals['all_parts_fully_masked']['root'],
        totals['body_after_gt_root']['body'],
        totals['hand_after_gt_root_body']['hand'],
    )
    calibration = {}
    for value in args.calibration_temperatures:
        key = str(float(value))
        token_count = sum(record['tokens'] for record in cascade_records)
        calibration[key] = sum(
            record['temperature_cross_entropy'][key] * record['tokens']
            for record in cascade_records
        ) / max(token_count, 1)
    del model
    torch.cuda.empty_cache()
    return {
        'contract': contract,
        'path': str(checkpoint.resolve()),
        'sha256': sha256_file(checkpoint),
        'metrics': totals,
        'cascade_temperature_calibration': calibration,
        'nll_optimal_grid_temperature': (
            min(calibration, key=calibration.get) if calibration else None
        ),
        'sharpness_temperature': float(args.sharpness_temperature),
        'inference_conditioning_contract': (
            inference_contract or model_args.conditioning_contract
        ),
        'cfg_scale': getattr(args, 'cfg_scale', None),
        'token_window': max_length,
        'maximum_payload_tokens': max_length - 1,
        'first_commitment_counterfactual': {
            'scope': 'fixed fully masked logits; independent categorical proposals; one committed position per part; not an end-to-end rollout',
            'draws_per_sample': getattr(args, 'selection_draws', 0),
            'counts': selection_totals,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--historical-checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--token-window', type=int,
                        help='Total token positions including end token; omit for model capacity.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=90123)
    parser.add_argument('--cfg-scale', type=float)
    parser.add_argument('--selection-draws', type=int, default=0)
    parser.add_argument(
        '--inference-conditioning-contract',
        choices=('paper_text_cfg', 'historical_mmm_shape_dropout'),
    )
    parser.add_argument(
        '--sharpness-temperature', type=float, default=0.5,
        help='Softmax temperature used only for entropy/margin diagnostics.',
    )
    parser.add_argument(
        '--calibration-temperatures', type=float, nargs='*', default=(),
        help=(
            'Optional held-out NLL temperature grid. This diagnoses logit '
            'calibration only and never selects on test motion metrics.'
        ),
    )
    parser.add_argument(
        '--split-role', choices=('validation', 'training'), default='validation',
        help=(
            'Use held-out validation tokens by default. The training option is '
            'diagnostic only and can establish whether a worsening validation '
            'trajectory is caused by train/validation generalization rather '
            'than checkpoint corruption.'
        ),
    )
    args = parser.parse_args()
    if args.selection_draws < 0 or (args.selection_draws and args.cfg_scale is None):
        parser.error('--selection-draws must be nonnegative and requires --cfg-scale')
    if not args.checkpoint and not args.historical_checkpoint:
        parser.error('at least one --checkpoint or --historical-checkpoint is required')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = json.loads(args.training_manifest.read_text(encoding='utf-8'))
    resolved = manifest['resolved_arguments']
    if args.token_window is not None and not 2 <= args.token_window <= resolved['block_size'] - 1:
        parser.error('--token-window must be in [2, block_size - 1]')
    split_file = (
        resolved['val_split_file']
        if args.split_role == 'validation'
        else resolved['train_split_file']
    )
    dataset = Text2MotionDatasetV2HHI_Token(
        split_file, resolved['codebook_dir'],
        unit_length=2 ** resolved['down_t'], codebook_size=resolved['nb_code'],
        text_dir='dataset/interx/texts_processed',
        max_motion_length=resolved['block_size'] - 1,
        contact_dir=str(Path(resolved['cache_root']) / 'contact_token_label'),
        beta_dir='dataset/interx/betas', strict_assets=True,
    )
    # Match the training loader's active >=20-token validation bucket.
    active_names = sorted(dataset.name_list[dataset.pointer:])
    if len(active_names) < args.samples:
        raise RuntimeError(
            f'requested {args.samples} samples, found {len(active_names)}'
        )
    selection_rng = np.random.default_rng(args.seed)
    names = sorted(selection_rng.choice(
        active_names, size=args.samples, replace=False
    ).tolist())
    result = {
        'schema': 'hhc_mgt_mask_exposure_audit_v1',
        'purpose': (
            'validation-token diagnosis only; no test-set selection'
            if args.split_role == 'validation'
            else 'training-token generalization comparator; no model selection'
        ),
        'split_role': args.split_role,
        'split_file': str(Path(split_file).resolve()),
        'split_file_sha256': sha256_file(split_file),
        'seed': args.seed,
        'samples': len(names),
        'sample_selection': 'uniform_without_replacement_over_active_split',
        'token_window': args.token_window or resolved['block_size'] - 1,
        'implementation_sha256': sha256_file(__file__),
        'active_split_samples': len(active_names),
        'sample_names_sha256': hashlib.sha256(
            '\n'.join(names).encode('utf-8')
        ).hexdigest(),
        'training_manifest': str(args.training_manifest.resolve()),
        'training_manifest_sha256': sha256_file(args.training_manifest),
        'checkpoints': [],
    }
    for checkpoint in args.checkpoint:
        result['checkpoints'].append(
            audit_checkpoint(args, resolved, dataset, names, checkpoint, args.device)
        )
    for checkpoint in args.historical_checkpoint:
        result['checkpoints'].append(
            audit_checkpoint(
                args, resolved, dataset, names, checkpoint, args.device,
                contract='historical_mmm',
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
