#!/usr/bin/env python3
"""Guarded post-formal wiring for mandatory online W&B training telemetry."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ROOT / "options/option_transformer.py"
TRAIN = ROOT / "train/train_t2i_trans.py"
EVAL = ROOT / "utils/eval_trans.py"
TELEMETRY = ROOT / "utils/training_telemetry.py"
POLICY = ROOT / "configs/online_wandb_training_policy.json"
OUTPUT = ROOT / "output/post_formal_online_wandb_metrics_application.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one preimage, found {count}")
    return source.replace(old, new, 1)


def transform_sources(options_source: str, train_source: str,
                      eval_source: str) -> tuple[str, str, str]:
    train_source = replace_once(
        train_source,
        "    return (right_num * 100 / total_num).item()\n",
        "    # Preserve a scalar tensor for type-stable sparse telemetry.\n"
        "    return right_num * 100.0 / total_num\n",
        "type-stable accuracy metric",
    )
    options_source = replace_once(
        options_source,
        'parser.add_argument("--wandb-mode", default=\'offline\', '
        "choices=['online', 'offline', 'disabled'])",
        'parser.add_argument("--wandb-mode", default=\'online\', '
        "choices=['online', 'offline', 'disabled'])",
        "online W&B default",
    )

    train_source = replace_once(
        train_source,
        "from utils.runtime import generate_src_mask, init_save_folder, preserving_rng_state\n",
        "from utils.runtime import generate_src_mask, init_save_folder, preserving_rng_state\n"
        "from utils.training_telemetry import (\n"
        "    ThroughputMeter,\n"
        "    gpu_peak_memory_gib,\n"
        "    parameter_and_gradient_telemetry,\n"
        ")\n",
        "telemetry import",
    )
    train_source = replace_once(
        train_source,
        "    args = option_trans.get_args_parser()\n"
        "    os.environ['HHC_INTERX_STATS_DIR'] = os.path.abspath(args.stats_dir)\n",
        "    args = option_trans.get_args_parser()\n"
        "    non_training_mode = (\n"
        "        args.cache_only or args.generate_only or args.evaluate_only\n"
        "    )\n"
        "    if not non_training_mode and args.wandb_mode != 'online':\n"
        "        raise ValueError(\n"
        "            'all newly launched MGT training requires --wandb-mode online'\n"
        "        )\n"
        "    os.environ['HHC_INTERX_STATS_DIR'] = os.path.abspath(args.stats_dir)\n",
        "online-only MGT training gate",
    )
    train_source = replace_once(
        train_source,
        "    'utils/motion_process.py',\n    'utils/runtime.py',\n)",
        "    'utils/motion_process.py',\n    'utils/runtime.py',\n"
        "    'utils/training_telemetry.py',\n)",
        "telemetry provenance source",
    )
    post_root_resume = (
        "    'contact_guidance_scale', 'seed',\n"
        "    'token_cache_contract',\n)"
    )
    if post_root_resume in train_source:
        train_source = replace_once(
            train_source,
            post_root_resume,
            "    'contact_guidance_scale', 'seed', 'wandb_mode',\n"
            "    'token_cache_contract',\n)",
            "online W&B resume identity after root-tokenizer wiring",
        )
    else:
        train_source = replace_once(
            train_source,
            "    'contact_guidance_scale', 'seed',\n)",
            "    'contact_guidance_scale', 'seed', 'wandb_mode',\n)",
            "online W&B resume identity",
        )
    train_source = replace_once(
        train_source,
        "        self.historical_replay = (\n"
        "            args.training_validation_contract == 'historical_test_side_effect'\n"
        "        )\n",
        "        self.historical_replay = (\n"
        "            args.training_validation_contract == 'historical_test_side_effect'\n"
        "        )\n"
        "        self.telemetry_interval_optimizer_steps = 50\n"
        "        self.training_throughput = ThroughputMeter()\n",
        "telemetry initialization",
    )
    train_source = replace_once(
        train_source,
        "    prepare_mgt_output_directory(args)\n"
        "    os.makedirs(args.out_dir, exist_ok=True)\n",
        "    run_state = prepare_mgt_output_directory(args)\n"
        "    os.makedirs(args.out_dir, exist_ok=True)\n"
        "    args.wandb_run_id = None\n"
        "    if args.wandb_mode == 'online':\n"
        "        if run_state.get('mode') == 'strict_in_place_resume':\n"
        "            with open(run_state['prior_manifest'], encoding='utf-8') as handle:\n"
        "                prior_observability = json.load(handle).get('observability', {})\n"
        "            args.wandb_run_id = prior_observability.get('wandb_run_id')\n"
        "            if not args.wandb_run_id:\n"
        "                raise ValueError('online MGT resume manifest lacks wandb_run_id')\n"
        "        else:\n"
        "            args.wandb_run_id = wandb.util.generate_id()\n",
        "stable online W&B run identity",
    )
    train_source = replace_once(
        train_source,
        "        'launch_time': _PROCESS_START_TIME,\n"
        "        'repository': repository_provenance(),\n",
        "        'launch_time': _PROCESS_START_TIME,\n"
        "        'observability': {\n"
        "            'wandb_mode': args.wandb_mode,\n"
        "            'wandb_project': 'NewInter',\n"
        "            'wandb_run_id': args.wandb_run_id,\n"
        "            'metric_profile': 'masked_token_classification',\n"
        "            'telemetry_interval_optimizer_steps': 50,\n"
        "        },\n"
        "        'repository': repository_provenance(),\n",
        "online W&B manifest provenance",
    )
    train_source = replace_once(
        train_source,
        "    wandb_logger = False if args.wandb_mode == 'disabled' else WandbLogger(\n"
        "        project=\"NewInter\", config=args, group='mask_gen',\n"
        "        name=args.exp_name, offline=(args.wandb_mode == 'offline'),\n"
        "    )\n",
        "    wandb_logger = False if args.wandb_mode == 'disabled' else WandbLogger(\n"
        "        project=\"NewInter\", config=args, group='mask_gen',\n"
        "        name=args.exp_name, offline=(args.wandb_mode == 'offline'),\n"
        "        id=args.wandb_run_id,\n"
        "        resume='allow' if args.wandb_mode == 'online' else None,\n"
        "    )\n",
        "stable online W&B logger identity",
    )
    train_source = replace_once(
        train_source,
        "        opt = self.optimizers()\n"
        "        opt.zero_grad()\n"
        "        self.manual_backward(loss_cls)\n"
        "        opt.step()\n"
        "        self.lr_schedulers().step()\n",
        "        opt = self.optimizers()\n"
        "        opt.zero_grad()\n"
        "        self.manual_backward(loss_cls)\n"
        "        self.training_throughput.update(batch_size)\n"
        "        next_step = int(self.global_step) + 1\n"
        "        if next_step % self.telemetry_interval_optimizer_steps == 0:\n"
        "            telemetry = parameter_and_gradient_telemetry(\n"
        "                self.trans_encoder, fail_on_nonfinite=False\n"
        "            )\n"
        "            telemetry.update({\n"
        "                'train/optimizer_step': float(next_step),\n"
        "                'train/loss': loss_cls.detach(),\n"
        "                'train/accuracy_all': acc_all.detach(),\n"
        "                'train/accuracy_masked': acc_masked.detach(),\n"
        "                'train/accuracy_visible': acc_nomask.detach(),\n"
        "                'train/mask_ratio_body': mask_ratios[:, 0].mean(),\n"
        "                'train/mask_ratio_hand': mask_ratios[:, 1].mean(),\n"
        "                'train/mask_ratio_root': mask_ratios[:, 2].mean(),\n"
        "                'train/learning_rate': float(opt.param_groups[0]['lr']),\n"
        "                'train/samples_per_second': (\n"
        "                    self.training_throughput.measure_and_reset()\n"
        "                ),\n"
        "                'train/gpu_peak_memory_gib': gpu_peak_memory_gib(\n"
        "                    self.device\n"
        "                ),\n"
        "            })\n"
        "            self.log_dict(telemetry, on_step=True, logger=True)\n"
        "            if (telemetry['train/nonfinite_gradients']\n"
        "                    or telemetry['train/nonfinite_weights']):\n"
        "                raise FloatingPointError(\n"
        "                    f'nonfinite training telemetry: {telemetry}'\n"
        "                )\n"
        "        opt.step()\n"
        "        self.lr_schedulers().step()\n",
        "gradient and weight telemetry",
    )
    # The source computes accuracy after the optimizer step for ordinary
    # logging. Sparse telemetry first fires at step 50, so move the same
    # quantities before the optimizer block instead of referencing locals that
    # do not exist yet.
    train_source = replace_once(
        train_source,
        "        opt = self.optimizers()\n",
        "        with torch.no_grad():\n"
        "            cls_pred_index = cls_pred.detach().argmax(dim=-1)\n"
        "            right_token = (cls_pred_index == target) & seq_mask_no_end_3c\n"
        "            acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()\n"
        "            real_mask_token = mask_token & seq_mask_no_end_3c\n"
        "            acc_masked = get_acc(cls_pred.detach(), target, real_mask_token)\n"
        "            no_mask_token = (~mask_token) & seq_mask_no_end_3c\n"
        "            acc_nomask = get_acc(cls_pred.detach(), target, no_mask_token)\n"
        "\n"
        "        opt = self.optimizers()\n",
        "accuracy available before sparse telemetry",
    )
    train_source = replace_once(
        train_source,
        "        # 这里的 global_step 是 Lightning 自动管理的\n"
        "        probs = torch.softmax(cls_pred, dim=-1)\n"
        "        _, cls_pred_index = torch.max(probs, dim=-1)\n"
        "        right_token = (cls_pred_index == target) & seq_mask_no_end_3c\n"
        "        acc_all = right_token.sum() * 100.0 / seq_mask_no_end_3c.sum()\n"
        "        real_mask_token = mask_token & seq_mask_no_end_3c\n"
        "        acc_masked = get_acc(cls_pred, target, real_mask_token)\n"
        "        no_mask_token = (~mask_token) & seq_mask_no_end_3c\n"
        "        acc_nomask = get_acc(cls_pred, target, no_mask_token)\n",
        "",
        "remove post-step duplicate accuracy computation",
    )
    train_source = replace_once(
        train_source,
        "        self.log('Train/mask_ratio_root', mask_ratios[:, 2].mean())\n",
        "        self.log('Train/mask_ratio_root', mask_ratios[:, 2].mean())\n"
        "        self.log('train/mask_ratio_body', mask_ratios[:, 0].mean(), on_step=True, logger=True)\n"
        "        self.log('train/mask_ratio_hand', mask_ratios[:, 1].mean(), on_step=True, logger=True)\n"
        "        self.log('train/mask_ratio_root', mask_ratios[:, 2].mean(), on_step=True, logger=True)\n"
        "        self.log('train/loss', loss_cls, on_step=True, logger=True)\n"
        "        self.log('train/accuracy_all', acc_all, on_step=True, logger=True)\n"
        "        self.log('train/accuracy_masked', acc_masked, on_step=True, logger=True)\n"
        "        self.log('train/accuracy_visible', acc_nomask, on_step=True, logger=True)\n"
        "        self.log(\n"
        "            'train/learning_rate', opt.param_groups[0]['lr'],\n"
        "            on_step=True, logger=True,\n"
        "        )\n",
        "standard online training metrics",
    )

    validation_preimage = (
        "    def validation_step(self, batch, batch_idx):\n"
        "        pass\n"
    )
    validation_replacement = '''    def validation_step(self, batch, batch_idx):
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
'''
    train_source = replace_once(
        train_source, validation_preimage, validation_replacement,
        "validation loss and accuracy",
    )
    train_source = replace_once(
        train_source,
        "                    self.best_matching, self.best_multi, \\\n"
        "                    _, _ = eval_trans.evaluation_transformer(\n",
        "                    self.best_matching, self.best_multi, \\\n"
        "                    _, _, current_metrics = eval_trans.evaluation_transformer(\n",
        "current validation metrics return",
    )
    train_source = replace_once(
        train_source,
        "                    contact_map_path=self.args.contact_map_path,\n"
        "                )\n\n"
        "            self.log('val/FID', fid)\n"
        "            self.log('val/Diversity', self.best_div)\n"
        "            self.log('val/TOP1', self.best_top1)\n"
        "            self.log('val/TOP2', self.best_top2)\n"
        "            self.log('val/TOP3', self.best_top3)\n"
        "            self.log('val/Matching', self.best_matching)\n",
        "                    contact_map_path=self.args.contact_map_path,\n"
        "                    return_current_metrics=True,\n"
        "                )\n\n"
        "            self.log('val/FID', current_metrics['FID'])\n"
        "            self.log('val/Diversity', current_metrics['Diversity'])\n"
        "            self.log('val/TOP1', current_metrics['R_precision_top1'])\n"
        "            self.log('val/TOP2', current_metrics['R_precision_top2'])\n"
        "            self.log('val/TOP3', current_metrics['R_precision_top3'])\n"
        "            self.log('val/Matching', current_metrics['Matching'])\n"
        "            self.log('val/R_precision_top1', current_metrics['R_precision_top1'])\n"
        "            self.log('val/R_precision_top2', current_metrics['R_precision_top2'])\n"
        "            self.log('val/R_precision_top3', current_metrics['R_precision_top3'])\n"
        "            self.log('val/Multimodality', current_metrics['Multimodality'])\n",
        "exact current semantic validation metrics",
    )

    eval_source = replace_once(
        eval_source,
        "contact_map_path='checkpoints/vq_paper/code_contact_probs.npy'):\n",
        "contact_map_path='checkpoints/vq_paper/code_contact_probs.npy', "
        "return_current_metrics=False):\n",
        "optional exact current metrics API",
    )
    eval_source = replace_once(
        eval_source,
        "    trans.train()\n"
        "    return fid, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger\n",
        "    trans.train()\n"
        "    result = (\n"
        "        fid, pose, m_length, clip_text, best_fid, best_iter, best_div,\n"
        "        best_top1, best_top2, best_top3, best_matching, multimodality,\n"
        "        writer, logger,\n"
        "    )\n"
        "    if return_current_metrics:\n"
        "        return result + ({\n"
        "            'FID': float(fid),\n"
        "            'Diversity': float(diversity),\n"
        "            'R_precision_top1': float(R_precision[0]),\n"
        "            'R_precision_top2': float(R_precision[1]),\n"
        "            'R_precision_top3': float(R_precision[2]),\n"
        "            'Matching': float(matching_score_pred),\n"
        "            'Multimodality': float(multimodality),\n"
        "        },)\n"
        "    return result\n",
        "return exact current validation metrics",
    )
    return options_source, train_source, eval_source


def online_wandb_metrics_already_applied(train_source: str) -> bool:
    """Recognize the actual post-transform call sites, not imported key text."""
    return all(
        marker in train_source
        for marker in (
            "parameter_and_gradient_telemetry(",
            "'train/optimizer_step': float(next_step)",
            "'train/accuracy_masked': acc_masked.detach()",
            "return right_num * 100.0 / total_num",
            "'val/loss', loss, on_step=False, on_epoch=True",
            "self.log('val/R_precision_top1'",
            "telemetry_interval_optimizer_steps",
            "all newly launched MGT training requires --wandb-mode online",
            "resume='allow' if args.wandb_mode == 'online' else None",
        )
    )


def atomic_write(path: Path, source: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("status") != "active" or policy.get("mode") != "online":
        raise ValueError("online W&B policy is not active")
    if not TELEMETRY.is_file():
        raise FileNotFoundError(TELEMETRY)

    before = {
        "options": OPTIONS.read_text(encoding="utf-8"),
        "train": TRAIN.read_text(encoding="utf-8"),
        "eval": EVAL.read_text(encoding="utf-8"),
    }
    already = online_wandb_metrics_already_applied(before["train"])
    if already:
        after = (before["options"], before["train"], before["eval"])
    else:
        after = transform_sources(before["options"], before["train"], before["eval"])
    for label, source in zip(("options", "train", "eval"), after):
        ast.parse(source, filename=label)

    payload = {
        "schema": "hhc_post_formal_online_wandb_metrics_application_v1",
        "status": "already_applied" if already else ("applied" if args.apply else "check_pass"),
        "write_requested": bool(args.apply),
        "policy_sha256": sha256_file(POLICY),
        "telemetry_sha256": sha256_file(TELEMETRY),
        "files": {},
    }
    for label, path, source in zip(
        ("options", "train", "eval"), (OPTIONS, TRAIN, EVAL), after
    ):
        payload["files"][label] = {
            "path": str(path),
            "before_sha256": sha256_bytes(before[label].encode()),
            "after_sha256": sha256_bytes(source.encode()),
        }
    if args.apply and not already:
        atomic_write(OPTIONS, after[0])
        atomic_write(TRAIN, after[1])
        atomic_write(EVAL, after[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
