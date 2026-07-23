import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import random
import sys
from pathlib import Path
import re

import clip
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import options.option_transformer as option_trans
from dataset import dataset_interx_TM
from options.get_eval_option import get_opt
from train.train_t2i_trans import MaskGITLightning
from utils.motion_process import MotionNormalizerTorchInterX, recover_interx
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper


def parse_cli():
    parser = argparse.ArgumentParser(description="Generate multimodality samples from dataset selection.")
    parser.add_argument("--ckpt", required=True, help="Transformer checkpoint path.")
    parser.add_argument("--split", default="val", choices=["val", "test", "all"], help="Dataset split to draw samples from.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--mm-num-samples", type=int, required=True, help="Number of dataset items selected for MM generation.")
    parser.add_argument("--mm-num-repeats", type=int, default=30, help="Number of repeats per selected sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed controlling sample selection and dataset randomness.")
    parser.add_argument("--save-dir", required=True, help="Flat output directory for repeat files.")
    parser.add_argument("--eval-opt-path", default="checkpoints/hhi/Comp_v6_KLD01/opt.txt", help="Evaluator opt path.")
    parser.add_argument("--glove-dir", default="dataset/interx/glove", help="GloVe directory.")
    parser.add_argument("--vocab-prefix", default="hhi_vab", help="Vectorizer vocab prefix.")
    parser.add_argument("--cfg", type=float, default=3.5, help="Classifier-free guidance scale.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation dataloader batch size. Use 1 to match sample-wise selection.")
    parser.add_argument("--sampler", default="sample", choices=["sample", "sample_intra_cas"], help="Sampling method on the transformer.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when the sampler supports it.")
    parser.add_argument("--reference-dir", default=None, help="Optional flat directory like results/best_fid.tar used to select sample ids.")
    parser.add_argument("--sample-names", nargs="+", default=None, help="Explicit InterX sample names to generate.")
    parser.add_argument("--chi3d-beta-groups", nargs="+", type=int, default=None, help="Optional Chi3D beta group ids used to override InterX betas.")
    parser.add_argument("--force-gender", choices=["male", "female"], default=None, help="Optional gender override applied to both persons.")
    parser.add_argument("--zero-beta", action="store_true", help="Override both persons' betas with zeros.")
    parser.add_argument("--zero-beta-for-mgt", action="store_true", help="Use all-zero betas only during transformer sampling.")
    parser.add_argument("--zero-beta-for-vq", action="store_true", help="Use all-zero betas only during VQ decode and saved outputs.")
    parser.add_argument("--default-stem-suffix", default="", help="Suffix appended to every saved sample stem.")
    extra_args, remaining = parser.parse_known_args()
    return extra_args, remaining


def load_model_args(remaining_argv):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining_argv
        args = option_trans.get_args_parser()
    finally:
        sys.argv = old_argv
    return args


def set_all_seeds(seed):
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_and_loader(model_args, cli_args, w_vectorizer):
    split_file = f"dataset/interx/{cli_args.split}.txt"
    motion_file = f"dataset/interx/motions/{cli_args.split}.h5"
    dataset = dataset_interx_TM.Text2MotionDatasetV2HHI(
        split_file=split_file,
        w_vectorizer=w_vectorizer,
        motion_file=motion_file,
        unit_length=2 ** model_args.down_t,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_interx_TM.collate_fn,
        drop_last=False,
    )
    return dataset, loader


def build_model(model_args, eval_wrapper, val_loader):
    model = MaskGITLightning(model_args, eval_wrapper, val_loader)
    model = model.to(model_args.device)
    model.eval()
    return model


def load_chi3d_group_betas(group_ids):
    betas = {}
    for group_id in group_ids:
        path = Path("dataset/chi3d/betas") / f"{int(group_id)}.npy"
        arr = np.load(path).astype(np.float32)
        if arr.shape != (2, 10):
            raise ValueError(f"Expected chi3d betas shape (2, 10), got {arr.shape} from {path}")
        betas[int(group_id)] = torch.from_numpy(arr.reshape(-1))
    return betas


@torch.no_grad()
def generate_repeat_files(
    model,
    eval_wrapper,
    batch,
    save_dir,
    repeat_idx,
    cfg_scale,
    sampler,
    temperature,
    stem_suffix="",
    trans_betas_override=None,
    decode_betas_override=None,
    gender_override=None,
    beta_source=None,
):
    normalizer = MotionNormalizerTorchInterX("raw")
    device = torch.device(model.args.device)

    word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, betas, gender, contact_mask = batch
    del token, contact_mask

    pose = pose.to(device).float()
    betas = betas.to(device).float()
    gender = gender.to(device)
    trans_betas = betas
    decode_betas = betas
    if trans_betas_override is not None:
        trans_betas = trans_betas_override.to(device=device, dtype=torch.float32).reshape(1, -1).repeat(betas.shape[0], 1)
    if decode_betas_override is not None:
        decode_betas = decode_betas_override.to(device=device, dtype=torch.float32).reshape(1, -1).repeat(betas.shape[0], 1)
    if gender_override is not None:
        gender = gender_override.to(device=device, dtype=torch.long).reshape(1, -1).repeat(gender.shape[0], 1)
    m_length = m_length.to(device)
    text = clip.tokenize(clip_text, truncate=True).to(device)
    feat_clip_text, word_emb = model.clip_model(text)
    pred_tok_len = torch.ceil(m_length / 4)
    seq = pose.shape[1]
    pred_pose_eval = torch.zeros((pose.shape[0], seq, pose.shape[-1]), device=device)
    new_motion_lengths = torch.zeros_like(m_length)

    sample_kwargs = dict(
        m_length=m_length,
        rand_pos=True,
        betas=trans_betas,
        gender=gender,
        CFG=cfg_scale,
    )
    if sampler == "sample_intra_cas":
        sample_kwargs["temperature"] = temperature
    index_motion = model.trans_encoder(
        feat_clip_text,
        word_emb,
        type=sampler,
        **sample_kwargs,
    )

    saved_records = []
    for batch_idx in range(pose.shape[0]):
        sample_name = str(name[batch_idx])
        pred_pose = model.net(
            index_motion[batch_idx:batch_idx + 1, : int(pred_tok_len[batch_idx].item())],
            decode_betas[batch_idx:batch_idx + 1],
            gender[batch_idx:batch_idx + 1],
            type="decode",
        )
        pred_pose = normalizer.backward(pred_pose)
        motion_length = int(m_length[batch_idx].item())
        pred_pose_eval[batch_idx:batch_idx + 1, :motion_length] = pred_pose[:, :motion_length]
        new_motion_lengths[batch_idx] = motion_length

        trans1, pose1, trans2, pose2 = recover_interx(pred_pose.squeeze())
        gender_word = ["male" if g == 0 else "female" for g in gender[batch_idx]]

        stem = f"{sample_name}{stem_suffix}_repeat{repeat_idx}"
        p1_path = save_dir / f"{stem}_p1.npz"
        p2_path = save_dir / f"{stem}_p2.npz"
        np.savez(
            p1_path,
            poses=pose1,
            trans=trans1,
            betas=decode_betas[batch_idx, 0:10].detach().cpu().numpy(),
            gender=gender_word[0],
            mocap_framerate=30,
        )
        np.savez(
            p2_path,
            poses=pose2,
            trans=trans2,
            betas=decode_betas[batch_idx, 10:].detach().cpu().numpy(),
            gender=gender_word[1],
            mocap_framerate=30,
        )
        saved_records.append((sample_name, p1_path, p2_path))

    _, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, new_motion_lengths)
    for batch_idx, (sample_name, p1_path, p2_path) in enumerate(saved_records):
        emb_path = save_dir / f"{sample_name}{stem_suffix}_repeat{repeat_idx}_emb.npy"
        np.save(emb_path, em_pred[batch_idx].detach().cpu().numpy())

    return [
        {
            "sample_name": sample_name,
            "repeat_id": repeat_idx,
            "stem_suffix": stem_suffix,
            "motion_p1_path": p1_path.name,
            "motion_p2_path": p2_path.name,
            "embedding_path": f"{sample_name}{stem_suffix}_repeat{repeat_idx}_emb.npy",
            "text": clip_text[batch_idx],
            "beta_source": beta_source,
        }
        for batch_idx, (sample_name, p1_path, p2_path) in enumerate(saved_records)
    ]


def main():
    cli_args, remaining_argv = parse_cli()
    model_args = load_model_args(remaining_argv)
    set_all_seeds(cli_args.seed)

    model_args.device = cli_args.device
    model_args.contact = False
    model_args.resume_trans = cli_args.ckpt
    model_args.resume_pth = "./checkpoints/vq/net_175000.pth"

    save_dir = Path(cli_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    w_vectorizer = WordVectorizer(cli_args.glove_dir, cli_args.vocab_prefix)
    dataset, loader = build_dataset_and_loader(model_args, cli_args, w_vectorizer)
    if cli_args.mm_num_samples > len(dataset):
        raise ValueError(f"mm_num_samples={cli_args.mm_num_samples} must be no larger than dataset size {len(dataset)}")

    eval_wrapper = EvaluatorModelWrapper(get_opt(cli_args.eval_opt_path, cli_args.device))
    model = build_model(model_args, eval_wrapper, loader)

    rng = np.random.default_rng(cli_args.seed)
    selected_names = None
    if cli_args.sample_names is not None:
        if cli_args.mm_num_samples != len(cli_args.sample_names):
            raise ValueError(
                f"mm_num_samples={cli_args.mm_num_samples} must match number of sample_names={len(cli_args.sample_names)}"
            )
        selected_names = list(cli_args.sample_names)
        mm_idx_set = None
        selected_name_set = set(selected_names)
    elif cli_args.reference_dir is not None:
        pattern = re.compile(r"^(?P<name>.+)_repeat\d+_p1\.npz$")
        selected_names = []
        seen = set()
        for path in sorted(Path(cli_args.reference_dir).glob("*_repeat*_p1.npz")):
            match = pattern.match(path.name)
            if match is None:
                continue
            sample_name = match.group("name")
            if sample_name not in seen:
                seen.add(sample_name)
                selected_names.append(sample_name)
        if len(selected_names) != cli_args.mm_num_samples:
            raise ValueError(
                f"reference_dir produced {len(selected_names)} unique ids, expected mm_num_samples={cli_args.mm_num_samples}"
            )
        mm_idx_set = None
        selected_name_set = set(selected_names)
    else:
        mm_idxs = np.sort(rng.choice(len(dataset), cli_args.mm_num_samples, replace=False))
        mm_idx_set = set(int(idx) for idx in mm_idxs.tolist())
        selected_names = [dataset.name_list[dataset.pointer + int(idx)] for idx in mm_idxs.tolist()]
        selected_name_set = set(selected_names)

    metadata = {
        "ckpt": cli_args.ckpt,
        "split": cli_args.split,
        "device": cli_args.device,
        "seed": cli_args.seed,
        "mm_num_samples": cli_args.mm_num_samples,
        "mm_num_repeats": cli_args.mm_num_repeats,
        "selected_indices": None if (cli_args.reference_dir is not None or cli_args.sample_names is not None) else [int(idx) for idx in mm_idxs.tolist()],
        "selected_names": selected_names,
        "sampler": cli_args.sampler,
        "temperature": cli_args.temperature,
        "chi3d_beta_groups": cli_args.chi3d_beta_groups,
        "force_gender": cli_args.force_gender,
        "zero_beta": cli_args.zero_beta,
        "zero_beta_for_mgt": cli_args.zero_beta_for_mgt,
        "zero_beta_for_vq": cli_args.zero_beta_for_vq,
        "records": [],
    }

    chi3d_betas = None
    if cli_args.chi3d_beta_groups is not None and cli_args.zero_beta:
        raise ValueError("Use either --chi3d-beta-groups or --zero-beta, not both.")
    if cli_args.zero_beta and (cli_args.zero_beta_for_mgt or cli_args.zero_beta_for_vq):
        raise ValueError("Use --zero-beta alone or the stage-specific zero-beta flags, not both.")

    if cli_args.chi3d_beta_groups is not None:
        chi3d_betas = load_chi3d_group_betas(cli_args.chi3d_beta_groups)

    gender_override = None
    if cli_args.force_gender is not None:
        gender_code = 0 if cli_args.force_gender == "male" else 1
        gender_override = torch.tensor([gender_code, gender_code], dtype=torch.long)

    for dataset_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating MM samples"):
        batch_names = batch[7]
        batch_sample_name = str(batch_names[0])
        if cli_args.sample_names is not None or cli_args.reference_dir is not None:
            if batch_sample_name not in selected_name_set:
                continue
        else:
            if dataset_idx not in mm_idx_set:
                continue
        if chi3d_betas is None:
            if cli_args.zero_beta:
                shape_jobs = [(None, cli_args.default_stem_suffix, "zero_beta")]
            else:
                shape_jobs = [(None, cli_args.default_stem_suffix, "dataset_beta")]
        else:
            shape_jobs = [
                (group_id, f"{cli_args.default_stem_suffix}_chi3dG{group_id}", f"chi3d_group_{group_id}")
                for group_id in cli_args.chi3d_beta_groups
            ]

        for group_id, stem_suffix, beta_source in shape_jobs:
            group_betas = None if group_id is None else chi3d_betas[group_id]
            if cli_args.zero_beta:
                trans_betas_override = torch.zeros(20, dtype=torch.float32)
                decode_betas_override = torch.zeros(20, dtype=torch.float32)
            else:
                trans_betas_override = torch.zeros(20, dtype=torch.float32) if cli_args.zero_beta_for_mgt else group_betas
                decode_betas_override = torch.zeros(20, dtype=torch.float32) if cli_args.zero_beta_for_vq else group_betas
            for repeat_idx in range(1, cli_args.mm_num_repeats + 1):
                metadata["records"].extend(
                    generate_repeat_files(
                        model=model,
                        eval_wrapper=eval_wrapper,
                        batch=batch,
                        save_dir=save_dir,
                        repeat_idx=repeat_idx,
                        cfg_scale=cli_args.cfg,
                        sampler=cli_args.sampler,
                        temperature=cli_args.temperature,
                        stem_suffix=stem_suffix,
                        trans_betas_override=trans_betas_override,
                        decode_betas_override=decode_betas_override,
                        gender_override=gender_override,
                        beta_source=beta_source,
                    )
                )

    metadata_path = save_dir / "mm_samples.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"saved_mm_samples: {len(metadata['records'])}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
