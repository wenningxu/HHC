"""Generate dataset/interx/skeleton/{1..59}.npy (dual-person SMPLX rest-pose joints).

Each skeleton i.npy is shape (2, 55, 3): [P1_rest_joints, P2_rest_joints] in
SMPLX_TREE joint order. Computed as J_regressor @ (v_template + shapedirs[...,:10] @ betas)
using the gender-appropriate SMPLX model, so FK in VQVAE_SEP3_shape gets correct
bone-length offsets per InterX group.
"""
import os
import re

import numpy as np

BM_DIR = "body_models/smplx"
MOTIONS_RAW = "/tmp/motions_raw/motions"
OUT_DIR = "dataset/interx/skeleton"
SMPLX_FILES = {
    "male": os.path.join(BM_DIR, "SMPLX_MALE.npz"),
    "female": os.path.join(BM_DIR, "SMPLX_FEMALE.npz"),
    "neutral": os.path.join(BM_DIR, "SMPLX_NEUTRAL.npz"),
}
SPLITS = ["dataset/interx/train.txt", "dataset/interx/val.txt", "dataset/interx/test.txt"]


def load_smplx(path):
    d = np.load(path, allow_pickle=True)
    return {
        "v_template": d["v_template"].astype(np.float32),       # (10475,3)
        "shapedirs": d["shapedirs"].astype(np.float32),         # (10475,3,400)
        "J_regressor": d["J_regressor"].astype(np.float32),     # (55,10475)
    }


def shaped_joints(model, betas):
    """betas: (10,) -> (55,3) rest-pose joints in SMPLX order."""
    betas = betas.reshape(10).astype(np.float32)
    v = model["v_template"] + np.einsum("vdb,b->vd", model["shapedirs"][..., :10], betas)
    return model["J_regressor"] @ v  # (55,3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}

    # group id (1..59) -> first sample name
    group2name = {}
    for sp in SPLITS:
        for line in open(sp):
            line = line.strip()
            if not line:
                continue
            m = re.search(r"G(\d+)", line)
            if m:
                g = int(m.group(1))
                group2name.setdefault(g, line)

    groups = sorted(group2name)
    assert groups == list(range(1, 60)), f"expected 1..59, got {groups}"
    print(f"generating {len(groups)} skeletons -> {OUT_DIR}")

    for g in groups:
        name = group2name[g]
        sample_dir = os.path.join(MOTIONS_RAW, name)
        pair = []
        for pid in ("P1", "P2"):
            d = np.load(os.path.join(sample_dir, f"{pid}.npz"), allow_pickle=True)
            betas = d["betas"].reshape(-1)[:10]          # (10,)
            gender = str(d["gender"])                    # 'male'/'female'
            key = gender if gender in SMPLX_FILES else "neutral"
            if key not in cache:
                cache[key] = load_smplx(SMPLX_FILES[key])
            pair.append(shaped_joints(cache[key], betas))
        skel = np.stack(pair, axis=0).astype(np.float32)  # (2,55,3)
        np.save(os.path.join(OUT_DIR, f"{g}.npy"), skel)
    print(f"done. wrote {len(groups)} files. shape check:",
          np.load(os.path.join(OUT_DIR, "1.npy")).shape)


if __name__ == "__main__":
    main()
