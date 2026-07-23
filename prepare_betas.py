"""Build dataset/interx/betas/<name>.npz per InterX sample by merging raw P1/P2.
Each output .npz has: betas (2,10) and gender [str,str], matching what
dataset/dataset_interx.py (beta_data['betas'].reshape(-1), beta_data['gender']) expects.
"""
import os

import numpy as np

MOTIONS_RAW = "/tmp/motions_raw/motions"
OUT_DIR = "dataset/interx/betas"
SPLITS = ["dataset/interx/train.txt", "dataset/interx/val.txt", "dataset/interx/test.txt"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = []
    for sp in SPLITS:
        for line in open(sp):
            line = line.strip()
            if line:
                names.append(line)
    names = sorted(set(names))
    print(f"building betas for {len(names)} samples -> {OUT_DIR}")
    missing = 0
    done = 0
    for i, name in enumerate(names):
        out = os.path.join(OUT_DIR, name + ".npz")
        if os.path.isfile(out):
            done += 1
            continue
        p1 = os.path.join(MOTIONS_RAW, name, "P1.npz")
        p2 = os.path.join(MOTIONS_RAW, name, "P2.npz")
        if not (os.path.isfile(p1) and os.path.isfile(p2)):
            missing += 1
            continue
        d1 = np.load(p1, allow_pickle=True)
        d2 = np.load(p2, allow_pickle=True)
        betas = np.stack([d1["betas"].reshape(-1)[:10], d2["betas"].reshape(-1)[:10]], axis=0)  # (2,10)
        gender = [str(d1["gender"]), str(d2["gender"])]
        np.savez(out, betas=betas.astype(np.float32), gender=gender)
        done += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(names)} done")
    print(f"done={done} missing={missing} total={len(names)}")


if __name__ == "__main__":
    main()
