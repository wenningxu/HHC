import os

import numpy as np


def build_code_contact_map(
    indices_dir, contact_dir, codebook_size, alpha=0.0, included_names=None
):
    """Estimate P(contact | code) for Body, Hand and Root codebooks.

    Observed codes use the paper's empirical contact frequency exactly.  An
    optional symmetric Beta prior can be requested for ablations; unseen/dead
    codes remain neutral at 0.5 because their empirical probability is
    undefined and the sampler subsequently takes a logarithm.
    """
    total = np.zeros((3, codebook_size), dtype=np.float64)
    positive = np.zeros_like(total)
    used_files = 0

    included_names = None if included_names is None else set(included_names)
    for filename in sorted(os.listdir(indices_dir)):
        if not filename.endswith('.npy'):
            continue
        if included_names is not None and os.path.splitext(filename)[0] not in included_names:
            continue
        contact_path = os.path.join(contact_dir, filename)
        if not os.path.isfile(contact_path):
            continue
        indices = np.load(os.path.join(indices_dir, filename)).reshape(-1, 3)
        contact = np.load(contact_path).reshape(-1).astype(np.float64)
        length = min(len(indices), len(contact))
        indices, contact = indices[:length], contact[:length]
        if indices.size == 0:
            continue
        if indices.min() < 0 or indices.max() >= codebook_size:
            raise ValueError(f'{filename}: code index outside [0, {codebook_size})')
        contact = np.clip(contact, 0.0, 1.0)
        for channel in range(3):
            np.add.at(total[channel], indices[:, channel], 1.0)
            np.add.at(positive[channel], indices[:, channel], contact)
        used_files += 1

    if used_files == 0:
        raise RuntimeError(f'no matching token/contact files in {indices_dir} and {contact_dir}')
    probabilities = np.full_like(total, 0.5)
    observed = total > 0
    if alpha:
        probabilities[observed] = (
            (positive[observed] + alpha) / (total[observed] + 2.0 * alpha)
        )
    else:
        probabilities[observed] = positive[observed] / total[observed]
    return probabilities.astype(np.float32), total, used_files
