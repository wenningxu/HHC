"""Replay augmentation with the original bidirectional-Chamfer distance helper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimization.losses as losses
from tools.check_signed_distance_knn_parity import reference
from tools.run_interx_shape_augmentation import main


def original_chamfer(x, y, x_normals=None, y_normals=None, return_vector=False):
    result = reference(x, y, y_normals)
    return result if return_vector else result[0]


if __name__ == '__main__':
    losses.point2point_signed = original_chamfer
    print('Diagnostic override: original bidirectional Chamfer, all other losses unchanged.', flush=True)
    main()
