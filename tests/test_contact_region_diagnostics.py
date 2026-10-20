import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.score_reviewed_contact_comparison import region_diagnostics


def check():
    a = torch.tensor([[[0., 0., 0.], [10., 0., 0.]], [[0., 0., 0.], [10., 0., 0.]]])
    b = torch.tensor([[[1., 0., 0.], [10.25, 0., 0.]], [[2., 0., 0.], [10.5, 0., 0.]]])
    regions = {3: torch.tensor([0]), 7: torch.tensor([1])}
    pairs = [{'p1_ids': [3, 7], 'p2_ids': [3, 7]}]
    rows = region_diagnostics(a, b, pairs, regions, {0: (regions[3], regions[3])})
    assert rows == [{'union_distance_m_by_frame': [.25, .5], 'winner_region_ids': [3, 3], 'winner_distance_m_by_frame': [1., 2.]}]
    assert 'winner_region_ids' not in region_diagnostics(a, b, pairs, regions, {})[0]


if __name__ == '__main__':
    check()
