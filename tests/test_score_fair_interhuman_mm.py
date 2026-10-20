import numpy as np

from tools.score_fair_interhuman_mm import multimodality_from_embeddings


class FakeMetrics:
    @staticmethod
    def calculate_multimodality(activation, times, scale, divisor):
        assert activation.shape == (100, 30, 4)
        assert (times, scale, divisor) == (10, 6, 2)
        first = np.random.choice(30, times, replace=False)
        second = np.random.choice(30, times, replace=False)
        return np.linalg.norm(
            (activation[:, first] * scale - activation[:, second] * scale) / divisor,
            axis=2,
        ).mean()


def test_multimodality_scoring_is_seeded_and_preserves_numpy_state():
    embeddings = np.arange(100 * 30 * 4, dtype=np.float32).reshape(100, 30, 4)
    np.random.seed(77)
    before = np.random.get_state()
    first = multimodality_from_embeddings(embeddings, FakeMetrics, random_seed=1236)
    after = np.random.get_state()
    second = multimodality_from_embeddings(embeddings, FakeMetrics, random_seed=1236)
    assert first == second
    assert all(np.array_equal(left, right) for left, right in zip(before, after))


def test_multimodality_scoring_rejects_wrong_protocol_shape():
    try:
        multimodality_from_embeddings(np.zeros((99, 30, 4)), FakeMetrics)
    except ValueError as exc:
        assert '[100,30,D]' in str(exc)
    else:
        raise AssertionError('wrong MM protocol shape was accepted')
