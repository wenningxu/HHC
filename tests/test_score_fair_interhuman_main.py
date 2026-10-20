import numpy as np

from models.interhuman_evaluator_wrapper import load_official_metrics_module
from tools.score_fair_interhuman_main import main_metrics_from_embeddings


def test_main_metric_core_matches_identity_retrieval_and_is_deterministic():
    rng = np.random.default_rng(55)
    embeddings = rng.normal(size=(384, 32))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    generated = embeddings + rng.normal(scale=1e-4, size=embeddings.shape)
    metrics = load_official_metrics_module()
    first = main_metrics_from_embeddings(
        embeddings, generated, generated, metrics,
        batch_size=96, random_seed=77,
    )
    second = main_metrics_from_embeddings(
        embeddings, generated, generated, metrics,
        batch_size=96, random_seed=77,
    )
    assert first == second
    assert first['samples'] == 384
    assert first['matching'] > 0
    assert first['top1'] == first['top2'] == first['top3'] == 1
    assert abs(first['fid']) < 1e-5
    assert first['diversity'] > 0 and first['diversity_gt'] > 0


def test_retrieval_is_computed_within_released_batches_not_globally():
    metrics = load_official_metrics_module()
    rng = np.random.default_rng(91)
    text = rng.normal(size=(384, 64))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    generated = text + rng.normal(scale=1e-5, size=text.shape)
    gt = text.copy()
    result = main_metrics_from_embeddings(
        text, generated, gt, metrics, batch_size=96, random_seed=1
    )
    assert result['top1'] == 1.0
