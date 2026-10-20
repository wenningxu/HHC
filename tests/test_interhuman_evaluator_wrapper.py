import numpy as np
import torch

from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    evaluator_provenance,
    load_official_metrics_module,
)


class FakeInterCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_motion(self, batch):
        # Exposes sorted first-person marker and received temporal width.
        marker = batch['motions'][:, 0, 0]
        width = torch.full_like(marker, batch['motions'].shape[1])
        return {'motion_emb': torch.stack([marker, width], dim=-1)}

    def encode_text(self, batch):
        values = torch.arange(len(batch['text']), device=self.anchor.device).float()
        return {'text_emb': torch.stack([values, values + 10], dim=-1)}


def _batch():
    motion1 = torch.zeros(3, 5, 262)
    motion2 = torch.zeros_like(motion1)
    motion1[:, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    return (
        ['a', 'b', 'c'], ['text-a', 'text-b', 'text-c'],
        motion1, motion2, torch.tensor([2, 5, 3]),
    )


def test_wrapper_matches_released_descending_length_alignment():
    wrapper = InterHumanEvaluatorWrapper(FakeInterCLIP())
    text, motion = wrapper.get_co_embeddings(_batch())
    # align_idx is [1,2,0]; text is encoded in original order, then aligned.
    torch.testing.assert_close(text, torch.tensor([[1., 11.], [2., 12.], [0., 10.]]))
    torch.testing.assert_close(motion[:, 0], torch.tensor([20., 30., 10.]) * wrapper.std[0] + wrapper.mean[0])
    assert torch.all(motion[:, 1] == 5)


def test_motion_only_path_uses_same_alignment():
    wrapper = InterHumanEvaluatorWrapper(FakeInterCLIP())
    motion = wrapper.get_motion_embeddings(_batch())
    torch.testing.assert_close(motion[:, 0], torch.tensor([20., 30., 10.]) * wrapper.std[0] + wrapper.mean[0])


def test_wrapper_denormalizes_both_people_exactly_once():
    wrapper = InterHumanEvaluatorWrapper(FakeInterCLIP())
    batch = _batch()
    batch[3][:] = 2.0
    prepared, align = wrapper._prepare(batch)
    expected = torch.stack((batch[2], batch[3]), dim=2)
    expected = (expected * wrapper.std + wrapper.mean).flatten(2)[align]
    torch.testing.assert_close(prepared['motions'], expected, rtol=0, atol=0)
    assert not torch.equal(prepared['motions'], torch.cat((batch[2], batch[3]), dim=-1)[align])


def test_normalized_adapter_matches_released_wrapper_on_raw_coordinates():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    source = Path('/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator.py')
    tree = ast.parse(source.read_text())
    original_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == 'EvaluatorModelWrapper')
    methods = [node for node in original_class.body if isinstance(node, ast.FunctionDef)
               and node.name in ('get_co_embeddings', 'get_motion_embeddings')]
    namespace = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    ours = InterHumanEvaluatorWrapper(FakeInterCLIP())
    normalized = _batch()
    raw = (*normalized[:2], normalized[2] * ours.std + ours.mean,
           normalized[3] * ours.std + ours.mean, normalized[4])
    reference = SimpleNamespace(model=ours.model, device=ours.device)
    expected = namespace['get_co_embeddings'](reference, raw)
    actual = ours.get_co_embeddings(normalized)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(ours.get_motion_embeddings(normalized),
                               namespace['get_motion_embeddings'](reference, raw), rtol=0, atol=0)


def test_released_metric_module_is_hash_bound_and_callable():
    metrics = load_official_metrics_module()
    embeddings = np.arange(20, dtype=np.float64).reshape(4, 5)
    mean, covariance = metrics.calculate_activation_statistics(embeddings, 6)
    np.testing.assert_array_equal(mean, (embeddings * 6).mean(axis=0))
    np.testing.assert_array_equal(covariance, np.cov(embeddings * 6, rowvar=False))


def test_provenance_does_not_hash_large_checkpoint_by_default():
    provenance = evaluator_provenance()
    assert provenance['checkpoint']['bytes'] == 1513626777
    assert 'sha256' not in provenance['checkpoint']
    assert provenance['config']['sha256'] == (
        'df6643ccdc84d0b27ce220603d62f302191d7138030378a834f7b538db2bb61f'
    )
