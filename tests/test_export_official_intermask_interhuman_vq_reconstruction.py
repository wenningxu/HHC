import ast
from pathlib import Path


def test_official_vq_export_is_hash_bound_and_non_overwriting():
    source = Path('tools/export_official_intermask_interhuman_vq_reconstruction.py').read_text()
    tree = ast.parse(source)
    assert 'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272' in source
    assert "model.load_state_dict(checkpoint['vq_model'], strict=True)" in source
    assert "raise FileExistsError" in source
    assert "compression='lzf'" in source
    assert "InterHumanFrozenRequestDataset" in source
    assert any(isinstance(node, ast.FunctionDef) and node.name == 'reconstruct_person' for node in ast.walk(tree))


def test_official_vq_has_frozen_val_only_export_and_matching_scorer():
    exporter = Path(
        'tools/export_official_intermask_interhuman_vq_reconstruction.py'
    ).read_text()
    scorer = Path('tools/score_interhuman_vq_validation.py').read_text()
    ast.parse(exporter)
    ast.parse(scorer)
    assert 'InterHumanValidationRequestDataset' in exporter
    assert "parser.add_argument('--validation-manifest')" in exporter
    assert "split = 'val'" in exporter
    assert 'expected_count = 480' in exporter
    assert 'expected_batches = 5' in exporter
    assert "handle.attrs['status'] = 'complete'" in exporter
    assert "handle.attrs['exporter_sha256']" in exporter
    assert "handle.attrs['official_source_contract_sha256']" in exporter
    assert 'InterHumanValidationRequestDataset' in scorer
    assert 'InterHumanGeneratedH5Dataset' in scorer
    assert 'main_metrics_from_embeddings' in scorer
    assert 'reconstruction_errors' in scorer
    assert "random_seed=20260130" in scorer

    hhc_scorer = Path('tools/evaluate_hhc_interhuman_vq_validation.py').read_text()
    ast.parse(hhc_scorer)
    assert 'load_hhc_interhuman_vq' in hhc_scorer
    assert 'evaluate_interhuman_vq' in hhc_scorer
    assert "'samples': 480" in hhc_scorer
    assert 'evaluator_provenance(include_checkpoint_sha256=True)' in hhc_scorer
