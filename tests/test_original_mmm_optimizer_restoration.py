"""Regression guard: formal optimization must use original MMM numerics."""

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMM = Path('/home/wenning/projects/MMM')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


def test_original_mmm_numeric_sources_are_restored():
    assert digest(ROOT / 'optimization/losses.py') == digest(
        MMM / 'optimization/losses.py'
    )
    assert function_ast(
        ROOT / 'optimize/optimize_full_body.py', 'optimize_full_body'
    ) == function_ast(MMM / 'optimize_full_body.py', 'optimize_full_body')


def test_invalid_experimental_rewrite_is_not_imported():
    forbidden = (
        'losses_experimental_invalid',
        'optimize_full_body_experimental_invalid',
    )
    for path in list((ROOT / 'tools').glob('*.py')) + list(
        (ROOT / 'train').glob('*.py')
    ):
        source = path.read_text(encoding='utf-8')
        assert not any(name in source for name in forbidden), path


def test_shape_augmentation_wrapper_cannot_change_original_mmm_numerics():
    """Keep the dataset wrapper from silently becoming a new optimizer."""
    path = ROOT / 'tools/run_interx_shape_augmentation.py'
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'optimize_full_body'
    ]
    assert len(calls) == 1
    call = calls[0]
    assert [keyword.arg for keyword in call.keywords] == [
        'temporal_mask',
        'interaction_pairs',
        'num_epochs',
        'lr',
        'device',
    ]
    assert ast.unparse(call.keywords[2].value) == 'args.num_epochs'
    assert ast.unparse(call.keywords[3].value) == 'args.learning_rate'

    # These runtime guards make the declared original-MMM contract executable:
    # unsupported backends/policies cannot merely be selected by CLI accident.
    required_guards = (
        "choices=('nearest_vertex',)",
        "choices=('original_mmm',)",
        "choices=('released_all_per_frame',)",
        "if fine_stage_start != 100:",
        "if loss_weights:",
        "raise ValueError('loss overrides are disabled for original-MMM numerics')",
    )
    for guard in required_guards:
        assert guard in source, guard


if __name__ == '__main__':
    test_original_mmm_numeric_sources_are_restored()
    test_invalid_experimental_rewrite_is_not_imported()
    test_shape_augmentation_wrapper_cannot_change_original_mmm_numerics()
    print('original MMM optimizer restoration tests passed')
