import ast
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.apply_post_formal_online_wandb_metrics import transform_sources


def test_root_then_online_then_global_patch_sequence_is_composable():
    live_sources = {
        relative: (ROOT / relative).read_text(encoding='utf-8')
        for relative in (
            'options/option_transformer.py',
            'train/train_t2i_trans.py',
            'utils/eval_trans.py',
            'utils/motion_masks.py',
            'utils/training_telemetry.py',
        )
    }
    live_combined = '\n'.join(live_sources.values())
    final_markers = (
        'deterministic_historical_root_v1',
        'wandb_run_id',
        'train/gradient_l2_norm',
        'paper_masked_global_token_mean',
        'DiagnosticStopAfterValidation',
    )
    # After the guarded post-formal queue applies all three changes, validate
    # the real live state directly.  Before that transition, exercise the
    # complete composition from the launch-time source below.
    if all(marker in live_combined for marker in final_markers):
        for relative, source in live_sources.items():
            ast.parse(source, filename=relative)
        train_tree = ast.parse(
            live_sources['train/train_t2i_trans.py'],
            filename='train/train_t2i_trans.py',
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in train_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'MGT_RESUME_IDENTITY_KEYS'
        }
        resume_keys = assignments['MGT_RESUME_IDENTITY_KEYS']
        assert 'wandb_mode' in resume_keys
        assert 'diagnostic_stop_iter' not in resume_keys
        return

    with tempfile.TemporaryDirectory(prefix='hhc-post-formal-sequence-') as tmp:
        target = Path(tmp)
        for relative in (
            'options/option_transformer.py',
            'train/train_t2i_trans.py',
            'utils/eval_trans.py',
            'utils/motion_masks.py',
            'utils/training_telemetry.py',
            'dataset/dataset_interx_tokenize_historical_root.py',
        ):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)

        root_patch = (
            ROOT / 'tools/provenance_snapshots/'
            'post_formal_shape_root_tokenizer_wiring.patch'
        )
        global_patch = (
            ROOT / 'tools/provenance_snapshots/'
            'post_formal_global_masked_loss_diagnostic.patch'
        )
        subprocess.run(
            ['git', 'apply', '--check', str(root_patch)],
            cwd=target, check=True,
        )
        subprocess.run(
            ['git', 'apply', str(root_patch)], cwd=target, check=True,
        )

        option_path = target / 'options/option_transformer.py'
        train_path = target / 'train/train_t2i_trans.py'
        eval_path = target / 'utils/eval_trans.py'
        transformed = transform_sources(
            option_path.read_text(encoding='utf-8'),
            train_path.read_text(encoding='utf-8'),
            eval_path.read_text(encoding='utf-8'),
        )
        for path, source in zip(
            (option_path, train_path, eval_path), transformed
        ):
            path.write_text(source, encoding='utf-8')

        subprocess.run(
            ['git', 'apply', '--check', str(global_patch)],
            cwd=target, check=True,
        )
        subprocess.run(
            ['git', 'apply', str(global_patch)], cwd=target, check=True,
        )

        sources = {
            relative: (target / relative).read_text(encoding='utf-8')
            for relative in (
                'options/option_transformer.py',
                'train/train_t2i_trans.py',
                'utils/eval_trans.py',
                'utils/motion_masks.py',
                'utils/training_telemetry.py',
            )
        }
        for relative, source in sources.items():
            ast.parse(source, filename=relative)
        combined = '\n'.join(sources.values())
        for marker in final_markers:
            assert marker in combined

        train_tree = ast.parse(
            sources['train/train_t2i_trans.py'],
            filename='train/train_t2i_trans.py',
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in train_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'MGT_RESUME_IDENTITY_KEYS'
        }
        resume_keys = assignments['MGT_RESUME_IDENTITY_KEYS']
        assert 'wandb_mode' in resume_keys
        # The 50k stop is a diagnostic pause within a 300k optimization
        # horizon, not a different experiment identity.  It must therefore be
        # removable when the same checkpoint lineage is promoted to 300k.
        assert 'diagnostic_stop_iter' not in resume_keys


def test_global_control_launchers_preserve_50k_to_300k_identity():
    initial = (ROOT / 'tools/run_hhc_global_masked_loss_control.sh').read_text(
        encoding='utf-8'
    )
    continuation = (
        ROOT / 'tools/continue_hhc_global_masked_loss_control_300k.sh'
    ).read_text(encoding='utf-8')

    for source in (initial, continuation):
        assert '--total-iter 300000' in source
        assert '--wandb-mode online' in source
        assert '--training-loss-contract paper_masked_global_token_mean' in source
        assert '--sampler sample_intra_cas' in source
        assert '--sampling-steps 50' in source
        assert '--temperature 0.5' in source
        assert '--cfg-scale 3.5' in source
    assert '--diagnostic-stop-iter 50000' in initial
    assert '--diagnostic-stop-iter' not in continuation
    assert '--resume-training-ckpt "${resume_checkpoint}" --resume-in-place' in continuation

    # Both launchers must address the exact same experiment lineage.  Extract
    # the literal instead of accepting two similar-looking names.
    pattern = re.compile(r'^experiment="([^"]+)"$', re.MULTILINE)
    assert pattern.search(initial).group(1) == pattern.search(continuation).group(1)


if __name__ == '__main__':
    test_root_then_online_then_global_patch_sequence_is_composable()
    test_global_control_launchers_preserve_50k_to_300k_identity()
