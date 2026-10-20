import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from options.option_transformer import get_args_parser, resolve_text_drop_probability
from models.two_person_transformer import CrossCondTransBase


class TrainingTextDropProbabilityTest(unittest.TestCase):
    def test_resolution_and_rejection(self):
        args = SimpleNamespace(conditioning_contract='paper_text_cfg')
        self.assertEqual(resolve_text_drop_probability(args), .1)
        for value in (0., .2, 1.):
            args.text_drop_prob = value
            self.assertEqual(resolve_text_drop_probability(args), value)
        for value in (-.1, 1.1, float('nan'), float('inf')):
            args.text_drop_prob = value
            with self.assertRaises(ValueError):
                resolve_text_drop_probability(args)
        args.conditioning_contract = 'historical_mmm_shape_dropout'
        args.text_drop_prob = .2
        with self.assertRaises(ValueError):
            resolve_text_drop_probability(args)
        args.text_drop_prob = None
        self.assertEqual(resolve_text_drop_probability(args), .1)

    def test_actual_joint_mask_and_eval_invariance(self):
        state = SimpleNamespace(training=True, text_drop_prob=.2)
        words, global_text = torch.ones(4096, 3, 2), torch.ones(4096, 2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(90210)
            w, g = CrossCondTransBase.mask_text(state, words, global_text)
        dropped = g.eq(0).all(1)
        self.assertTrue(torch.equal(w.eq(0).all((1, 2)), dropped))
        self.assertTrue(.17 < dropped.float().mean().item() < .23)
        state.training = False
        for probability in (0., .2, 1.):
            state.text_drop_prob = probability
            w, g = CrossCondTransBase.mask_text(state, words, global_text)
            self.assertTrue(torch.equal(w, words) and torch.equal(g, global_text))
            w, g = CrossCondTransBase.mask_text(state, words, global_text, force_mask=True)
            self.assertEqual(w.count_nonzero().item() + g.count_nonzero().item(), 0)

    def test_training_binding_and_resume_identity(self):
        source = (Path(__file__).resolve().parents[1] / 'train/train_t2i_trans.py').read_text()
        self.assertIn('self.trans_encoder.trans_base.text_drop_prob =', source)
        tree = ast.parse(source)
        keys = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'MGT_RESUME_IDENTITY_KEYS'
                            for t in n.targets))
        self.assertIn('text_drop_prob', keys)
        with patch('sys.argv', ['train']):
            self.assertIsNone(get_args_parser().text_drop_prob)
        with patch('sys.argv', ['train', '--text-drop-prob', '0.2']):
            self.assertEqual(get_args_parser().text_drop_prob, .2)

    def test_old_manifest_default_and_override_resume(self):
        from train.train_t2i_trans import (
            MGT_RESUME_IDENTITY_KEYS, MGT_TRAINING_SOURCE_PATHS,
            prepare_mgt_output_directory, sha256_file,
        )
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'fixture_text_dropout'
            run.mkdir()
            checkpoint = run / 'latest-000000100.ckpt'
            checkpoint.touch()
            values = dict.fromkeys(MGT_RESUME_IDENTITY_KEYS)
            values.update(exp_name='text_dropout', total_iter=300)
            old_values = {k: v for k, v in values.items() if k != 'text_drop_prob'}
            (run / 'training_manifest.json').write_text(json.dumps({
                'resolved_arguments': old_values,
                'sha256': {p: sha256_file(root / p) for p in MGT_TRAINING_SOURCE_PATHS},
            }))
            def args(probability):
                return SimpleNamespace(**{**values, 'text_drop_prob': probability},
                                       out_dir=directory, resume_in_place=True,
                                       resume_training_ckpt=str(checkpoint))
            self.assertEqual(prepare_mgt_output_directory(args(None))['checkpoint_step'], 100)
            with self.assertRaisesRegex(ValueError, 'text_drop_prob'):
                prepare_mgt_output_directory(args(.2))


if __name__ == '__main__':
    unittest.main()
