"""Audit literal mirror-caption expressions without changing training sources.

The candidate below fixes only clockwise/counterclockwise replacement overlap.
It deliberately preserves the historical left/right substring behavior.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_expression(path, target):
    tree = ast.parse(path.read_text())
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(x, ast.Name) and x.id == target for x in node.targets):
            continue
        value = node.value.elt if isinstance(node.value, ast.ListComp) else node.value
        if 'counterclockwise' in ast.unparse(value):
            matches.append(value)
    assert len(matches) == 1, (str(path), len(matches))
    code = compile(ast.Expression(matches[0]), str(path), 'eval')

    def transform(text):
        return eval(code, {'__builtins__': {}}, {'text': text, 'item': text, 'caption': text})

    return transform


def candidate(text):
    # This is an offline candidate, not an installed training augmentation.
    text = text.replace('left', 'tmp').replace('right', 'left').replace('tmp', 'right')
    return re.sub('counterclockwise|clockwise',
                  lambda m: {'clockwise': 'counterclockwise', 'counterclockwise': 'clockwise'}[m[0]],
                  text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists()
    os.nice(15)
    sources = {
        'archived_custom': (ROOT/'dataset/interhuman_root_rotation.py', 'text'),
        'current_official': (ROOT/'third_party/intergen_official_baseline_20261017/datasets/interhuman.py', 'texts_swap'),
        'hhc_adapter': (ROOT/'dataset/interhuman_hhc_adapter.py', 'caption'),
    }
    source_before = {name: digest(path) for name, (path, _) in sources.items()}
    transforms = {name: literal_expression(path, target) for name, (path, target) in sources.items()}
    cases = [
        ('counterclockwise', 'clockwise'),
        ('clockwise', 'counterclockwise'),
        ('left and right, clockwise then counterclockwise.',
         'right and left, counterclockwise then clockwise.'),
        ('both walk forward', 'both walk forward'),
        ('counterclockwise, counterclockwise', 'clockwise, clockwise'),
        ('', ''),
    ]
    synthetic = []
    for text, expected in cases:
        observed = {name: fn(text) for name, fn in transforms.items()}
        assert len(set(observed.values())) == 1, observed
        fixed = candidate(text)
        assert fixed == expected and candidate(fixed) == text
        synthetic.append({'input': text, 'original': observed, 'candidate': fixed, 'expected': expected})
    assert synthetic[0]['original']['current_official'] == 'countercounterclockwise'

    sys.path.insert(0, str(ROOT))
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    import torch
    torch.set_num_threads(1)
    adapter = OfficialInterHumanAdapter(data_root=ROOT/'dataset/interhuman_official')
    identifiers = adapter.split_ids('train', require_text=True, min_raw_length=15)
    assert len(identifiers) == 5637
    affected_ids, changed_count, captions_count, counter_count = [], 0, 0, 0
    examples = []
    expected_probability = 0.0
    for identifier in identifiers:
        captions = adapter.texts(identifier)
        changed_for_id = 0
        for index, caption in enumerate(captions):
            # Strip before applying all three expressions so whitespace policy is
            # not conflated with the replacement-order bug.
            caption = caption.strip()
            old = {name: fn(caption) for name, fn in transforms.items()}
            assert len(set(old.values())) == 1, (identifier, index)
            old = old['archived_custom']
            fixed = candidate(caption)
            captions_count += 1
            counter_count += 'counterclockwise' in caption
            if old != fixed:
                assert 'counterclockwise' in caption
                changed_count += 1
                changed_for_id += 1
                if len(examples) < 5:
                    examples.append({'id': identifier, 'caption_index': index,
                                     'text': caption, 'original_mirror': old, 'candidate_mirror': fixed})
            else:
                assert 'counterclockwise' not in caption
        if changed_for_id:
            affected_ids.append(identifier)
        expected_probability += .5 * changed_for_id / len(captions) / len(identifiers)
    assert changed_count == counter_count
    assert source_before == {name: digest(path) for name, (path, _) in sources.items()}
    assert not torch.cuda.is_initialized()
    report = {
        'schema': 'intergen_mirror_text_cpu_audit_v1', 'status': 'passed',
        'observed_at_unix': time.time(), 'cuda_initialized': False,
        'source': {'path': str(Path(__file__)), 'sha256': digest(Path(__file__))},
        'source_files': {name: {'path': str(path), 'sha256': source_before[name]}
                         for name, (path, _) in sources.items()},
        'synthetic_cases': synthetic,
        'train_scan': {'eligible_ids': len(identifiers), 'captions': captions_count,
                       'captions_changed_by_candidate': changed_count,
                       'affected_ids': affected_ids, 'examples': examples,
                       'expected_bad_caption_probability_per_uniform_training_draw': expected_probability,
                       'probability_assumptions': 'uniform ID and mirror bit, uniform caption within ID'},
        'source_files_unchanged': True,
        'limits': 'Bug inherited by official and custom; cannot explain their FID gap. No training source modified, checkpoint loaded, GPU work, or generated metric. Candidate not connected to any run; official reproduction keeps official behavior. Case-sensitive exact counterclockwise only; no claim to repair other lexical variants.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'report_sha256': digest(args.output), 'train_scan': report['train_scan'],
                      'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
