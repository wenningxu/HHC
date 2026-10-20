"""One-off, exact-diff provenance migration for the stopped 10k v2 pilot."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_vq_interhuman import atomic_json, canonical_sha256, sha256_file


def identical(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            identical(left, right)
    else:
        assert a == b


def main():
    run = Path('/home/wenning/HHC_interhuman_runs/vq_interhuman/hhc_interhuman_vq_bodyroot_evalfix_v2')
    backup = run / 'rng_resume_fix_20260930'
    manifest_path = run / 'training_manifest.json'
    old = json.loads((backup / 'training_manifest.before.json').read_text())
    assert json.loads(manifest_path.read_text()) == old, 'already migrated or changed'
    assert old['status'] == 'pilot_decision_ready' and old['current_optimizer_step'] == 10000
    source = str(ROOT / 'train/train_vq_interhuman.py')
    before = backup / 'train_vq_interhuman.before.py'
    assert sha256_file(before) == old['sources'][source]
    expected = before.read_text().replace("torch.set_rng_state(state['torch'])", "torch.set_rng_state(state['torch'].cpu())").replace("torch.cuda.set_rng_state_all(state['cuda'])", "torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])")
    assert Path(source).read_text() == expected, 'migration permits ONLY RNG device fix'
    for path, digest in old['sources'].items():
        if path != source:
            assert sha256_file(Path(path)) == digest, path
    new = json.loads(json.dumps(old))
    new['sources'][source] = sha256_file(Path(source))
    contract = {key: new[key] for key in ('schema', 'resolved_arguments', 'dataset', 'sources', 'observability')}
    contract['observability'] = {key: value for key, value in contract['observability'].items() if key != 'wandb_run_id'}
    new['contract_sha256'] = canonical_sha256(contract)
    checkpoint_records = []
    for archived, destination in (('latest.before.pth', 'latest.pth'), ('best.before.pth', 'best_validation_fid.pth')):
        original = backup / archived
        target = run / 'checkpoints' / destination
        assert sha256_file(target) == sha256_file(original), 'checkpoint changed'
        payload = torch.load(original, map_location='cpu', weights_only=False)
        assert payload['training_contract_sha256'] == old['contract_sha256']
        payload['training_contract_sha256'] = new['contract_sha256']
        temporary = target.with_suffix('.rngfix.tmp')
        torch.save(payload, temporary)
        reread = torch.load(temporary, map_location='cpu', weights_only=False)
        identical(payload, reread)
        baseline = torch.load(original, map_location='cpu', weights_only=False)
        for key in baseline:
            if key != 'training_contract_sha256':
                identical(baseline[key], reread[key])
        os.replace(temporary, target)
        checkpoint_records.append({'path': str(target), 'before_sha256': sha256_file(original), 'after_sha256': sha256_file(target), 'only_changed_field': 'training_contract_sha256'})
    new['best_validation_checkpoint_sha256'] = sha256_file(Path(new['best_validation_checkpoint']))
    record = {'reason': 'CPU and CUDA RNG byte states must be on CPU after checkpoint map_location=cuda; failed continuation made no optimizer steps',
              'before_contract': old['contract_sha256'], 'after_contract': new['contract_sha256'],
              'old_source_sha256': old['sources'][source], 'new_source_sha256': new['sources'][source],
              'checkpoints': checkpoint_records, 'archived_manifest': str(backup / 'training_manifest.before.json')}
    atomic_json(backup / 'migration.json', record)
    atomic_json(manifest_path, new)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
