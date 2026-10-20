"""Check the frozen 175k VQ gate; never resumes training or changes criteria."""
import argparse
import json
import math
from pathlib import Path

from compare_fair_interx_methods_paired import atomic_json, sha256_file

BASE = Path('output/hhc_original_vq_reproduction_20260910')
REFERENCE_SHA = 'fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb'
REQUEST_SHA = 'ade004f1e586a6c94afa41c1dd15defc90d1493c09ae5b9b968f2fecb1e99175'
GT_SHA = 'ea8ed90af73293f6487bda51fb96b1ca41ad307a93d45ae47aa36a02b82e9804'
REFERENCE_SCORE_SHAS = [
    '02faec8d97695ea6a94d242ac758ad4b6aa22c1fa454e21fe1e007f57a5fc389',
    'd3c24a25ed605b433ff57e4b84cc3c5fb86aba21d05a1a5763f2617e4ac95406',
    '338a8fbbe80e4e4dd6984ac7d2fc6803643b284e0e714d1c5375c818d36cf100',
]


def no_worse(old, new, higher=False):
    if not math.isfinite(old) or not math.isfinite(new):
        raise ValueError('nonfinite metric')
    return new >= old if higher else new <= old


def load(prefix):
    paths = [Path(str(prefix) + '_' + suffix + '.json') for suffix in
             ('reconstruction', 'joint_cpu', 'physical_per_sample')]
    records = [json.loads(p.read_text()) for p in paths]
    r, j, p = records
    assert r['samples'] == j['samples'] == p['samples'] == 544
    assert r['frames'] == 71026 and j['status'] == 'complete'
    assert r['input_motion_rep'] == r['output_motion_rep'] == 'raw'
    assert r['requests_sha256'] == j['requests']['sha256'] == p['requests_sha256'] == REQUEST_SHA
    assert r['ground_truth_sha256'] == j['raw_ground_truth']['sha256'] == GT_SHA
    assert sha256_file(Path(r['requests'])) == REQUEST_SHA
    assert sha256_file(Path(r['ground_truth'])) == GT_SHA
    assert sha256_file(Path(r['checkpoint'])) == r['checkpoint_sha256']
    h5 = Path(r['reconstruction_h5']).resolve()
    assert h5 == Path(j['generations']['path']).resolve() == Path(p['generations']).resolve()
    assert sha256_file(h5) == r['reconstruction_h5_sha256'] == j['generations']['sha256'] == p['generations_sha256']
    counts = p['per_sample_counts']
    assert len(counts) == len({x['name'] for x in counts}) == 544
    sums = {k: sum(x[k] for x in counts) for k in counts[0] if k != 'name'}
    for key, value in sums.items():
        assert math.isclose(value, p['counts'][key], abs_tol=1e-12, rel_tol=1e-12), key
    slide = sums['grounded_foot_speed_sum_mps'] / sums['grounded_foot_transitions']
    tp = sums['true_positive_frames']
    f1 = 2 * tp / (2 * tp + sums['false_positive_frames'] + sums['false_negative_frames'])
    assert abs(slide - p['foot_sliding_mps']) < 1e-12
    assert abs(f1 - p['contact_f1']) < 1e-12
    return records, {str(path): sha256_file(path) for path in paths}


def main():
    assert no_worse(1., 1.) and no_worse(1., .9) and not no_worse(1., 1.1)
    assert no_worse(1., 1.1, higher=True) and not no_worse(1., .9, higher=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, reference_hashes = load(BASE / 'original_step000175000')
    assert list(reference_hashes.values()) == REFERENCE_SCORE_SHAS, 'frozen reference scores changed'
    candidate, candidate_hashes = load(args.candidate_prefix)
    assert reference[0]['checkpoint_sha256'] == REFERENCE_SHA, 'reference cannot be replaced'
    assert candidate[0]['checkpoint_iteration'] == 175000, 'no cross-step substitution'
    protocols = (
        ('fk_skeleton_contract', 'temporal_stride', 'codebook_size_per_part'),
        ('schema', 'evaluation_role', 'evaluator_provenance', 'seed', 'effective_length_policy'),
        ('protocol', 'floor_height_policy', 'contact_threshold_m', 'ground_height_threshold_m', 'repeat'),
    )
    for old, new, keys in zip(reference, candidate, protocols):
        for key in keys:
            assert old[key] == new[key], 'protocol differs: ' + key
    assert [x['name'] for x in reference[2]['per_sample_counts']] == [x['name'] for x in candidate[2]['per_sample_counts']]
    comparisons = {}

    def compare(name, old, new, higher=False):
        assert math.isfinite(old) and math.isfinite(new), name + ': nonfinite'
        comparisons[name] = dict(reference=old, candidate=new, delta=new-old,
                                 direction='higher' if higher else 'lower',
                                 passed=no_worse(old, new, higher))

    for key in ('rmse_raw_features', 'mpjpe_m', 'body_mpjpe_m',
                'hand_mpjpe_m', 'root_translation_error_m'):
        compare(key, reference[0][key], candidate[0][key])
    for part in ('JointsFull', 'JointsBody', 'JointsHands'):
        compare(part + '_FID', reference[1]['metrics'][part]['fid'], candidate[1]['metrics'][part]['fid'])
    compare('foot_sliding_mps', reference[2]['foot_sliding_mps'], candidate[2]['foot_sliding_mps'])
    compare('contact_f1', reference[2]['contact_f1'], candidate[2]['contact_f1'], higher=True)
    passed = all(item['passed'] for item in comparisons.values())
    report = dict(schema='hhc_frozen_vq_175k_gate_v1', status='pass' if passed else 'fail',
                  policy='all ten metrics no worse; zero added tolerance; original175k fixed',
                  protocol_and_recorded_artifact_hashes='verified',
                  scope='score/artifact consistency gate, not historical source or multi-seed certification',
                  reference_score_sha256=reference_hashes, candidate_score_sha256=candidate_hashes,
                  comparisons=comparisons)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
