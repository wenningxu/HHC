"""Four-method formal20rep main/MM comparison, with strict shared-evaluator evidence."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.hhc_text_only_formal_contract import file_sha, require, validate_manifests, validate_selection
from tools.interhuman_formal_score_records import read_record, verify_artifacts

METHODS = ('hhc', 'intergen', 'intermask', 'timotion')
MAIN = ('fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'diversity_gt')


def summarize(config):
    main_manifest, mm_manifest = validate_manifests(config['main_manifest'], config['mm_manifest'])
    selected = validate_selection(config['selection_receipt'])
    expected_selection = file_sha(config['selection_receipt'])
    require(set(config['methods']) == set(METHODS), 'Require HHC and all three reproducible public methods')
    all_records = {}; provenance = None; selection = None; artifact_states = {}
    for method in METHODS:
        identities = []
        for mode, manifest in (('main', main_manifest), ('mm', mm_manifest)):
            paths = config['methods'][method][mode + '_scores']
            require(len(paths) == 20 and len(set(paths)) == 20, 'Need twenty distinct score files per mode/method')
            records = [read_record(path, method, mode) for path in paths]
            require(sorted(row['replication'] for row in records) == list(range(20)), 'Replications must be exactly0..19')
            entries = {row['replication']: row for row in manifest['replications']}
            for row in records:
                require(row['request_sha256'] == entries[row['replication']]['request_sha256'], 'Wrong frozen request content')
                needed = MAIN if mode == 'main' else ('multimodality',)
                require(all(key in row['metrics'] and math.isfinite(float(row['metrics'][key])) for key in needed), 'Missing/nonfinite metric')
                if mode == 'main':
                    require(0 <= row['metrics']['top1'] <= row['metrics']['top2'] <= row['metrics']['top3'] <= 1,
                            'Retrieval rates must be ordered probabilities')
                if provenance is None: provenance = row['evaluator']
                require(row['evaluator'] == provenance, 'Evaluator provenance differs across methods/modes/reps')
                identities.append(row['identity'])
                receipt = row['identity']['selection_receipt_sha256']
                if selection is None: selection = receipt
                require(receipt == selection == expected_selection, 'Methods do not share the real frozen comparison selection')
                if method == 'hhc':
                    require(row['identity']['winner'] == selected['winner']
                        and row['identity']['mgt_checkpoint_sha256'] == selected['mgt_checkpoint']['sha256']
                        and row['identity']['shape_policy'] == selected['shape_policy'], 'HHC differs from validation winner')
                artifact_states[str(row['path'])] = verify_artifacts(row)
            all_records[method, mode] = {row['replication']: row for row in records}
        require(all(identity == identities[0] for identity in identities), 'Model/sampler changed between main/MM/reps')
    # Exact GT/text equality was verified during scoring, tied to one actual
    # InterGen reference embedding file for each shared replication.
    for method in ('hhc', 'intermask', 'timotion'):
        for replication in range(20):
            row = all_records[method, 'main'][replication]['value']
            reference = all_records['intergen', 'main'][replication]
            require(row['reference_embeddings_sha256'] == reference['embedding_sha256']
                and all(row['reference_gt_text'][key]['equal'] is True for key in ('GT', 'text')),
                'Missing exact shared GT/text parity for ' + method)
            require(all_records[method, 'main'][replication]['metrics']['diversity_gt']
                    == reference['metrics']['diversity_gt'], 'GT diversity/metric seed differs across methods')
    summaries = {}; values = {}
    for method in METHODS:
        summaries[method] = {}
        for metric in MAIN + ('multimodality',):
            mode = 'mm' if metric == 'multimodality' else 'main'
            array = np.asarray([all_records[method, mode][i]['metrics'][metric] for i in range(20)], np.float64)
            values[method, metric] = array
            summaries[method][metric] = {'mean': float(array.mean()),
                'ci95': float(1.96 * array.std(ddof=0) / math.sqrt(20)), 'replication_values': array.tolist()}
    paired = {}
    indices = np.random.Generator(np.random.PCG64(20261016)).integers(0, 20, size=(10000, 20))
    for baseline in METHODS[1:]:
        paired[baseline] = {}
        for metric in MAIN + ('multimodality',):
            delta = values['hhc', metric] - values[baseline, metric]
            low, high = np.quantile(delta[indices].mean(1), [.025, .975])
            paired[baseline][metric] = {'hhc_minus_baseline': float(delta.mean()),
                'paired_replication_bootstrap95': [float(low), float(high)]}
    return {'schema': 'interhuman_text_only_four_method_20rep_comparison_v1',
        'status': 'complete_four_method_comparison', 'replications': 20,
        'main_samples_per_replication': 1056, 'mm_prompts': 100, 'mm_repeats': 30,
        'ci_definition': '1.96 * population_std / sqrt(20), matching published protocol',
        'paired_ci_definition': '10000 paired replication bootstrap means; seed20261016; not per-motion/model-training uncertainty',
        'selection_receipt_sha256': selection, 'metrics': summaries, 'paired_differences': paired,
        'evaluator': provenance, 'artifact_verification_states': artifact_states,
        'score_files_sha256': {str(row['path']): file_sha(row['path']) for rows in all_records.values() for row in rows.values()},
        'source_sha256': file_sha(Path(__file__)),
        'scope': 'Four reproduced models only; no automatic global SOTA claim. CoShMDM paper4.801 is literature-only pending independent matching public architecture/weights. TIMotion is Transformer+LPA, not RWKV.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Never overwrite a comparison')
    result = summarize(json.loads(args.config.read_text()))
    result.update(config_sha256=file_sha(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({'status': result['status'], 'metrics': result['metrics']}))


if __name__ == '__main__': main()
