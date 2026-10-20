"""Real cloud GPU sampler acceptance on validation min/max/first lengths only.

No benchmark score, test requests, saved motion/checkpoint or model selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import mm_repeat_seed
from dataset.interhuman_native_baseline_h5 import native_length
from tools.export_interhuman_native_baselines_formal_cloud import load_public_model
from tools.hhc_text_only_formal_contract import file_sha, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['intergen', 'intermask', 'timotion'], required=True)
    parser.add_argument('--baseline-cpu-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Existing cloud GPU only')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'output') and not output.exists(), 'New cloud smoke output required')
    cpu = json.loads(args.baseline_cpu_receipt.read_text())
    require(cpu['status'] == 'public_baseline_strict_CPU_load_passed' and cpu['metadata']['method'] == args.method,
            'Completed CPU acceptance required')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    require(len(free) == 1 and int(free[0]) >= 13 * 1024, 'Insufficient single-GPU headroom; preserve training')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    data = InterHumanValidationRequestDataset()
    requests = data.requests
    indices = sorted({0, min(range(480), key=lambda i: requests[i]['processed_length']),
                        max(range(480), key=lambda i: requests[i]['processed_length'])})
    started = time.time()
    generate, metadata, _ = load_public_model(args.method, 'cuda:0')
    for key in ('method', 'model', 'checkpoint_sha256', 'source_files_sha256', 'sampler', 'source_commit',
                'exporter_sha256', 'dataset_sha256', 'official_VQ_bridge_sha256'):
        require(metadata[key] == cpu['metadata'][key], 'CPU/GPU acceptance identity changed: ' + key)
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    class RestrictedRequest(dict):
        def __getitem__(self, key):
            require(key in ('caption', 'processed_length'), 'Unexpected generator input: ' + key)
            return super().__getitem__(key)
    rows = []
    for index in indices:
        request = requests[index]
        seed = mm_repeat_seed(request['sample_seed'], 0)
        first = generate(RestrictedRequest(request), seed)
        require(tuple(first.shape) == (native_length(args.method, request['processed_length']), 2, 262)
                and first.dtype == torch.float32 and torch.isfinite(first).all().item(), 'Invalid real GPU output')
        repeated = generate(RestrictedRequest(request), seed)
        difference = float((first - repeated).abs().max())
        require(difference <= 1e-5, 'Same-seed generation drift')
        require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state()),
                'Request sampling changed outer RNG states')
        rows.append({'validation_index': index, 'identifier': request['identifier'],
            'requested_length': request['processed_length'], 'generated_length': len(first),
            'shape': list(first.shape), 'all_finite': True, 'same_seed_bitwise_equal': torch.equal(first, repeated),
            'same_seed_max_abs': difference})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'passed_real_validation_sampler_GPU_smoke', 'method': args.method,
        'validation_request_sha256': data.payload['request_sha256'], 'model_metadata': metadata,
        'CPU_acceptance_sha256': file_sha(args.baseline_cpu_receipt), 'cases': rows,
        'outer_CPU_CUDA_RNG_preserved': True, 'GPU_peak_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'elapsed_s': time.time() - started, 'formal_test': False, 'benchmark_score_computed': False,
        'GT_motion_or_shape_inputs': False, 'motion_or_checkpoint_saved': False,
        'source_sha256': file_sha(Path(__file__)),
        'scope': 'Real official sampler length/RNG/input contract acceptance on val only; no quality improvement or SOTA evidence'}
    with output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({'status': result['status'], 'method': args.method, 'cases': rows,
                      'GPU_peak_gib': result['GPU_peak_gib'], 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__': main()
