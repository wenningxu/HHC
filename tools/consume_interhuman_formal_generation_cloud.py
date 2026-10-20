"""Release only a verified formal H5 after preserving its score and embeddings.

Cloud-only explicit action, not automatic cleanup. Never consumes validation,
checkpoints, model assets, directories or another task's files.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hhc_text_only_formal_contract import file_sha, require
from tools.interhuman_formal_score_records import read_record, verify_artifacts


def consume(path, method, mode, root=ROOT):
    record = read_record(path, method, mode)
    generation = record['generation']
    require(generation.is_relative_to((root / 'output').resolve())
        and generation.suffix == '.h5' and generation.is_file(), 'Only this cloud output formal H5 may be consumed')
    require(verify_artifacts(record) == 'present_and_rehashed', 'H5 must be present and hash verified')
    require(record['embedding'] is not None, 'Keep H5 when the scorer did not retain MM embeddings')
    receipt = record['path'].with_suffix('.consumed.json')
    require(not receipt.exists(), 'Consumption already recorded; never replay')
    report = {'schema': 'interhuman_formal_generation_consumption_v1', 'status': 'hash_verified_before_generation_delete',
        'score': str(record['path']), 'score_sha256': file_sha(record['path']),
        'generation': str(generation), 'generation_sha256': record['generation_sha256'],
        'generation_bytes': generation.stat().st_size, 'embedding': str(record['embedding']),
        'embedding_sha256': record['embedding_sha256'], 'method': method, 'mode': mode,
        'replication': record['replication'], 'started': time.time(),
        'source_sha256': file_sha(Path(__file__)),
        'scope': 'H5 hash verified before deletion; original H5 is no longer retained. Reproduce from frozen model/source/request/seed. Score and embedding archive remain.'}
    with receipt.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())
    # Recheck after durable receipt, immediately before the single-file unlink.
    require(file_sha(generation) == report['generation_sha256'], 'Generation changed before unlink')
    generation.unlink()
    report.update(status='hash_verified_then_generation_deleted', ended=time.time())
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n'); os.replace(temporary, receipt)
    require(verify_artifacts(record) == 'hash_verified_at_consumption_then_deleted', 'Consumption readback failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path, required=True)
    parser.add_argument('--method', choices=['hhc', 'intergen', 'intermask', 'timotion'], required=True)
    parser.add_argument('--mode', choices=['main', 'mm'], required=True)
    args = parser.parse_args()
    require(str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Cloud CPU only')
    print(json.dumps(consume(args.score, args.method, args.mode)), flush=True)


if __name__ == '__main__': main()
