"""Read-only cloud observation; never authorizes a resume or changes a process."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/intergen_rotation_scratch_runpod_20261015'
WAIT = ROOT / 'output/hhc_after_intergen_fast_v3_resume_control_20261017'
A_RUN = ROOT / 'output/hhc_cfg_A_historical_40k_20261016'


def record(path):
    try:
        data = path.read_bytes()
        return {'observation': 'read', 'path': str(path), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'data': json.loads(data)}
    except FileNotFoundError:
        return {'observation': 'missing', 'path': str(path)}
    except (OSError, ValueError) as error:
        return {'observation': 'incomplete', 'path': str(path), 'error': str(error)}


def process(pid):
    path = Path('/proc') / str(int(pid))
    try:
        # Fields after the final ')' start with state (field 3).
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (path / 'cmdline').read_bytes().split(b'\0')
        return {'pid': int(pid), 'observation': 'read', 'state': stat[0],
                'active': stat[0] not in {'Z', 'X'},
                'cpu_ticks': int(stat[11]) + int(stat[12]),
                'argv': [arg.decode(errors='replace') for arg in argv if arg]}
    except FileNotFoundError:
        return {'pid': int(pid), 'observation': 'absent_at_observation'}
    except (OSError, ValueError, IndexError) as error:
        return {'pid': int(pid), 'observation': 'incomplete', 'error': str(error)}


def trainers():
    found, errors = [], []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if (any(Path(arg.decode(errors='replace')).name in ('train_intergen_rotation.py', 'train_intergen_rotation_fast_cloud.py')
                    for arg in argv if arg) and str(RUN).encode() in argv):
                found.append(process(path.parent.name))
        except FileNotFoundError:
            pass  # A process exiting while /proc is scanned is ordinary.
        except OSError as error:
            errors.append({'path': str(path), 'error': str(error)})
    return {'processes': found, 'scan_errors': errors}


def last_log(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            chunk = stream.read(65536)
        lines = chunk.split(b'\n')
        if size > 65536:
            lines = lines[1:]
        # Ignore a trailing partial write and search complete JSON lines only.
        for line in reversed(lines[:-1]):
            if line.strip():
                try:
                    return {'observation': 'read', 'data': json.loads(line), 'file_bytes': size}
                except ValueError:
                    continue
        return {'observation': 'incomplete', 'error': 'No complete JSON record in bounded tail'}
    except FileNotFoundError:
        return {'observation': 'missing'}
    except OSError as error:
        return {'observation': 'incomplete', 'error': str(error)}


def gpu_apps():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=20, check=True)
        rows = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if row:
                if len(row) != 3:
                    raise ValueError('Expected exactly pid,name,memory columns')
                rows.append({'pid': int(row[0].strip()), 'name': row[1].strip(),
                             'used_memory_mib': int(row[2].strip())})
        return {'observation': 'read', 'compute_apps': rows, 'scope': 'cloud'}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {'observation': 'incomplete', 'error': str(error), 'scope': 'cloud'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-step', type=int, required=True)
    args = parser.parse_args()
    if str(ROOT) != '/workspace/HHC' or os.name != 'posix':
        parser.error('This observer may run only inside /workspace/HHC on the cloud host')
    if args.validation_step <= 0 or args.validation_step > 704000:
        parser.error('validation-step must be between 1 and 704000')
    waiter = record(WAIT / 'STATUS.json')
    pid = waiter.get('data', {}).get('pid')
    snapshot = {
        'schema': 'intergen_exclusive_fast_read_only_observation_v2',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'intergen': {'status': record(RUN / 'STATUS.json'), 'trainers': trainers(),
                     'last_training_record': last_log(RUN / 'training.jsonl'),
                     'validation': record(RUN / f'validation_{args.validation_step:06d}' /
                                          'validation_metrics.json')},
        'hhc_waiter': waiter,
        'hhc_waiter_process': process(pid) if pid else {'observation': 'pid_unknown'},
        'hhc_A_manifest': record(A_RUN / 'training_manifest.json'),
        'expected_stopped_processes': [process(pid) for pid in (17259, 7178, 2353, 17952)],
        'gpu': gpu_apps(),
        'interpretation': 'Observation only. Missing/read errors are not successful completion; '
                          'this script never opens checkpoints or authorizes a GPU job.'}
    print(json.dumps(snapshot, ensure_ascii=True))


if __name__ == '__main__':
    main()
