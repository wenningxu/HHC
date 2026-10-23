"""Read the bounded Edmond ASAN experiment; never launch or restart anything."""
import json
import time
from pathlib import Path
from migrate_edmond_transport_local import remote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/hhc_sota_20261016'
STATE = OUT / 'asan_v2_verified_20261023.json'
REMOTE = r'''
import json,time,re
from pathlib import Path
p=Path('/mnt/8T-HDD/wenning/hhc_migration_20261020/workspace/HHC/output/intergen_official_baseline_20261017/native_abort_20261023_1144/asan_single_device_v2')
def read(name):
 f=p/name
 return json.loads(f.read_text()) if f.exists() else None
i=read('inferior_identity.json')
q=Path('/proc')/str(i['pid'])
try:
 s=(q/'stat').read_text().rsplit(')',1)[1].split()
 process={'birth':s[19],'state':s[0],'cmdline':(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()}
except (FileNotFoundError,ProcessLookupError):process=None
with (p/'stdout.log').open('rb') as f:
 f.seek(0,2);f.seek(max(0,f.tell()-65536));tail=f.read().decode(errors='replace')
steps=re.findall(r'niter:\s*(\d+)',tail)
print(json.dumps({'observed_at':time.time(),'identity':i,'process':process,
 'exit':read('exit.json'),'completed':read('completed.json'),'actual_stop':read('actual_stop.json'),
 'last_step':int(steps[-1]) if steps else None,'tail':tail}))
'''


def main():
    previous = json.loads(STATE.read_text())
    due = previous['next_normal_poll_not_before']
    if time.time() < due:
        print(json.dumps({'status': 'not_due', 'not_observed': True, 'due': due}))
        return
    result = remote('edmond', REMOTE, timeout=90)
    if result.returncode:
        raise RuntimeError('SSH observation failed; do not infer trainer exit: ' + result.stderr)
    packet = json.loads(result.stdout)
    (OUT / 'asan_v2_latest.json').write_text(json.dumps(packet, indent=2))
    identity = packet['identity']
    assert (identity['pid'], identity['birth']) == (previous['pid'], previous['birth'])
    proc = packet['process']
    live = bool(proc and proc['birth'] == identity['birth'] and proc['state'] not in ('Z', 'X')
                and 'probe_official_intergen_asan_cloud.py' in proc['cmdline'])
    status = 'live' if live else 'terminal_needs_audit'
    # Completion/exit artifacts require interpretation together with actual process identity.
    if live and packet['exit']:
        status = 'inconsistent_needs_audit'
    updated = dict(previous, observed_at=packet['observed_at'],
                   next_normal_poll_not_before=packet['observed_at'] + 600,
                   status=status, observed_step=packet['last_step'])
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(updated, indent=2))
    temp.replace(STATE)
    print(json.dumps({**updated, 'exit': packet['exit'], 'completed': packet['completed']}))


if __name__ == '__main__':
    main()
