"""Consume the existing cloud watcher plus live /proc proofs; never launch jobs."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
from pathlib import Path
import hashlib,json,time
root=Path('/workspace/HHC');run=root/'output/intergen_official_baseline_20261017'
def read(p):return json.loads(p.read_text())
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'birth':s[19],'state':s[0],'cpu_ticks':int(s[11])+int(s[12]),'command':(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()}
 except (FileNotFoundError,ProcessLookupError):return None
launch=read(run/'LAUNCH.json');watch=read(run/'OBSERVER_LAUNCH.json')
out={'launch':launch,'launch_sha256':hashlib.sha256((run/'LAUNCH.json').read_bytes()).hexdigest(),'observation':read(run/'OBSERVER_STATUS.json'),'live_trainer':proc(launch['pid']),'live_watcher':proc(watch['pid']),'watcher_launch':watch,'queues':{}}
for name,folder in [('hhc','hhc_after_official_intergen_resume_control_20261017'),('eval','intergen_official_eval20_control_20261017')]:
 control=root/'output'/folder;state=read(control/'STATUS.json')
 out['queues'][name]={'state':state,'process':proc(state['pid']),'intents':[str(p) for p in control.glob('*intent*')]}
out['readback_verified_at_unix']=time.time()
print(json.dumps(out))
'''


def atomic(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def iso(seconds):
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='For recovery acceptance, not routine polling')
    args = parser.parse_args()
    path = ROOT/'migration/control/compute-owner.json'
    before = path.read_bytes()
    owner = json.loads(before)
    migration = owner.get('migration', {})
    if migration and migration.get('status') != 'complete':
        print(json.dumps({'status': 'migration_in_progress', 'not_observed': True,
                          'migration_status': migration.get('status'),
                          'note': 'Use the explicit handoff receipts; do not restart either host.'}))
        return
    due = dt.datetime.fromisoformat(owner['next_ordinary_observation_not_before_utc'])
    if not args.force and dt.datetime.now(dt.timezone.utc) < due:
        print(json.dumps({'status': 'not_due', 'not_observed': True, 'next_ordinary_observation_not_before_utc': due.isoformat()}))
        return
    if owner['backend'] == 'edmond':
        ssh = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=15', owner['ssh_alias']]
        interpreter = 'python3 '+shlex.quote(owner['namespace_launcher'])+' /opt/hhc-env/bin/python'
    else:
        assert owner['backend'] == 'runpod'
        ssh = ['ssh', '-i', str(Path.home()/'.ssh/id_ed25519_newstation'), '-p', str(owner['ssh_port']),
               '-oBatchMode=yes', '-oConnectTimeout=15', 'root@'+owner['ssh_host']]
        interpreter = '/opt/hhc-env/bin/python'
    result = subprocess.run(ssh+[interpreter+' -c '+shlex.quote(REMOTE)],
        capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=180)
    assert result.returncode == 0, (result.returncode, result.stderr)
    packet = json.loads(result.stdout)
    output = ROOT/'output/hhc_sota_20261016'
    atomic(output/'official_cloud_readback_latest.json', packet)
    obs = packet['observation']
    trainer, watcher = packet['live_trainer'], packet['live_watcher']
    assert trainer and watcher and trainer['state'] not in ('Z', 'X') and watcher['state'] not in ('Z', 'X')
    assert (trainer['pid'], trainer['birth']) == (owner['current_trainer_pid'], owner['current_trainer_birth'])
    assert (watcher['pid'], watcher['birth']) == (owner['current_cloud_observer_pid'], owner['current_cloud_observer_birth'])
    assert packet['launch_sha256'] == owner['current_launch_sha256']
    assert 'observe_official_intergen_baseline_cloud.py --watch' in watcher['command']
    assert obs['status'] == 'training_verified' and obs['attributed_trainer_live']
    assert (obs['trainer']['pid'], obs['trainer']['birth']) == (trainer['pid'], trainer['birth'])
    assert obs['observer_pid'] == watcher['pid'] and trainer['cpu_ticks'] >= obs['trainer']['cpu_ticks']
    assert 0 <= packet['readback_verified_at_unix']-obs['observed_at_unix'] <= 660
    for name, pid_key, birth_key in [('hhc','current_hhc_waiter_pid','current_hhc_waiter_birth'),
                                    ('eval','current_official_eval_waiter_pid','current_official_eval_waiter_birth')]:
        queue = packet['queues'][name]
        actual, state = queue['process'], queue['state']
        assert actual and actual['state'] not in ('Z', 'X')
        assert (actual['pid'], actual['birth']) == (owner[pid_key], owner[birth_key])
        assert state['status'] == 'waiting_for_official_training' and not queue['intents']
        assert not state.get('phase') and not state.get('child_pid') and not state.get('phase_exit_codes')
        assert not state.get('dispatch_started') and 'child_exit_code' not in state
    step = obs['latest_training_log']['step']
    assert step > owner['last_verified_step'], 'No new logged update; inspect without automatically restarting'
    previous_class = owner.get('last_turn_classification', 'verified_wait')
    owner.update(status='official_baseline_training_verified', last_verified_step=step,
        last_verified_utc=iso(obs['observed_at_unix']), last_live_trainer_readback_utc=iso(packet['readback_verified_at_unix']),
        next_ordinary_observation_not_before_utc=iso(packet['readback_verified_at_unix']+600),
        last_turn_classification='verified_wait')
    owner['latest_training_checkpoint_observation'] = {'observed_at_utc': iso(obs['observed_at_unix']),
        'files': [p for p in obs['checkpoints'] if '/official_intergen_scratch_20261017/model/' in p['path']],
        'content_validation': 'Stat only; no checkpoint reopened or accepted'}
    atomic(path, owner)
    queues = {'identities': [dict(q['process'], queue_status=q['state']['status']) for q in packet['queues'].values()]+[watcher],
              'observed_at_unix': packet['readback_verified_at_unix']}
    atomic(output/'official_queue_verified_wait_latest.json', queues)
    atomic(output/'official_intergen_verified_wait_latest.json', {
        'schema': 'official_intergen_verified_wait_v2', 'turn_classification': 'verified_wait',
        'previous_turn_classification': previous_class, 'observed_at_utc': iso(obs['observed_at_unix']),
        'observation': obs, 'live_readback': packet, 'new_quality_result': False,
        'owner_before_sha256': hashlib.sha256(before).hexdigest(),
        'owner_after_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'local_checkpoint_downloads': 0, 'local_array_downloads': 0})
    print(json.dumps({'classification': 'verified_wait', 'step': step, 'observed_at_utc': iso(obs['observed_at_unix']),
                     'live_trainer_pid': trainer['pid'], 'live_queue_pids': [q['process']['pid'] for q in packet['queues'].values()],
                     'new_quality_result': False}))


if __name__ == '__main__':
    main()
