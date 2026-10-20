"""Bounded, cloud-only InterGen throughput probe; original trainer is resumed."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/intergen_rotation_scratch_runpod_20261015'
SOURCE_SHA = '5c5f732163eeeb5e8f5d16768aad5646076dcf0fc304b82860a85e2113354fc8'


def write(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'birth': int(fields[19]),
                'cpu_ticks': int(fields[11]) + int(fields[12]),
                'command': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def same_live(identity):
    p = proc(identity['pid'])
    return p is not None and p['birth'] == identity['birth'] and p['state'] not in ('Z', 'X')


def stop_worker(folder):
    path = folder / 'worker_identity.json'
    identities = []
    if path.exists():
        identity = json.loads(path.read_bytes())
        if identity:
            identities.append(identity)
    # Cover a controller dying between Popen and writing the child receipt.
    for candidate in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = candidate.read_bytes().split(b'\0')
            if (str(Path(__file__).resolve()).encode() in argv and b'--worker' in argv
                    and str(folder).encode() in argv):
                identity = proc(int(candidate.parent.name))
                if identity:
                    identities.append(identity)
        except FileNotFoundError:
            pass
    for identity in {i['pid']: i for i in identities}.values():
        if same_live(identity):
            os.killpg(identity['pid'], signal.SIGTERM)
            deadline = time.monotonic() + 5
            while same_live(identity) and time.monotonic() < deadline:
                time.sleep(.1)
            if same_live(identity):
                os.killpg(identity['pid'], signal.SIGKILL)


def watchdog(folder):
    guard = json.loads((folder / 'guard.json').read_bytes())
    while not (folder / 'controller_done.json').exists():
        if time.time() >= guard['deadline_unix'] or not same_live(guard['controller']):
            stop_worker(folder)
            if same_live(guard['trainer']):
                os.kill(guard['trainer']['pid'], signal.SIGCONT)
            write(folder / 'watchdog_result.json', {'action': 'worker_stopped_trainer_continued',
                                                   'utc_unix': time.time()})
            return
        time.sleep(1)
    write(folder / 'watchdog_result.json', {'action': 'controller_completed'})


def control(folder, verify_only=False, grouped_only=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Controller must not own CUDA'
    assert hashlib.sha256((ROOT / 'tools/train_intergen_rotation.py').read_bytes()).hexdigest() == SOURCE_SHA
    status = json.loads((RUN / 'STATUS.json').read_bytes())
    assert status['status'] in ('training', 'running') and status['step'] < 704000
    trainer = proc(2353)
    assert trainer and 'tools/train_intergen_rotation.py' in trainer['command'] and str(RUN) in trainer['command']
    assert trainer['state'] not in ('T', 't', 'Z', 'X')
    a = json.loads((ROOT / 'output/hhc_cfg_A_historical_40k_20261016/training_manifest.json').read_bytes())
    assert a['status'] == 'paused_by_user_priority' and a['step'] == 4000
    query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert [int(row[0]) for row in csv.reader(query.splitlines()) if row] == [2353]
    folder.mkdir(exist_ok=False)
    guard = {'trainer': trainer, 'controller': proc(os.getpid()), 'deadline_unix': time.time()+900,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'intergen_status_before': status, 'purpose': 'InterGen throughput only; HHC stays paused'}
    write(folder / 'guard.json', guard)
    log = (folder / 'watchdog.log').open('w')
    watcher = subprocess.Popen([sys.executable, str(Path(__file__)), '--watchdog', '--output', str(folder)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    write(folder / 'watchdog_identity.json', proc(watcher.pid))
    worker_process = None
    paused = False
    exit_code = None
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Controller terminated')))
        os.kill(trainer['pid'], signal.SIGSTOP)
        paused = True
        deadline = time.monotonic()+5
        while proc(trainer['pid'])['state'] not in ('T', 't'):
            assert time.monotonic() < deadline, 'Trainer did not stop'
            time.sleep(.05)
        time.sleep(1)  # Let kernels already submitted by the trainer finish.
        write(folder / 'STATUS.json', {'status': 'original_suspended_probe_starting',
                                       'trainer': proc(2353), 'watchdog_pid': watcher.pid})
        with (folder / 'worker.log').open('w') as output:
            worker_command = [sys.executable, '-u', str(Path(__file__)), '--worker', '--output', str(folder)]
            if verify_only:
                worker_command.append('--verify-only')
            if grouped_only:
                worker_command.append('--grouped-only')
            worker_process = subprocess.Popen(worker_command, cwd=ROOT,
                                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'},
                                               stdin=subprocess.DEVNULL, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
            write(folder / 'worker_identity.json', proc(worker_process.pid))
            try:
                exit_code = worker_process.wait(timeout=780)
            except subprocess.TimeoutExpired:
                stop_worker(folder)
                worker_process.wait(timeout=10)
                raise RuntimeError('Probe exceeded780seconds; original trainer will resume')
        assert exit_code == 0, f'Probe worker exit{exit_code}; inspect worker.log'
    finally:
        if worker_process is not None and worker_process.poll() is None:
            stop_worker(folder)
            worker_process.wait(timeout=10)
        if paused and same_live(trainer):
            os.kill(trainer['pid'], signal.SIGCONT)
        write(folder / 'controller_done.json', {'worker_exit_code': exit_code,
                                                 'trainer_after': proc(2353), 'unix': time.time()})
        try:
            watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # The done receipt tells the independent guard to leave; trainer
            # restoration already happened, so a slow guard is not probe failure.
            write(folder / 'watchdog_wait.json', {'status': 'done_receipt_written_guard_exit_pending'})
    print(json.dumps({'status': 'probe_finished_original_continued', 'worker_exit_code': exit_code,
                      'trainer': proc(2353), 'result': str(folder / 'results.json')}), flush=True)


def worker(folder, verify_only=False, grouped_only=False):
    import random
    import statistics
    from types import SimpleNamespace
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.65)
    sys.path.insert(0, str(ROOT))
    from dataset.interhuman_root_rotation import RootRotationData
    from tools.intergen_rotation_objective import rotation_objective
    from tools.intergen_rotation_grouped_objective import rotation_objective as grouped_objective
    config = json.loads((RUN / 'config.json').read_bytes())
    source = Path(config['source'])
    sys.path.insert(0, str(source))
    os.chdir(source)
    from models.intergen import InterGen
    saved = torch.load(RUN / 'recovery.pt', map_location='cpu', mmap=True, weights_only=False)
    checkpoint = saved['checkpoint']
    assert checkpoint['step'] == saved['step'] and checkpoint['config']['effective_batch'] == 32
    cfg = SimpleNamespace(**yaml.safe_load((source / 'configs/model.yaml').read_text()))
    cfg.INPUT_DIM = 268
    model = InterGen(cfg)
    model.positional_embedding.requires_grad_(False)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(checkpoint['trainable_state']) == names
    model = model.cuda()
    mean, std = checkpoint['mean'].cuda(), checkpoint['std'].cuda()
    data = RootRotationData()
    results = []
    optimizer = None
    arms = [(4, 'original_logging'), (4, 'deferred_logging')]
    if not verify_only:
        arms += [(8, 'deferred_logging'), (16, 'deferred_logging'), (32, 'deferred_logging')]
    if grouped_only:
        arms = [(4, 'original_logging'), (16, 'grouped_deferred_logging'), (32, 'grouped_deferred_logging')]
    for micro, mode in arms:
        if optimizer is not None:
            del optimizer
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        record = {'micro_batch': micro, 'accumulation': 32//micro, 'effective_batch': 32,
                  'mode': mode, 'checkpoint_step': checkpoint['step'], 'warmup_updates': 2,
                  'measured_updates': 8}
        write(folder / 'STATUS.json', {'status': 'benchmarking', 'arm': record, 'worker_pid': os.getpid()})
        try:
            model.load_state_dict(checkpoint['trainable_state'], strict=False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=config['lr'], weight_decay=2e-5)
            optimizer.load_state_dict(saved['optimizer'])
            # AdamW keeps non-capturable step counters on CPU. load_state_dict
            # can alias these tensors, unlike CUDA moments copied from CPU.
            # Each arm must own counters instead of mutating the recovery view.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                        state[key] = value.clone()
            random.seed(20261017)
            np.random.seed(20261017)
            torch.manual_seed(20261017)
            torch.cuda.manual_seed_all(20261017)
            loader = torch.utils.data.DataLoader(data, batch_size=micro, shuffle=True,
                                                 drop_last=True, num_workers=0)
            iterator = iter(loader)
            timings, data_timings, metrics = [], [], []
            inputs = hashlib.sha256()
            torch.cuda.reset_peak_memory_stats()
            for update in range(10):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                totals = {}
                data_seconds = 0.
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(32//micro):
                    tick = time.perf_counter()
                    texts, raw, lengths, offsets = next(iterator)
                    data_seconds += time.perf_counter()-tick
                    for index, text in enumerate(texts):
                        for tensor in (raw[index], lengths[index], offsets[index]):
                            inputs.update(tensor.numpy().tobytes())
                        inputs.update(json.dumps(text, ensure_ascii=True).encode())
                    raw, lengths, offsets = raw.cuda(), lengths.cuda(), offsets.cuda()
                    target = ((raw-mean)/std).flatten(2)
                    t, _ = model.decoder.sampler.sample(len(raw), 'cuda:0')
                    mask = (torch.arange(raw.shape[1], device='cuda')[None] < lengths[:, None]).float()[..., None].expand(-1, -1, 2)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        cond = model.text_process({'text': list(texts)})['cond']
                        cond, _ = model.decoder.mask_cond(cond, .1)
                        noisy = model.decoder.diffusion.q_sample(target, t)
                        pred = model.decoder.net(noisy, t, mask=mask, cond=cond)
                    objective = grouped_objective if mode == 'grouped_deferred_logging' else rotation_objective
                    losses = objective(pred, target, offsets, lengths, t, mean, std)
                    assert torch.isfinite(losses['total'])
                    (losses['total']/(32//micro)).backward()
                    # Last update is a logging update in both arms; earlier
                    # updates match the ordinary49/50 non-logging cadence.
                    if mode == 'original_logging':
                        for key, value in losses.items():
                            totals[key] = totals.get(key, 0.)+float(value.detach())/(32//micro)
                    elif update == 9:
                        for key, value in losses.items():
                            contribution = value.detach()/(32//micro)
                            totals[key] = contribution if key not in totals else totals[key]+contribution
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                optimizer.step()
                if update == 9 and mode != 'original_logging':
                    totals = dict(zip(totals, torch.stack(list(totals.values())).cpu().tolist()))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                if update >= 2:
                    timings.append(elapsed)
                    data_timings.append(data_seconds)
                if update == 9:
                    metrics.append(totals)
            digest = hashlib.sha256()
            for name, parameter in model.named_parameters():
                if name in names:
                    digest.update(name.encode())
                    digest.update(parameter.detach().cpu().numpy().tobytes())
            opt_digest = hashlib.sha256()
            optimizer_state = optimizer.state_dict()
            opt_digest.update(json.dumps(optimizer_state['param_groups'], sort_keys=True).encode())
            for parameter_id, state in optimizer_state['state'].items():
                opt_digest.update(str(parameter_id).encode())
                for key, value in sorted(state.items()):
                    opt_digest.update(key.encode())
                    opt_digest.update(value.detach().cpu().numpy().tobytes()
                                      if isinstance(value, torch.Tensor) else repr(value).encode())
            record.update(status='complete', seconds_per_update=timings,
                          median_seconds=statistics.median(timings),
                          mean_data_seconds=statistics.mean(data_timings),
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                          final_metrics=metrics[-1], final_trainable_sha256=digest.hexdigest(),
                          final_optimizer_sha256=opt_digest.hexdigest(),
                          input_sha256=inputs.hexdigest(),
                          host_rng_sha256=hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))+torch.get_rng_state().numpy().tobytes()).hexdigest(),
                          cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
            del optimizer_state, state, value
        except torch.OutOfMemoryError as error:
            record.update(status='oom_rejected', error=str(error))
        results.append(record)
        write(folder / 'results.json', {'status': 'partial', 'arms': results})
    first, second = results[:2]
    equivalent = (first.get('status') == second.get('status') == 'complete' and
                  first['final_trainable_sha256'] == second['final_trainable_sha256'] and
                  first['final_optimizer_sha256'] == second['final_optimizer_sha256'] and
                  first['input_sha256'] == second['input_sha256'] and
                  first['host_rng_sha256'] == second['host_rng_sha256'] and
                  first['cuda_rng_sha256'] == second['cuda_rng_sha256'])
    if grouped_only:
        equivalent = None  # Different microbatch sizes are not exact replays.
    write(folder / 'results.json', {'status': 'complete', 'arms': results,
                                    'same_micro_logging_weights_and_rng_exact': equivalent,
                                    'scope': 'Short InterGen performance probe; no saved model, no val/test scoring. Larger micro batches change stochastic execution; not an exact training replay. Original GPU allocation remains resident; worker memory capped at65percent, so OOM does not establish standalone24GB infeasibility. Input hashing adds minor common timing overhead.',
                                    'original_checkpoint_written': False,
                                    'original_source_modified': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--grouped-only', action='store_true')
    args = parser.parse_args()
    assert str(ROOT) == '/workspace/HHC' and os.name == 'posix', 'Cloud only'
    folder = args.output.resolve()
    assert folder.is_relative_to(ROOT / 'output')
    if args.worker:
        worker(folder, args.verify_only, args.grouped_only)
    elif args.watchdog:
        watchdog(folder)
    else:
        control(folder, args.verify_only, args.grouped_only)


if __name__ == '__main__':
    main()
