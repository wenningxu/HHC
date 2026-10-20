"""Restore pinned cloud input archives after tmpfs loss, without launching jobs."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import zipfile

ROOT = Path('/workspace/HHC')
OUT = ROOT/'output/intergen_official_baseline_20261017/recovery_20261020'
FAST = Path('/dev/shm/hhc_inputs')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not FAST.exists()
    staging = FAST.with_name('hhc_inputs.restore_20261020')
    assert not staging.exists()
    archives = [
        (Path('/workspace/runpod-raw-20261015.tar.gz'), 642329772,
         '1246ce6cb461daae02063a84fb6df31b2e24511429af1ce2ad934c8dd55e419c'),
        (ROOT/'dataset/interhuman_official/motions_processed.zip', 2462028219,
         'ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325'),
    ]
    for path, size, expected in archives:
        assert path.stat().st_size == size and digest(path) == expected, path
        print(json.dumps({'verified_archive': str(path), 'sha256': expected}), flush=True)
    staging.mkdir()
    with tarfile.open(archives[0][0]) as archive:
        members = archive.getmembers()
        for member in members:
            assert member.isfile() or member.isdir()
            assert (staging/member.name).resolve().is_relative_to(staging.resolve())
            assert any(member.name.startswith('dataset/interhuman_official/'+key)
                       for key in ('motions', 'annots', 'split'))
        raw_bytes = sum(member.size for member in members)
        with zipfile.ZipFile(archives[1][0]) as motions:
            processed_bytes = sum(row.file_size for row in motions.infolist())
            stat = os.statvfs(staging)
            assert stat.f_bavail*stat.f_frsize > raw_bytes+processed_bytes+1024**3
        archive.extractall(staging, filter='data')
    actual = staging/'dataset/interhuman_official'
    with zipfile.ZipFile(archives[1][0]) as archive:
        for entry in archive.infolist():
            assert entry.filename.startswith('motions_processed/')
            assert (actual/entry.filename).resolve().is_relative_to(actual.resolve())
            archive.extract(entry, actual)
    repair = json.loads((ROOT/'output/intergen_official_baseline_20261017/annotation_eligibility_repair.json').read_text())
    ignore = actual/'ignore_list.txt'
    ignore.write_text(''.join(name+'\n' for name in sorted(repair['excluded_train_annotations'])))
    assert digest(ignore) == repair['ignore_list_sha256']
    # The official training entry points at this root, not its split/ child.
    for split in ('train', 'val', 'test'):
        assert (actual/'split'/(split+'.txt')).is_file()
        (actual/(split+'.txt')).symlink_to(FAST/'dataset/interhuman_official/split'/(split+'.txt'))
    counts = {p: len(list((actual/'motions_processed'/p).glob('*.npy'))) for p in ('person1', 'person2')}
    assert counts == {'person1': 7777, 'person2': 7777}, counts
    staging.rename(FAST)
    evaluation = json.loads((ROOT/'output/intergen_official_reproduction_setup_20261017/SOURCE_PROVENANCE.json').read_text())
    view = Path(evaluation['view'])
    links = {str(p.relative_to(view)): str(p.resolve()) for p in view.rglob('*') if p.is_symlink()}
    assert all((view/relative).exists() for relative in links)
    link_hash = hashlib.sha256(json.dumps(links, sort_keys=True).encode()).hexdigest()
    assert link_hash == evaluation['view_links_sha256']
    config = json.loads((ROOT/'migration/official-intergen-eval20-after-final.json').read_text())
    for path, expected in config['pins'].items():
        assert digest(path) == expected, path
    result = {'status': 'restored_pinned_inputs_no_gpu_launch', 'counts': counts,
              'raw_expanded_bytes': raw_bytes, 'processed_expanded_bytes': processed_bytes,
              'archives': [{'path': str(p), 'bytes': size, 'sha256': sha} for p, size, sha in archives],
              'ignore_list_sha256': digest(FAST/'dataset/interhuman_official/ignore_list.txt'),
              'eval_view_links_sha256': link_hash, 'eval_view_links_verified': len(links),
              'local_array_downloads': 0, 'time_unix': time.time()}
    with (OUT/'input_restore_acceptance.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
