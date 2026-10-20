"""Download the public official raw folder, with paginated metadata and MD5 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time

import requests

FOLDER = '1cr46yZHEgdlaMR61TJojjZ05Hlq9DC70'
URL = f'https://drive.google.com/drive/folders/{FOLDER}'
LOCAL = threading.local()


def valid_name(name):
    return re.fullmatch(r'\d+\.pkl', name) is not None


def inventory():
    # Public Drive web-client keys; no account credentials are read or persisted.
    session = requests.Session()
    html = session.get(URL, timeout=60)
    html.raise_for_status()
    params = {'q': f"'{FOLDER}' in parents and trashed=false", 'pageSize': 1000,
              'fields': 'nextPageToken,files(id,name,size,md5Checksum)'}
    for key in dict.fromkeys(re.findall(r'AIza[\w-]+', html.text)):
        params['key'] = key
        response = session.get('https://www.googleapis.com/drive/v3/files',
                               params=params, headers={'Referer': 'https://drive.google.com/'}, timeout=60)
        if response.ok:
            break
    else:
        raise RuntimeError('Public folder listing unavailable')
    files = []
    while True:
        response.raise_for_status()
        page = response.json()
        files.extend(page['files'])
        token = page.get('nextPageToken')
        if not token:
            break
        params['pageToken'] = token
        response = session.get('https://www.googleapis.com/drive/v3/files',
                               params=params, headers={'Referer': 'https://drive.google.com/'}, timeout=60)
    assert files and all(valid_name(f['name']) and f.get('md5Checksum') for f in files)
    assert len({f['id'] for f in files}) == len(files)
    unique = {}
    for item in files:
        previous = unique.get(item['name'])
        if previous:
            if (previous['size'], previous['md5Checksum']) != (item['size'], item['md5Checksum']):
                raise RuntimeError(f'Conflicting official duplicate filename: {item["name"]}')
            previous.setdefault('identical_duplicate_ids', []).append(item['id'])
        else:
            unique[item['name']] = item
    print(f'official entries={len(files)}, unique filenames={len(unique)}', flush=True)
    return sorted(unique.values(), key=lambda f: int(f['name'][:-4]))


def verified(path, item):
    return (path.is_file() and path.stat().st_size == int(item['size'])
            and hashlib.md5(path.read_bytes()).hexdigest() == item['md5Checksum'])


def download(item, destination):
    path = destination / item['name']
    if verified(path, item):
        return
    if path.exists():
        raise RuntimeError(f'Existing file does not match official metadata: {path}')
    partial = path.with_suffix('.pkl.part')
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(5):
        try:
            response = LOCAL.session.get('https://drive.usercontent.google.com/download',
                                         params={'id': item['id'], 'export': 'download', 'confirm': 't'},
                                         timeout=20)
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}')
            if (len(response.content) == int(item['size'])
                    and hashlib.md5(response.content).hexdigest() == item['md5Checksum']):
                partial.write_bytes(response.content)
                partial.replace(path)
                return
        except Exception as exc:
            if attempt == 4:
                raise RuntimeError(f'Download failed: {item["name"]}: {type(exc).__name__}') from None
        time.sleep(2 ** attempt)
    raise RuntimeError(f'Download integrity failure: {item["name"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path('dataset/interhuman_official/motions'))
    parser.add_argument('--manifest', type=Path, default=Path('output/interhuman_raw_download_20261003.json'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--reuse-manifest', action='store_true')
    args = parser.parse_args()
    assert valid_name('7778.pkl') and not valid_name('../7778.pkl') and not valid_name('abc.pkl')
    if args.reuse_manifest:
        previous = json.loads(args.manifest.read_text())
        assert previous['listing_complete'] and previous['source_folder'] == URL
        files = previous['files']
        assert len(files) == previous['count'] and all(valid_name(f['name']) for f in files)
    else:
        files = inventory()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    total = sum(int(f['size']) for f in files)
    result = {'source_folder': URL, 'listing_complete': True, 'count': len(files),
              'bytes': total, 'status': 'listed', 'files': files}
    args.manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)
    if args.list_only:
        return
    required = sum(int(f['size']) for f in files if not verified(args.destination / f['name'], f))
    if shutil.disk_usage(args.destination).free < required + 1024 ** 3:
        raise RuntimeError(f'Insufficient disk space: need {required} bytes plus 1 GiB reserve')
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(download, f, args.destination): f for f in files}
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                job.result()
            except Exception as exc:
                failures.append({'name': jobs[job]['name'], 'error': str(exc)})
                print(f'failed {jobs[job]["name"]}: {exc}', flush=True)
            if i % 250 == 0 or i == len(files):
                print(f'checked {i}/{len(files)}, failures={len(failures)}', flush=True)
    result.update(status='complete' if not failures else 'partial', failures=failures,
                  verified_count=sum(verified(args.destination / f['name'], f) for f in files))
    args.manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
