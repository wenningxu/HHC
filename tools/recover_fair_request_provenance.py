#!/usr/bin/env python
"""Recover immutable source bytes bound by the frozen Inter-X requests.

The request manifest predates the train-only augmentation loader extensions.
Those extensions do not affect the evaluator dataset class, but they changed
the live file hash.  This script reconstructs the exact previously recorded
file from the known later revision, verifies historical hashes, and stores
it, the request-time protocol/verifier, and the original request manifest in
content-addressed snapshots.
It never rewrites the canonical request manifest or any request NPZ.
"""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / 'tools' / 'provenance_snapshots'
SOURCE = ROOT / 'dataset' / 'dataset_interx_TM.py'
VERIFIER = ROOT / 'tools' / 'verify_fair_eval_requests.py'
MANIFEST = ROOT / 'output' / 'fair_interx_requests' / 'manifest.json'
AUDIT = ROOT / 'output' / 'fair_interx_requests' / 'provenance_recovery.json'
SESSION = Path(
    '/home/wenning/.codex/sessions/2026/08/23/'
    'rollout-2026-08-23T14-18-21-01a02ec5-abdc-7583-8567-f0cfd4e75a89.jsonl'
)
CREATION_COMMAND = (
    'OMP_NUM_THREADS=8 conda run --no-capture-output -n New_MMM python -u '
    'tools/prepare_fair_eval_requests.py '
    '--output-dir output/fair_interx_requests --device cpu '
    '--replications 20 --seed 1234'
)
CREATION_RUNTIME = {
    'contract': 'pytorch_official_evaluator_single_device_v1',
    'requested_device': 'cpu',
    'device_type': 'cpu',
    'device_index': None,
    'device_name': 'cpu',
}
CREATION_LAUNCH_LINE_SHA256 = (
    '4da1a2adf58b2b529963ebe8c47524a29c04c761ae31a3fb714e74a2d289e560'
)
CREATION_COMPLETION_LINE_SHA256 = (
    'df2a9895370b19d390b5740ec7d586c7593739a05c2a13928213d45b2e3078b1'
)
CREATION_RECORDS_SHA256 = (
    'e2ae556fb8774e2aefb9ddd2166bc53e27d1922ea35039c81c4ce1da10f450d1'
)
CREATION_RECORDS_BYTES = 16063

CURRENT_SOURCE_SHA256 = (
    'a7d199ba8fdb245343c48fe17822b6fe57637bdbd7ee4bd9aa353c1346c4c9d3'
)
FROZEN_SOURCE_SHA256 = (
    'c3993a2c1f3dce799a5673b46e47341a0571c90aae567e6e33f82ab384fc145d'
)
FROZEN_SOURCE_BYTES = 19551
FROZEN_MANIFEST_SHA256 = (
    'd4007bf0666c92d54ea05dc5117d0d4c7b5baccf386e05a4556654964ec54070'
)
FROZEN_MANIFEST_BYTES = 7836
CURRENT_VERIFIER_SHA256 = (
    'b3265b1e314bbab09a299ac7f9e0059b6c48442fd005bb5a13794b8ece71bae4'
)
FROZEN_VERIFIER_SHA256 = (
    '6b357b1db6f646808820977227b53d4b1dfa8d321103b4db25f23f9931dcb7b1'
)
FROZEN_VERIFIER_BYTES = 4803
FROZEN_PROTOCOL_SHA256 = (
    '1db8e9389cdc70c0a875383b1be64dba32ca9fdff92115e04916992a0c5ad11f'
)
FROZEN_PROTOCOL_BYTES = 3721
SOURCE_REVERSE_EXTENSIONS = (
    'token_dataset_signature',
    'token_dataset_asset_fields',
    'beta_fallback',
    'text_fallback',
    'strict_asset_failure',
    'train_loader_augmentation_assets',
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text, later, earlier, label):
    count = text.count(later)
    if count != 1:
        raise ValueError(f'{label}: expected one later-source block, got {count}')
    return text.replace(later, earlier, 1)


def reconstruct_frozen_source():
    payload = SOURCE.read_bytes()
    if sha256_bytes(payload) != CURRENT_SOURCE_SHA256:
        raise ValueError(
            'live dataset loader is no longer the audited later revision; '
            'refusing heuristic reconstruction'
        )
    text = payload.decode('utf-8')
    replacements = (
        (
            "                 max_motion_length=50, contact_dir='checkpoints/vq/contact_token_label',\n"
            "                 beta_dir='dataset/interx/betas', fallback_beta_dir=None,\n"
            "                 fallback_text_dir=None, strict_assets=False):",
            "                 max_motion_length=50, contact_dir='checkpoints/vq/contact_token_label'):",
            'token_dataset_signature',
        ),
        (
            "        self.beta_dir = beta_dir\n"
            "        self.fallback_beta_dir = fallback_beta_dir\n"
            "        self.fallback_text_dir = fallback_text_dir\n"
            "        self.strict_assets = strict_assets\n"
            "        self.contact_dir = contact_dir",
            "        self.beta_dir = 'dataset/interx/betas'\n"
            "        self.contact_dir = contact_dir",
            'token_dataset_asset_fields',
        ),
        (
            "                beta_path = pjoin(self.beta_dir, name + '.npz')\n"
            "                if not os.path.isfile(beta_path) and self.fallback_beta_dir:\n"
            "                    beta_path = pjoin(self.fallback_beta_dir, name + '.npz')\n"
            "                beta_data = np.load(beta_path, allow_pickle=True)",
            "                beta_data = np.load(pjoin(self.beta_dir, name + '.npz'), allow_pickle=True)",
            'beta_fallback',
        ),
        (
            "                text_path = pjoin(text_dir, name + '.txt')\n"
            "                if not os.path.isfile(text_path) and self.fallback_text_dir:\n"
            "                    text_path = pjoin(self.fallback_text_dir, name + '.txt')\n"
            "                with cs.open(text_path) as f:",
            "                with cs.open(pjoin(text_dir, name + '.txt')) as f:",
            'text_fallback',
        ),
        (
            "                if self.strict_assets:\n"
            "                    raise RuntimeError(\n"
            "                        f'failed to load MGT token asset for {name}: {e}'\n"
            "                    ) from e",
            "                # print(e)\n"
            "                pass",
            'strict_asset_failure',
        ),
        (
            "               shuffle=True,\n"
            "               contact_dir='checkpoints/vq/contact_token_label',\n"
            "               max_motion_length=50,\n"
            "               text_dir='./dataset/interx/texts_processed',\n"
            "               beta_dir='dataset/interx/betas',\n"
            "               fallback_beta_dir=None,\n"
            "               fallback_text_dir=None,\n"
            "               strict_assets=False):\n"
            "    trainSet = Text2MotionDatasetV2HHI_Token(\n"
            "        split_file, motion_file, unit_length, codebook_size,\n"
            "        contact_dir=contact_dir, max_motion_length=max_motion_length,\n"
            "        text_dir=text_dir, beta_dir=beta_dir,\n"
            "        fallback_beta_dir=fallback_beta_dir,\n"
            "        fallback_text_dir=fallback_text_dir,\n"
            "        strict_assets=strict_assets,\n"
            "    )",
            "               shuffle=True,\n"
            "               contact_dir='checkpoints/vq/contact_token_label',\n"
            "               max_motion_length=50):\n"
            "    trainSet = Text2MotionDatasetV2HHI_Token(\n"
            "        split_file, motion_file, unit_length, codebook_size,\n"
            "        contact_dir=contact_dir, max_motion_length=max_motion_length,\n"
            "    )",
            'train_loader_augmentation_assets',
        ),
    )
    for later, earlier, label in replacements:
        text = replace_once(text, later, earlier, label)
    frozen = text.encode('utf-8')
    if len(frozen) != FROZEN_SOURCE_BYTES:
        raise ValueError(
            f'reconstructed source is {len(frozen)} bytes, expected '
            f'{FROZEN_SOURCE_BYTES}'
        )
    if sha256_bytes(frozen) != FROZEN_SOURCE_SHA256:
        raise ValueError('reconstructed source does not match frozen SHA-256')
    labels = [label for _, _, label in replacements]
    if tuple(labels) != SOURCE_REVERSE_EXTENSIONS:
        raise ValueError('source reconstruction labels changed')
    return frozen, labels


def reconstruct_frozen_verifier():
    payload = VERIFIER.read_bytes()
    if sha256_bytes(payload) != CURRENT_VERIFIER_SHA256:
        raise ValueError(
            'live verifier is no longer the audited non-mutating revision; '
            'refusing heuristic reconstruction'
        )
    text = payload.decode('utf-8')
    text = replace_once(
        text, 'from pathlib import Path\n', '', 'verifier_path_import'
    )
    start = text.index('\ndef resolve_frozen_artifact(')
    end = text.index('\ndef main():', start)
    text = text[:start] + text[end:]
    argument_extension = """
    parser.add_argument(
        '--refresh-provenance', action='store_true',
        help=(
            'Explicitly replace the provenance manifest with current source '
            'records. By default an existing frozen manifest is checked '
            'without modification.'
        ),
    )
    parser.add_argument(
        '--snapshot-root', type=Path,
        default=Path('tools/provenance_snapshots'),
    )
"""
    text = replace_once(
        text, argument_extension, '\n', 'verifier_arguments'
    )
    non_mutating_output = """
    output = Path(args.output).resolve()
    if output.is_file() and not args.refresh_provenance:
        frozen = json.loads(output.read_text(encoding='utf-8'))
        report = verify_frozen_manifest(
            frozen, manifest, args.snapshot_root.resolve()
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(output) + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
"""
    original_output = """
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = os.path.abspath(args.output) + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, os.path.abspath(args.output))
    print(json.dumps(manifest, indent=2, sort_keys=True))
"""
    text = replace_once(
        text, non_mutating_output, original_output, 'verifier_output_mode'
    )
    frozen = text.encode('utf-8')
    if len(frozen) != FROZEN_VERIFIER_BYTES:
        raise ValueError(
            f'reconstructed verifier is {len(frozen)} bytes, expected '
            f'{FROZEN_VERIFIER_BYTES}'
        )
    if sha256_bytes(frozen) != FROZEN_VERIFIER_SHA256:
        raise ValueError(
            'reconstructed verifier does not match frozen SHA-256: '
            f'{sha256_bytes(frozen)}'
        )
    return frozen


def materialize_exact(path, payload, expected_sha256):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if sha256_file(path) != expected_sha256:
            raise ValueError(f'corrupt existing snapshot: {path}')
        return False
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f'new snapshot hash mismatch: {path}')
    return True


def artifact(path):
    return {
        'path': str(path.resolve()),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def command_from_event(item):
    command = item.get('command')
    if not isinstance(command, list) or len(command) != 3:
        return None
    if command[:2] != ['/bin/bash', '-lc']:
        return None
    return command[2]


def event_cwd_path(value):
    parsed = urlparse(value)
    if parsed.scheme != 'file':
        raise ValueError(f'creation event cwd is not a file URI: {value}')
    return Path(unquote(parsed.path)).resolve()


def validate_creation_records(payload):
    lines = payload.splitlines(keepends=True)
    if len(lines) != 2:
        raise ValueError(f'expected two selected creation records, got {len(lines)}')
    expected_hashes = (
        CREATION_LAUNCH_LINE_SHA256, CREATION_COMPLETION_LINE_SHA256,
    )
    if tuple(sha256_bytes(line) for line in lines) != expected_hashes:
        raise ValueError('selected creation record line hashes changed')
    launch = json.loads(lines[0])
    completion = json.loads(lines[1])
    launch_payload = launch.get('payload', {})
    completion_payload = completion.get('payload', {})
    item = completion_payload.get('item', {})
    if not (
        launch.get('type') == 'response_item'
        and launch_payload.get('type') == 'custom_tool_call'
        and launch_payload.get('name') == 'exec'
        and CREATION_COMMAND in launch_payload.get('input', '')
        and completion.get('type') == 'event_msg'
        and completion_payload.get('type') == 'item_completed'
        and item.get('type') == 'CommandExecution'
        and command_from_event(item) == CREATION_COMMAND
        and item.get('status') == 'completed'
        and item.get('exit_code') == 0
        and event_cwd_path(item.get('cwd', '')) == ROOT.resolve()
    ):
        raise ValueError('selected records do not prove a successful creation run')
    stdout = item.get('stdout', '')
    for replication in range(20):
        expected = (
            f"{{'replication': {replication}, 'samples': 1696, 'request': "
            f"'output/fair_interx_requests/replication_{replication:02d}.npz'}}"
        )
        if expected not in stdout:
            raise ValueError(
                f'creation completion omitted replication {replication}'
            )
    return {
        'source_session_path': str(SESSION),
        'launch_line_number': 2371,
        'completion_line_number': 2522,
        'launch_timestamp': launch.get('timestamp'),
        'completion_timestamp': completion.get('timestamp'),
        'launch_line_sha256': expected_hashes[0],
        'completion_line_sha256': expected_hashes[1],
        'command': CREATION_COMMAND,
        'cwd': str(ROOT.resolve()),
        'status': item.get('status'),
        'exit_code': item.get('exit_code'),
        'runtime': dict(CREATION_RUNTIME),
        'replications_reported_complete': 20,
        'samples_per_replication_reported': 1696,
    }


def recover_creation_records():
    snapshot = (
        SNAPSHOT_ROOT
        / f'{CREATION_RECORDS_SHA256}__fair_interx_request_creation_records.jsonl'
    )
    if snapshot.is_file():
        payload = snapshot.read_bytes()
        created = False
    else:
        if not SESSION.is_file():
            raise ValueError(
                'request creation session is unavailable and its immutable '
                'selected-record snapshot has not been materialized'
            )
        selected = {}
        with SESSION.open('rb') as handle:
            for line_number, line in enumerate(handle, 1):
                line_hash = sha256_bytes(line)
                if line_hash == CREATION_LAUNCH_LINE_SHA256:
                    selected['launch'] = (line_number, line)
                elif line_hash == CREATION_COMPLETION_LINE_SHA256:
                    selected['completion'] = (line_number, line)
        if set(selected) != {'launch', 'completion'}:
            raise ValueError('could not recover both frozen request-creation records')
        if (selected['launch'][0], selected['completion'][0]) != (2371, 2522):
            raise ValueError('request-creation source line numbers changed')
        payload = selected['launch'][1] + selected['completion'][1]
        if (len(payload) != CREATION_RECORDS_BYTES
                or sha256_bytes(payload) != CREATION_RECORDS_SHA256):
            raise ValueError('selected request-creation record payload changed')
        created = materialize_exact(
            snapshot, payload, CREATION_RECORDS_SHA256
        )
    if (len(payload) != CREATION_RECORDS_BYTES
            or sha256_bytes(payload) != CREATION_RECORDS_SHA256):
        raise ValueError(f'corrupt request-creation snapshot: {snapshot}')
    return snapshot, created, validate_creation_records(payload)


def verify_request_files():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    records = manifest.get('records', [])
    if not (
        manifest.get('status') == 'pass'
        and manifest.get('replications') == 20
        and manifest.get('samples_per_replication') == 1696
        and len(records) == 20
    ):
        raise ValueError('frozen request manifest contract changed')
    for replication, record in enumerate(records):
        path = Path(record.get('path', ''))
        if not path.is_absolute():
            path = ROOT / path
        if not (
            record.get('replication') == replication
            and record.get('seed') == 1234 + replication
            and path.is_file()
            and path.stat().st_size == record.get('bytes')
            and sha256_file(path) == record.get('sha256')
        ):
            raise ValueError(f'frozen request replication {replication} changed')
    return len(records)


def main():
    source_snapshot = (
        SNAPSHOT_ROOT / f'{FROZEN_SOURCE_SHA256}__dataset_interx_TM.py'
    )
    if source_snapshot.is_file():
        frozen_source = source_snapshot.read_bytes()
        replacement_labels = list(SOURCE_REVERSE_EXTENSIONS)
        if (len(frozen_source) != FROZEN_SOURCE_BYTES
                or sha256_bytes(frozen_source) != FROZEN_SOURCE_SHA256):
            raise ValueError(f'corrupt source snapshot: {source_snapshot}')
        source_created = False
    else:
        frozen_source, replacement_labels = reconstruct_frozen_source()
        source_created = materialize_exact(
            source_snapshot, frozen_source, FROZEN_SOURCE_SHA256
        )

    if (not MANIFEST.is_file()
            or MANIFEST.stat().st_size != FROZEN_MANIFEST_BYTES
            or sha256_file(MANIFEST) != FROZEN_MANIFEST_SHA256):
        raise ValueError('canonical request manifest is not the frozen original')
    manifest_snapshot = (
        SNAPSHOT_ROOT / f'{FROZEN_MANIFEST_SHA256}__fair_interx_requests_manifest.json'
    )
    manifest_created = materialize_exact(
        manifest_snapshot, MANIFEST.read_bytes(), FROZEN_MANIFEST_SHA256
    )

    verifier_snapshot = (
        SNAPSHOT_ROOT / f'{FROZEN_VERIFIER_SHA256}__verify_fair_eval_requests.py'
    )
    if verifier_snapshot.is_file():
        verifier_created = False
        if (verifier_snapshot.stat().st_size != FROZEN_VERIFIER_BYTES
                or sha256_file(verifier_snapshot) != FROZEN_VERIFIER_SHA256):
            raise ValueError(f'corrupt verifier snapshot: {verifier_snapshot}')
    else:
        verifier_created = materialize_exact(
            verifier_snapshot, reconstruct_frozen_verifier(),
            FROZEN_VERIFIER_SHA256,
        )

    protocol_snapshot = (
        SNAPSHOT_ROOT
        / f'{FROZEN_PROTOCOL_SHA256}__fair_interx_protocol.yaml'
    )
    if (not protocol_snapshot.is_file()
            or protocol_snapshot.stat().st_size != FROZEN_PROTOCOL_BYTES
            or sha256_file(protocol_snapshot) != FROZEN_PROTOCOL_SHA256):
        raise ValueError(
            'request-time protocol snapshot is missing or corrupt: '
            f'{protocol_snapshot}'
        )

    creation_snapshot, creation_created, creation = recover_creation_records()
    request_files_verified = verify_request_files()

    report = {
        'schema': 'fair_interx_request_provenance_recovery_v1',
        'status': 'pass',
        'canonical_manifest_unchanged': True,
        'request_npz_files_modified': 0,
        'request_npz_files_verified': request_files_verified,
        'live_source': artifact(SOURCE),
        'frozen_source_snapshot': artifact(source_snapshot),
        'frozen_manifest_snapshot': artifact(manifest_snapshot),
        'frozen_verifier_snapshot': artifact(verifier_snapshot),
        'frozen_protocol_snapshot': artifact(protocol_snapshot),
        'frozen_creation_records_snapshot': artifact(creation_snapshot),
        'request_creation_runtime_recovery': creation,
        'source_snapshot_created': source_created,
        'manifest_snapshot_created': manifest_created,
        'verifier_snapshot_created': verifier_created,
        'creation_records_snapshot_created': creation_created,
        'reverse_extensions': replacement_labels,
        'implementation': artifact(Path(__file__)),
    }
    temporary = AUDIT.with_suffix(AUDIT.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, AUDIT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
