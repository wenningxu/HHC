#!/usr/bin/env python
"""Deterministic lexical contact-region baseline for the robustness study."""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimization.text_2_regions import ContactExtractor
from utils.paramUtil import CANDIDATES


SCHEMA = 'hhc_interx_contact_pairs_rule_v1'
PART_PATTERNS = (
    (r'left (?:palm|hand)', 'left hand (palm)'),
    (r'right (?:palm|hand)', 'right hand (palm)'),
    (r'left shoulder', 'left shoulder'),
    (r'right shoulder', 'right shoulder'),
    (r'left (?:thigh|calf|knee|leg)', 'left leg'),
    (r'right (?:thigh|calf|knee|leg)', 'right leg'),
    (r'left foot', 'left foot'),
    (r'right foot', 'right foot'),
    (r'left (?:forearm|elbow|arm)', 'left arm'),
    (r'right (?:forearm|elbow|arm)', 'right arm'),
    (r'cheek|face|head', 'head'),
    (r'neck', 'neck'),
    (r'chest', 'chest'),
    (r'stomach|belly', 'stomach'),
    (r'waist|hip', 'waist'),
    (r'back', 'back'),
    (r'butt', 'butt'),
    (r'shoulder', 'right shoulder'),
    (r'thigh|calf|knee|leg', 'right leg'),
    (r'foot', 'right foot'),
    (r'forearm|elbow|arm', 'right arm'),
    (r'palm|hand|finger', 'right hand (palm)'),
)
CONTACT_CUES = re.compile(
    r'hug|embrac|hold|grab|grip|touch|tap|pat|push|pull|punch|hit|strike|'
    r'kick|shake|high[- ]?five|carry|lift|kiss|slap|collid|clap|support|'
    r'wrap|sit(?:s|ting)? (?:on|upon)',
)
NEGATED_CONTACT_CUES = re.compile(
    r'without (?:ever )?touch|does not touch|do not touch|never touch|'
    r'but misses|and misses|without making contact|avoid(?:s|ing)? contact',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def atomic_pickle(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def mentioned_parts(caption):
    matches = []
    text = caption.lower()
    for pattern_index, (pattern, part) in enumerate(PART_PATTERNS):
        for match in re.finditer(pattern, text):
            matches.append((match.start(), pattern_index, part))
    matches.sort()
    selected = []
    occupied_positions = set()
    for position, _, part in matches:
        if position in occupied_positions:
            continue
        occupied_positions.add(position)
        if not selected or selected[-1] != part:
            selected.append(part)
    return selected


def make_row(p1_part, p2_part, contact_type='flat', role='5:5'):
    return {
        'person 1 body part': p1_part,
        'person 2 body part': p2_part,
        'contact area': contact_type,
        'role ratio (p1:p2)': role,
    }


def rule_rows(caption):
    text = caption.lower()
    if NEGATED_CONTACT_CUES.search(text):
        return [], 'explicitly_negated_contact'
    if not CONTACT_CUES.search(text):
        return [], 'no_lexical_contact_cue'
    parts = mentioned_parts(caption)
    if 'hug' in text or 'embrac' in text or 'wrap' in text:
        return [
            make_row('left arm', 'back', 'hug'),
            make_row('right arm', 'back', 'hug'),
        ], 'hug_template'
    if 'handshake' in text or 'shake' in text or 'high-five' in text \
            or 'high five' in text or 'clap' in text:
        return [make_row(
            'right hand (palm)', 'right hand (palm)', 'flat'
        )], 'hand_to_hand_template'
    if 'kick' in text:
        target = next((part for part in reversed(parts) if part != 'right foot'),
                      'stomach')
        return [make_row('right foot', target, 'flat', '9:1')], 'kick_template'
    if 'collid' in text or 'sit on' in text or 'sits on' in text \
            or 'sitting on' in text:
        return [make_row('chest', 'chest', 'flat')], 'body_contact_template'

    contact_type = 'wrap' if any(
        cue in text for cue in ('hold', 'grab', 'grip', 'carry', 'lift', 'support')
    ) else 'flat'
    if len(parts) >= 2:
        return [make_row(parts[0], parts[1], contact_type, '9:1')], (
            'ordered_explicit_parts'
        )
    if len(parts) == 1:
        target = parts[0]
        source = 'right hand (palm)' if target != 'right hand (palm)' \
            else 'left hand (palm)'
        return [make_row(source, target, contact_type, '9:1')], (
            'single_explicit_part_with_hand_default'
        )
    return [make_row(
        'right hand (palm)', 'chest', contact_type, '9:1'
    )], 'contact_cue_with_hand_chest_default'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--requests', type=Path,
        default=Path('output/interx_llm_robustness/requests.npz'),
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_rule_robustness_test'),
    )
    args = parser.parse_args()
    requests = args.requests.resolve()
    with np.load(requests, allow_pickle=False) as request:
        names = request['names'].astype(str).tolist()
        captions = request['captions'].astype(str).tolist()
        strata = request['strata'].astype(str).tolist()
    if len(names) != len(captions) or len(names) != len(set(names)):
        raise ValueError('request names/captions are not uniquely aligned')

    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / '_metadata'
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mapper = ContactExtractor.__new__(ContactExtractor)
    records = []
    nonempty = 0
    inference_wall_seconds = []
    for index, (name, caption, stratum) in enumerate(
        zip(names, captions, strata)
    ):
        started = time.perf_counter()
        rows, rule = rule_rows(caption)
        contacts = mapper.map_contacts_to_ids(rows, CANDIDATES)
        sample_wall_seconds = time.perf_counter() - started
        inference_wall_seconds.append(sample_wall_seconds)
        pair_path = output_dir / f'{name}.pkl'
        metadata_path = metadata_dir / f'{name}.json'
        atomic_pickle(pair_path, contacts)
        metadata = {
            'schema': SCHEMA,
            'sample_name': name,
            'request_index': index,
            'caption': caption,
            'caption_sha256': sha256_text(caption),
            'stratum': stratum,
            'rule': rule,
            'raw_rule_rows': rows,
            'structured_contact_rows': len(contacts),
            'inference_wall_seconds': sample_wall_seconds,
            'pair_sha256': sha256_file(pair_path),
            'requests': str(requests),
            'requests_sha256': sha256_file(requests),
            'status': 'success',
        }
        atomic_json(metadata_path, metadata)
        records.append(metadata)
        nonempty += int(bool(contacts))

    manifest = {
        'schema': SCHEMA,
        'status': 'complete',
        'provider': 'deterministic_lexical_rule_baseline',
        'sample_count': len(records),
        'completed_count': len(records),
        'nonempty_samples': nonempty,
        'requests': str(requests),
        'requests_sha256': sha256_file(requests),
        'selected_names_sha256': canonical_sha256(names),
        'selected_captions_sha256': canonical_sha256(captions),
        'inference_wall_timing': {
            'scope': 'rule_rows_and_contact_region_mapping',
            'total_seconds': float(sum(inference_wall_seconds)),
            'mean_seconds_per_sample': float(np.mean(inference_wall_seconds)),
            'median_seconds_per_sample': float(np.median(inference_wall_seconds)),
            'p95_seconds_per_sample': float(np.percentile(
                inference_wall_seconds, 95,
            )),
        },
        'implementation': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(__file__),
        },
        'records_sha256': canonical_sha256(records),
    }
    atomic_json(output_dir / '_extraction_manifest.json', manifest)
    print(json.dumps({
        'output_dir': str(output_dir), 'sample_count': len(records),
        'nonempty_samples': nonempty,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
