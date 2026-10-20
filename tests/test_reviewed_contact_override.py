"""Run with python -m tests.test_reviewed_contact_override from the repo root."""
import copy
import json
from pathlib import Path
from tools.check_shape_ground_after_optimization import reviewed_contacts


def test_reviewed_contact_override():
    rows = json.loads(Path('docs/contact_pair_corrections_proposed_20260912.json').read_text())['entries']
    entry = next(r for r in rows if r['source_name'] == 'G026T001A009R022')
    args = (entry['old_contacts'], entry['pickle_sha256'], entry['caption'])
    assert reviewed_contacts(entry, *args) == entry['new_contacts']
    for key, value in [('do_not_apply', True), ('review_status', 'proposed'),
                       ('pickle_sha256', 'wrong'), ('caption', 'wrong'), ('new_contacts', [])]:
        bad = copy.deepcopy(entry); bad[key] = value
        try:
            reviewed_contacts(bad, *args)
        except AssertionError:
            continue
        raise AssertionError(f'Accepted invalid override: {key}')
    blocked = next(r for r in rows if r['source_name'] == 'G013T007A030R001')
    try:
        reviewed_contacts(blocked, blocked['old_contacts'], blocked['pickle_sha256'], blocked['caption'])
    except AssertionError:
        return
    raise AssertionError('Accepted incomplete kick correction')


if __name__ == '__main__':
    test_reviewed_contact_override()
    print('reviewed contact override checks passed')
