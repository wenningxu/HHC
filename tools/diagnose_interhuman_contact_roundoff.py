"""Explain the sparse contact-bit differences without changing official thresholds."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reconstruct_interhuman_raw_body import load_model, reconstruct_person
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter


def main():
    full = json.loads((ROOT/'output/interhuman_raw_body_reconstruction_full_20261003.json').read_text())
    cases = [r['id'] for r in full['records'] if r.get('online262', {}).get('contact_disagreements')]
    adapter, model = OfficialInterHumanAdapter(), load_model()
    rows = []
    for name in cases:
        raw = pickle.loads((ROOT/f'dataset/interhuman_official/motions/{name}.pkl').read_bytes())
        for who in ('person1', 'person2'):
            a, _ = reconstruct_person(raw[who], raw['mocap_framerate'], model)
            y = np.load(ROOT/f'dataset/interhuman_official/motions_processed/{who}/{name}.npy')
            b = np.concatenate([y[:, :66], y[:, 186:312]], axis=1)
            # Before the adapter's final float32 cast: same arrays as foot_detect.
            a = adapter.interhuman.process_motion_np(a, 0.001, 0, 22)[0]
            b = adapter.interhuman.process_motion_np(b, 0.001, 0, 22)[0]
            for frame, channel in np.argwhere(a[:, 258:] != b[:, 258:]):
                joint = (7, 10, 8, 11)[channel]
                threshold = (0.12, 0.05, 0.12, 0.05)[channel]
                signals = {}
                for label, data in [('reconstructed', a), ('released', b)]:
                    velocity = data[frame, 66+3*joint:66+3*(joint+1)]
                    v2 = float(velocity @ velocity)
                    height = float(data[frame, 3*joint+1])
                    bit = int(data[frame, 258+channel])
                    assert bit == int(v2 < 0.001 and height < threshold)
                    signals[label] = {'bit': bit, 'speed_squared': v2, 'height_m': height,
                                      'speed_margin': v2-0.001, 'height_margin_m': height-threshold}
                rows.append({'id': name, 'person': who, 'frame': int(frame), 'foot_channel': int(channel),
                             'joint': joint, 'height_threshold_m': threshold, **signals})
    assert len(rows) == full['summary']['contact_disagreements'] == 9
    destination = ROOT/'output/interhuman_raw_contact_roundoff_20261003.json'
    destination.write_text(json.dumps({'count': len(rows), 'records': rows}, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
