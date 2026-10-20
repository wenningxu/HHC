"""A download ban must preserve cloud monitoring and suppress network transfers."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import local_compute_control as control
import local_runpod_control as runpod


class CheckpointPolicyTests(unittest.TestCase):
    def test_download_and_old_backup_do_not_touch_network_or_destination(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'local-checkpoint-policy.json').write_text(json.dumps({'checkpoint_downloads_enabled':False}))
            with patch.object(control,'CONTROL',root), patch.object(control,'event'), patch.object(control.subprocess,'run',side_effect=AssertionError('network forbidden')):
                with self.assertRaisesRegex(RuntimeError,'disabled by user policy'):
                    control.download('remote/recovery.pt',root/'new/recovery.pt','unused',100)
                control.initial_backup()
            self.assertFalse((root/'new').exists())

    def test_existing_pending_transfer_cannot_bypass_ban_but_status_is_read(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'old-compute-fenced.json').write_text('{}')
            (root/'runpod-transfer.json').write_text('{"step":99999}')
            current={'processes':'training','recovery':{'step':99999},'status':{'status':'training'}}
            with patch.object(runpod,'CONTROL',root), patch.object(runpod,'remote',return_value=current) as read, patch.object(runpod,'atomic_json') as save, patch.object(runpod,'checkpoint_downloads_allowed',return_value=False), patch.object(runpod.subprocess,'run',side_effect=AssertionError('download forbidden')):
                runpod.cycle()
            self.assertEqual(read.call_count,1)
            self.assertEqual(save.call_count,1)
            self.assertEqual(save.call_args.args[0].name,'runpod-status.json')
            self.assertEqual(json.loads((root/'runpod-transfer.json').read_text()),{'step':99999})


if __name__=='__main__':
    unittest.main()
