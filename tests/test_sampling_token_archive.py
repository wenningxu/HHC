import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from tools.export_hhc_official_interx import token_archive, validate_resume_attributes


class TokenArchiveTest(unittest.TestCase):
    def test_archive_roundtrip_and_resume_contract(self):
        validate_resume_attributes(SimpleNamespace(attrs={}), {})
        for old, new in [({}, {'save_token_ids': True}), ({'save_token_ids': True}, {})]:
            with self.assertRaises(ValueError):
                validate_resume_attributes(SimpleNamespace(attrs=old), new)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.h5'
            shape = (2, 1, 5, 3)
            values = np.arange(30, dtype=np.int64).reshape(shape)
            with h5py.File(path, 'w') as target:
                with self.assertRaises(ValueError): token_archive(target, shape, True)
                token_archive(target, shape, False)[:] = values
            with h5py.File(path, 'r') as target:
                np.testing.assert_array_equal(token_archive(target, shape, True)[:], values)
                with self.assertRaises(ValueError): token_archive(target, (2, 2, 5, 3), True)


if __name__ == '__main__':
    unittest.main()
