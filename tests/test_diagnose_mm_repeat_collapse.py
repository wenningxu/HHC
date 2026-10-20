import tempfile
import unittest
from pathlib import Path
import h5py, numpy as np
from tools.diagnose_mm_repeat_collapse import inspect

class TestCollapse(unittest.TestCase):
    def test_three_repeats_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.h5'; x=np.zeros((1,3,4,2),np.float32); x[0,1,:2]=0; x[0,2,:2]=2; x[0,:,2:]=np.arange(12).reshape(3,2,2)
            with h5py.File(p,'w') as f: f['motions']=x; f['lengths']=[2]; f['completed_repeats']=[True] * 3
            self.assertAlmostEqual(inspect(p,3,1,3)['exact_pair_rate'],1/3)
            with h5py.File(p,'a') as f: f['motions'][0, :, 2:] = np.nan
            self.assertAlmostEqual(inspect(p,3,1,3)['exact_pair_rate'],1/3)
            with self.assertRaises(ValueError): inspect(p,4,1,3)
            with self.assertRaises(ValueError): inspect(p,3,1,30)
            with h5py.File(p,'a') as f: f['completed_repeats'][:]=False
            with self.assertRaises(ValueError): inspect(p,3,1,3)

if __name__=='__main__': unittest.main()
