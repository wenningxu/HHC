"""Train-only paired shape draws, independent of model RNG and evaluation GT."""
import hashlib
import json
from pathlib import Path
import numpy as np

POLICIES=('train_mean','empirical_train_pair')
SEED_RECIPE='sha256(hhc-trainshape-v1:sample_seed:replication), first8 little-endian, NumPy PCG64 row-uniform'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


class TrainShapePrior:
    def __init__(self,spec_path,expected_data_sha=None,expected_cache_source_sha=None):
        self.spec_path=Path(spec_path)
        self.spec=json.loads(self.spec_path.read_text())
        spec=self.spec
        if spec['schema']!='hhc_train_only_paired_shape_prior_v1' or spec['status']!='complete':
            raise ValueError('Incomplete or unsupported train shape prior')
        if spec['validation_test_beta_arrays_read'] or spec['motion_arrays_read']:
            raise ValueError('Prior must use training betas only')
        if expected_data_sha is not None and spec['data_manifest_sha256']!=expected_data_sha:
            raise ValueError('Shape prior data contract mismatch')
        if expected_cache_source_sha is not None and spec['mgt_cache_source_sha256']!=expected_cache_source_sha:
            raise ValueError('Shape prior MGT support mismatch')
        path=Path(spec['array'])
        if path.stat().st_size!=spec['array_bytes'] or file_sha(path)!=spec['array_sha256']:
            raise ValueError('Shape prior array integrity failure')
        with np.load(path,allow_pickle=False) as loaded:
            self.betas=loaded['betas'].copy();self.identifiers=loaded['identifiers'].astype(str)
        if self.betas.dtype!=np.float32 or list(self.betas.shape)!=spec['shape'] or self.betas.shape[1:]!=(2,10):
            raise ValueError('Shape prior array schema mismatch')
        if len(self.betas)!=spec['rows'] or len(self.identifiers)!=len(self.betas) or not np.isfinite(self.betas).all():
            raise ValueError('Shape prior rows invalid')
        names_sha=hashlib.sha256('\n'.join(self.identifiers).encode()).hexdigest()
        if names_sha!=spec['eligible_train_names_sha256']:
            raise ValueError('Train shape membership digest mismatch')
        mean=self.betas.astype(np.float64).mean(0)
        if not np.allclose(mean,spec['beta_mean_pair'],rtol=0,atol=1e-12):
            raise ValueError('Shape prior mean inconsistent with rows')
        self.mean=mean.astype(np.float32)
        self.betas.setflags(write=False);self.mean.setflags(write=False)

    def draw(self,policy,sample_seed,replication=0):
        if policy not in POLICIES:raise ValueError('Unknown train shape policy')
        if int(sample_seed)!=sample_seed or int(replication)!=replication or min(sample_seed,replication)<0:
            raise ValueError('Shape seeds must be nonnegative integers')
        if policy=='train_mean':return self.mean.copy(),None
        payload=f'hhc-trainshape-v1:{int(sample_seed)}:{int(replication)}'.encode()
        seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')
        index=int(np.random.Generator(np.random.PCG64(seed)).integers(len(self.betas)))
        return self.betas[index].copy(),index

    def request_betas(self,requests,policy,replication=0):
        # Only identifiers for output lookup and the pre-existing random seed;
        # caption, GT beta, root and motion fields are never accessed.
        result={};indices={}
        for request in requests:
            name=request['identifier']
            if name in result:raise ValueError('Duplicate request identifier requires per-request lookup')
            beta,index=self.draw(policy,request['sample_seed'],replication)
            result[name]=beta;indices[name]=index
        return result,indices
