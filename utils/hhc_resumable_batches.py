"""CPU-only resumable random-permutation batch iterator."""
import torch


class ResumableWindowBatches:
    SCHEMA = 'hhc_resumable_window_batches_v1'

    def __init__(self, dataset_size, batch_size=256, seed=123):
        if not isinstance(dataset_size, int) or dataset_size < 0:
            raise ValueError('dataset_size must be a nonnegative int')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive int')
        if not isinstance(seed, int):
            raise ValueError('seed must be an int')
        self.size, self.batch, self.seed = dataset_size, batch_size, seed
        self.epoch, self.cursor, self.permutation = 0, 0, None
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(seed)

    def _new_epoch(self):
        self.permutation = torch.randperm(self.size, generator=self.generator, device='cpu')
        self.cursor = 0

    def draw(self):
        usable = (self.size // self.batch) * self.batch
        if usable < self.batch:
            raise StopIteration('dataset has no complete batch')
        if self.permutation is None or self.cursor + self.batch > usable:
            if self.permutation is not None:
                self.epoch += 1
            self._new_epoch()
        out = self.permutation[self.cursor:self.cursor + self.batch].tolist()
        self.cursor += self.batch
        return out

    def state_dict(self):
        return {'schema': self.SCHEMA, 'size': self.size, 'batch': self.batch,
                'seed': self.seed, 'epoch': self.epoch, 'cursor': self.cursor,
                'permutation': None if self.permutation is None else self.permutation.clone(),
                'generator_state': self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('schema') != self.SCHEMA:
            raise ValueError('invalid schema')
        for key, value in (('size', self.size), ('batch', self.batch), ('seed', self.seed)):
            if state.get(key) != value:
                raise ValueError(f'incompatible {key}')
        epoch, cursor, perm, gen = state.get('epoch'), state.get('cursor'), state.get('permutation'), state.get('generator_state')
        usable = (self.size // self.batch) * self.batch
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(cursor, int) or cursor < 0 or cursor > usable or cursor % self.batch:
            raise ValueError('invalid epoch/cursor')
        if perm is None:
            if cursor != 0 or epoch != 0: raise ValueError('missing permutation')
        else:
            if not isinstance(perm, torch.Tensor) or perm.device.type != 'cpu' or perm.dtype != torch.int64 or tuple(perm.shape) != (self.size,):
                raise ValueError('invalid permutation shape')
            if torch.unique(perm).numel() != self.size or (perm.numel() and (int(perm.min()) < 0 or int(perm.max()) >= self.size)):
                raise ValueError('invalid permutation values')
        if not isinstance(gen, torch.Tensor) or gen.device.type != 'cpu' or gen.dtype != torch.uint8:
            raise ValueError('invalid generator state')
        g = torch.Generator(device='cpu'); g.set_state(gen.clone())
        self.epoch, self.cursor = epoch, cursor
        self.permutation = None if perm is None else perm.clone()
        self.generator = g
