"""InterHuman-only EMA precision/reset fix; historical Inter-X stays unchanged."""
import torch

from models.quantize_cnn import QuantizeEMAReset


class InterHumanQuantizeEMAReset(QuantizeEMAReset):
    """Keep EMA and assignment in FP32, with a finite 21-update usage window."""

    @torch.no_grad()
    def init_codebook(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            super().init_codebook(x.float())

    def quantize(self, x):
        # Also covers encode(), which bypasses forward().
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.codebook.dtype != torch.float32:
                raise ValueError('InterHuman codebook must remain FP32; use autocast, not model.bfloat16()')
            return super().quantize(x.float())

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.code_sum.dtype != torch.float32 or self.code_count.dtype != torch.float32:
                raise ValueError('mixed-precision legacy EMA state requires a separate explicit migration')
            reset_due = self.reset_count >= 20
            perplexity = super().update_codebook(x.float(), code_idx)
            if reset_due:
                # Parent's reset branch reads, but does not overwrite, usage.
                dead = (self.code_count < 1) & (self.usage[:, 0] < 1)
                # A replacement must start with matching EMA statistics, or
                # the next non-reset update restores its stale old centroid.
                self.code_sum[dead] = self.codebook[dead]
                self.code_count[dead] = 1
                self.usage.zero_()
            return perplexity

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())
