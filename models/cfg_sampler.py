import torch.nn as nn


class ClassifierFreeSampleModel(nn.Module):
    """Classifier-free guidance wrapper used by legacy diffusion evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0, (
            "Cannot run guided sampling on a model that was not trained with condition masking"
        )

    def forward(self, x, timesteps, y=None):
        y["uncond"] = False
        out = self.model(x, timesteps, y)
        y["uncond"] = True
        out_uncond = self.model(x, timesteps, y)
        return out_uncond + (y["scale"].view(-1, 1, 1) * (out - out_uncond))
