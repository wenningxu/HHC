from types import SimpleNamespace

import torch

from models.vqvae_sep3_interhuman import VQVAE_SEP2_InterHuman
from tools.diagnose_interhuman_vq_quantization import BypassProbe


def test_probe_quantized_and_continuous_identity():
    model = VQVAE_SEP2_InterHuman(SimpleNamespace(mu=.99, interhuman_latent_dims='24,8'), nb_code=16, code_dim=32, down_t=2, width=32, depth=1).eval()
    motion = torch.randn(2, 64, 524)
    with torch.no_grad():
        for quantizer in model.quantizers:
            quantizer.codebook.normal_()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        probe = BypassProbe(model, ())
        assert torch.equal(model(motion)[0], probe(motion)[0])
        assert all(int(count.sum()) == 32 for count in probe.counts)
        expected = model._postprocess(model.decoder(torch.cat(model._encode_latents(motion), dim=1)))
        assert torch.equal(BypassProbe(model, (0, 1))(motion)[0], expected)
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
