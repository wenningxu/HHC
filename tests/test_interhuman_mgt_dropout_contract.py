import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.vqvae_sep3_interhuman import VQVAE_SEP3_InterHuman


def test_dropout_candidate_changes_only_dropout_and_binds_its_contract():
    baseline = json.loads(Path('configs/interhuman_mgt_pilot_v1.json').read_text())
    candidate = json.loads(Path('configs/interhuman_mgt_dropout03_v2.json').read_text())
    assert candidate.pop('experiment')['changed_training_or_model_fields'] == {'model.dropout': [0.1, 0.3]}
    assert candidate['model']['dropout'] == 0.3
    candidate['model']['dropout'] = 0.1
    assert candidate == baseline
    source = Path('train/train_mgt_interhuman.py').read_text()
    assert 'contract_path = Path(args.contract).resolve()' in source
    assert "'contract_sha256': sha256_file(contract_path)" in source
    launcher = Path('tools/run_hhc_interhuman_mgt.sh').read_text()
    assert '--contract "$contract"' in launcher
    assert '${name}_common50' in launcher


def test_dropout_changes_probability_not_initial_weights_or_model_shape():
    models = []
    for probability in (0.1, 0.3):
        torch.manual_seed(123)
        vq = VQVAE_SEP3_InterHuman(
            SimpleNamespace(mu=0.99, interhuman_latent_dims='24,8'),
            nb_code=16, code_dim=32, output_emb_width=32, down_t=2,
            stride_t=2, width=32, depth=1, dilation_growth_rate=2)
        model = InterHumanText2MotionTransformer(
            vq, num_vq=16, embed_dim=32, clip_dim=32, block_size=75,
            num_layers=2, num_local_layer=1, n_head=4, drop_out_rate=probability,
            fc_rate=2, paper_architecture=False,
            conditioning_contract='historical_mmm_shape_dropout')
        drops = [layer.p for name, layer in model.named_modules()
                 if isinstance(layer, torch.nn.Dropout) and '.pos_embed.' not in name]
        assert drops and set(drops) == {probability}
        models.append(model)
    assert all(torch.equal(value, models[1].state_dict()[key])
               for key, value in models[0].state_dict().items())
