import json
from pathlib import Path


def test_interhuman_vq_pilot_is_val_selected_online_and_not_launched_early():
    contract = json.loads(Path('configs/interhuman_vq_pilot_v1.json').read_text())
    assert contract['status'] == 'preregistered_not_launched'
    assert contract['execution_gate']['launch_now'] is False
    assert contract['data']['test_use_during_training'] is False
    assert contract['data']['representation'].endswith('524D')
    assert contract['model']['feature_partition'] == [512, 12]
    assert contract['model']['latent_partition'] == [384, 128]
    assert contract['model']['codebooks'] == 2
    assert contract['model']['parameters'] == 22337548
    assert contract['model']['codes_per_codebook'] == 1024
    assert contract['optimization']['pilot_optimizer_steps'] == 10000
    assert contract['automatic_validation']['samples'] == 480
    assert contract['automatic_validation']['selection_metric'].startswith('lowest official-val')
    assert contract['wandb']['mode'] == 'online'
    required = contract['wandb']['required_train_metrics']
    assert any('gradient norm' in item for item in required)
    assert any('weight norm' in item for item in required)
