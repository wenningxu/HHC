import json
from pathlib import Path


CONTRACT = Path('configs/interhuman_mgt_pilot_v1.json')


def test_interhuman_mgt_pilot_reuses_selected_high_performance_hhc_contract():
    value = json.loads(CONTRACT.read_text())
    model = value['model']
    training = value['training']
    assert value['status'] == 'preregistered_not_launched'
    assert value['data']['token_shape'] == [75, 2]
    assert model['storage_channels'] == ['body', 'root']
    assert model['sampling_cascade'] == ['root', 'body']
    assert model['embed_dim'] == 1024 and model['attention_heads'] == 16
    assert (model['global_base_layers'], model['local_cross_attention_layers']) == (7, 2)
    assert model['output_head_layers'] == 9 and model['paper_architecture'] is False
    assert training['mask_contract'] == 'historical_pipeline_paper_kinematic_ratio'
    assert training['loss_contract'] == 'historical_all_valid'
    assert training['effective_batch_pairs'] == (
        training['micro_batch_pairs'] * training['gradient_accumulation']
    )


def test_interhuman_mgt_pilot_is_online_val_selected_and_common50():
    value = json.loads(CONTRACT.read_text())
    validation = value['automatic_validation']
    assert value['launch_gate']['requires_interx_stages_complete'] is True
    assert value['launch_gate']['requires_selected_hhc_interhuman_vq'] is True
    assert value['observability']['wandb_mode'] == 'online'
    assert validation['outer_maskgit_steps'] == 50
    assert validation['temperature'] == 0.5 and validation['cfg_scale'] == 3.5
    assert validation['checkpoint_selection'] == 'lowest frozen official-val FID only'
    assert value['data']['test_selection_forbidden'] is True
    required = set(value['observability']['required_process_metrics'])
    assert {'gradient_l2_norm', 'gradient_l2_norm_post_clip', 'weight_l2_norm'} <= required
