from copy import deepcopy

from tools.assess_fast_selective_shape_pilot import speed_work_contract


def completion():
    return {
        'requested_samples': 25,
        'completed_samples': 25,
        'completed_names': [f'sample_{index}' for index in range(25)],
        'plan_records_sha256': 'plan',
        'contact_manifest_sha256': 'contact',
        'source_motion_sha256': 'motion',
        'optimizer': {
            'iterations': 120,
            'learning_rate': 0.01,
            'fine_stage_start': 100,
            'seed': 1234,
            'seed_policy': 'base_seed_plus_100x_full_selection_position_before_slicing',
            'foot_contact_mask_policy': 'released_dynamic',
            'contact_loss_reduction_policy': 'original_mmm',
            'optimization_variable_policy': 'released_all_per_frame',
            'temporal_mask': 'exact cropped manual binary timing',
            'mode': 'optimized',
            'signed_distance_backend': 'nearest_vertex',
            'query_chunk': 512,
        },
    }


def test_speed_contract_ignores_only_backend_execution_parameters():
    nearest = completion()
    volumetric = deepcopy(nearest)
    volumetric['optimizer'].update({
        'signed_distance_backend': 'volumetric_smpl',
        'query_chunk': 2048,
        'volumetric_smpl_frame_batch_size': 4,
        'volumetric_smpl_penetration_vertex_count': 256,
        'volumetric_smpl_penetration_frame_count': 16,
        'volumetric_smpl_body_encoding_samples': 64,
    })
    assert speed_work_contract(nearest) == speed_work_contract(volumetric)


def test_speed_contract_rejects_different_iteration_budget():
    nearest = completion()
    volumetric = deepcopy(nearest)
    volumetric['optimizer']['iterations'] = 200
    assert speed_work_contract(nearest) != speed_work_contract(volumetric)


def test_speed_contract_rejects_different_loss_or_sample_contract():
    baseline = completion()
    changed_loss = deepcopy(baseline)
    changed_loss['optimizer']['contact_loss_reduction_policy'] = 'different'
    assert speed_work_contract(baseline) != speed_work_contract(changed_loss)

    changed_samples = deepcopy(baseline)
    changed_samples['completed_names'] = list(reversed(changed_samples['completed_names']))
    assert speed_work_contract(baseline) != speed_work_contract(changed_samples)
