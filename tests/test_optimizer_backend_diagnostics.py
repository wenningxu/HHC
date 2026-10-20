from types import SimpleNamespace

from optimize.optimize_full_body import _volumetric_smpl_diagnostics


def test_nearest_vertex_diagnostics_do_not_require_volumetric_backend():
    checkpoints, representation = _volumetric_smpl_diagnostics(
        SimpleNamespace(volumetric_smpl=None)
    )

    assert checkpoints == {}
    assert representation == {
        'status': 'not_used',
        'reason': 'nearest_vertex signed-distance backend',
    }


def test_volumetric_backend_diagnostics_preserve_provenance():
    backend = SimpleNamespace(
        checkpoint_records={'male': {'sha256': 'abc'}},
        representation_contract={'pose_layout': '55-joint axis-angle'},
    )

    checkpoints, representation = _volumetric_smpl_diagnostics(
        SimpleNamespace(volumetric_smpl=backend)
    )

    assert checkpoints == {'male': {'sha256': 'abc'}}
    assert representation == {'pose_layout': '55-joint axis-angle'}
