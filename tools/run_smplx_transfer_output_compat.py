"""Run installed transfer_model with a TensorOutput namedtuple -> mapping adapter.

Use PYTHONPATH to select the existing licensed SMPL-X transfer checkout.
Bind the configured beta count explicitly (this checkout ignores nested config).
Fitting losses and optimizer are untouched; the output adapter preserves tensors.
"""
import runpy

import smplx


def output_mapping(module, inputs, output):
    if isinstance(output, tuple) and hasattr(output, '_asdict'):
        result = output._asdict()
        assert result['vertices'] is output.vertices
        return result
    return output


if __name__ == '__main__':
    build_layer = smplx.build_layer

    def compatible_layer(*args, **kwargs):
        shape_count = int(kwargs['smplx']['betas']['num'])
        kwargs['num_betas'] = shape_count
        model = build_layer(*args, **kwargs)
        assert model.num_betas == shape_count
        model.register_forward_hook(output_mapping)
        return model

    smplx.build_layer = compatible_layer
    runpy.run_module('transfer_model', run_name='__main__')
