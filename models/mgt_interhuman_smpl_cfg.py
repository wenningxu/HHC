"""Single-factor text CFG ablation for native-SMPL HHC.

The historical checkpoint format and network topology are unchanged. Both
arms retain independent 20% shape dropout; only the text null condition changes.
"""


def configure_conditioning(model, arm, *, sampling=False):
    if arm not in {'historical', 'matched'}:
        raise ValueError(f'Unknown conditioning arm: {arm}')
    # Historical training drops only global text; its released sampler drops
    # both text paths. Matched training and sampling use the same null text.
    contract = 'paper_text_cfg' if sampling or arm == 'matched' else 'historical_mmm_shape_dropout'
    model.conditioning_contract = model.trans_base.conditioning_contract = contract
    model.trans_base.text_drop_prob = .2
    model.trans_base.shape_mask_prob = .2
    return model

