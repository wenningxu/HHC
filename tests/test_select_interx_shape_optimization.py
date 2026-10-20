from tools.select_interx_shape_optimization import accept_candidate


def test_accept_candidate_requires_all_three_gates():
    baseline = {"mismatch": 0.2, "pene": 0.3, "skate": 0.4}
    accepted, delta = accept_candidate(
        baseline,
        {"mismatch": 0.1, "pene": 0.2, "skate": 0.400001},
        2e-6,
    )
    assert accepted
    assert delta["mismatch"] < 0

    assert not accept_candidate(
        baseline,
        {"mismatch": 0.2, "pene": 0.2, "skate": 0.4},
        2e-6,
    )[0]
    assert not accept_candidate(
        baseline,
        {"mismatch": 0.1, "pene": 0.31, "skate": 0.4},
        2e-6,
    )[0]
    assert not accept_candidate(
        baseline,
        {"mismatch": 0.1, "pene": 0.2, "skate": 0.400003},
        2e-6,
    )[0]
