"""Small, dependency-free checks for the physical-ground helpers."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

# Make direct ``python tests/...`` execution independent of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.interx_physical_ground import (
    decode_physical_pair,
    encode_physical_pair,
    place_pair_on_mesh_ground,
    translate_person,
)


def _people():
    return [
        {
            "trans": np.array([[3.0, 4.0, 5.0], [4.0, 6.0, 8.0]], dtype=np.float32),
            "poses": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "betas": np.array([0.1, -0.2]),
        },
        {
            "trans": np.array([[-2.0, 7.0, 1.0], [-1.0, 8.5, 2.0]], dtype=np.float32),
            "rotation": np.array([0.2, 0.3, 0.4]),
            "betas": np.array([0.5, 0.6]),
        },
    ]


def test_physical_ground_and_roundtrip():
    people = _people()
    original = deepcopy(people)

    grounded = place_pair_on_mesh_ground(people, [4.0, 7.0])
    assert np.allclose(grounded[0]["trans"][:, 1], [0.0, 2.0])
    assert np.allclose(grounded[1]["trans"][:, 1], [0.0, 1.5])
    assert np.array_equal(grounded[0]["trans"][1] - grounded[0]["trans"][0], [1.0, 2.0, 3.0])
    assert np.array_equal(grounded[1]["trans"][1] - grounded[1]["trans"][0], [1.0, 1.5, 1.0])
    for before, after in zip(original, grounded):
        for key in before:
            if key != "trans":
                assert np.array_equal(before[key], after[key])
    assert np.array_equal(people[0]["trans"], original[0]["trans"])
    assert np.array_equal(people[1]["trans"], original[1]["trans"])

    encoded, origin = encode_physical_pair(grounded)
    decoded = decode_physical_pair(encoded, origin)
    assert np.array_equal(origin, grounded[0]["trans"][0])
    for expected, actual in zip(grounded, decoded):
        assert np.allclose(expected["trans"], actual["trans"])
        assert np.array_equal(expected["betas"], actual["betas"])
    relative = grounded[1]["trans"] - grounded[0]["trans"]
    assert np.allclose(encoded[1]["trans"] - encoded[0]["trans"], relative)


def test_invalid_inputs_rejected():
    person = _people()[0]
    for offset in ([np.nan, 0, 0], [0, np.inf, 0]):
        try:
            translate_person(person, offset)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite offset accepted")
    try:
        translate_person({"trans": np.empty((0, 3))}, [0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("empty translation accepted")
    for floors in ([np.nan, 1], [0, np.inf]):
        try:
            place_pair_on_mesh_ground(_people(), floors)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite floor accepted")


if __name__ == "__main__":
    test_physical_ground_and_roundtrip()
    test_invalid_inputs_rejected()
    print("ok")
