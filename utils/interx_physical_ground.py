"""Explicit constant translations for shape-aware physical ground coordinates.

Floor estimation must use the complete source sequence and actual mesh shape.
This does not infer whether an actor was suspended or supported by an object.
"""
import numpy as np


def translate_person(payload, offset):
    trans = np.asarray(payload['trans'], dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3 or len(trans) == 0:
        raise ValueError('expected nonempty [T,3] translation')
    if offset.shape != (3,) or not np.isfinite(offset).all() or not np.isfinite(trans).all():
        raise ValueError('translation and offset must be finite')
    return {**payload, 'trans': trans + offset}


def place_pair_on_mesh_ground(people, full_sequence_mesh_floors):
    floors = np.asarray(full_sequence_mesh_floors, dtype=np.float64)
    if len(people) != 2 or floors.shape != (2,) or not np.isfinite(floors).all():
        raise ValueError('expected two people and two finite full-sequence mesh floors')
    return [translate_person(person, [0, -floor, 0]) for person, floor in zip(people, floors)]


def encode_physical_pair(people):
    """Use one P1 initial origin, never independently recenter the two people."""
    if len(people) != 2:
        raise ValueError('expected two people')
    origin = translate_person(people[0], [0, 0, 0])['trans'][0].copy()
    return [translate_person(person, -origin) for person in people], origin


def decode_physical_pair(people, origin):
    if len(people) != 2:
        raise ValueError('expected two people')
    return [translate_person(person, origin) for person in people]
