#!/usr/bin/env python3
"""Replay TiMotion export with the standard Inter-X 6D root-fit convention.

This deliberately lives beside, rather than modifying, the exporter frozen by
the active TiMotion training run. It is an offline diagnostic until matched-
seed scoring proves the effect and a new fair protocol is preregistered.
"""

from __future__ import annotations

import os
import sys

import h5py
import torch
import torch.nn.functional as functional

import export_timotion_official_interx as base


def standard_rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"required wrapper argument missing: {name}") from error


def main() -> None:
    if "--resume" in sys.argv:
        raise ValueError(
            "rotation-fix diagnostic forbids resume so provenance attributes "
            "cannot be confused with the frozen active exporter"
        )
    output = os.path.abspath(argument_value("--output"))
    base.rotation_6d_to_matrix = standard_rotation_6d_to_matrix
    base.main()
    with h5py.File(output, "r+") as handle:
        handle.attrs["adapter"] = (
            "22-body rotations + shape-specific all-joint rigid root fit v3; "
            "standard row-concatenated rotation6d; identity hands; diagnostic"
        )
        handle.attrs["exporter_sha256"] = base.sha256_file(__file__)
        handle.attrs["adapter_source_sha256"] = base.sha256_file(__file__)
        handle.attrs["base_exporter_sha256"] = base.sha256_file(base.__file__)
        handle.attrs["rotation6d_contract"] = (
            "a1=d6[...,0:3], a2=d6[...,3:6], Gram-Schmidt, row-stack"
        )
        handle.attrs["formal_result_eligible"] = False
        handle.attrs["diagnostic_reason"] = (
            "matched-seed isolation of TiMotion root-fit 6D convention bug"
        )


if __name__ == "__main__":
    main()
