#!/usr/bin/env python3
"""Diagnostic-only HHC exporter that changes only token update support.

All CLI arguments are forwarded to the frozen production HHC exporter.  The
production ``sample_intra_cas`` method is monkey-patched in memory to the
all-valid-position re-prediction diagnostic for this process only.  Resume is
deliberately forbidden so no production-labelled HDF5 can be mixed with this
diagnostic contract.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.two_person_transformer import Text2Motion_Transformer
from tools import export_hhc_official_interx as production_exporter


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_value(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        production_exporter.main()
        return
    if "--resume" in sys.argv:
        raise ValueError(
            "diagnostic re-prediction export forbids --resume; use a new output"
        )
    output_text = option_value(sys.argv[1:], "--output")
    if not output_text:
        raise ValueError("--output is required")
    output = Path(output_text).resolve()
    if output.exists():
        raise FileExistsError(
            f"diagnostic output already exists and cannot be resumed: {output}"
        )

    original = Text2Motion_Transformer.sample_intra_cas
    Text2Motion_Transformer.sample_intra_cas = (
        Text2Motion_Transformer.sample_intra_cas_repredict
    )
    try:
        production_exporter.main()
    finally:
        Text2Motion_Transformer.sample_intra_cas = original

    if not output.is_file():
        raise RuntimeError(f"production exporter did not create {output}")
    model_source = ROOT / "models/two_person_transformer.py"
    exporter_source = ROOT / "tools/export_hhc_official_interx.py"
    with h5py.File(output, "r+") as target:
        if target.attrs.get("sampler") != "sample_intra_cas":
            raise RuntimeError("unexpected production sampler metadata")
        if target.attrs.get("sampling_steps") != 50:
            raise RuntimeError("diagnostic must retain the fixed 50 outer steps")
        if target.attrs.get("temperature") != 0.5:
            raise RuntimeError("diagnostic must retain temperature 0.5")
        if target.attrs.get("cfg_scale") != 3.5:
            raise RuntimeError("diagnostic must retain CFG 3.5")
        if target.attrs.get("contact_guidance_scale") != 1.0:
            raise RuntimeError("diagnostic must retain contact scale 1.0")
        target.attrs["sampler"] = "sample_intra_cas_repredict"
        target.attrs["sampler_semantics"] = (
            "diagnostic_repredict_all_valid_positions_each_outer_iteration"
        )
        target.attrs["diagnostic_changed_factor"] = "token_update_support_only"
        target.attrs["diagnostic_not_formal_result"] = True
        target.attrs["diagnostic_wrapper"] = str(Path(__file__).resolve())
        target.attrs["diagnostic_wrapper_sha256"] = sha256_file(Path(__file__))
        target.attrs["production_exporter"] = str(exporter_source.resolve())
        target.attrs["production_exporter_sha256"] = sha256_file(exporter_source)
        target.attrs["model_source_sha256"] = sha256_file(model_source)
        target.flush()


if __name__ == "__main__":
    main()
