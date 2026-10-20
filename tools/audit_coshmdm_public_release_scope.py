#!/usr/bin/env python3
"""Classify what the public CoShMDM repository actually makes runnable."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path("/mnt/8T-HDD/wenning/coshmdm-code")
EXPECTED_REVISION = "c5116ab3bead55051ffec2ec9a161a38d2ab5116"
EXPECTED_ORIGIN = "https://github.com/AliManjotho/coshmdm-code.git"
PAPER_DOI = "10.1109/TVCG.2026.3675725"
PROJECT_URL = "https://alimanjotho.github.io/coshmdm/"
OUTPUT = ROOT / "output/coshmdm_public_release_scope_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str, binary: bool = False):
    value = subprocess.check_output(["git", "-C", str(REPOSITORY), *args])
    return value if binary else value.decode().strip()


def method_arguments(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and child.name == method_name:
                    return [argument.arg for argument in child.args.args]
    raise ValueError(f"missing {class_name}.{method_name} in {path}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def intergen_source_comparison(reference: Path, verify_reference: bool = False) -> dict:
    """Compare source without importing models or touching any checkpoint."""
    aliases = {
        'CoShMDM': 'InterGen', 'bert_model': 'clip_model',
        'bert_transformer': 'clip_transformer',
        'bertTransEncoderLayer': 'clipTransEncoderLayer',
        'bertTransEncoder': 'clipTransEncoder', 'bert_ln': 'clip_ln',
        'bert_out': 'clip_out', 'coshmdm': 'intergen',
    }

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = aliases.get(node.id, node.id)
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            node.attr = aliases.get(node.attr, node.attr)
            return self.generic_visit(node)

        def visit_ClassDef(self, node):
            node.name = aliases.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            node.module = aliases.get(node.module, node.module)
            return self.generic_visit(node)

    rows = []
    for path in sorted((REPOSITORY/'models').glob('*.py')):
        other = reference/'models'/('intergen.py' if path.name == 'coshmdm.py' else path.name)
        if not other.is_file():
            rows.append({'coshmdm':path.name,'intergen_missing':True})
            continue
        first, second = path.read_text(encoding='utf-8'), other.read_text(encoding='utf-8')
        rows.append({'coshmdm':path.name,'intergen':other.name,
                     'bytes_equal':path.read_bytes() == other.read_bytes(),
                     'ast_equal_after_explicit_identifier_renaming':
                         ast.dump(Rename().visit(ast.parse(first))) == ast.dump(ast.parse(second)),
                     'coshmdm_source':artifact(path),'intergen_source':artifact(other)})
    first = yaml.safe_load((REPOSITORY/'configs/model.yaml').read_text())
    second = yaml.safe_load((reference/'configs/model.yaml').read_text())
    differences = {k:{'coshmdm':first.get(k),'intergen':second.get(k)}
                   for k in sorted(first.keys() | second.keys()) if first.get(k) != second.get(k)}
    checkpoint_ids = []
    import re
    for directory in (REPOSITORY, reference):
        checkpoint_ids.append(re.findall(r'uc\?id=([A-Za-z0-9_-]+)',
            (directory/'prepare/download_pretrain_model.sh').read_text()))
    loader_strict_values = {}
    for relative in ('tools/infer.py', 'tools/eval.py'):
        path = REPOSITORY/relative
        calls = [node for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                 and node.func.attr=='load_state_dict']
        loader_strict_values[relative] = [ast.literal_eval(k.value)
            for call in calls for k in call.keywords if k.arg=='strict']
    reference_integrity = {'verified':False, 'reason':'Live reference verification was not requested'}
    if verify_reference:
        import urllib.request
        commit='b1750df4a859443b8a69329f759ce3a7503ae179'
        url=f'https://api.github.com/repos/tr3e/InterGen/git/trees/{commit}?recursive=1'
        request=urllib.request.Request(url,headers={'User-Agent':'HHC-source-evidence-audit'})
        with urllib.request.urlopen(request,timeout=30) as response:
            tree=json.load(response)
        assert tree['sha']==commit and not tree['truncated']
        blobs={entry['path']:entry['sha'] for entry in tree['tree'] if entry['type']=='blob'}
        paths=[f'models/{row["intergen"]}' for row in rows]+['configs/model.yaml','prepare/download_pretrain_model.sh']
        verified=[]
        for relative in paths:
            content=(reference/relative).read_bytes()
            digest=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert digest==blobs[relative],f'Reference differs from official commit: {relative}'
            verified.append({'path':relative,'git_blob_sha1':digest})
        reference_integrity={'verified':True,'repository':'https://github.com/tr3e/InterGen',
                             'commit':commit,'api_url':url,'files':verified}
    return {'scope':'Static model/config/source and download-link comparison; not inference or independent checkpoint validation',
            'identifier_aliases':aliases, 'model_sources':rows,
            'all_model_ast_equal_after_explicit_identifier_renaming':
                len(rows)==9 and all(r.get('ast_equal_after_explicit_identifier_renaming') for r in rows),
            'config_differences':differences,
            'coshmdm_download_ids':checkpoint_ids[0],'intergen_download_ids':checkpoint_ids[1],
            'same_download_ids':checkpoint_ids[0]==checkpoint_ids[1],
            'reference_integrity':reference_integrity,
            'loader_strict_values':loader_strict_values,
            'loader_sources':{relative:artifact(REPOSITORY/relative) for relative in loader_strict_values},
            'checkpoint_namespace_warning':'The bert_* attribute renames change state_dict keys; public infer uses strict=False and eval uses strict=True. Names alone do not make a separately trained model.'}


def main() -> None:
    global REPOSITORY, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=REPOSITORY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--claim-scope', choices=('historical_interx_port', 'interhuman_release_only'),
                        default='historical_interx_port')
    parser.add_argument('--intergen-reference', type=Path)
    parser.add_argument('--verify-reference-github', action='store_true')
    args = parser.parse_args()
    REPOSITORY, OUTPUT = args.repository.resolve(), args.output.resolve()
    revision = git("rev-parse", "HEAD")
    origin = git("remote", "get-url", "origin")
    tracked = git("ls-tree", "-r", "--name-only", revision).splitlines()
    remote_branches = git(
        "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"
    ).splitlines()
    tags = git("tag", "--list").splitlines()
    runnable_python = [
        path for path in tracked
        if path.endswith(".py")
        and path.split("/", 1)[0] in {"models", "datasets", "tools"}
    ]
    model_path = REPOSITORY / "models/coshmdm.py"
    diffusion_path = REPOSITORY / "models/nets.py"
    config_path = REPOSITORY / "configs/model.yaml"
    readme_path = REPOSITORY / "README.md"
    download_path = REPOSITORY / "prepare/download_pretrain_model.sh"
    model_source = model_path.read_text(encoding="utf-8")
    diffusion_source = diffusion_path.read_text(encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")
    download = download_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    path_inventory = "\n".join(runnable_python).lower()

    checks = {
        "official_repository_revision": revision == EXPECTED_REVISION,
        "official_repository_origin": origin == EXPECTED_ORIGIN,
        "readme_identifies_paper_doi": PAPER_DOI.lower() in readme.lower(),
        "released_model_is_direct_262d_diffusion": (
            config.get("INPUT_DIM") == 262
            and "x_start = batch[\"motions\"]" in diffusion_source
            and "MotionDiffusion(" in diffusion_source
        ),
        "released_model_forward_has_no_shape_argument": (
            method_arguments(model_path, "CoShMDM", "forward")
            == ["self", "batch"]
            and method_arguments(diffusion_path, "InterDiffusion", "forward")
            == ["self", "batch"]
            and 'batch["betas"]' not in model_source + diffusion_source
            and 'batch["shape"]' not in model_source + diffusion_source
        ),
        "no_vq_or_quantizer_in_runnable_tree": not any(
            token in path_inventory
            for token in ("vqvae", "vq_vae", "quantizer", "codebook")
        ) and not any(
            token in (model_source + diffusion_source).lower()
            for token in ("vectorquant", "vector_quant", "codebook", "quantizer")
        ),
        "no_rl_sdf_policy_in_runnable_tree": not any(
            token in path_inventory
            for token in ("reinforcement", "policy_network", "signed_distance", "sdf")
        ),
        "released_checkpoint_is_interhuman_only": (
            "# InterHuman" in download and download.count("gdown ") == 1
        ),
        "no_tracked_interx_checkpoint": not any(
            "interx" in path.lower()
            and Path(path).suffix.lower() in {".ckpt", ".pt", ".pth", ".tar"}
            for path in tracked
        ),
        "no_alternate_public_branch_or_tag_in_checkout": (
            set(remote_branches) in (
                {"origin", "origin/main"},
                {"origin/HEAD", "origin/main"},
            )
            and tags == []
        ),
    }
    payload = {
        "schema": "hhc_coshmdm_public_release_scope_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "paper_identity": {
            "title": (
                "CoShMDM: Contact and Shape-Aware Latent Motion Diffusion "
                "Model for Human Interaction Generation"
            ),
            "doi": PAPER_DOI,
            "project_page": PROJECT_URL,
            "full_text_access": (
                "DOI and project abstract are public; project Paper/arXiv "
                "buttons were disabled and the IEEE PDF was not retrievable "
                "in the audited environment on 2026-09-05"
            ),
            "project_page_claims": [
                "contact-and-shape-aware latent motion representation",
                "dual-encoder VQ-VAE for motion and contact",
                "text-and-body-shape-conditioned diffusion",
                "reinforcement-learning SDF penetration avoidance policy",
            ],
            "abstract_reported_fid": {
                "interhuman": 4.801,
                "interx": 0.013,
                "comparison_policy": (
                    "headline abstract values only; full evaluator/data protocol "
                    "is unavailable, so these values must not be compared "
                    "numerically with the local official-Inter-X evaluator"
                ),
            },
        },
        "released_runnable_pipeline": {
            "motion_representation": "262D per person, direct motion diffusion",
            "text_conditioning": "CLIP ViT-L/14@336 plus two-layer text transformer",
            "diffusion": "1000-step cosine training diffusion; DDIM50 release config",
            "contact_support": (
                "geometric/contact loss code is present, but the project-page "
                "latent VQ/shape/RL-SDF pipeline is not present in the runnable tree"
            ),
            "shape_conditioning": False,
            "vq_latent_pipeline": False,
            "rl_sdf_policy": False,
            "released_checkpoint_dataset": "InterHuman",
            "released_interx_checkpoint": False,
        },
        "classification": (
            "public_repository_runnable_diffusion_port_not_paper_architecture_reproduction"
        ),
        "allowed_claim_scope": (
            "user-requested public-repository code ported and retrained on the "
            "official Inter-X split under a two-day cap; not the paper-described "
            "CoShMDM architecture and not a paper-exact result"
            if args.claim_scope == 'historical_interx_port' else
            "Static audit of the author's released InterHuman source only; no "
            "new training, inference, independent checkpoint or paper-result reproduction."
        ),
        "forbidden_claims": [
            "reproduction of the paper-reported Inter-X FID 0.013",
            "paper architecture trained on Inter-X",
            "official CoShMDM Inter-X checkpoint evaluation",
            "independent reproduction of the paper-reported InterHuman FID 4.801",
        ],
        "repository": {
            "path": str(REPOSITORY),
            "revision": revision,
            "origin": origin,
            "tracked_files": len(tracked),
            "runnable_python_files": runnable_python,
            "remote_branches": remote_branches,
            "tags": tags,
            "live_remote_ref_audit_2026_09_05": (
                "git ls-remote found only refs/heads/main at the audited revision"
            ),
        },
        "artifacts": {
            "readme": artifact(readme_path),
            "model": artifact(model_path),
            "diffusion": artifact(diffusion_path),
            "model_config": artifact(config_path),
            "checkpoint_download": artifact(download_path),
            "implementation": artifact(Path(__file__)),
        },
    }
    if args.intergen_reference:
        payload['intergen_source_comparison'] = intergen_source_comparison(
            args.intergen_reference.resolve(), args.verify_reference_github)
    atomic_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "checks": len(checks),
        "output": str(OUTPUT),
    }, indent=2))
    if payload["status"] != "pass":
        raise RuntimeError(
            f"CoShMDM public release scope audit failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )


if __name__ == "__main__":
    main()
