#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

echo "Archived duplicate post-shape queue: use run_current_interx_review_experiments_before_shape.sh and the explicit Interact2AR/shape entrypoints." >&2
exit 2

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# These reviewer-facing studies are intentionally scheduled after the core
# weak-FID diagnosis/control and the requested shape-augmentation retraining.
# They consume the same frozen baseline generations and do not determine which
# HHC training contract is launched.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_interx_volumetric_smpl_contract.py \
  --query-chunk 10475 \
  --output output/interx_volumetric_smpl_contract_gate.json

conda run --no-capture-output -n New_MMM python -u \
  tools/export_fair_h5_to_optimizer_npz.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --generations output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact/replication_00_main.h5 \
  --output-dir output/optimization_inputs/hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0 \
  --repeat-id 0 --resume

if conda run --no-capture-output -n New_MMM python -u \
  tools/audit_optimization_ablation_readiness.py --require-ready
then
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_paper_optimization_ablation.py \
    --motion-dir output/optimization_inputs/hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0 \
    --generation-manifest output/optimization_inputs/hhc_formal_raw12288_repaired_mgt_common50_no_contact_rep0/generation_records.jsonl \
    --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_test \
    --expected-split dataset/interx/splits/test.txt \
    --output-root output/optimization_ablation_pilot \
    --device cuda:1 --query-chunk 10475 --limit 25

  conda run --no-capture-output -n New_MMM python -u \
    tools/run_paper_optimization_formal.py \
    --device cuda:1 --query-chunk 10475 \
    --output-root output/optimization_formal_rep0
else
  echo 'Strict optimization cache is not ready; readiness artifact recorded, continuing independent experiments.'
fi

if ! conda run --no-capture-output -n New_MMM python -u \
  tools/test_volumetric_smpl_penetration.py \
  --device cuda:1 --frames 150 --query-chunk 10475 --seed 1234 \
  --output output/optimization_backend_audit/volumetric_smpl_penetration_cuda_f150_exact.json
then
  echo 'VolumetricSMPL 150-frame CUDA benchmark failed; continuing to sampler grid.'
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/run_hhc_sampling_pareto.py --device cuda:1 --batch-size 32

conda run --no-capture-output -n New_MMM python -u \
  tools/summarize_hhc_sampling_pareto.py \
  --manifest output/fair_eval/hhc_sampling_pareto_rep0/grid_manifest.json \
  --output output/fair_eval/hhc_sampling_pareto_rep0/summary.json

conda run --no-capture-output -n New_MMM python -u \
  tools/run_hhc_contact_timing_ablation.py \
  --device cuda:1 --batch-size 32

conda run --no-capture-output -n New_MMM python -u \
  tools/summarize_hhc_contact_timing_ablation.py
