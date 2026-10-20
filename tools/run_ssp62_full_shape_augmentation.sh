#!/usr/bin/env bash
# Frozen 62-donor optimizer; full eligible set, including zero-timing crops.
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1
python=/home/wenning/miniconda3/envs/New_MMM/bin/python
plan=configs/interx_shape_augmentation/ssp3d_all62_fulltrain_20260911/plan_manifest.json
base=output/ssp3d_all62_full6238_20260911
overlay=dataset/interx_shape_aug_ssp62_full6238_gated_20260911
[[ ! -e "$base" && ! -e "$overlay" ]]
[[ $(df --output=avail -B1 . | tail -1) -gt 8589934592 ]]
mkdir "$base"

wait_pair() {
    local first_status=0 second_status=0
    wait "$1" || first_status=$?
    wait "$2" || second_status=$?
    [[ $first_status == 0 && $second_status == 0 ]]
}

generate() {
    local shard=$1
    "$python" tools/run_interx_shape_augmentation.py --plan "$plan" \
        --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
        --mode optimized --selection-seed 20260911 --offset "$((shard * 3119))" --limit 3119 \
        --num-epochs 30 --learning-rate .001 --signed-distance-backend nearest_vertex \
        --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed \
        --penetration-loss-reduction frame_sum --aux-temporal-weighting released_broadcast \
        --ground-alignment-policy pair_shared_joint_min_restore_raw \
        --device "cuda:$shard" --output-dir "$base/shard$shard"
}

score() {
    local shard=$1
    "$python" tools/diagnose_shape_optimization_gt.py --condition "shared62=$base/shard$shard" \
        --output "$base/gt_diagnostic$shard.json" --device "cuda:$shard"
    "$python" tools/export_shape_gt_references.py --candidate-dir "$base/shard$shard" \
        --output-dir "$base/references$shard"
    "$python" tools/score_volumetric_smpl_npz.py --folder "$base/references$shard/original_gt" \
        --output-dir "$base/volume_gt$shard" --device "cuda:$shard"
    "$python" tools/score_volumetric_smpl_npz.py --folder "$base/shard$shard" \
        --output-dir "$base/volume_candidate$shard" --device "cuda:$shard"
    "$python" tools/build_shape_volume_quality_gate.py --report "$base/gt_diagnostic$shard.json" \
        --condition shared62 --references "$base/references$shard/references.json" \
        --gt-dir "$base/volume_gt$shard" --candidate-dir "$base/volume_candidate$shard" \
        --output "$base/gate$shard.json"
}

generate 0 > "$base/generate0.log" 2>&1 & first=$!
generate 1 > "$base/generate1.log" 2>&1 & second=$!
echo "Generation jobs: $first $second; logs: $base/generate{0,1}.log"
wait_pair "$first" "$second"
score 0 > "$base/score0.log" 2>&1 & first=$!
score 1 > "$base/score1.log" 2>&1 & second=$!
echo "Scoring jobs: $first $second; logs: $base/score{0,1}.log"
wait_pair "$first" "$second"
"$python" - "$base" <<'PY'
import json, sys
from pathlib import Path
gates = [json.loads((Path(sys.argv[1])/f'gate{i}.json').read_text()) for i in (0, 1)]
total = sum(g['denominator'] for g in gates)
accepted = sum(g['accepted_count'] for g in gates)
assert total == 6238
print('Full requested/accepted:', total, accepted, 'pass fraction:', accepted/total)
assert accepted / total > .60, 'Keep scored outputs, but do not promote a failing full run'
PY
"$python" tools/materialize_interx_shape_augmentation.py --plan "$plan" \
    --optimization-dir "$base/shard0" --optimization-dir "$base/shard1" \
    --quality-gate "$base/gate0.json" --quality-gate "$base/gate1.json" --output-root "$overlay"
"$python" - "$overlay" <<'PY'
import sys
from utils.interx_shape_augmentation import validate_materialized_interx_shape_augmentation
root = sys.argv[1]
result = validate_materialized_interx_shape_augmentation(root, root+'/materialization_manifest.json')
print('Full overlay validated:', result['counts'], result['quality_selection']['pass_fraction'])
PY
