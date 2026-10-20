#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda_run=(/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM)
random="output/interx_shape_aug_pilot_random_seed20260828"
g024_p1="${random}/G024T003A019R024__shapeaug_s00000123_p1.npz"
g024_p2="${random}/G024T003A019R024__shapeaug_s00000123_p2.npz"
g005_p1="${random}/G005T003A016R004__shapeaug_s00000123_p1.npz"
g005_p2="${random}/G005T003A016R004__shapeaug_s00000123_p2.npz"
query_g024="output/volumetric_smpl_current_query_chunk_G024.json"
query_g005="output/volumetric_smpl_current_query_chunk_G005.json"
query_gate="output/volumetric_smpl_current_query_chunk_assessment.json"
frame_g024="output/volumetric_smpl_current_frame_batch_G024.json"
frame_g005="output/volumetric_smpl_current_frame_batch_G005.json"
frame_gate="output/volumetric_smpl_current_frame_batch_assessment.json"

for person in "${g024_p1}" "${g024_p2}" "${g005_p1}" "${g005_p2}"; do
  test -s "${person}"
done

# Re-run against current source hashes. Historical pass files are deliberately
# not reused after backend changes.
for spec in \
  "${g024_p1}|${g024_p2}|12|${query_g024}" \
  "${g005_p1}|${g005_p2}|120|${query_g005}"
do
  IFS='|' read -r p1 p2 frames output <<<"${spec}"
  "${conda_run[@]}" python -u tools/benchmark_volumetric_smpl_query_chunk.py \
    --p1 "${p1}" --p2 "${p2}" --frames "${frames}" \
    --baseline-query-chunk 512 \
    --candidate-query-chunks 1024 2048 4096 8192 10475 \
    --device "cuda:${gpu}" --output "${output}"
done
"${conda_run[@]}" python -u tools/assess_volumetric_smpl_query_chunk.py \
  --benchmark "${query_g024}" "${query_g005}" --output "${query_gate}"

authorized_query="$(python -c "import json; print(json.load(open('${query_gate}'))['query_chunk_authorized'])")"
if [[ ! "${authorized_query}" =~ ^[0-9]+$ ]] || (( authorized_query < 1024 )); then
  echo "Current query gate did not authorize a usable query chunk: ${authorized_query}" >&2
  exit 2
fi

for spec in \
  "${g024_p1}|${g024_p2}|12|${frame_g024}" \
  "${g005_p1}|${g005_p2}|120|${frame_g005}"
do
  IFS='|' read -r p1 p2 frames output <<<"${spec}"
  "${conda_run[@]}" python -u tools/benchmark_volumetric_smpl_frame_batching.py \
    --p1 "${p1}" --p2 "${p2}" --frames "${frames}" \
    --frame-batch-sizes 2 4 8 --query-chunk "${authorized_query}" \
    --device "cuda:${gpu}" --output "${output}"
done
"${conda_run[@]}" python -u tools/assess_volumetric_smpl_frame_batching.py \
  --benchmark "${frame_g024}" "${frame_g005}" \
  --expected-query-chunk "${authorized_query}" --output "${frame_gate}"

echo "Current-source VolumetricSMPL query and frame-batching gates passed."
