#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

patch_path="tools/provenance_snapshots/post_formal_shape_root_tokenizer_wiring.patch"
expected_options_sha="2535372b000283497bf63d51163020a30ac85ecc891db2cd2d7205ab7182c60b"
expected_train_sha="79235536ca8e935e1e643ce18d9f8ece608e9ca21177561dcb2683497237eb0d"

# The historical V7 replay and the formal paper-config HHC run both bind the
# current training source bytes.  Wiring the augmentation-only tokenizer is
# authorized only after both have exited and the formal baseline artifacts
# prove that its training/evaluation chain completed.
if pgrep -f 'python -u train/train_t2i_trans.py' >/dev/null 2>&1; then
  echo 'An MGT process is still active; refusing post-formal source wiring.'
  exit 2
fi

for required in \
  output/hhc_formal_raw12288_repaired_mgt_selection.json \
  output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact/summary.json \
  output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/summary.json
do
  if [[ ! -s "${required}" ]]; then
    echo "Formal HHC completion artifact is missing: ${required}"
    exit 2
  fi
done

formal_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_formal_raw12288_repaired_mgt_selection.json'))['training_manifest'])")"
formal_run_dir="$(dirname "${formal_training_manifest}")"
formal_source_snapshot="${formal_run_dir}/launch_source_snapshot_manifest.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/freeze_mgt_training_sources.py \
  --training-manifest "${formal_training_manifest}" \
  --output-dir "${formal_run_dir}/launch_sources" \
  --output-manifest "${formal_source_snapshot}"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_formal_launch_contract.py \
  --source-snapshot-manifest "${formal_source_snapshot}"

# Idempotent after a successful application.
if conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_root_tokenizer_wiring.py >/dev/null 2>&1
then
  echo 'Shape root-tokenizer wiring already passes.'
  exit 0
fi

actual_options_sha="$(sha256sum options/option_transformer.py | awk '{print $1}')"
actual_train_sha="$(sha256sum train/train_t2i_trans.py | awk '{print $1}')"
if [[ "${actual_options_sha}" != "${expected_options_sha}" \
      || "${actual_train_sha}" != "${expected_train_sha}" ]]; then
  echo 'Post-formal wiring source hash mismatch; refusing contextual patching.'
  echo "options expected=${expected_options_sha} actual=${actual_options_sha}"
  echo "train expected=${expected_train_sha} actual=${actual_train_sha}"
  exit 2
fi

git apply --check "${patch_path}"
git apply --whitespace=error-all "${patch_path}"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_root_tokenizer_wiring.py
