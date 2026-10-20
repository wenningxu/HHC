#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

driver="tools/run_fair_interx_method.py"
patch="tools/provenance_snapshots/post_hhc_timotion_selection_binding.patch"
manifest="output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/run_manifest.json"
summary="output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/summary.json"
launch_sha="014a720cdb53eb43bd4834391e8b97bc70763d8b31b43dc00a398b66fdaed6ec"
timotion_sha="7b83a6b748dfcff07082973778d188925308164c58ced498504483a68e888d7e"

if pgrep -f 'python.*tools/run_fair_interx_method.py.*hhc_formal_raw12288_repaired_mgt_common50_contact' >/dev/null 2>&1; then
  echo 'Formal HHC common-50 driver is still active; refusing post-HHC patch.'
  exit 2
fi
if [[ ! -s "${manifest}" || ! -s "${summary}" ]]; then
  echo 'Formal HHC common-50 completion artifacts are missing.'
  exit 2
fi

completion_stable="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${manifest}')).get('implementation_source_stability_verified_at_completion') is True)")"
if [[ "${completion_stable}" != True ]]; then
  echo 'Formal HHC driver did not record stable launch sources at completion.'
  exit 2
fi

current_sha="$(sha256sum "${driver}" | cut -d' ' -f1)"
if [[ "${current_sha}" == "${timotion_sha}" ]]; then
  echo 'Post-HHC TiMotion selection binding already applied.'
  exit 0
fi
if [[ "${current_sha}" != "${launch_sha}" ]]; then
  echo "Unexpected fair driver hash before post-HHC patch: ${current_sha}"
  exit 2
fi

git apply --check "${patch}"
git apply --whitespace=error-all "${patch}"
python -m py_compile "${driver}"

current_sha="$(sha256sum "${driver}" | cut -d' ' -f1)"
if [[ "${current_sha}" != "${timotion_sha}" ]]; then
  echo "Post-HHC fair driver hash mismatch: ${current_sha}"
  exit 2
fi
echo 'Post-HHC TiMotion selection binding applied and verified.'
