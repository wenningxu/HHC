# Historical HHC VQ recovery after Edmond became reachable

The original 185000-step VQ is available again. CPU recovery succeeded on Edmond;
no GPU task was launched and no model or numerical array was downloaded locally.

- Original: `/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004/best_fid.pth`
- Preserved snapshot inside the migrated runtime:
  `/workspace/HHC/output/historical_vq185k_recovered_20261020/best_fid.pth`
- Size: 115068263 bytes.
- SHA-256: `08ba7d0def3c73a94c7e0b63bc1ac6b48b757b4757bbf66a8641cf7062f8d608`.
- Strict CPU state loading passed for all 107 tensors. Model and normalization
  tensors are finite; normalization is float32 with 276 channels and positive std.
- Normalization SHA matches the active rebuilt 170k VQ exactly:
  `3eb9cfbb627322add3bcb0c362c747173cd577a21160fc5491e9ce06644d9fb6`.

The original manifest records completed 300000-step training, best step 185000,
and historical native fixed-480 reconstruction FID **2.4238196275200607**.
That score was not rerun here; it is neither generated-motion FID nor evidence of
a new improvement over the active pipeline.

Seven of the ten source fingerprints in the original training manifest match
current files. The other three original files were recovered with exact matching
hashes from the Edmond pre-sync archive:
`models/vqvae_sep3_shape.py`, `utils/losses.py`, and
`models/interhuman_evaluator_wrapper.py`. They are saved under the snapshot's
`provenance_sources/`, without replacing production files. This establishes the
listed source provenance; it does not establish current-vs-historical forward or
evaluation equivalence.

The recovery tool is `tools/recover_historical_vq185k_cpu.py`.
Cloud receipts are `acceptance.json` and `source_recovery.json` in the snapshot
directory; the small acceptance receipt is also saved locally at
`output/hhc_sota_20261016/historical_vq185k_recovered_20261020.json`.
The tool and acceptance receipt were independently read back and hashed.

The active frozen 170k specification remains byte-identical
(`b1688d04a2337bb8215486b9ce13bfc70360b4dc9f87787c89788d5a6ece247e`).
A4000 remains bound to that codebook. The recovered historical codebook must not
be substituted into an already-trained MGT; any future historical comparison
requires a matching tokenizer/model contract and the same validation protocol.
Official InterGen priority and the existing HHC/evaluation release gates remain.
