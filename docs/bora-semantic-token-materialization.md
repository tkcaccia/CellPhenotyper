# Source-bound semantic token materialization

`bin/materialize_bora_semantic_tokens.py` reconstructs one registered,
full-frame **scale-1** Virchow2 spatial-token field and its matching H&E
token means. It runs on CPU with disk-backed arrays and bounded row reads.
It does not run Virchow2, fit Bora, predict tissue classes, or change a project.

The input JSON has `format:
cellphenotyper-bora-semantic-token-materialization-spec-v1`, an absolute
`bora_source_root` (reviewed Bora `src`), its `bora_source_sha256` map for
`__init__.py`, `admission.py`, `chunk_plan.py`, `chunk_store.py`,
`token_grid.py`, `local_semantic.py`, and these absolute `{path, sha256}`
`inputs`: `registration`, `native_image`, `working_image`, `working_he`,
`he_evidence`, `token_inventory`, and `virchow2_checkpoint`. The inventory
path must end in `CHUNK_INVENTORY.json`. The working H&E must be a registered
`uint8` YXC-3 NPY linked by the H&E evidence receipt to that exact image and
registration. The checkpoint must match the inventory. The two calibrated
images must have the same physical field and pixel-edge affine.

```bash
python3 bin/materialize_bora_semantic_tokens.py \
  --spec /absolute/frozen-token-materialization-spec.json \
  --output /absolute/new-materialized-tokens \
  --memory-budget-bytes 134217728 \
  --disk-budget-bytes 17179869184
```

Before writing, the stage validates **all three** planned scales, every
chunk hash, shape and dtype, complete absolute placement with no gaps or
overlaps, lifecycle completion, working frame, checkpoint and reviewed source
hashes. Only scale 1 is materialized. The first output is
`virchow2-spatial-tokens.npy`, YXC-1280 float32 in absolute token coordinates.
The second is `registered-he-tokens.npy`, YXC-3 float32: for each 14×14
working-pixel token, mean the RGB values and divide by 255; white-pad only
the true bottom/right image edge. This is the same H&E feature rule used by
the expert ROI fit. Both outputs and the complete receipt are staged and
published to a new folder only after source rechecks. A failed stage removes
its staging folder; an existing destination is preserved.

The completion receipt is `MATERIALIZATION_COMPLETE.json` with exact source
spec, inventory, checkpoint, frame, source-code and output SHA-256 values.
`verify_materialization(folder)` rechecks those sources, all chunk hashes and
the output arrays. A Bora semantic token scorer must call that verifier,
pass **only the two returned folder files** as `token_path` and
`he_token_path`, then check that its
`TOKEN_SEMANTIC_COMPLETE.json` records the same array hashes, dual-MPP
registration hash, model hash and native/working frames. The semantic project
exporter now requires this materialization folder and repeats the lineage
check; arbitrary raw token/H&E NPYs cannot enter its project handoff.

This proves **data lineage and geometry**, not cell or tissue classification
accuracy. The expert ROI holdout and native H&E boundary/biological review
gates remain separate.

```bash
python3 -m pytest -q tests/test_materialize_bora_semantic_tokens.py
```
