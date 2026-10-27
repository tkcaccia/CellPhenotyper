# Bora semantic project export (review-only candidate)

`bin/export_bora_semantic_project.py` makes a **new immutable project JSON**
from a completed, expert-trained Bora semantic chain. It does no inference,
GPU work, image conversion, or modification of the input project. The current
`bora-wsi-semantic-chain-v1` has `native_he_boundary_refinement_applied: false`
and `biological_accuracy_qualified: false`, so the only permitted output is a
locked **predicted** review artifact. It is never a ground-truth label, a PLS
training region, or canonical `tissue_annotations`.

```bash
python3 bin/export_bora_semantic_project.py \
  --base-project /absolute/base.cellphenotyper.json \
  --roi-spec /absolute/frozen-semantic-roi-spec.json \
  --evaluation-folder /absolute/passing-expert-roi-evaluation \
  --registration /absolute/registration.json \
  --registered-support-tiff /absolute/registered-grandqc.tif \
  --materialization-folder /absolute/source-bound-materialized-tokens \
  --token-folder /absolute/semantic-token-field \
  --token-source /absolute/source-bound-materialized-tokens/virchow2-spatial-tokens.npy \
  --he-token-source /absolute/source-bound-materialized-tokens/registered-he-tokens.npy \
  --shard-folder /absolute/semantic-shard-0 \
  --shard-folder /absolute/semantic-shard-1 \
  --bora-output /absolute/semantic-global-chain \
  --bora-source-root /absolute/reviewed-bora/src \
  --output /absolute/new-semantic-review.cellphenotyper.json
```

Give **every** shard folder exactly once. The exporter verifies the frozen
expert ROI spec and all source bytes; a completed passing same-slide holdout
and the saved model; source-bound Virchow2/H&E token arrays and prediction
receipt; the [materialization receipt](bora-semantic-token-materialization.md)
against all three frozen chunk scales and registered H&E; every native shard
receipt and probability; dual-MPP registration,
GrandQC support and native image; complete core assembly; global exact-mask
and smoothed-vector receipts; and all published output hashes. It scans the
cleaned exact native mask in bounded row blocks to reject missing labels,
out-of-support pixels, and single-class collapse. It requires every declared
class to retain at least 100 µm². Its GeoJSON headers must bind the same native
frame, image, semantic mask and global model. The Bora chain's own exact
mask/vector parity and ≥100 µm² part/hole cleanup receipts remain required.

The new artifact has role `bora_exploratory_boundary_review`, producer
`bora-wsi-semantic-chain-v1`, identity
`exploratory_same_slide_holdout`, `predicted_not_expert_annotation: true`,
`review_only: true`, and `training_eligible: false`. The unchanged base project
remains available. An existing destination is rejected. The `--canonical-tissue-annotations`
switch intentionally fails for this chain; it cannot be used to relabel a
review result by changing one flag.

Future canonical publication needs a **new, source-bound H&E-refined chain
contract**, not an altered `v1` receipt. Its exact output inventory must bind
the H&E image/refinement algorithm and parameters to the same native frame,
the passing expert holdout/model and all probability shards; verify geometry
against the refined native mask; preserve review versus exact geometry roles;
pass the 100 µm² and class-collapse gates; and undergo native H&E overlay and
independent biological review. A new exporter/consumer version must verify
those receipt and source hashes before adding a clearly predicted
`tissue_annotations` artifact. A passing same-slide holdout alone does not
establish independent-patient accuracy.

Synthetic contract tests:

```bash
python3 -m pytest -q tests/test_export_bora_semantic_project.py
```
