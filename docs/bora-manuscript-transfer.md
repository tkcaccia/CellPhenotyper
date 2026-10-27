# Exploratory manuscript Bora transfer

This is a separate, opt-in **CPU-only** continuation for a complete, verified
Virchow2 spatial-token materialization. It transfers the frozen V5 tumor/stroma
manuscript head to a registered slide, preserves out-of-domain/low-confidence
tokens as **unknown**, refines the known tissue boundary against native H&E,
then creates **one globally vectorized, smooth GeoJSON**. The registered
GrandQC mask may be coarser than the native H&E: each native pixel centre
samples its full-frame GrandQC pixel by exact integer nearest mapping,
`((2*i+1)*support_size)//(2*native_size)` on both axes. The converter and
exporter verify this mapping without resizing the mask. It does not run an
encoder, use a GPU, fit a new model, or replace the expert-evaluated Bora
workflow. On PM016 its tissue names remain an exploratory manuscript transfer;
there is no independent PM016 biological-accuracy claim.

The action is `bin/bora_manuscript_transfer_action.py`. It requires the exact
V5 model, policy and centroid hashes, the registered native H&E and GrandQC
support, a verified dual-MPP registration, a complete materialization receipt,
the pinned Bora source-tree digest, and the running Python/dependency digest.
Use `identity --bora-source <source>` first to obtain the two digests. A run
requires `--acknowledge-exploratory` and a **new** result directory. Its new
project file is `experiment.cellphenotyper.json`; the input project is never
rewritten. The output artifact has role `bora_exploratory_boundary_review`,
`review_only: true`, and explicit unvalidated provenance. It is not an expert
annotation and supplies no PLS/training ROIs.

The action creates a single result folder containing `PREFLIGHT.json`, native
`support.npy`, `tokens/`, `shards/`, `assembly/`, `vectors/`, the new project,
and `RUN_COMPLETE.json`. If a stage fails, it retains `RUN_FAILED.json` and all
completed work for diagnosis; it does not retry or publish a successful
project. The project exporter rejects the old hard-group/stair-step chain and
requires the unknown-aware native assembly, exact categorical halo parity,
positive guarded smoothing, a reduction in contour complexity, no macroregion
cuts, and the registered frame/image/support hashes.

## PM016 admission and execution scope to review before remote use

- Use the existing PM016 native slide and registered 0.5-µm Virchow2 field.
  Materialize the saved 60/60 three-scale token inventory first. The
  materializer verifies every chunk, source hash, coordinate placement and
  H&E token mean. `MATERIALIZATION_COMPLETE.json` is mandatory. An inventory
  alone is insufficient. For PM016 the registered GrandQC TIFF is 1,845 ×
  2,988, while the native H&E is 14,762 × 23,904; both dimensions must remain
  explicit in the support-conversion receipt.
- Stage an immutable copy of the reviewed Bora source and exact V5 files.
  Obtain `identity` on the **actual CPU runtime**. Confirm Shapely ≥2.1,
  rasterio, model and policy hashes, native TIFF/support hashes, registration,
  sufficient free disk and no active writer to the result folder. Never use
  the deployed old hard-group exporter as a fallback.
- Native PM016 is 14,762 × 23,904 pixels (about 353 million). The default
  2,048-pixel core yields **96** bounded shards. Each native source window is
  checked against Bora's 20-million-pixel limit; cores cover the slide exactly
  once, and adjacent halos must agree categorically. The action requires at
  least 1 GiB and at most 12 GiB declared RAM and a 1–128 GiB disk budget;
  defaults are 3 GiB RAM and 16 GiB disk. Its conservative disk estimate,
  plus the host's free-space reserve, is checked before any output folder is
  made. Measure actual peak RAM and disk on the first admitted bounded case
  before changing those limits.
- The built-in deadline is checked before every shard and global vector step.
  Native library calls are in process, so this check **does not guarantee**
  shutdown of a hung call. A real Chiamaka run needs a separately reviewed
  outer process/cgroup supervisor with finite walltime, shutdown and owned
  descendant cleanup checks. If cleanup is unknown, stop; do not retry or
  start a successor. Chiamaka requires its own exact authorized execution
  scope. UCT remains paused under the coordinator HPC GPU policy.
- After a successful CPU run, check `RUN_COMPLETE.json`, every receipt/hash,
  unknown preservation, global vector parity/IoU and the opened wsiTools
  review layer on the **full** slide. Verify that no stair-step fallback or
  macroregion seam appears. Do not promote exploratory tissue labels to a
  validated semantic annotation without independent PM016 labels/review.

The current exact V5 source files are under the coordinator's
`validation/pm016-token-local-expanded-20261024/` folder. This document
describes an execution contract; it is not authorization to stage or start a
Chiamaka job.
