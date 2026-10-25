# Standard contextual Bora continuation for three KODAMA groups

`bora_contextual_fullslide.nf` is the explicit CellPhenotyper continuation for
the registered **Virchow2 → KODAMA (three direct groups) → contextual Bora**
route. It starts after a complete 0.5 µm/pixel Virchow2 spatial-token inventory
has been projected to one 32-channel PCA lattice on the same image. It does
not silently use the earlier `main.nf` tissue route. That main workflow has
not yet been upgraded to produce the complete three-scale spatial-token
inventory or this KODAMA binding, so it remains unchanged.

Use Bora `main` at commit `fa3e3d41f7b61500a442d9c0c52f3f86b739ff9d`
or a later compatible revision. This continuation calls its public registered
three-group fit, tiled score, native refinement, compact label assembly, and
guarded contour APIs.

The current standard is a method, not PM016's fitted weights. Every new slide
must fit its own three-group numeric head from its KODAMA-labeled tile
representatives or supply a head receipt pinned to the *same image* and PCA
lattice. PM016's fitted coefficients, 2.9371-token radius and 0.7806 prior
exponent are not defaults and are not included in Git. KODAMA group IDs are
anonymous domains; the output does not claim tumor/stroma identity.

## Frozen input contract

Start with [the manifest template](../params/bora_contextual_fullslide.example.json).
Every file is an ordinary absolute path paired with its SHA-256. The manifest
identifies one CellPhenotyper base project, the exact native RGB TIFF and frame,
native and token tissue-support arrays, the complete float32 `[Y,X,32]` PCA
lattice, the frozen PCA projection file, and the direct three-group KODAMA binding. The working grid must be
0.5 µm/pixel, with a spatial token every 14 working pixels; the native slide
MPP and physical extent must agree with it. The binding must pin the native
image, frame, PCA field, token support, working shape and three group IDs.
It also verifies and binds the frozen PCA projection file's SHA-256; an unrelated projection cannot
reuse the fitted model even if its array dimensions match.

The recommended `head_mode` is `fit_from_kodama_tiles`. Its bound UTF-8 CSV
has the exact columns `tile_id,native_x_px,native_y_px,kodama_group`, one
registered representative point per tile, group IDs exactly 1, 2 and 3, and
at least ten tiles per group. The KODAMA binding must record the CSV SHA-256.
The action samples the **contextualized** PCA lattice at those points, fits
Bora's balanced three-class logistic head, and saves both
`model/numeric_head.npz` and Bora's `model/FIT_COMPLETE.json` in this slide's
result folder.
The declared training prior must equal the three CSV group fractions.

For a deterministic replay, `head_mode: pinned_same_slide_head` replaces the
tile CSV with `numeric_head.npz` and Bora's `FIT_COMPLETE.json` from one
existing model folder. The receipt must bind this native image and frame,
PCA projection, KODAMA binding and model hash. A head from PM016 cannot be
passed for another image.

The context radius, prior correction, regularization, core size and halo are
explicit in the manifest. The recovered native H&E boundary-refinement
specification is versioned and checked exactly. The source tree of Bora is
hashed (every Python module and `pyproject.toml`) and must match the launch
pin. Install the matching Bora source and CPU dependencies in the actual
runtime; use no Docker/Singularity profile for this entrypoint yet.

## Run one slide

```bash
python3 bin/bora_contextual_fullslide.py identity \
  --bora-source /absolute/reviewed/Bora

nextflow run bora_contextual_fullslide.nf \
  --bora_contextual_manifest /absolute/slide/bora-contextual.json \
  --bora_contextual_source /absolute/reviewed/Bora \
  --bora_contextual_source_sha256 HASH_FROM_IDENTITY \
  --bora_contextual_output /absolute/new/CellPhenotyper_slide_result
```

This uses two CPUs and a local CPU Python environment. Its default application
deadline is 14,400 seconds and its declared disk budget is 32 GiB. Set
`--bora_contextual_deadline_seconds` and
`--bora_contextual_disk_budget_bytes` explicitly for an admitted larger slide.
The caller must still enforce an outer bounded lifetime and terminal cleanup
when running on a remote host. UCT GPU work remains paused by coordinator
policy. The entrypoint never starts a GPU job or retries a failed result.
Use a fresh output folder for each corrected run; failed folders retain
`RUN_FAILED.json` and their completed evidence.

## Products and acceptance

The action scores one complete, pooled probability lattice across encoder
patch seams. Every native core is projected at **absolute image pixel
centres**, refined against native H&E with finite context, and written as
compact uint8 labels with overlapping halos. Bora's global assembler requires exact
adjacent-halo agreement and one complete source-frame mask. It then creates
the authoritative cleaned mask and global exact GeoJSON, removing parts and
holes strictly below **1,000 µm²**. Finally Bora chooses a guarded whole
boundary simplification that reduces stair-step segments while preserving
component identities and passing class-IoU, displacement and separate
lost/gained-support checks. No macroregion polygons are independently fitted.

The result folder contains `token_scores/three_group_probability.npy`,
`model/FIT_COMPLETE.json` when fitted there, `native_shards/`,
`assembly/`, `contours/global_vectors/exact/`, and
`contours/tissue_annotations/BORA_TISSUE_ANNOTATIONS.geojson`. The new
`experiment.cellphenotyper.json` imports the latter as an ordinary editable
`tissue_annotations` layer, with `predicted_not_expert_annotation: true` and
`training_eligible: false`. Existing base-project assets are referenced by
rebased relative paths, not copied. `RUN_COMPLETE.json` is written only after
the project validates. Keep the original project and exact mask for audit.
Open the project from the canonical `--bora_contextual_output` folder. The
Nextflow task work directory contains only a completion receipt; it does not
publish a detached copy of the project with broken relative asset paths.

The PM016 case that motivated this method produced 49 components and 83.1%
diagonal segments. Those counts are **not** universal acceptance thresholds.
The small integration test uses a synthetic registered image; it tests
identity, same-slide fitting, native assembly, guarded geometry and project
import. It does not establish biological accuracy on another specimen.
