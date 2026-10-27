# H&E-refined semantic Bora project continuation

## Run as a CellPhenotyper workflow

The selectable Nextflow entrypoint is `bora_refined_semantic_fullslide.nf`.
First finish the complete three-scale Virchow2 inventory, materialize its
registered scale-1 spatial tokens with
`bin/materialize_bora_semantic_tokens.py`, and fit/evaluate the local tissue
classifier against **independently marked train and holdout ROIs** with
`bin/bora_semantic_roi_intake.py`. Keep the two evaluated classes, source
image, token inventory, and dual-MPP registration bound to the same slide.
The ordinary `main.nf` DAG does not yet create that complete source-bound
inventory or expert evaluation, so it rejects `bora_refined_*` parameters
instead of silently using its older per-window KODAMA-group implementation.
That older method requires its explicit `legacy_per_window_kodama_mask_v1`
selector and does not produce the recovered semantic H&E-refined result.

Copy `params/bora_refined_semantic_fullslide.example.yml`, replace every
absolute placeholder with the verified slide artifacts, then run `identity`
in the exact CPU Python environment to fill its two SHA-256 pins:

```bash
python3 bin/bora_refined_semantic_fullslide_action.py identity \
  --bora-source /absolute/path/to/reviewed-bora
nextflow run bora_refined_semantic_fullslide.nf \
  -params-file /absolute/path/to/bora-refined-slide.yml
```

The new workflow admits one slide and one new output folder. Its sole process
calls the source-bound Python controller below, with no retries, GPU steps,
hard-group fallback, or automatic promotion to ground truth. It exports a
review-only `experiment.cellphenotyper.json` and `RUN_COMPLETE.json` only
after the saved independent-expert same-slide holdout gates, model/source
hashes, calibrated registration, complete token field, native H&E,
global-seam checks and project validation pass. The output remains
**exploratory** because same-slide holdout is not independent-patient
validation. A failed run keeps `RUN_FAILED.json` and cannot resume or publish
the project. No PM016 prediction is qualified merely by selecting this route:
its tissue-specific expert labels and passing evaluation are still missing.

The holdout gate recomputes each class's support, precision, recall and IoU,
the macro IoU, and all pass/fail gates from the integer confusion matrix. It
checks retained train/holdout polygon and ROI counts against the registered
token grid and frozen expert annotation, then binds selected training counts
to the saved model metadata. Missing, stale, nonfinite or contradictory
metrics fail before project publication. This verifies internal consistency
and source lineage; an independently drawn annotation and independent-patient
evaluation remain the scientific evidence for tissue identity.

The Nextflow process runs in a pinned local CPU Python environment.
The identity receipt pins scikit-learn's logistic solver module, tifffile,
threadpoolctl and joblib in addition to OpenCV and the base numeric/image
stack; obtain new identity hashes after changing any of these dependencies.
The `-stub-run` mode emits deliberately invalid stub receipts and cannot be
used as scientific evidence. On Chiamaka, a new run requires the exact
authorized scope and an independent outer lifetime/owned-descendant cleanup
controller; Nextflow's process time limit alone does not prove cleanup.

Use `bin/bora_refined_semantic_fullslide_action.py` for one CPU-only slide after
Virchow2 extraction and a complete three-scale CellPhenotyper token
materialization. The independent expert ROI intake must have produced a passing
same-slide holdout `EVALUATION.json` and its frozen model. A colon-trained model
transferred to another tissue does not satisfy this gate.

The action uses the verified materialization outputs directly. It scores one
global semantic Virchow2 token field, projects native probability shards,
refines each against the registered native H&E, publishes one source-evidence
chain and one refined global mask/GeoJSON, then exports a new
`experiment.cellphenotyper.json`. The project contains a locked **Predicted by
Bora · exploratory** review layer. It never writes `tissue_annotations`, edits
the input project, runs a GPU, or selects the old hard-group algorithm.

The default H&E boundary parameters now match the recovered 4090 × 4090 colon
result: uncertainty margin 0.30, maximum search 12 µm, H&E blur 0.5 µm, edge
sigma 0.12, smoothness 1.5 and four iterations. The controller records the
complete spec in `PREFLIGHT.json`; an explicit `--refinement-spec` overrides it.
These settings do not repair an incorrect semantic probability field. They
must be paired with a locally validated token-level model and checked against
the native image before a PM016 annotation is accepted.

All products are contained beneath one new slide result folder:

| Path inside result folder | Purpose |
|---|---|
| `PREFLIGHT.json`, `PROGRESS.json` | Frozen source/runtime/input identity and most recent atomic checkpoint |
| `01_inputs/support.npy` | Registered native tissue support |
| `02_semantic_token/` | One global semantic token probability field |
| `03_native_source_shards/`, `04_native_he_refined_shards/` | Bounded, paired source and H&E-refined native shards |
| `05_source_global_evidence/`, `06_refined_global/` | Whole-frame source proof and final refined mask/GeoJSON |
| `experiment.cellphenotyper.json`, `RUN_COMPLETE.json` | Verified viewer project and accepted completion |
| `RUN_FAILED.json` | Latched failure; no successor shard, retry, or accepted project |

First run `identity` **in the exact CPU Python environment** to obtain
`bora_source_sha256` and `runtime_sha256`. The runtime identity includes the
interpreter, core numeric/image dependencies, and the OpenCV native binary.
Supply those hashes verbatim to `run`:

```bash
python bin/bora_refined_semantic_fullslide_action.py identity \
  --bora-source /absolute/path/to/staged-bora

python bin/bora_refined_semantic_fullslide_action.py run \
  --roi-spec /absolute/path/to/roi-intake-spec.json \
  --evaluation-folder /absolute/path/to/passing-evaluation \
  --materialization-folder /absolute/path/to/materialized-tokens \
  --registration /absolute/path/to/dual-mpp-registration.json \
  --registered-support-tiff /absolute/path/to/native-grandqc.tif \
  --base-project /absolute/path/to/existing/experiment.cellphenotyper.json \
  --bora-source /absolute/path/to/staged-bora \
  --expected-bora-sha256 SHA256_FROM_IDENTITY \
  --expected-runtime-sha256 SHA256_FROM_IDENTITY \
  --output /absolute/path/to/new-slide-result \
  --compute-device cpu \
  --core-pixels 2048 --source-halo-pixels 96 \
  --output-halo-pixels 3 \
  --memory-budget-bytes 3221225472 \
  --disk-budget-bytes 68719476736 \
  --deadline-seconds 86400
```

The source halo must cover the output halo plus the physical H&E search,
Gaussian-blur, and smoothing context at the slide's native µm/pixel. Admission
computes that physical context and rejects a too-small explicit halo. The
96-pixel default admits the recovered profile at 0.25 µm/pixel; finer native
pixel spacings may require a larger explicit halo. The historical integration checkout at `/absolute/path/to/reviewed/Bora`
must contain the optional `wsi_semantic_*` producer modules and recovered
crop/window refinement APIs. These are distinct historical source APIs; the
accepted contextual Bora v0.1 package does not supply them. The current
[contextual three-group standard](BORA_CONTEXTUAL_THREE_GROUP.md) has a separate
source-bound route and frozen PLS tissue-name step. An archived 4090 × 4090
colon replay of the historical integration matched its saved cleaned mask at
every pixel; that repeatability result is not a current tissue-accuracy claim. Run `identity` again after any source change: previous source
hash pins are stale. This integration does not establish PM016 tissue accuracy;
the required local holdout evaluation remains a separate admission gate.
Admission
rejects source windows above 16,777,216 pixels, refined windows above
20,000,000 pixels or 8192 pixels per side, more than 4096 cores, a native
frame above one billion pixels, and insufficient declared or free disk/RAM.
The controller checks a monotonic deadline before each bounded stage. Native
library calls themselves do not have a proven hard interrupt; an overrun or
uncertain external termination must be treated as failed, with the folder
preserved. **Before any Chiamaka execution, wrap the whole process/container in
an independently enforced outer timeout with bounded owned-process cleanup,
and obtain the exact Chiamaka execution authorization.** The controller's
`--deadline-seconds` alone is not a hard wall-time guarantee. A failed folder
cannot be resumed by this command. Use a new output
path only after examining the failure and establishing a new authorized scope.

This is a local executable contract, not evidence of PM016 biological
accuracy or a Chiamaka run. H&E refinement changes boundary placement but
does not create expert tissue labels. Independent-patient validation and expert
review remain necessary before any prediction is promoted beyond exploratory
review.
