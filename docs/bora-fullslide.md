# Historical hard-group Bora continuation

This page documents the older, explicitly selected `bora-global-hard-group-v1`
research route. It is not the accepted smooth PM016 standard. Use
[the contextual three-group route](BORA_CONTEXTUAL_THREE_GROUP.md) for that
standard and its separate frozen PLS tissue-name step. Historical KODAMA group
counts below must not be copied as defaults for a new slide.

`bora_orchestrate_fullslide.nf` is a CPU-only continuation for
one slide whose Virchow2 GPU inventory and KODAMA fit are **already complete**.
It builds a `bora-wsi-global-v1` manifest from exact on-disk hashes, asks Bora
to admit every frozen token case and lifecycle receipt, then runs the
whole-field CPU chain. An optional third CPU task attaches its exact GeoJSON
as an exploratory, review-only boundary artifact in a new CellPhenotyper project. No process
in this entry point submits, retries or queues a GPU case.

The manifest binds one complete Virchow2 token inventory at scales 1, 2 and
4; a 0.5 µm/pixel, nonoverlapping 224-pixel tile grid; the working image and
classifier; and the exact CSV and receipt from a **direct** KODAMA group fit.
The superseded PM016 trial used `required_group_count: 2`; the accepted current
PM016 result uses three groups in the separate contextual route. Declare and
validate the group count for every slide rather than hard-coding either count.
Post hoc group merging is rejected. The builder derives plan, inventory,
image, model, classifier, source and KODAMA hashes; it accepts no handwritten
hash claims. Bora's own admission is authoritative and must pass before
`BUILD_COMPLETE.json` is published.

Copy `params/bora_global_inputs.example.json` and
`params/bora_orchestrate_fullslide.example.yml` to local files. Fill in
absolute paths and choose new manifest and Bora output directories. After a
separately qualified CPU SIF exists, run one slide with:

```bash
nextflow run bora_orchestrate_fullslide.nf -profile singularity \
  -params-file params/bora_orchestrate_fullslide.local.yml --compute_device cpu
```

The Bora task invokes `bora wsi-global --manifest ... --output ...` with
`PYTHONNOUSERSITE=1`, two CPUs, at most 12 GiB of configured RAM, no GPU and
no retry. It verifies `ADMISSION.json`, `CHAIN_COMPLETE.json`, RGB, semantic
mask, the source-bound `semantic/ASSEMBLY_COMPLETE.json` hard-group receipt,
exact raster, exact/review GeoJSON, and their hashes. This route's algorithm
identity is **`bora-global-hard-group-v1`**: one predicted tissue identity per
complete anonymous Bora-AF group, projected onto the native image. The exact
project artifact uses role `bora_exploratory_boundary_review`, is labelled
“Bora anonymous-group boundaries (exploratory review; exact)”, and carries
`review_only: true`, the exact GeoJSON SHA-256, matching image/frame identity,
the algorithm ID, anonymous-group prediction unit and receipt digests. It is
not a `tissue_annotations` artifact or a validated tumor/stroma annotation.
It is not the opt-in colon H&E-guided boundary method. Passing
`--requested-algorithm bora-colon-he-guided-v1` to the action or project-export
script fails before publication because no source-bound whole-slide completion
contract for that method exists. The generalized algorithm remains the default
for this existing Nextflow entry point; request it explicitly when invoking
the action script directly. A visually smoother review GeoJSON does not alter
the semantic algorithm or establish H&E refinement.

Each output directory is durable. A failed directory remains for diagnosis and
cannot be overwritten. Optional project export runs only after a genuine
`CHAIN_COMPLETE.json`, validates the existing project, and requires exactly
one tissue matching the Bora image SHA and frame ID.

`bora_fullslide.nf` is a lower-level single-task entry point for an already
admitted `bora-wsi-global-v1` manifest. `-preview` and `-stub-run` validate
the DAG only; their stub receipts say `complete:false` and do not create
durable scientific output.

**Release gate:** `main.nf` still emits single-scale Virchow2 embeddings and
does not emit the direct-fit KODAMA receipt required by generalized Bora. Its
`REFINE_VIRCHOW2_BORA` is an older per-window GPU wrapper. Passing global
manifest parameters to `main.nf` fails explicitly to prevent a misleading
route. The three-scale producer and direct KODAMA fit must be joined under a
single source-bound controller before claiming a complete GPU-to-project DAG.
The current GPU SIF uses CPython 3.11 and cannot load pinned CPython 3.10
fastPLS; a separate CPU SIF with the exact runtime lock is required. Current
Generalized Bora now has a versioned frozen-PM016 verifier, but the authentic
inventory, KODAMA inputs and separate CPU SIF must still pass together.
Local stub tests do not certify full-slide Chiamaka execution.
