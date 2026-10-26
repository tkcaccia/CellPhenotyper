# Full-slide CellViT++ and Bora project workflow

`cellvit_fullslide.nf` is the supported full-slide cell-identification entry
point. It calls official CellViT++ `process_wsi` on sequential bounded image
regions, using nonoverlapping 1920-pixel ownership cores and 960-pixel context
halos. The halo supplies context; a cell is retained only by its core, so
neighboring cores cannot duplicate it. All planned cores must have checked
completion records, including explicit records for empty clean-tissue cores.

This is separate from the legacy ROI/consensus route in `main.nf`. Do not
interpret that older route as full-slide completion. The accepted tissue
continuation is documented in [BORA_CONTEXTUAL_THREE_GROUP.md](BORA_CONTEXTUAL_THREE_GROUP.md).

## Inputs and runtime

Keep a calibrated 0.25 µm/pixel RGB TIFF, its full-image zero-origin shift
JSON, the full-image GrandQC clean-tissue mask and the eight classifier
checkpoints in persistent paths. The full-slide route requires the SAM
backbone and AMP. It runs segmentation once per core and applies all eight
heads to the retained cell vectors: PanNuke, CoNSeP, Lizard, NuCLS main,
NuCLS super, Ocelot, MIDOG and PanopTILs. Their labels are independent
ontologies; they are never voted into a single replacement label.

The Linux host runs the lifecycle supervisor, `systemd --user`/cgroup checks,
disk admission and GPU ownership checks. Official CellViT inference and
per-core packing execute inside the pinned Singularity SIF. Supply the exact
SIF SHA-256, in-image Python and inference-command paths. Historical images
missing pyvips may use a separately hashed immutable supplemental dependency
directory; new runtime recipes install pyvips directly. Do not modify an
active campaign's source, SIF, checkpoint or dependency directory. Runtime
recipe changes in Git are instructions for a future build, not evidence that
a new image was built or GPU-qualified. See [the runtime maintainer guide](../singularity/README.md).

Copy `params/cellvit_fullslide.example.yml` to a local file, set all paths,
the assigned GPU UUID and `cellvit_fullslide_expected_shards` to the complete
plan count. First use `preflight` and `plan` if that count is not known.
The PM016 validation case has 104 planned cores and completed all 104, with
151,257 cells and all eight heads. This is execution/coverage evidence on
one slide, not biological validation of every cell label.

## Run

Invoke this host entry point **without** a Docker/Singularity Nextflow
profile: it launches the selected SIF itself and must observe the actual host
workload. No new image is pulled implicitly.

```bash
nextflow run cellvit_fullslide.nf \
  -params-file /absolute/params/cellvit-fullslide.local.yml
```

Set `cellvit_fullslide_action: run` for the complete remaining slide.
`cellvit_fullslide_max_new_shards: 1` can bound a qualification to one fresh
core; `0` processes the full declared plan. A completion receipt is published
only when every planned core verifies. `preflight`, `plan`, `adopt`, `pack`,
`compact` and `finalize` are separate CPU actions; `adopt` requires the exact
source checkpoint and independently verified terminal cleanup. The legacy
standalone `shard` action is disabled: `run` owns each bounded GPU lifecycle.

The default per-core envelope is one GPU, two CPUs, 12 GiB RAM, a
7,200-second application timeout and finite shutdown checks. Ray receives
8 GiB within that envelope; the remaining memory covers model/runtime
storage. Disk admission reserves 40 GiB plus a 4 GiB core budget. The host
must provide a working user systemd manager, readable cgroup v2 controls,
Singularity and NVIDIA context telemetry. This is a single-GPU Linux-host
implementation; it does not qualify scheduler or multi-GPU behavior.

Every case checks resource admission before execution and two terminal
observations after execution. A failure latches the bundle and blocks its
successors. Preserve the failed bundle; after cleanup has been verified,
make any correction in a fresh versioned campaign. Do not retry an existing
failed core in place. UCT work remains paused under coordinator policy.

## Storage and viewer handoff

Each core writes compressed classification tables and retains every cell's
1,280-dimensional feature vector. CPU packing creates a checked float32 gzip
matrix with its ID index; lossless archives retain scientific records and
hash receipts without thousands of separate top-level output trees. The
final manifest includes all cores, one SQLite viewport index, a feature
pointer and `experiment.cellphenotyper.json`. The archive consumer can reopen
this project after relocation, using checked relative references.

`CELLVIT_PROJECT_COMPLETE.json` records the
`cellphenotyper-cellvit-fullslide-complete-v2` schema, equal planned/complete
counts, all eight heads, retained vectors, exact source-image SHA-256 and
project/index hashes. Partial pilot indexes must not be presented as complete.
The viewer reads the SQLite index by viewport, rather than materializing a
full-slide cell table. The index includes cell contours when provided by the
source segmentation; centroid-only cells remain explicit.

Attach the accepted smooth Bora GeoJSON using
`bin/combine_cellvit_bora_project.py` or the four matching
`cellvit_fullslide_bora_*`/combined-output settings in the example parameter
file. Native image bytes, frame, dimensions and MPP must agree exactly. Open
the resulting combined CellPhenotyper JSON in wsiTools. KODAMA group IDs are
anonymous domains; the separate frozen Neighbor-context PLS identity stage
must run before those regions can carry predicted tissue names. Smooth
geometry alone does not establish tissue identity.
