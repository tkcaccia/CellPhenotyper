# Optional multiscale tissue hierarchy

This experimental route discovers subdomains **inside unchanged broad parent
domains**. It does not change tissue K=2, KODAMA `ncomp=50`, the canonical cell
profiles, or upstream tissue support. No expert GeoJSON is consumed. There is
no claim of improved biological accuracy without a held-out evaluation.

## Inputs and execution

Enable `tissue_hierarchy_enable` only with the `grid` or `both` UNI-2 route.
Provide `tissue_hierarchy_model_snapshot`, an existing local UNI2-h snapshot
containing `config.json` and a compatible checkpoint. The default search order
is `model.safetensors`, then `pytorch_model.bin`; the filename may be explicitly
selected with `tissue_hierarchy_weights_filename`. No model is downloaded.
The input files themselves are staged separately to support snapshots whose
checkpoint is a symlink to a shared cache blob.

The workflow takes the native analysis crop, its grid objects/metadata,
crop-to-slide shift, passed original-resolution report, cropped tissue support,
and matching canonical parent-domain and categorical-uncertainty rasters.
Geometry disagreements fail; an old uncalibrated grid is not relabelled with a
new MPP. Parent labels outside the supplied support also fail. Coarser support
uses the same floor-index nearest-neighbor mapping as the refinement stage,
with no tissue filling or parent-mask edits.

`tissue_hierarchy_local_field_um=56` and
`tissue_hierarchy_context_field_um=224` are experimental defaults. Each field is
a separate source-image crop and encoder forward, resized directly to 224 model
pixels. The larger field is genuine wider context, not a renamed token subset.
This downsampling changes effective model MPP and therefore the model's input
distribution; the best scales are not established by the implementation tests.
Incomplete crop coverage abstains by default (`minimum_field_coverage=1`).

Both feature blocks are retained separately in keyed float32 NPY bundles.
Their manifests bind exact image, grid, shift, calibration, checkpoint/config,
preprocessing and field definitions; complete reuse also verifies array and row
checksums. The process uses deep input caching and code fingerprints. The
producer fingerprints sources before parsing or image reads and checks them
again before each field completion and the final receipt. A changed image,
grid, calibration, model, or producer dependency prevents completion; incomplete
arrays are preserved but are not reusable completed fields. The final receipt
distinguishes model-derived features from whether this invocation actually
loaded the encoder (`encoder_loaded_this_invocation`), including exact reuse.
Repeated source hashing adds I/O; its whole-slide cost is not yet benchmarked.
The default device is explicitly `cuda`, with no silent fallback; choose `cpu` or
`mps` deliberately where appropriate. No extra GPU work is enabled by default.

## Discovery and uncertainty

Within each parent, the two feature blocks are standardized separately,
optionally reduced in numerical PCA space, normalized by total variance and
weighted explicitly. Clustering never uses a two-dimensional display UMAP.
The default CLI and the optional Nextflow route now fit native KODAMA corrected
graphs separately for local, context and combined features **within each fixed
parent**. Leiden partitions those graphs at resolutions 0.1, 0.2, 0.3, 0.5, 0.75,
1, 1.5 and 2. No coordinate, expert polygon or target cluster count enters the
native fit. These are feature-similarity edges, not gap-constrained physical
neighbour edges. Region rasterization remains restricted to unchanged parents.

`kodama_ncomp=50` is passed to `KODAMA.matrix`, separately from the per-block
PCA cap. The current native classifier is kNN: this parameter is **not a PLS
rank in that classifier**, as recorded in the graph receipt. Native M, Tcycle
and neighbours have separate hierarchy settings (100, 20 and 100 by default).
Actual task CPUs are passed explicitly. The required native graph-handle API
must already be installed; an incompatible CRAN API fails. An optional
`tissue_hierarchy_kodama_r_library` names an explicit task-visible library;
there is no automatic installation or host-library discovery.
Standalone discovery normally finds `Rscript` on PATH; an explicitly set
`CELLPHENOTYPER_HIERARCHY_RSCRIPT` must be an absolute executable and never
falls back if invalid. Native R does not inherit the atlas Python library path.

Native interpreter/package bytes are verified during execution but are not yet
bound into the Nextflow scheduler's pre-execution cache key. Consequently this
optional discovery stage uses `cache false`; its downstream derivatives may also
rerun. Upstream local/context feature extraction retains deep caching. This is a
deliberate runtime-identity limitation and speed cost, not a completed resumable
native-runtime design. Existing containers still need the updated runtime
overlay and a successful native functional preflight; no new image build is
claimed by local source tests.

Automatic selection considers actual graph partitions with 2..`max_k` groups,
mean repeated-seed ARI at least 0.75, mean graph affinity margin at least 0.1,
and smallest group at least `max(2,min_observations/2)` rounded down. It maximizes
`mean_seed_ARI * max(mean_affinity_margin,0) - 0.01*K`. Pipeline `fixed_k=0`
selects automatically; a positive setting filters candidates to that actual
**subdomain** count. It never forces a split or centroid merge; if no graph
partition has that count, the parent remains unresolved.

The base clustering seed and `repeats` additional seeds are evaluated on each
frozen graph. Per-observation agreement uses one-to-one label matching, including
unequal community counts. Local/context agreement includes both fields over all
these seeds; a constant field provides no vote. These are conditional graph
stability checks, not repeated KODAMA graph fits or independent biological tests.
The raw-data native graph builder can vary with parallel thread scheduling, so
the requested seed alone is not an exact reproducibility guarantee. Each actual
graph is retained and contributes to the realized hierarchy identity.

`fit_limit` caps normalization/PCA training observations and native landmarks,
not the graph population: all complete eligible observations within the parent
enter fitting. Reduced matrices are resident per parent; the graph is sparse
and no dense all-pairs distance matrix is constructed. Whole-slide memory and
runtime still require measurement. The previous k-means implementation remains
only as the standalone CLI's explicit `--discovery-method legacy_kmeans`
comparison; Nextflow does not silently fall back to it.

Parent uncertainty code 0 is eligible for accepted subdomains. Codes 1–254
remain upstream uncertain/inferred assignments: their original categorical
codes are copied unchanged, and those pixels cannot receive an accepted
subdomain. This includes growth-only assignments; acellular or sparsely
observed tissue is retained in the parent rather than silently classified.

| Hierarchy status | Meaning |
|---|---|
| 0 | Parent background |
| 1 | Accepted exploratory subdomain |
| 2 | Parent tissue without a grid observation |
| 3 | Mixed-parent core |
| 4 | Missing feature block |
| 5 | Insufficient eligible observations |
| 6 | No supported stable subdivision |
| 7 | Seed instability |
| 8 | Local/context disagreement |
| 9 | Ambiguous centroid assignment (explicit legacy comparison only) |
| 10 | Parent assignment uncertain/inferred |
| 11 | Native graph isolate |
| 12 | Ambiguous graph-affinity assignment |

Positive parent with subdomain/region 0 means unresolved parent tissue, not
background. Regions are four-connected components: neither tissue gaps,
diagonal contact, nor a parent boundary joins two regions. Stability scores
and graph affinity margins are descriptive, not calibrated probabilities.
Native rows carry degree, own-community affinity fraction and own-versus-
strongest-other affinity margin; they are not relabelled centroid distances.

## Outputs and reference mapping

The producer publishes `22_tissue_hierarchy/<sample>/features/` with `local/`,
`context/`, `embedding_metadata.json`, `hierarchy_feature_request.json`, and
`hierarchy_features_summary.json`. Model provenance identifies the exact local
config/checkpoint bytes; no license status is invented from a filename.

Discovery publishes `22_tissue_hierarchy/<sample>/<parent_variant>/`:

- `parent_domains.ome.tif` and `parent_uncertainty.ome.tif`: byte-identical copies.
- `subdomain_mask.ome.tif`, `region_mask.ome.tif`, `hierarchy_status.ome.tif`.
- `grid_subdomains.csv`: source identities, physical coordinates, assignments
  and descriptive uncertainty metrics.
- `region_profiles/`: raw local/context feature means weighted by the assigned
  grid-core tissue area, region identities, real representative grid sites,
  and a versioned `tissue_region` manifest for atlas use.
- `kodama_graphs/`: exact per-parent native input matrices/row identities,
  frozen producer code, portable local/context/combined graphs, partitions and
  source/runtime/checkpoint-independent execution receipts. Inputs, producer
  code and outputs are checked again after consumption and raster/profile export.
- `hierarchy_summary.json`: input lineage, output filename→SHA256 mapping,
  geometry, support checks, feature definitions, discovery diagnostics and
  categorical status counts. The summary does not hash itself.

The optional region-reference mapping is a separate output at
`23_region_reference_mapping/<sample>/<parent_variant>/reference_assignments.csv`.
It compares with an explicitly supplied frozen region atlas; it never trains a
reference from the query or overwrites discovery labels. Feature/model/scale
definitions must match. Missing features, insufficient reference examples,
ambiguity and out-of-reference observations remain `unknown`. Unreviewed
subdomain labels are sample-scoped, not assumed to be shared biological types.

## Verification boundary

When cell profiles are also enabled, stage 24 links canonical nuclei and optional
perinuclear rings to this hierarchy using exact raster overlaps. The resulting
SpatialData export retains the five hierarchy rasters, region features and
many-to-many cell membership. See [cell-to-tissue linkage](CELL_TISSUE_LINKS.md).
The hierarchy and cell profiles must use the same refined primary parent variant.
Grid status 10 may cover an entire mixed-uncertainty core; the original parent
uncertainty remains pixel-specific. These two fractions are not interchangeable.
Linking and export require a nonempty categorical status legend declaring every
observed code. Native graph isolates (11) and ambiguous graph assignments (12)
are retained as unresolved parent tissue; they never become accepted regions
or cause the canonical cells overlapping them to be dropped. Unsupported codes
above 12, missing legends, and undeclared observed codes fail validation.

Synthetic tests cover known partitions, missing observations, parent/support
holes, categorical uncertainty, image lineage, immutable copies, binary feature
reuse, and actual Nextflow stub wiring. A separate actual Nextflow mapping test
uses independent synthetic reference/query profiles. These are engineering
checks, not an evaluation on independent human tissue. Real checkpoint loading,
full-slide runtime/memory, scale selection and blinded subdomain/atlas accuracy
remain to be validated before scientific adoption.
