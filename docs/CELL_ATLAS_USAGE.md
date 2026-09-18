# Linked cell profiles and spatial atlas

These opt-in stages extend, rather than replace, the existing tissue-domain
route. They are research outputs. Technical identity checks and successful
exports do not establish cell-type or measured-marker accuracy.

## Full workflow

Set these parameters in the normal pipeline parameter file:

```yaml
cell_profiles_enable: true
cell_profiles_uni2_enable: true
cell_profiles_markers_enable: true
cell_profiles_domain_source: refined
cell_neighborhood_radii_um: '25,50,100'
cell_neighborhood_feature_groups: available
cell_neighborhood_feature_weights: '{}'
cell_neighborhood_feature_storage: table
cell_neighborhood_row_batch_size: 4096
cell_neighborhood_column_batch_size: 64
cell_niche_fit_limit: 20000
cell_neighborhood_support_mode: provided
cell_niche_fixed_k: 0
cellvit_export_embeddings: false
cell_profiles_spatialdata: false
cell_reference_atlas: null
cell_measured_assays: null
cohort_niches_enable: false
cohort_niches_bundle: null
```

With `uni2_sampling_mode: grid`, cell-profile UNI-2 extraction adds cell-centred
local/context features without a second cell-based tissue clustering/refinement
branch. `both` reuses its existing auxiliary cell features. Local and contextual
vectors remain separate, explicitly contextual H&E representations; neither is
an isolated-cell protein measurement.

CellViT features require the CellViT/consensus detector route. They are optional
mean token representations within detected nuclear bounding boxes, aligned via
the source CellViT ID. They are not pure nuclear-mask pooled features.

The profile join fails on duplicate, foreign or missing required sample IDs.
Every canonical cell is retained; absent modalities remain missing. Optional
feature blocks do not change the canonical population. No expert evaluation
annotation is consulted by these stages.

### Own-cell features, physical texture and shared niches

The default `cell_neighborhood_feature_groups: available` selects every feature
block actually present in the canonical profile; absent modalities are not
fabricated. An explicit comma-separated list restricts the selection, and an
empty string requests only neighbourhood statistics. Each selected block has
an `own:<block>` group containing the cell's own values and a `<block>` group
containing finite-neighbour means at the requested radii. Both are retained in
the default wide table (or optional array store below), including raw NaNs and coverage. Representation 2.0.0 keeps missingness
indicators even when observed values are constant. Groups are standardized and
dimension-balanced separately. For example, use
`cell_neighborhood_feature_weights: '{"own:morphology":2,"morphology":1}'`
to emphasize the cell's geometry over its neighbourhood's geometry. Explicitly
named weights require their groups to be available.

Morphology 2.0.0 uses normalized RGB optical density with fixed quantization.
Equivalent full-range uint8, uint16 and float-[0,1] intensity encodings share the
same definition; nonstandard white calibration is separately recorded. Texture
uses a target axial lag of 0.5 micrometres (standalone
`profile_cell_morphology.py --texture-lag-um`), with rounded pixel offsets and
effective physical distances recorded. Sampling and grid-dependent geometry
are not claimed to be invariant across arbitrary MPPs. Keep
`cell_morphology.csv`, `morphology_summary.json` and completion-last
`morphology_completion.json` together. The profile consumer verifies exact
image/mask/object/transform/calibration identities and actual producer settings.
Legacy tables remain readable but are not reference-compatible.

With at least two specimens, `cohort_niches_enable: true` adds one shared
exploratory fit after specimen-specific profile assembly/hierarchy linking.
`cohort_niches_max_k` (8), `cohort_niches_fixed_k` (0=automatic),
`cohort_niches_seed` (17), `cohort_niches_repeats` (5), and
`cohort_niches_fit_limit` (20000) control this separate discovery. Tissue K=2
is not imposed on niches. The same feature-group weights are used for the
shared fit. Missing source blocks remain NaN; genuinely absent phenotype
categories become zero only where neighbours exist. Incompatible/unverified
feature definitions fail pooling. Informative phenotype composition enters the
shared fit only with matching verified taxonomy definitions; otherwise it is
explicitly omitted from the fit, not erased from the source profiles. An
explicit composition weight in that case fails rather than being ignored.

The shared model records common scaling, centroids, source fingerprints,
runtime, feature axes and stability diagnostics. Per-slide discovery labels
and all physical graphs remain untouched; there are no cross-specimen physical
edges. This is not frozen-reference assignment, batch correction or a validated
biological taxonomy. Sampling is per-cell, so large specimens can influence
the shared result more strongly. Inspect missingness/site effects before
interpreting niches. The dense-representation preflight rejects estimates over
`cohort_niches_max_working_mb` (1024 by default); it does not silently discard
features or cells. Profile assembly reserves half its allocated task memory for
that estimate, leaving the rest for graph/raster/interpreter overhead.

### Optional array-backed neighbourhoods

`cell_neighborhood_feature_storage: arrays` replaces the wide own-cell/neighbor
numeric columns with an explicitly indexed store. The canonical Parquet/CSV
table retains IDs, coordinates, support, scalar statistics, coverage and niche
results. `neighborhood_features/feature_store.json` binds the exact
`feature_rows.csv` identity/order, each logical group/axis, and every payload's
hash/shape/dtype. Own-cell groups reference the unchanged `feature_blocks/*.npy`
matrices; each radius adds only its finite-neighbour-mean float64 matrix.
These intermediate payloads are memory-mapped NPY, **not compressed/chunked
Zarr**. SpatialData exports the groups as separately chunked `cells.obsm`
arrays with exact axes and aggregation definitions, without widening `obs`.
Keep the complete profile directory together when moving it.

The array route uses the same support geometry, structural zero-distance edges,
group weights, missingness axes and sorted-cell seeded clustering order.
Neighbour aggregation uses `cell_neighborhood_row_batch_size` (4096) and
`cell_neighborhood_column_batch_size` (64); full-population scaling and
prediction use row batches. A disk-backed temporary scaled matrix contains
every eligible cell and retained numeric/missingness axis. The seeded training
subset remains dense, capped by `cell_niche_fit_limit` (20000); this cap already
existed internally and is now explicit. It does not drop cells from outputs.
The cohort route automatically uses streamed fitting if any source has an
array store, including mixtures with verified legacy-wide profiles. Legacy-only
cohorts keep the dense route; cohort training retains its separate fit limit.

Memory preflights estimate numerical working buffers, not total process RSS.
Sparse graphs, canonical/scalar tables, identity sets and OS file-backed pages
still consume resources; this is not constant-memory or unlimited-cohort
execution. Storage preflight includes neighbour payloads and worst-case scaled
scratch. Budgets fail explicitly without reducing dimensions, training limits
or canonical populations. The fitting scratch is removed by its owning run.

Batched float64 reductions can change rounding versus the dense implementation,
especially for ill-conditioned large-offset features and near-tied solutions.
Regression tolerances are evidence for the tested fixtures, not a universal
bitwise-equivalence guarantee or proof of biological accuracy. `table` remains
the default pending broader real-feature/large-cohort acceptance.

Standalone assembly accepts `--feature-storage arrays --row-batch-size 4096
--column-batch-size 64 --fit-limit 20000`. Programmatic consumers can use
`FeatureColumns(profile_directory).read(row_slice, logical_columns)` from
`bin/neighborhood_feature_io.py`; it returns ordered float64 values including
NaNs without constructing a global wide DataFrame. Its source receipt and
`recheck()` validate the original files. Hierarchy linking copies the store
unchanged, and shared-niche source identities bind its payloads explicitly.

The inspector restores array-backed neighbour fields using single-cell reads;
its original similarity groups do not change. Changed source-file identities
trigger digest verification and, when needed, memory-map refresh. This is
per-request source monitoring, not continuous cryptographic auditing.

The eight atlas Nextflow stages also isolate Python bytecode-cache reads using
a fresh task-private prefix with bytecode writes disabled. This prevents an old
timestamp/size-based `.pyc` from surviving a correctly detected source-code
change. Shared caches are neither deleted nor rewritten. An empty private
directory remains in the task work directory; imports can incur extra startup
time. This runtime safeguard is separate from Nextflow task-cache identity.

An existing-artifact invocation, requiring new representation-2.0.0 profiles:

```bash
python bin/fit_cohort_niches.py --profiles /absolute/specimen_A/cell_profiles \
  /absolute/specimen_B/cell_profiles --outdir /absolute/new_cohort_niches
```

### Attach shared niches without changing canonical profiles

With both `cohort_niches_enable: true` and `cell_profiles_spatialdata: true`,
the workflow waits for the single shared fit and attaches the corresponding
rows to each specimen's new SpatialData export. The fit uses the same final
profile registry as export, including hierarchy linkage when enabled. There
is no Cartesian product of specimen cells or cross-specimen physical graph.

To reuse a completed shared fit, set `cohort_niches_bundle` to its directory
and leave `cohort_niches_enable: false`. This requires SpatialData export and
is mutually exclusive with fitting. The artifact workflow's `existing_profiles`
route preserves the exact profile identity needed for this operation. Each
selected profile must occur in the bundle with the exact original manifest,
table, row inventory, graph and selected feature-array hashes. A bundle fitted
before a later profile/hierarchy change is not silently rebound to that change.
Use a new output location; existing profile directories are never edited.

The portable bundle contains all four files:
`cohort_niche_model.json`, `cohort_niche_assignments.parquet`,
`cohort_niche_summary.json`, and completion-last `cohort_niches_completion.json`.
Keep them together. A consumer needs the explicitly selected specimen's profile
and this bundle, not the other specimens' source directories. It never follows
recorded upstream paths and never imports a fitting, scikit-learn or Torch
runtime. The reader verifies global assignment identities/counts/statuses in
65,536-row batches, then exact source row order, coordinates, support and
neighbourhood eligibility for the selected specimen. Global identity sets still
use memory proportional to cohort size; this is not constant-memory validation.
Array-backed source stores and their referenced payload hashes are also bound
to the shared model and verified without importing the fitting runtime.

Five additional `cells.obs` fields are exported: `cohort_niche_id`,
`cohort_niche_status`, `cohort_niche_centroid_margin`, `cohort_niche_stability`,
and `cohort_niche_model_id`. Original `niche_*`, `reference_*`, measured data,
profiles, embeddings and graphs remain separate. Missing niche IDs remain
nullable integers; single-niche/unassigned margins and stability remain missing.
Full model/summary/completion/source evidence is retained in
`cells.uns['cellphenotyper']['cohort_niches_json']` and package provenance
`cellphenotyper.cohort_niches`. Export checks the source bytes before and after
writing and verifies exact interpretation values/types and metadata on readback.
These checks establish lineage, not biological accuracy or calibrated confidence.

### Source-bound CellViT features

With `cellvit_export_embeddings: true`, stage 03c writes a portable bundle:
`cellvit_embeddings.npy`, `cellvit_embedding_ids.csv`,
`cellvit_embeddings_metadata.json`, `cellvit_raw_population.csv`,
`cellvit_cells.json`, and completion-last `cellvit_embeddings_completion.json`.
Keep these six files together. Raw `cells.pt` is converted with PyTorch's
restricted weights-only loader; downstream profiles and exports read only
numeric arrays, CSV and JSON, never pickle or the recorded upstream paths.

The workflow passes the exact source-resolution report to CellViT. Standalone
`run_cellvitpp.py --export-embeddings` now also requires `--resolution-json`.
The receipt binds image, shift, resolution, optional support mask, native crop
geometry, prepared TIFF identity, physical preprocessing, segmentation checkpoint
and actual configured CLI runtime. Unsupported/non-Python launchers remain
explicitly unverified. Runtime inspection does not import Torch or execute a
learned model. Exact source/package/checkpoint checks add I/O and do not establish
biological accuracy or identical behaviour on every GPU/driver.

Consensus retains `cellvitpp_x_px`, `cellvitpp_y_px` and
`cellvitpp_source_sha256` separately from its fused centroid. Profile attachment
requires the source ID, the detector's own centroid and the normalized source
population to agree; a legitimate fused-centroid displacement is not an error.
All canonical cells remain in order, including cells with no CellViT source,
whose vectors are NaN and whose missingness is explicit. The additive object
columns change the object-table hash for newly generated consensus results;
the existing segmentation-derived cell-UID policy is unchanged. Historical
profiles, IDs and result files are not rewritten.

Reference definitions include physical scale, preprocessing, requested AMP
enforcement, segmentation-checkpoint identity and configured runtime identity.
AMP false means no forced `--enforce_amp`; the checkpoint/runtime may still
enable mixed precision. Taxonomies requiring an additional external classifier
are not reference-compatible without its bound identity. Receipt-free legacy
features remain inspectable but are explicitly unverified and cannot silently
become reference-compatible. A new-format bundle without completion is an error,
not a legacy fallback. Real learned-token extraction and independent validation
remain required before making accuracy claims.

## Outputs

- `19_cell_profiles/<sample>/cell_profiles/`: CSV and Parquet master table,
  `feature_rows.csv`, row-aligned float32 `.npy` feature blocks, sparse `.npz`
  physical-radius graphs, manifest and neighbourhood summary.
- `19_cell_profiles/<sample>/morphology/`: nuclear raster morphology and
  RGB-optical-density texture proxies. Texture is not stain-deconvolved chromatin.
- `25_cohort_niches/cohort_niches/`: optional shared model JSON, canonical-ID
  assignment Parquet, summary and completion receipt. The separate
  `cohort_niche_*` fields do not overwrite per-slide niche IDs. When export is
  enabled, new SpatialData packages receive source-bound additive fields and
  the full shared-model provenance; existing packages are not modified in place.
- `20_spatialdata/<sample>/spatialdata.zarr/` when enabled: native crop image,
  canonical instance labels, tissue polygons, cell/domain tables, embeddings,
  sparse neighbourhood graphs and physical transforms. Export is read back
  through SpatialData/AnnData APIs before success is reported.
- `21_reference_mapping/<sample>/reference_assignments.csv` when a reference
  atlas is supplied: resemblance assignments and explicit unknown reasons,
  separate from the discovered tissue/niche labels.
- `22_tissue_hierarchy/<sample>/`: optional separately encoded physical fields,
  immutable broad parents, subdomains, categorical status/provenance and region
  profiles. See the multiscale configuration below.
- `23_region_reference_mapping/<sample>/<variant>/reference_assignments.csv`:
  optional mapping against a frozen region reference, separate from discovery.
- `24_cell_tissue_links/<sample>/cell_profiles/`: when hierarchy and profiles
  are enabled together, an immutable derived profile plus exact nuclear/ring
  overlaps with parents, subdomains, regions and uncertainty. Downstream cell
  reference mapping and SpatialData use this linked version. Stage 19 remains
  unchanged. See [cell-to-tissue membership](CELL_TISSUE_LINKS.md).

Each new pipeline cell profile has an immutable identity based on sample, the
combined segmentation-table/label-raster hashes and canonical label. Legacy
standalone profiles without a label raster record their weaker identity basis.
Coordinates are both crop/original pixels and original-slide
micrometres. Detector IDs are strings so leading zeros survive.

Neighbourhood graphs are built per specimen, use physical distance and reject
edges crossing excluded tissue pixels on the supplied support raster. Bounded
TIFF windows preserve that raster's pixels for endpoint tests, full line-of-sight
traversal and density denominators; the graph does not use a downsampled proxy.
A native-resolution crop mask can be supplied as the ordinary `support` artifact
without the old 50-million-pixel full-array limit. The optional domain-distance
grid is capped separately and remains an approximation. Its connected-component
and distance-transform arrays can still exceed 1 GiB at the default 50-million-
pixel limit; bounded graph-window I/O is not a constant-memory guarantee for the
entire profile stage. The assembly CLI's `--max-support-pixels` now controls only
that optional analysis grid, not graph or density resolution. Tiny gaps absent from
the supplied raster still cannot be detected: the pipeline's ordinary GrandQC
mask is low resolution and is not a native-gap detector. Isolated cells remain
explicit; a failed stability criterion can leave a single niche or no supported
assignment. Niche count is independent of tissue K.

For standalone `assemble_spatial_cell_profiles.py`, a separately produced native
mask may be declared with `--native-support-mask` and an explicit
`--native-support-coordinates crop_pixels|original_pixels`. Native dimensions
and original-slide crop offsets are validated. This changes only the graph and
density support, not the canonical cell population or historical marker/mask
definitions. The profile manifest records the exact mask hash, pixel geometry
and source frame; it does not certify the biological correctness of that mask.
No native mask is fabricated from domain labels, nuclear labels or an enlarged
GrandQC raster, and expert evaluation annotations must not be used as this
inference input.

### Native image-derived neighbourhood support

`cell_neighborhood_support_mode: provided` is the compatibility default and uses
the supplied raster exactly as before. The experimental `brightfield_native`
mode builds a native crop mask from the H&E image and supplied GrandQC support,
subtracting confidently bright background without adding tissue outside the
original support. It uses no expert annotations, cell labels or tissue-domain
labels to decide where image gaps occur. This is image-only bright-background
subtraction, not a newly trained tissue model or proof that every biological gap
has been detected; faint/transparent tissue and subtle gaps remain limitations.
To accommodate tinted scanner backgrounds, it conservatively estimates the
slide background colour from bright, consistent pixels on a fixed sampling
lattice within upstream exclusions. If there are too few samples or they fail
brightness/consistency checks, it records a nominal-white fallback. The estimate,
sample diagnostics, fallback reason and experimental thresholds are retained in
the provenance manifest; bright artifacts can still bias this adaptation.

Only neighbourhood graphs, support-area density and consequent niche assignments
use this derivative. Canonical cells, nuclear masks, marker compartments and
tissue-domain segmentation are unchanged. A canonical cell newly outside support
is retained, flagged and isolated. Existing marker compartments can consequently
include pixels excluded from this newer neighbourhood support; this is not a
marker-requantification step. Support-dependent density/niche comparisons must
record the support mode and mask identity.

The generated mask and provenance manifest are copied into
`19_cell_profiles/<sample>/cell_profiles/neighborhood_support/`, inside the
published profile bundle, rather than published as a separate tissue-domain
result. The manifest records the image/support/calibration identities and
algorithm settings. Native geometry and provenance are checked during assembly;
those checks certify input identity and sampling, not biological performance.
Domain-boundary distances remain their separately documented bounded-grid
approximation using the original support definition, not the new gap filter.
Historical outputs are not changed by selecting this mode in a
new run.

SpatialData exports include this bundle as two separate native categorical label
elements, `neighborhood_tissue_support` and `neighborhood_tissue_support_reasons`.
These are not cell-instance masks: the cell table remains linked to
`canonical_cells`. The exact producer receipt, interpretation limits and
profile-path-to-element mapping are embedded in the cell table and package
attributes. All native pixels and physical transforms are read back and checked;
the original source paths recorded in the receipt need not be accessible.

See the [real-crop audit](../audits/spatial_atlas_20260904/native_brightfield_support.md)
for removal counts, graph comparisons, runtime costs and outstanding validation.

## UNI2 storage and identity

UNI2 feature extraction can use `uni2_embedding_storage: binary` for both the
cell-profile and grid routes. It preserves explicit observation identities and
feature precision while avoiding large feature-CSV parsing. Existing hierarchy
NPY bundles remain distinct. See `UNI2_BINARY_STORAGE.md` for format, source
verification, reference-metadata migration and measured saved-feature results.
The storage choice does not make grid observations into cell profiles.
The default CSV KODAMA route also rejects duplicate, missing or foreign IDs and
invalid feature values rather than silently reducing the cell/tile population.
Selected families must cover all annotation IDs; unselected modalities remain
optional. CSV source hashes and row mappings are now carried into PCA/KODAMA.

## Physical compartments and marker provenance

`expand_um: 3.0` now produces unchanged nuclear labels, a tissue-constrained
perinuclear ring, and a whole-cell approximation including the nucleus.
`expand_um: -1` explicitly selects legacy pixel expansion. Expanded regions are
not inferred membranes. Crowding, image/tissue truncation and retained nuclear
pixels outside support are recorded in compartment QC and joined into profiles.

GigaTIME storage defaults to all 23 channels in float32. Two background channels
are kept for QC but excluded from default biological-marker feature blocks.
Integrated tables use the blended float32 field with float64 reductions;
actual model arithmetic is recorded separately. Marker restart validates the
ordered schema, checkpoint, stored precision, completion and compartment-mask
identity. Old subset/quantized images are not equivalent restart sources.

Legacy quantification tables remain inspectable, but unverified provenance is
explicit and blocks reference-atlas use of those marker features. Verified
marker feature definitions include checkpoint, order, precision, physical scale
and compartment construction; specimen-specific mask hashes remain provenance,
not cross-slide feature definitions. Never mix measured and predicted scores.
This compatibility check also applies when a predicted marker is used only as
a difference filter in reference similarity search, while morphology defines
distance. Atlas versions record marker-filter contracts separately. Missing or
incompatible marker provenance disables the requested filter, not ordinary
morphology-only search. Old atlases must be rebuilt from compatible verified
profiles before using marker discordance; their existing assignments remain
unchanged. See the input-identity and marker-search audit for verification scope.

New UNI-2 extraction receipts bind the actual encoder state, physical settings,
source inputs and every shard checksum. Profile export verifies these receipts
against the canonical image, labels and passed resolution report. Legacy shards
remain inspectable, but cannot become verified reference features merely because
their IDs, coordinates and model-name metadata look compatible.

The capacity preflight now accounts for channel count, dtype, inference scale,
retained formats/pyramids, scratch arrays and optional profile/export copies.
Its estimates remain planning assumptions, not measured peak-disk guarantees.

## Rebuild from existing artifacts without rerunning models

`cell_profiles.nf` accepts a JSON list. Each record requires `sample_id`, `image`
(native crop), `labels`, `objects`, `shift`, `resolution` (passed report), and
`support`. Optional fields are `markers` (directory or list), `uni2_context` and
`uni2_local` (supply both or neither), `cellvit`, `domains`, `domain_uncertainty`,
and `compartments` (directory containing QC and construction summary).

For SpatialData also supply `tissue_geojson` and an explicit
`tissue_coordinates`: `crop_pixels`, `original_pixels` or `original_um`.
Paths are resolved relative to the JSON file. Outputs must be a new results
location to preserve previous experiments.

To export or reference-map an existing immutable profile, supply
`existing_profiles` instead of `objects`/`support` and rebuilding fields. This
skips profile reconstruction and preserves the exact manifest identity used for
measured-assay registration. SpatialData still requires image, labels, shift,
passed resolution and explicitly framed tissue polygons.

Each record may include `measured_assays`, a list of immutable importer package
directories, and `measured_region_shapes`, a list of self-contained directory
bundles whose `link.json` declares the registered region geometry. These are
validated separately and never merged into predicted feature blocks. Relative
link resources must remain inside the staged bundle. For `main.nf`, use
`cell_measured_assays` to name a JSON list with `sample_id` and these same fields;
unknown specimen IDs and disabled exports fail instead of ignoring attachments.
All packages must match the exact output profile/region registry they reference.
See `MEASURED_ASSAY_IMPORT.md` for the independent-registration contract.

Records may also supply `cell_reference_mapping` and/or
`region_reference_mapping`, each pointing to a `.mapping.json` receipt with its
sibling `.csv` and `.atlas.json` files. All three are staged together. These
attachments require SpatialData export; region mappings also require the exact
`hierarchy` that owns the region profiles. Cell mappings must bind to the final
profile after any hierarchy linkage. A mapping made before linkage cannot be
attached to the changed registry. To generate a fresh mapping after linkage,
use `cell_reference_atlas` instead. Do not combine a per-record receipt with a
global atlas for the same observation unit. `region_reference_atlas` can also
generate fresh mappings for records with a hierarchy; records without one remain
without region assignments. Neither route rebuilds the frozen atlas.

```bash
nextflow run cell_profiles.nf \
  --cell_profile_samples /absolute/path/samples.json \
  --outdir_base /absolute/path/new_results \
  --cell_atlas_python /absolute/path/profile_environment/bin/python
```

Profile runtime dependencies include NumPy, pandas, SciPy, scikit-learn,
PyArrow, tifffile and the morphology dependencies. The separately pinned
`requirements-spatialdata.txt` defines the tested export environment. Set
`cell_profiles_spatialdata: true` and `spatialdata_python` to that environment's
Python to enable export. Current published containers are not claimed to contain
these new dependencies until rebuilt and tested.
The optional [atlas container runtime](ATLAS_CONTAINER_RUNTIME.md) supplies
source-controlled Docker/Singularity overlays and separate interpreter wrappers;
no newly built or published image is claimed by those recipes alone.

## Reference atlas

Build an immutable version from compatible reference profiles:

```bash
python bin/cell_reference_atlas.py build \
  --profiles /absolute/path/reference_a /absolute/path/reference_b \
  --outdir /absolute/path/atlas_v1 --version v1 \
  --feature-groups morphology --label-column tissue_domain
```

Arbitrary numeric tissue/niche labels are specimen-scoped by default, not shared
biological names. Reviewed descriptions are separate from predictions. Reference
feature schemas and model definitions must match. Missing vectors, unsupported
small groups, outliers and ambiguous matches remain unknown; distances are not
calibrated probabilities. Build/map/search supports cell and region profile
manifests, with observation units kept separate.

The `map` CLI writes a three-file bundle: `reference_assignments.csv` (original
query columns plus reference interpretations), `reference_assignments.atlas.json`
(the exact frozen atlas manifest), and `reference_assignments.mapping.json`
(a completion-last, source-bound receipt). Existing destinations are not
overwritten. The receipt verifies the query manifest, table, feature-row order,
selected feature arrays, complete identities and assignment bytes. Keep the
three files together when moving a bundle. Older unreceipted CSV files remain
legacy results; rerun mapping against their exact profile and frozen atlas to
produce a verifiable bundle—do not fabricate a receipt.

With SpatialData enabled, the main workflow automatically waits for mapping and
adds seven separate fields to `cells.obs` and, when applicable,
`hierarchy_region_profiles.obs`: `reference_assignment`, `reference_status`,
`reference_distance`, `reference_radius`, `reference_margin`,
`reference_nearest_group`, and `reference_atlas_id`. Original discovery labels,
measurements, observation order and unknown/NaN values are retained. Exact atlas
metadata and receipt text are embedded in each table's
`uns['cellphenotyper']['reference_mapping_json']` and package metadata; downstream
readback does not require access to the original atlas directory. Export checks
bound sources before and after writing and verifies the stored interpretation
values exactly. This establishes provenance, not biological accuracy.

Cell/region mapping and SpatialData combine deep caching for explicit files
with an explicit upstream code-and-data fingerprint value input. Testing
showed that deep caching alone did not detect a same-size/timestamp edit inside
a directory in the installed Nextflow version. The added fingerprint covers
relative names and all file bytes without an mtime shortcut, before Nextflow
decides whether to resume a cached task. It also binds the stages' declared
Python dependencies and shared fingerprint helper. Computing hashes inside the
process script was insufficient for both data and code changes in the tested
version. All 41 modules now additionally expose code and directory fingerprints
through directly referenced `task.ext` properties in real and stub scripts.
The shared dependency resolver includes configured entrypoints, current local
Python imports, R helpers and static source/resource references. Full-main stub
resume plus four direct probes cover all 41 hooks; code-mutation tests verify
selective reruns, not learned-model equivalence. See the
[pipeline cache acceptance audit](../audits/spatial_atlas_20260904/pipeline_cache_acceptance.md).
This adds input-reading
cost, including the native image for export. Inputs must remain immutable while
a workflow runs; a cache fingerprint is not a filesystem lock. Query and
atlas directories have distinct staging names even when their source basenames
match. In the main workflow, explicitly supplying `cell_reference_atlas` with
`cell_profiles_enable=false` is an error, not a silently ignored request.

The `search` command supports morphology-only retrieval with a minimum difference
in a named predicted marker. Cross-slide resemblance still requires external
validation; same-slide self-mapping only tests engineering round trips.

## Inspector

The specimen atlas includes a manual launch command when the native image,
labels, calibration and profiles can be matched by provenance. Nothing is
automatically opened or started. `bin/cell_inspector.py --check-only` validates
inputs without starting a server. The read-only loopback server provides native
H&E windows, boundaries, profile details, explicit missing embeddings, review
reasons and weighted feature-group similarity. Expert notes are a separate
display-only resource. HTTP and PNG tests do not substitute for rendered UI QA.

To inspect shared niches, add an explicit
`--cohort-niches /absolute/path/cohort_niches` to a direct inspector invocation,
or to its portable atlas launch:

```bash
python bin/cell_inspector.py \
  --atlas-manifest /absolute/new_results/00_execution/specimen_atlas.json \
  --sample-id specimen_A --cohort-niches /absolute/path/cohort_niches --check-only
```

Omit `--check-only` only when deliberately starting the local server. Shared
niches appear in their own section with their model identity and separate
review flags; they do not replace per-slide labels, reference assignments or
similarity features. Startup verifies source hashes. During serving, path/file
identity changes trigger a digest recheck before cohort-bearing responses;
this is read-only source monitoring, not a filesystem lock. The ordinary output
index does not establish which run produced a discovered stage-25 directory,
so the inspector does not attach one merely because it is present. Explicit
selection is required unless the portable atlas was itself generated with an
explicitly selected, source-verified bundle. Conflicting explicit and atlas
cohort paths fail rather than silently replacing the atlas selection.

### Reviewing predictions excluded by detector fusion

New consensus runs additionally export `rejected_detector_candidates.geojson`.
This is a **separate detector-observation layer**, not extra canonical cells.
Each excluded `(detector, source_id)` has a deterministic `candidate_uid`, its
fusion exclusion reason, original detector evidence and source geometry. A
valid source polygon is preserved without repair; an invalid polygon remains
an explicitly invalid outline. Missing or malformed contours are shown as a
centroid cross, never an invented cell boundary. In particular, the StarDist
objects table does not contain nuclear contours. Excluded predictions are not
automatically false detections, and multiple observations may refer to the same
real cell. No canonical IDs, masks, marker profiles or similarity results change.

Add `--rejected-candidates /absolute/path/rejected_detector_candidates.geojson`
to a normal inspector launch (or to `--check-only`). The optional section has a
detector filter, paginated queue and bounded native H&E/outline windows. It is
read-only and independent of display-only expert annotations. Its source image,
canonical labels and crop shift must match exact hashes; dimensions/offsets and
detector-ID separation are checked. **Raster binding is not candidate-payload
verification:** without explicit original sources the UI/API label candidate
geometry and evidence as `unverified_source_payload`. The observed review-file
checksum is not an independent trusted checksum, and source hashes written
inside the GeoJSON cannot authenticate its geometry or evidence by themselves.

For source-verified review, additionally provide all four startup paths:

```bash
--candidate-stardist-objects /data/stardist/objects.csv \
--candidate-hovernet-cells /data/hovernet/hovernet_cells.json.gz \
--candidate-cellvit-cells /data/cellvit/cellvit_cells.json \
--candidate-alignment-csv /data/consensus/alignment.csv
```

The inspector verifies their hashes and rechecks the complete rejected
population, centroids, original geometry/geometry issues, fusion decisions,
alignment evidence and detector phenotype evidence against those files.
Changed geometry/evidence, omitted observations, partial source sets and
mismatched sources fail closed. Paths are accepted only as explicit startup
arguments, never followed from GeoJSON metadata. Successful rechecking is
reported as `verified_against_explicit_sources`; this is source consistency,
not independent biological validation. Older integrated review exports whose
fusion fields do not match the alignment-table schema require a fresh export
from the unchanged source artifacts, not relaxation of verification.

Physical coordinates remain unavailable in exports from legacy shifts without
MPP; the inspector separately reports coordinates from its supplied calibration.
Original fusion scores/distances are preserved, not recomputed with the review
calibration; this layer does not establish which MPP the historical fusion used.

Existing detector results can be exported without inference or changing any
source result. Choose a fresh output path:

```bash
python bin/detector_candidate_review.py \
  --stardist-objects /data/stardist/objects.csv \
  --hovernet-cells /data/hovernet/hovernet_cells.json.gz \
  --cellvit-cells /data/cellvit/cellvit_cells.json \
  --alignment-csv /data/consensus/alignment.csv \
  --labels /data/consensus/labels.tif --image /data/crop.tif \
  --shift /data/shift.json --resolution-json /data/passed_resolution.json \
  --output /data/new_review/rejected_detector_candidates.geojson
```

`--resolution-json` is optional; when provided it must pass the existing physical
resolution checks and is hash-bound for inspector verification. No guessed MPP
is used by this exporter. Export and startup hash checks add source-byte I/O.
Candidate geometry is loaded in memory (a 512 MiB GeoJSON startup limit); native
H&E is read only in requested windows of at most 1024 × 1024 pixels. This closes
review access to rejected observations, not the independent cell-recall or
false-detection validation requirement.

## Multiscale tissue hierarchy

The normal workflow can run the optional hierarchy on the `grid` or `both` route:

```yaml
tissue_hierarchy_enable: true
tissue_hierarchy_model_snapshot: /absolute/path/to/local/UNI2/snapshot
tissue_hierarchy_device: cuda
tissue_hierarchy_python: python3
tissue_hierarchy_local_field_um: 56.0
tissue_hierarchy_context_field_um: 224.0
tissue_hierarchy_fixed_k: 0
region_reference_atlas: null
```

These field widths are experimental defaults, not validated histological scales.
Both fields are separately cropped and encoded; widening the context also changes
the effective model MPP. The snapshot must contain config and weights locally;
no model is automatically downloaded. Incomplete fields remain missing by
default. The current broad parent domains—including your K=2 partition—are not
relabelled. Parent uncertainty is copied unchanged, and uncertain parent pixels
cannot become accepted subdomains. Missing/sparse observations retain parent-only
tissue instead of disappearing. The supplied support mask must contain the full
parent partition; a conflict fails rather than silently clipping it.

Automatic within-parent subdivision reports seed/scale agreement and explicit
abstention. It is not calibrated classification confidence. A supplied
`region_reference_atlas` is mapped separately and never rebuilt from query slides.
New pipeline hierarchy products and their input/output checksums can be passed to
the inspector through `--hierarchy-dir`; reviewed region annotations remain a
separate display-only resource. No inspector is launched automatically.

## Refinement and clustering migration

New refinement outputs include an original-uncertainty raster, a provenance
raster and a JSON code legend. Modified, newly inferred and unresolved pixels
remain distinguishable, including growth-only assignments. These are categorical
reasons, not confidence probabilities. Source codes survive inside tissue support.
Restarting vectorization from an older stage-14 result now requires regenerating
the refinement stage if these sidecars are absent; do not invent confidence for
an old result. Legacy stream checkpoints without provenance are rejected when
uncertainty preservation is requested.

Optional physical-unit refinement settings and multiclass appearance refinement
are implemented but are not default accuracy upgrades. Their GPU execution and
held-out biological accuracy still require benchmarking. The MedSAM encoder cache
reuses image encodings across prompts for each tile.

`cluster_representation: pca` selects saved higher-dimensional features for the
clustering comparison; `umap2d` remains the compatibility baseline. Neither
changes `kodama_ncomp: 50`, which is the KODAMA.matrix internal setting and not
the PCA component count. Two forced tissue domains remain an experiment setting,
not evidence of independently discovered biological classes.

`cluster_representation: kodama_graph` instead uses the opt-in native feature
graph directly, with no UMAP-edge reconstruction or silent fallback. Its supported
settings, graph provenance and limitations are in `KODAMA_GRAPH.md`. This feature
graph is not the physical tissue-gap-constrained neighbourhood graph. The current
native raw-data kNN classifier does not fit a 50-component PLS model; the actual
classifier and applicability of ncomp are recorded separately.
