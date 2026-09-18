# Parameters

## Linked cell profiles and spatial atlas

See [the cell-atlas guide](docs/CELL_ATLAS_USAGE.md) for the complete workflow,
runtime requirements and restart migration. Important opt-in parameters are
`cell_profiles_enable`, `cell_profiles_spatialdata`, `cell_reference_atlas`, and
`cellvit_export_embeddings`. Cell-local/context feature extraction is controlled
by `cell_profiles_uni2_enable`; it does not require a second tissue-clustering
route. `cell_neighborhood_radii_um` defaults to `25,50,100`; niche K is independent
of tissue K (`cell_niche_fixed_k=0` selects an exploratory stable solution).

`cell_neighborhood_feature_groups=available` includes present own-cell and
neighbour feature blocks; an explicit empty string selects spatial statistics
only. `cell_neighborhood_feature_weights='{}'` holds optional JSON group weights
(`own:morphology` and `morphology`, for example). `cohort_niches_enable=false`
can be enabled for at least two specimens to produce a separate shared niche
fit. Its `max_k=8`, `fixed_k=0`, `seed=17`, `repeats=5`, `fit_limit=20000`, and
`max_working_mb=1024` controls use the `cohort_niches_` prefix. See the guide for
compatibility gates, missingness, memory limits and interpretation constraints.
With SpatialData enabled, shared fits attach separate `cohort_niche_*` fields
and model provenance to each specimen's new export. Alternatively,
`cohort_niches_bundle=null` accepts an explicitly selected existing four-file
bundle directory; it requires export and cannot be combined with fitting.
All selected profiles must match their exact source identities in that bundle.
Use `existing_profiles` in the artifact workflow to avoid rebuilding a registry
to which a completed bundle is already bound. Inspector selection is explicit
through `--cohort-niches`; directory discovery alone is not a producing-run proof.

`cell_neighborhood_feature_storage=table` remains the default. Opt-in `arrays`
keeps the same logical own-cell/neighbour groups in source-bound memory-mapped
NPY segments outside the scalar table; no feature axes or canonical rows are
discarded. `cell_neighborhood_row_batch_size=4096` controls aggregation/fitting
row batches (including streamed cohort fits), and
`cell_neighborhood_column_batch_size=64` controls neighbour aggregation axes.
`cell_niche_fit_limit=20000` caps the seeded per-specimen training subset; all
eligible cells still contribute scaling and receive assignments. This is
separate from `cohort_niches_fit_limit`. A declared memory budget can reject
an oversized training matrix; it never silently lowers these settings.
See the guide for numerical-reduction and memory-accounting limitations.

`cell_neighborhood_support_mode=provided` preserves the supplied support raster.
The opt-in `brightfield_native` mode subtracts automatically detected bright
H&E background at native sampling for neighbourhood graphs and density only.
It is experimental: native pixels do not establish biological gap accuracy.
Canonical cells, marker compartments and tissue-domain maps remain unchanged;
no expert annotation is used. See the [support contract](docs/CELL_ATLAS_USAGE.md#native-image-derived-neighbourhood-support).

`tissue_hierarchy_enable` adds separate local/wider-field inference and within-parent
subdomains; it requires the grid route and a local `tissue_hierarchy_model_snapshot`.
Field widths default to 56/224 micrometres and require validation on held-out tissue.
Subdomain discovery uses within-parent native KODAMA graphs and Leiden, retaining
the broad parents unchanged. `kodama_ncomp=50` is forwarded separately from
`tissue_hierarchy_components_per_block`; the current kNN classifier does not
interpret ncomp as PLS rank. Hierarchy `kodama_m`, `kodama_tcycle`, and
`kodama_neighbors` settings default to 100, 20, and 100. The optional
`tissue_hierarchy_kodama_r_library` must point to an existing task-visible native
API library. `tissue_hierarchy_min_affinity_margin=0.1` controls graph evidence;
the legacy centroid-margin setting is not used by this route. Positive
`tissue_hierarchy_fixed_k` filters actual subdomain counts, without forcing
merges/splits or changing broad tissue K. The stage remains disabled by default.
`region_reference_atlas` maps resulting regions against a frozen reference.
When SpatialData export is enabled, both cell and region mappings are attached
to their own observation tables with source-verified receipts. The artifact-only
`cell_profiles.nf` entry point alternatively accepts per-record
`cell_reference_mapping`/`region_reference_mapping` receipt paths; do not combine
these with a global atlas for the same unit. See the
[portable mapping contract](docs/CELL_ATLAS_USAGE.md#reference-atlas).
`cell_measured_assays` supplies a JSON list of registered measured packages for
SpatialData; it requires both profile and export stages. See
[hierarchy details](docs/TISSUE_HIERARCHY.md) and
[measured-assay contracts](docs/MEASURED_ASSAY_IMPORT.md).

Physical expansion now defaults to `expand_um=3.0`; `-1` explicitly selects
legacy pixel expansion. GigaTIME default storage is all channels in float32.
`cluster_representation=pca` enables the higher-dimensional comparison while
`umap2d` remains the compatibility baseline; neither changes `kodama_ncomp=50`.

`cluster_representation=kodama_graph` opts into the saved native KODAMA
dissimilarity graph, with `cluster_landmark_cells=0`, Leiden, fixed resolution,
and no secondary fine variant. It automatically requests graph export;
`kodama_export_native_graph=false` otherwise preserves the existing export
default. [The graph contract](docs/KODAMA_GRAPH.md) explains its required
receipt, affinity rule, isolates, sensitivity-only forced counts, and limitations.

All parameters are defined in `nextflow.config` inside `params { ... }`.

Scientific-route, validation, input-QC, hardware, uncertainty and publication-critical parameters are also described by `nextflow_schema.json`. Validate a YAML or JSON parameter file before starting a run:

```bash
python bin/validate_pipeline_params.py --params pipeline_paramers.yml
```

The validator checks types, enums, ranges, route conditionals and cross-field invariants. Nextflow repeats the critical scientific checks at runtime, so bypassing this convenience command does not permit an incompatible route.

You can modify them in two ways:

1. edit defaults in `nextflow.config`
2. edit `pipeline_paramers.yml` and run with `-params-file pipeline_paramers.yml`
3. override at runtime with `--parameter value`

Example:

```bash
nextflow run main.nf -profile singularity \
  -params-file pipeline_paramers.yml \
  --max_cpus 4 \
  --max_memory_gb 8 \
  --uni2_batch 32
```

## Core

| Parameter | Default | Meaning |
|---|---|---|
| `folder_input` | `null` | Input folder for multi-sample mode. Supported image extensions: `.ome.tif`, `.ome.tiff`, `.btf`, `.czi`, `.vsi`, `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. Olympus VSI files must remain beside their `_<sample>_` companion directories. |
| `image_input` | `null` | Single-sample input image. Supports `.ome.tif`, `.ome.tiff`, `.btf`, `.czi`, `.vsi`, `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. Olympus VSI files must remain beside their `_<sample>_` companion directories. Ignored when `folder_input` is set. |
| `roi_geojson` | `null` | Single-sample ROI GeoJSON path. Ignored when `folder_input` is set. |
| `outdir_base` | `results` | Base output directory. |
| `vsi_series_index` | `1` | Bio-Formats series containing the primary whole-slide image for Olympus VSI input. The prostate VSI files use series 1; change only after inspecting another scanner export. |
| `convert_channel_order` | `RGB` | Meaning of source planes when a three-plane brightfield image must be joined. Accepts `RGB`, `RBG`, `GRB`, `GBR`, `BRG`, or `BGR`; the normalized pyramidal OME-TIFF always decodes to canonical RGB. JPEG-in-TIFF uses standard `YCbCr` storage, while lossless codecs retain an `RGB` photometric tag. |
| `storage_preflight_mode` | `fail` | `off`, `warn`, or `fail`. The default rejects expected output/work/cache demand that cannot fit while restart-only insufficiency is a review warning. |
| `storage_min_free_gib` | `20.0` | Per-filesystem free-space reserve retained after estimated demand. |
| `storage_safety_factor` | `1.25` | Safety multiplier for expected published-output and Nextflow-work growth. |
| `storage_restart_duplication_factor` | `1.0` | Additional retained-work allowance in the restart worst-case estimate. |
| `storage_source_expansion_factor` | `8.0` | Fallback expansion from compressed source bytes when host-side image dimensions cannot be probed. |
| `run_full_pipeline` | `true` | Run complete workflow including UNI-2 and KODAMA. |
| `analysis_intent` | `exploratory` | Declares the scientific purpose: `exploratory`, `cell_segmentation`, `cell_phenotyping`, `tissue_domain_discovery`, `virtual_staining`, or `outcome_prediction`. Incompatible scientific routes fail before execution, and the contract is written to `00_execution/analysis_contract.json`. |
| `study_manifest` | `null` | Optional JSON study-design declaration based on `resources/study_manifest.template.json`. It records the intended use, primary endpoint, statistical unit, cohort split, reference standard and prespecification. |
| `evidence_gate_mode` | `warn` | `off`, `warn`, or `fail`. A missing or incomplete declaration limits the recorded claim ceiling; strict `fail` mode stops the run. Passing checks verifies declared fields only, not scientific evidence. |
| `start_point` | `convert` | Stage where execution starts. |
| `end_point` | `auto` | Stage where execution stops (`auto` = `cluster_geojson` if full pipeline, else `tissue_mask`). |
| `compute_device` | `auto` | `auto` selects GPU consistently for all GPU-capable stages when NVIDIA visibility is detected, otherwise CPU; `gpu` and `cpu` force an explicit policy. |
| `cell_detection_mode` | `consensus` | Scientific cell-identification contract: `consensus` runs StarDist, HoVer-Net MoNuSAC and CellViT++ and requires GPU execution; `stardist` explicitly selects StarDist only. Hardware never changes this mode automatically. |
| `host_arch` | `auto` | `auto`, `amd64`, or `arm64` host architecture selector/override. |
| `enable_gpu_on_arm64` | `false` | Allow GPU selection on arm64 hosts when a compatible GPU container exists. |
| `enable_stardist_gpu_on_arm64` | `false` | On arm64, keep StarDist on CPU container by default; set `true` to force StarDist into GPU container. |
| `runtime_image_mode` | `auto` | `auto` uses architecture/device-aware image selection; `manual` uses `singularity_image`/`docker_image`. |
| `container_repo` | `ghcr.io/tkcaccia/cellphenotyper` | Base GHCR repository used by auto image selection. |
| `container_gpu_repo` | `ghcr.io/tkcaccia/cellphenotyper-runtime` | GHCR repository used for the current amd64 GPU runtime. |
| `container_cpu_tag` | `2.2-amd64` | Legacy generic fallback. Architecture-specific CPU tags below remain authoritative. |
| `container_cpu_tag_amd64` | `2.2-amd64` | Published CPU tag for amd64 hosts. |
| `container_cpu_tag_arm64` | `0.2.0` | Published CPU tag for arm64 hosts. |
| `container_gpu_tag` | `2.7-gpu-amd64` | Validated amd64 GPU tag used when `compute_device` resolves to GPU. |
| `singularity_image_source` | `auto` | `auto` tries local `.sif`, then GHCR ORAS SIF tags, then legacy release assets, then `docker://` fallback. Valid values: `auto`, `oras`, `release`, `docker`. |
| `singularity_gpu_image_source` | `docker` | GPU-specific Singularity/Apptainer source. The verified `2.7-gpu-amd64` runtime is pulled from its Docker OCI image by default. |
| `singularity_oras_repo` | `ghcr.io/tkcaccia/cellphenotyper` | GHCR repository used to resolve ORAS-hosted `.sif` tags. |
| `singularity_cpu_oras_tag_amd64` | `2.2-sif-amd64` | CPU ORAS tag for amd64 hosts. |
| `singularity_cpu_oras_tag_arm64` | `2.2-sif-arm64` | CPU ORAS tag for arm64 hosts. |
| `singularity_gpu_oras_tag_amd64` | `2.2-sif-gpu-amd64` | GPU ORAS tag for amd64 hosts. |
| `singularity_gpu_oras_tag_arm64` | `2.2-sif-gpu-arm64` | GPU ORAS tag for arm64 hosts. |
| `singularity_release_repo` | `tkcaccia/CellPhenotyper` | GitHub repo used to resolve release-hosted `.sif` assets. |
| `singularity_release_tag` | `v2.2` | Legacy GitHub release tag containing smaller `.sif` assets when available. |
| `singularity_cpu_asset_amd64` | `cellphenotyper-2.2-amd64.sif` | Legacy CPU Singularity asset name for amd64 hosts. |
| `singularity_cpu_asset_arm64` | `cellphenotyper-2.2-arm64.sif` | Legacy CPU Singularity asset name for arm64 hosts. |
| `singularity_gpu_asset_amd64` | `cellphenotyper-2.2-gpu-amd64.sif` | Legacy GPU Singularity asset name for amd64 hosts. |
| `singularity_gpu_asset_arm64` | `cellphenotyper-2.2-gpu-arm64.sif` | Legacy GPU Singularity asset name for arm64 hosts. |
| `singularity_local_dir` | `''` | Optional local directory with prebuilt `.sif`; checked before ORAS/release/docker fallback. |
| `singularity_cache_dir` | `''` | Optional Apptainer/Singularity cache path; default is `<repo>/.apptainer_cache`. |
| `cpu_container_image` | `''` | Optional explicit CPU container URI/path. |
| `gpu_container_image` | `''` | Optional explicit GPU container URI/path. |
| `singularity_image` | `''` | Manual container URI/path for `-profile singularity` (`runtime_image_mode: manual`). |
| `docker_image` | `''` | Manual image for `-profile docker` (`runtime_image_mode: manual`). |
| `publish_dir_mode` | `copy` | Nextflow `publishDir` mode. The durable default keeps `results/` independent of `work/`; use `rellink` only when the work cache will be retained. |
| `gpu_debug_diagnostics` | `false` | When true, GPU-capable processes print `nvidia-smi` and framework CUDA diagnostics. |
| `gpu_scheduler_enable` | `true` | Enable per-device task locks and VRAM-token admission for all `gpu_capable` processes. |
| `gpu_lock_dir` | `${baseDir}/.gpu_locks` | Shared lock directory visible to all local Nextflow tasks on the host. |
| `gpu_memory_token_gb` | `2.0` | VRAM represented by one exclusive scheduler token. |
| `gpu_memory_reserve_gb` | `2.0` | Free VRAM retained outside pipeline reservations for the driver and transient allocations. |
| `gpu_default_task_memory_gb` | `8.0` | Default reservation for a GPU process without a stage-specific value. |
| `gpu_max_tasks_per_device` | `0` | Maximum concurrent tasks per GPU; `0` derives the limit from VRAM tokens. |
| `gpu_scheduler_timeout_seconds` | `172800` | Maximum wait for a device satisfying free-VRAM and token requirements. |
| `max_cpus` | `auto` | Global CPU envelope. `auto` uses the scheduler/container allowance and preserves up to two host CPUs outside a scheduler. |
| `max_memory_gb` | `auto` | Global RAM envelope. `auto` uses the effective host/cgroup limit while preserving bounded system headroom. |
| `max_parallel_tasks` | `auto` | Executor queue size. `auto` derives it from the usable CPU count; aggregate CPU/RAM limits still gate task launches. |

Storage demand is evaluated before channels schedule image processes and written to `00_execution/storage_preflight.json`. Output, work and cache paths on the same filesystem are aggregated. See `docs/STORAGE_PREFLIGHT.md` for decision rules and limitations.
| `hardware_auto` | `true` | Resolve per-stage CPUs, RAM and runtime workers from the global hardware envelope. |
| `hardware_profile` | `auto` | `auto`, `conservative`, `balanced`, or `aggressive`; automatic selection uses usable CPU, RAM and GPU capacity. |
| `hardware_cpu_reserve` | `auto` | CPUs retained for the host outside schedulers; automatic mode preserves zero to two depending on host size. |
| `hardware_min_free_system_gb` | `6.0` | Requested host RAM reserve; it is bounded to avoid starving small machines. |
| `hardware_max_auto_batch` | `32` | Pipeline-wide automatic batch ceiling; live tool-specific RAM/VRAM limits can select a smaller value. |
| `hardware_max_auto_block_size` | `8192` | Pipeline-wide automatic block-size ceiling. StarDist can use the full value; GigaTIME applies its own lower live-memory cap. |
| `hardware_max_auto_output_gib` | `16.0` | Maximum automatic in-memory/output buffer budget for tools that use the shared limit. |
| `docker_extra_run_options` | empty | Optional site-specific Docker flags, such as bind mounts for shared model caches. |
| `hf_home` | `${baseDir}/.hf_cache` | Hugging Face cache root for UNI-2 model files. |
| `hf_hub_cache` | `${baseDir}/.hf_cache/hub` | Hugging Face Hub cache directory for UNI-2 model files. |
| `hf_hub_offline` | `false` | If `true`, UNI-2 runs in strict offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and only uses local cache. |
| `hf_token_env_var_name` | `HF_UNI2` | Env var name used to read the HuggingFace token for UNI-2. |
| `hf_token_env_file` | `tokens.env` | Env file path sourced at runtime before UNI-2 starts (recommended in repo root for Docker profile). |

ROI resolution rules:
- In `folder_input` mode, each image `<sample>.<supported_extension>` uses `<sample>.geojson` if present in the same folder.
- For multi-region CZI inputs, use region-specific ROI files named `<image>.czi - ScanRegionN.geojson`; the pipeline resolves one sample per matching region and converts only that region to TIFF.
- If `<sample>.geojson` is missing, the ROI defaults to the full image.
- In single-sample mode, `--roi_geojson` is optional; if omitted, `<image_root>.geojson` is searched next to `image_input`, otherwise full-image ROI is generated. For CZI, `--roi_geojson` may also be a region-specific file such as `<image>.czi - ScanRegion0.geojson`.

Every provided or generated ROI is hashed and validated before cropping. The validator requires a non-empty Polygon/MultiPolygon FeatureCollection, preserves and counts holes, checks validity and finite coordinates, tests all area against the exact level-0 image bounds, rejects geographic/world CRS declarations, and verifies any declared source-image identity. It never repairs self-intersections or clips coordinates silently. Missing source-image and coordinate-space declarations are retained as explicit warnings because many legacy QuPath exports omit them.

| Parameter | Default | Meaning |
|---|---|---|
| `roi_validation_mode` | `fail` | `fail` stops on invalid/mismatched ROI geometry. `warn` preserves a legacy ROI unchanged but records a warning quality signal; use only after manual review. |
| `roi_bounds_tolerance_px2` | `1.0e-6` | Numerical area tolerance outside the level-0 image frame before the ROI is considered out of bounds. |

GPU run notes:
- amd64: use `--compute_device gpu --host_arch amd64`.
- `cell_detection_mode=consensus` fails on a resolved CPU run. Use `--cell_detection_mode stardist` only when a StarDist-only scientific analysis is intended.
- GPU-capable stages reserve one specific device and a configurable amount of VRAM before launch. The defaults reserve 6 GB for GrandQC, 8 GB for StarDist/GigaTIME, 10 GB for UNI-2/MedSAM, 12 GB for HoVer-Net/CellViT++/TITAN, and 4 GB for GPU KODAMA.
- StarDist, HoVer-Net, CellViT++, UNI-2 and TITAN size memory-sensitive work from live free VRAM on the scheduler-assigned device, not nominal GPU 0 capacity. GigaTIME also considers live RAM and VRAM. Positive explicit batch parameters remain overrides.
- `00_execution/hardware_plan.json` records detected hardware, reserved host capacity, the resolved profile and every per-stage CPU/RAM allocation. Full behavior is documented in `docs/HARDWARE_AUTOTUNING.md`.
- arm64: set `--compute_device gpu --host_arch arm64 --enable_gpu_on_arm64 true` and provide an arm64 GPU container (`singularity_gpu_asset_arm64` or `gpu_container_image`).
- If no arm64 GPU container is available, GPU-capable processes can use CPU containers only for an explicitly selected CPU-compatible route such as `cell_detection_mode=stardist`; consensus mode fails rather than changing detectors.
- On arm64, StarDist defaults to CPU container unless `--enable_stardist_gpu_on_arm64 true`.
- On GB10 (`sm_121`), use an arm64 GPU SIF built with nightly `cu130` PyTorch (the `v2.2` arm64 GPU asset may fail with `no kernel image is available`).

`start_point` / `end_point` allowed values:
`convert`, `grandqc`, `stardist`, `cell_consensus`, `tma`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `gigatime`, `marker_quantification`, `grid_tiles`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `medsam_refine`, `cluster_geojson`, `neoplastic_section`, `titan`, `pathofmpred`. `grow_tissue` applies to cell-centred masks: it runs for `cells` and for the `<sample>__cells` branch of `both`, while the dense grid branch bypasses it.

## Input Resolution

The `convert` stage validates physical pixel size before doing expensive conversion and validates it again on the generated OME-TIFF. Source and converted reports are published in `01_input/<sample>/`. They include the full-file SHA-256, dimensions, axes, dtype, bit depth, channel/color interpretation, compression, tiling, pyramid levels, every detected MPP source and any metadata conflicts. Upsampling is still permitted within the accepted range, but the report states explicitly that resampling cannot create additional spatial detail.

| Parameter | Default | Meaning |
|---|---|---|
| `input_resolution_check` | `true` | Run source and post-conversion MPP validation. |
| `input_resolution_strict` | `true` | Stop before cell analysis when resolution metadata is missing or outside the configured contract. |
| `input_resolution_min_mpp` | `0.05` | Lower metadata sanity bound in µm/px. |
| `input_resolution_max_mpp` | `0.50` | Coarsest accepted native resolution in µm/px. This limits linear upsampling to 2× for 0.25-µm cell models. |
| `input_resolution_cell_target_mpp` | `0.25` | Reference cell-model MPP used to report the required upsampling factor. |
| `input_resolution_max_anisotropy_fraction` | `0.05` | Maximum relative difference between X and Y MPP. |
| `input_resolution_max_conversion_drift_fraction` | `0.02` | Maximum MPP change permitted during conversion. |
| `input_resolution_max_metadata_conflict_fraction` | `0.02` | Maximum disagreement among OME, TIFF-tag and OpenSlide MPP sources before strict mode requires an explicit verified override. |
| `input_resolution_override_mpp` | `0.0` | Explicit isotropic source MPP override; `0` disables the override. Use only for independently verified metadata errors. |
| `input_hash_enable` | `true` | Compute a full SHA-256 for both source and converted artifacts. Disable only when the extra sequential I/O is unacceptable and external checksums are recorded elsewhere. |
| `input_require_rgb` | `true` | Require the converted analysis image to be three-channel RGB-compatible color; JPEG-in-TIFF `YCbCr` storage is accepted because standard WSI readers decode it to RGB. |
| `input_require_pyramid` | `true` | Require the converted analysis image to expose at least two pyramid levels. |

## Conversion (`.btf` -> `.ome.tif`)

| Parameter | Default | Meaning |
|---|---|---|
| `convert_compression` | `JPEG` | Shared converted-image and analysis-crop codec: `JPEG`, `LZW`, `DEFLATE`, `NONE`, or `UNCOMPRESSED`. JPEG-in-TIFF stores canonical colour as `YCbCr` and readers decode it to RGB; `LZW` and `DEFLATE` are lossless. |
| `convert_jpeg_quality` | `75` | JPEG quality from 1 to 100. It applies to the converted OME-TIFF and `crop_roi.tif` when `convert_compression=JPEG`; lossless codecs ignore it. |
| `convert_downsample` | `GAUSSIAN` | Pyramid downsample algorithm. |
| `convert_rgb` | `true` | Convert to RGB. |
| `convert_overwrite` | `true` | Overwrite output if existing. |
| `convert_cpus` | `8` | CPU allocation. |
| `convert_memory_gb` | `16` | RAM allocation. |
| `convert_time` | `6h` | Time allocation. |

## GrandQC

| Parameter | Default | Meaning |
|---|---|---|
| `grandqc_device` | `auto` | Use CUDA when the pipeline resolves a GPU run; otherwise use the available CPU/MPS backend. |
| `grandqc_artifact_mpp_model` | `auto` | Select the official 1.0, 1.5, or 2.0 MPP artifact checkpoint. Auto uses 1.0 below 0.12 um/px, 1.5 below 0.20 um/px, and the validated 2.0 checkpoint otherwise. |
| `grandqc_tissue_mpp_model` | `10.0` | Physical scale requested by the tissue detector. |
| `grandqc_tissue_probability_threshold` | `0.5` | Minimum GrandQC tissue probability. Lower values increase tissue sensitivity; validate against true slide background. |
| `grandqc_clean_tissue_policy` | `tissue_minus_artifacts` | Uses tissue-detector support minus explicit artifact classes 2–6. `artifact_normal_only` restores the legacy strict class-1 mask. |
| `grandqc_patch_size` | `512` | Official tissue-detector patch size. It does not control artifact context tiling. |
| `grandqc_artifact_tile_size` | `0` | Artifact inference tile size. `0` uses the official 512 x 512 geometry on every device. Explicit experimental values must be at least 256 and divisible by 32. |
| `grandqc_artifact_overlap_fraction` | `0.5` | Fractional artifact-tile overlap used for probability blending. |
| `prepare_crop_memory_gb` | `12` | RAM allocation for the streaming analysis crop created around GrandQC clean tissue intersected with the optional ROI. |
| `convert_compression` / `convert_jpeg_quality` | `JPEG` / `75` | Shared codec and JPEG quality for both the normalized input OME-TIFF and the pyramidal RGB OME-TIFF analysis crop. |
| `grandqc_preview_max_side` | `4096` | Maximum long side of GrandQC preview assets. |
| `grandqc_crop_mask_memory_gb` | `4` | RAM allocation for cropping the low-resolution full-slide clean-tissue mask into the analysis frame. |

## StarDist

| Parameter | Default | Meaning |
|---|---|---|
| `stardist_model` | `2D_versatile_he` | StarDist model preset. |
| `stardist_prob` | `0.52` | Detection probability threshold. |
| `stardist_nms` | `0.28` | NMS threshold. |
| `stardist_keras_home` | `${baseDir}/.keras` | Project-local Keras cache directory for StarDist pretrained model files. This keeps the cache inside the repository so Docker reruns can reuse it offline. |
| `stardist_pretrained_zip` | `''` | Optional local zip path (for example `python_2D_versatile_he.zip`) copied into StarDist cache before execution. If the zip already lives under `stardist_keras_home`, the pipeline now reuses and auto-extracts it without needing this override. |
| `stardist_autoinstall_runtime` | `true` | If StarDist/TensorFlow runtime is missing in container, auto-install required Python deps into task-local `.pydeps`. |
| `stardist_tensorflow_version` | `2.16.2` | TensorFlow version used by StarDist runtime auto-install fallback. |
| `stardist_tiles_x` | `32` | Tiles in X. |
| `stardist_tiles_y` | `32` | Tiles in Y. |
| `stardist_pythonpath` | `''` | Optional extra `PYTHONPATH` for StarDist runtime dependencies (e.g. external TensorFlow path on M1). |
| `input_roi_mask_label_mode` | `auto` | Rasterize the crop-aligned input ROI GeoJSON as a labeled mask. `auto` prefers a numeric `value` property when present and otherwise assigns stable IDs from annotation labels. |
| `input_roi_mask_value_prop` | `value` | Numeric GeoJSON property used first when `input_roi_mask_label_mode=auto` or explicitly when `input_roi_mask_label_mode=property`. |
| `input_roi_mask_annotation_props` | `classification.name,classification.label,class,label,type,name` | Fallback annotation properties used to derive per-class mask values from the input ROI GeoJSON. |
| `input_roi_mask_default_value` | `1` | Value used for ROI polygons that lack both a numeric value and a recognized annotation label. |
| `input_roi_mask_compression` | `deflate` | Compression for the crop-aligned labeled mask rasterized from the provided input ROI GeoJSON. |
| `input_roi_mask_preview_factor` | `10` | Downsample factor used only when the ROI mask preview exceeds the preview memory threshold. |
| `input_roi_mask_preview_threshold_mb` | `100.0` | Preview downsampling threshold for the ROI mask overlay. |
| `input_roi_mask_preview_alpha` | `0.45` | Overlay alpha for the ROI mask preview rendered on the crop image. |
| `input_roi_mask_cpus` | `4` | CPU allocation for the ROI GeoJSON-to-mask step. |
| `input_roi_mask_memory_gb` | `8` | RAM allocation for the ROI GeoJSON-to-mask step. |
| `input_roi_mask_time` | `4h` | Time allocation for the ROI GeoJSON-to-mask step. |
| `write_full_labels` | `false` | Write a full-canvas StarDist label artifact only when a downstream stage requires it. |
| `full_format` | `zarr` | Full label file format. `zarr` is the safe default for large WSI runs. |
| `allow_huge_tif` | `false` | Refuse very large dense TIFF label writes unless explicitly overridden. |
| `stardist_cpus` | `16` | CPU allocation. |
| `stardist_memory_gb` | `48` | RAM allocation. |
| `stardist_time` | `24h` | Time allocation. |

## TMA Detection

| Parameter | Default | Meaning |
|---|---|---|
| `tma_enable` | `true` | Run the post-StarDist TMA detection and cell-to-spot assignment step. |
| `tma_thumbnail_max_side` | `2048` | Maximum thumbnail side used for spot detection. |
| `tma_min_spots` | `4` | Minimum compact tissue spots required before an image is called a TMA. |
| `tma_min_spot_area_fraction` | `0.001` | Minimum candidate spot area as a fraction of the crop. |
| `tma_max_spot_area_fraction` | `0.25` | Maximum candidate spot area as a fraction of the crop. |
| `tma_max_area_cv` | `0.75` | Maximum coefficient of variation in candidate spot area. |
| `tma_cpus` | `4` | CPU allocation. |
| `tma_memory_gb` | `12` | RAM allocation. |
| `tma_time` | `4h` | Time allocation. |

## GigaTIME

| Parameter | Default | Meaning |
|---|---|---|
| `gigatime_enable` | `true` | Enable the crop-image GigaTIME virtual mIF stage after StarDist. Set to `false` only when you intentionally want to skip both GigaTIME inference and the downstream marker quantification outputs. |
| `gigatime_repo_id` | `prov-gigatime/GigaTIME` | Hugging Face repo used for model weights. |
| `gigatime_hf_token_env_var_name` | `HF_GIGATIME` | Preferred environment variable name for the GigaTIME token. The workflow also falls back to `HF_TOKEN` and `HF_UNI2`. |
| `gigatime_page` | `0` | TIFF page index used as the crop-image source. |
| `gigatime_patch_size` | `256` | Patch size for tiled GigaTIME inference. |
| `gigatime_stride` | `128` | Patch stride for tiled GigaTIME inference. Lower than `gigatime_patch_size` by default so overlapping tiles and raised-cosine blending reduce visible seam artifacts. |
| `gigatime_batch_size` | `1` | Conservative batch-size floor. With hardware adaptation enabled, CUDA batch size scales from live free VRAM up to the configured automatic cap; an in-process OOM fallback halves the batch and retries the current region. |
| `gigatime_auto_hardware` | inherited | Inherit the pipeline-wide `hardware_auto` setting unless explicitly overridden. GPU batches are selected from free VRAM, while block and output-buffer sizes remain bounded by available host RAM. |
| `gigatime_skip_background_blocks` | `false` | Optional coarse-background optimization. It is disabled for publication/default outputs because whole zero-filled blocks can create rectangular seams. When explicitly enabled, only a context-expanded region with tissue fraction at or below `gigatime_skip_background_min_fraction` is skipped; labelled blocks remain protected. |
| `gigatime_skip_background_min_fraction` | `0.0` | Maximum tissue fraction in the patch-halo-expanded decision region for background skipping. The default permits skipping only regions with no coarse-mask tissue. |
| `gigatime_seam_qc_mode` | `fail` | `off`, `warn`, or `fail` action for systematic discontinuities at block boundaries. |
| `gigatime_seam_qc_max_p95_excess` | `0.10` | Maximum allowed excess of the boundary 95th-percentile absolute gradient over neighboring within-block gradients on the 0-1 probability scale. |
| `gigatime_seam_qc_min_affected_fraction` | `0.05` | Minimum sampled boundary fraction required to trigger the seam gate. |
| `gigatime_strict_target_mpp` | `true` | Enforce the requested GigaTIME physical scale from image metadata with exact floating-point resampling. This includes upsampling when the source MPP is coarser than the model target; GigaTIME fails if MPP cannot be resolved and does not coarsen the image to satisfy the output-size budget. |
| `gigatime_max_output_gib` | `8.0` | Maximum estimated uncompressed persisted GigaTIME image size before automatic extra downsampling is applied when strict target MPP is disabled. |
| `gigatime_output_format` | `zarr` | Persist the GigaTIME virtual-marker score image as chunked `gigatime_probs.zarr` by default. The filename is retained for compatibility; scores are not calibrated probabilities. When `ome_tiff` is requested, the saved `gigatime_probs.ome.tif` is pyramidal by default for QuPath/WSI viewing. |
| `gigatime_output_channels` | `DAPI,PD-1,CD3,CD8,PD-L1` | Marker channels persisted in the GigaTIME image store. Integrated single-cell quantification is still computed from all GigaTIME model channels. |
| `gigatime_export_ometiff` | `true` | Export the GigaTIME store to a pyramidal multichannel `gigatime_probs.ome.tif` after prediction. This keeps inference reliable with Zarr while still producing a QuPath-ready OME-TIFF. |
| `gigatime_integrated_quantification` | `true` | Quantify all 23 GigaTIME model markers over nuclei and cytoplasm during the same tiled inference pass. The persisted image may remain a smaller selected channel subset; full-marker quantification does not require storing a 23-channel WSI. |
| `gigatime_kodama_enable` | `true` | Run an independent PCA/KODAMA analysis on all matched nuclei and cytoplasm marker means. Requires integrated quantification. |
| `gigatime_jpg_markers` | `DAPI,PD-1,CD3,CD8,PD-L1` | Marker channels exported as lightweight JPEG previews for visual QC. |
| `gigatime_output_compression` | `jpeg` | Compression for `gigatime_probs.ome.tif` when `--gigatime_output_format ome_tiff` is explicitly requested. |
| `gigatime_cpus` | `8` | CPU allocation. |
| `gigatime_memory_gb` | `24` | RAM allocation. |
| `gigatime_time` | `12h` | Time allocation. |

## GigaTIME marker quantification

| Parameter | Default | Meaning |
|---|---|---|
| `marker_quantification_enable` | `true` | Quantify the GigaTIME marker stack over nuclei and cytoplasm label masks. |
| `marker_quantification_cpus` | `8` | CPU allocation. |
| `marker_quantification_memory_gb` | `24` | RAM allocation. |
| `marker_quantification_time` | `12h` | Time allocation. |

## Final cluster GeoJSON

| Parameter | Default | Meaning |
|---|---|---|
| `cluster_geojson_page` | `0` | TIFF pyramid level used for polygonization; the default uses the native-resolution label raster. |
| `cluster_geojson_polygon_backend` | `rasterio` | Topology-preserving polygonization backend. |
| `cluster_geojson_connectivity` | `4` | Area connectivity for raster polygonization. Four-neighbour connectivity prevents diagonal-only contacts from becoming self-touching rings. |
| `cluster_geojson_dissolve_by_value` | `true` | Dissolve polygons by label value. |
| `cluster_geojson_min_area` | `0` | Minimum polygon area. Zero preserves every annotated component; increase only when deliberate removal of small regions is acceptable. |
| `cluster_geojson_smooth_buffer` | `0.0` | Buffer-based smoothing radius; disabled by default for multiclass topology. |
| `cluster_geojson_smooth_passes` | `1` | Number of smoothing passes when smoothing is enabled. |
| `cluster_geojson_simplify` | `16.0` | Maximum native-pixel coverage-simplification tolerance considered by the fidelity-constrained search. |
| `cluster_geojson_shared_boundary_simplify` | `true` | Simplify multiclass geometries as one polygon coverage so shared boundaries remain coincident. Requires Shapely 2.1 or newer. |
| `cluster_geojson_simplify_max_categorical_difference_fraction` | `0.01` | Maximum fraction of native foreground area permitted to change label or tissue/background membership during simplification (0.01 = 1%). Set `0` for exact vector geometry. |
| `cluster_geojson_simplify_search_steps` | `10` | Binary-search iterations used to find the strongest simplification satisfying the annotation-error limit. |
| `cluster_geojson_shared_boundary_smooth` | `true` | After coverage simplification, selectively relax long, sharp internal cluster-boundary vertices without changing the external tissue footprint or creating class overlap. This production transform never reads an expert annotation. |
| `cluster_geojson_shared_boundary_smoothing_coefficient` | `0.05` | Fraction of the local Laplacian displacement applied to an eligible shared-boundary vertex. |
| `cluster_geojson_shared_boundary_smoothing_passes` | `1` | Number of endpoint-preserving selective smoothing passes. |
| `cluster_geojson_shared_boundary_minimum_turn_degrees` | `45.0` | Minimum internal turn angle eligible for smoothing. |
| `cluster_geojson_shared_boundary_minimum_adjacent_length` | `16.0` | Both adjacent segments must be at least this many native pixels, preventing short anatomical details from being rounded. |
| `cluster_geojson_preserve_topology` | `true` | Preserve topology on simplify. |
| `cluster_geojson_fill_holes` | `false` | Remove interior holes; disabled by default so interlocking cluster classes remain mutually exclusive. |
| `cluster_geojson_group_map` | `''` | Optional JSON/YAML map for class names. |
| `cluster_geojson_group_prefix` | `color_` | Default label prefix when map is absent. |
| `cluster_geojson_cpus` | `8` | CPU allocation. |
| `cluster_geojson_memory_gb` | `16` | RAM allocation. |
| `cluster_geojson_time` | `6h` | Time allocation. |

## Cytoplasm expansion

| Parameter | Default | Meaning |
|---|---|---|
| `expand_px` | `12` | Expansion radius (pixels). |
| `expand_full_labels` | `true` | Also expand `labels_full.tif` to `labels_full_cyto.tif`. |
| `expand_mode` | `auto` | `auto`, `full`, or `tiled` expansion strategy. |
| `expand_tile_size` | `2048` | Core tile size used in tiled mode. |
| `expand_auto_threshold_mpix` | `25.0` | In `auto` mode, switch to tiled when image is larger than this MP threshold. |
| `expand_compression` | `zlib` | TIFF compression for output mask. |
| `expand_cpus` | `8` | CPU allocation. |
| `expand_memory_gb` | `16` | RAM allocation. |
| `expand_time` | `4h` | Time allocation. |

## Cell assignment + cytoplasm + UNI-2 + KODAMA + post-KODAMA

| Parameter group | Key controls |
|---|---|
| Cell assignment | `assign_*` |
| Cytoplasm expansion | `expand_*` (`expand_mode`, `expand_tile_size`, `expand_auto_threshold_mpix` control memory-safe tiled expansion; `expand_full_labels=false` skips full-image expansion) |
| UNI-2 sampling units | `uni2_sampling_mode`, `uni2_grid_*` |
| UNI-2 embeddings | `uni2_*`, `hf_*`, `hf_token_env_var_name` |
| KODAMA R step | `r_*` |
| KODAMA clustering | `cluster_*`, `cluster_primary_variant`, `cluster_secondary_variant`, `cluster_secondary_profile`, `cluster_fine_resolution_multiplier`, `cluster_fine_score_margin` |
| Cluster mask build | `cluster_mask_*` |
| Grow clusters to tissue | `grow_*` |
| MedSAM border refinement | `medsam_refine_*`, `medsam_*` |
| Final cluster GeoJSON | `cluster_geojson_*` |

Grid sampling controls:

| Parameter | Default | Meaning |
|---|---|---|
| `uni2_sampling_mode` | `grid` | `grid` is the default regular spatial route and does not run cell-centred UNI-2. `cells` explicitly uses nucleus-centred observations; `both` keeps grid primary and adds a complete namespaced cell-centred route for comparison. |
| `uni2_grid_min_tissue_fraction` | `0.05` | Minimum tissue fraction in the visible inner-square core required to retain a grid observation. |
| `uni2_grid_stride_px` | `0` | Grid-centre stride in UNI2 model-input pixels. `0` uses `uni2_inner_square_fixed_px`; a positive value separates sampling density from local pooling width. |
| `uni2_grid_preview_max_side` | `2048` | Maximum grid QC preview dimension. |
| `uni2_grid_cpus`, `uni2_grid_memory_gb`, `uni2_grid_time` | `4`, `8`, `2h` | Resource request for tissue occupancy and grid-table construction. |

UNI-2 behavior:

- `uni2_sampling_mode=cells` preserves the original nucleus-centred pipeline. `uni2_sampling_mode=grid` creates regular GrandQC-filtered observations in `09_grid_tiles`, then uses those same observation IDs and coordinates for UNI-2, KODAMA, clustering, and cluster-mask generation. `uni2_sampling_mode=both` additionally writes a complete cell-centred route under `<sample>__cells`, from embeddings through GeoJSON. Cell UNI-2 waits for primary grid UNI-2 to finish to avoid loading two UNI-2 models on one GPU; subsequent route tasks follow the shared scheduler policy.
- In grid mode, the level-0 source stride is `round(effective_grid_stride_px * extraction_tile_size / uni2_tile_size)`, where `effective_grid_stride_px` is `uni2_grid_stride_px` when positive and otherwise `uni2_inner_square_fixed_px`. This separates observation spacing from the local token-pooling square while preserving the historical coupled default. Logical grid cores remain adjacent; the larger UNI2 context crops overlap by `extraction_tile_size - stride`.
- `uni2_grid_min_tissue_fraction=0.05` excludes grid cores with less than 5% tissue. Grid mode is intentionally restricted to fused `tile,inner_square` UNI2-h embeddings; cell-defined `nuclei` and `cyto` families are rejected rather than mixed with grid observations.
- `uni2_fuse_tile_inner_square=true` is the optimized default: one Python/model session loads UNI2-h once and performs one forward pass for each 224 x 224 tile. The `tile` vector uses the CLS token. The `inner_square` vector averages patch tokens whose centers fall inside the fixed central 90 x 90-pixel square; it does not use the cytoplasm mask and does not trigger a second image forward pass. Set this to `false` only for comparison with the slower legacy masked-image forward definition.
- `uni2_target_mpp=0.25` makes the source crop physically calibrated before the 224 x 224 UNI2 input transform. If a StarDist crop lost OME pixel-size metadata, UNI2 recovers the MPP from the crop `shift.json` and the original input image when possible; otherwise it uses `uni2_default_source_mpp`.
- For a source image at `0.08706 µm/px`, the default 224-pixel UNI2 input is extracted from a 643-pixel source crop, giving an effective resolution of about `0.25 µm/px`.
- `uni2_save_tiles=false` is the production default. KODAMA reads embedding shards, not per-cell PNG crops, so saving tiles is mainly for debugging/QC and can dominate disk I/O on large runs.
- `uni2_embedding_storage=csv` is the compatibility default. `binary` stores exact numerical feature arrays separately from small row metadata, with strict IDs, byte sizes and SHA256 receipts. R/KODAMA and Python profile/hierarchy loaders support it. `uni2_rows_per_csv` controls rows per shard in either format. Use a fresh output directory when changing storage; see `docs/UNI2_BINARY_STORAGE.md` for precision, restart and benchmark scope.
- `uni2_reuse_existing=false` is the fresh-run default. Set `--uni2_reuse_existing true` for recovery runs where `09_embeddings/<sample>/embeddings_<sample>_*` already exists and should be consumed by KODAMA instead of scheduling UNI-2 again, even if the requested stage window includes `uni2`.
- `kodama_r_library_dir` and `cluster_r_library_dir` isolate R package lookup to the bundled R 4.6 and StarDist R libraries, respectively. The pipeline also disables host `.Renviron` and `.Rprofile` files so Singularity cannot accidentally load ABI-incompatible packages from the user's home directory.
- `kodama_landmarks=10000` is the default for both UNI-2 and GigaTIME-marker KODAMA runs. PCA computes `kodama_dims_to_run` components (50 by default), and the raw feature matrix is deleted immediately after PCA to release RAM. `kodama_ncomp=50` independently sets the `ncomp` argument passed to `KODAMA.matrix`, bounded by the input PCA dimension count. Its meaning depends on the actual classifier: native raw-data kNN does not fit a 50-component PLS model. The classifier and ncomp applicability are recorded explicitly; no implicit classifier change is made.
- With `uni2_sampling_mode=both`, `COMPARE_UNI2_ROUTES` aggregates cell-centred KODAMA coordinates inside retained grid cores before comparison. Independently fitted axes are Procrustes-aligned; predefined endpoints include coverage, pairwise-distance correlation, 15-neighbor overlap, spatial coherence, and cross-validated prediction of GigaTIME marker proxies when available. The report does not choose a winning route automatically.

MedSAM behavior:

- MedSAM refinement runs per cluster label. For large WSI crops, each cluster border is split into overlapping tiles controlled by `medsam_cluster_tile_size` and `medsam_cluster_tile_overlap`, so a tile may legitimately contain only one cluster.
- The crop-aligned GrandQC clean-tissue/ROI mask is a hard negative constraint inside MedSAM. GrandQC-empty and artifact pixels are excluded from seeds, baseline labels, dilation envelopes, prompts, editable bands, hole filling and final labels. Stage 14 fails if its final zero-leakage assertion is violated.
- `<sample>_<variant>_medsam_grandqc_empty_exclusion.png` shows GrandQC-empty pixels in blue, labels removed from those pixels in red, and any forbidden final leakage in magenta. Corresponding pixel counts are stored in the MedSAM summary JSON.
- Large masks use downsampled morphology for protected-core and editable-band construction, and full-resolution work is limited to the cluster-border tile passed to MedSAM. The default `medsam_device=auto` follows the pipeline-wide resolved compute device; forcing `medsam_device=cuda` still requires CUDA.
- `medsam_pre_boundary_competition=true` runs before MedSAM and allows adjacent KODAMA labels to compete only inside `medsam_pre_boundary_radius=64` pixels of an internal boundary. It minimizes `robust Lab/optical-density data + edge-weighted Potts smoothness` by deterministic mean-field annealing from temperature 2.0 to 0.05 over 16 iterations. `medsam_pre_boundary_connectivity=8` uses distance-weighted diagonal neighbours to suppress grid-aligned staircase boundaries; set it to `4` to reproduce the earlier orthogonal-only competition. Candidate invasion remains local, protected cores cannot change, the foreground footprint is invariant, and GrandQC is a hard constraint. `medsam_pre_boundary_downsample=4` bounds memory; `medsam_pre_boundary_radius_um` can replace the pixel radius when verified MPP is available. The remaining data, smoothness and edge parameters are sensitivity settings and require held-out boundary validation before biological claims.
- `medsam_image_guided_internal_refine=true` adds native-resolution marker-controlled watershed after MedSAM cleanup. `medsam_internal_boundary_radius=64` bounds how far an internal label boundary can move; foreground/background membership is invariant. `medsam_internal_gradient_space` selects luminance, Lab, optical-density, or combined Lab/OD evidence; `medsam_internal_gradient_sigma_px` controls the image scale seen by the watershed, and `medsam_internal_watershed_compactness` adds an optional geometric regularizer. The historical settings are `luminance`, `0`, and `0`. `medsam_internal_gradient_sigma_um` can replace the pixel sigma when verified MPP is available. Disable image-guided refinement to compare against pure MedSAM output.
- `medsam_appearance_refine=true` (default) detects large coherent two-domain assignment errors after MedSAM. It clusters multiscale Lab/optical-density H&E features, names appearance groups from eroded pipeline-label cores, and accepts only connected disagreements of at least `medsam_appearance_min_region_area_px`. It never consumes expert annotations and abstains unless exactly two nonzero tissue labels are present. Set it to `false` to reproduce the earlier MedSAM-only behavior.

Optional dual clustering behavior:

- `cluster_primary_variant` defaults to `standard` and preserves the existing clustering choice.
- `cluster_target_clusters=0` preserves the graph-derived community count. A value of 2 or more performs nearest-centroid merging as a sensitivity analysis and requires `cluster_forced_count_sensitivity_acknowledged=true`. The cluster CSV, summary, plots and review landing page are stamped `sensitivity_only`; acknowledgement does not make a forced count a validated primary result.
- `cluster_secondary_variant` defaults to `none`. Set it to `fine` to run a second clustering branch with slightly higher cluster granularity.
- `cluster_secondary_profile=fine` tells `bin/Rcode_Clustering.R` to keep the same KODAMA embedding input but prefer a nearby higher-cluster solution when the score remains close to the standard branch.
- `cluster_fine_resolution_multiplier` is used when `cluster_resolution` is fixed instead of `auto`.
- `cluster_fine_score_margin` controls how far the fine branch is allowed to deviate from the best standard auto-clustering score.
- `cluster_seed=1` records the primary stochastic seed. `cluster_stability_runs=3` repeats the full landmark selection, graph clustering and KNN assignment with consecutive seeds; the per-run adjusted Rand index is written to `<sample>_<variant>_cluster_stability.csv`.
- `cluster_assignment_min_vote_margin=0.10` marks observations with weak separation between the two largest landmark KNN vote totals. `cluster_stability_min_fraction=0.67` requires the seed-aligned cluster label to agree in at least two of three runs.
- `cluster_spatial_k=15` controls the descriptive same-cluster spatial-neighbor calculation; `cluster_spatial_max_observations=50000` bounds its memory and runtime by stratified sampling.
- `cluster_review_per_cluster=20`, `cluster_review_radius_px=256`, and `cluster_assessment_seed=1` define a reproducible blinded region packet. The review form and GeoJSON omit cluster membership; the answer key is written separately.
- GigaTIME marker enrichment is reported when marker quantification is available, but it is labelled as a same-image virtual-marker proxy rather than independent biological validation.
- `cluster_abstain_uncertain=true` retains every algorithmic assignment in `cluster` but writes `NA` to `interpretable_cluster` for ambiguous or unstable observations. Canonical cell and grid masks prefer `interpretable_cluster`, so abstentions are excluded from definitive domains. Stage 11 records `uncertainty_reason`, `is_abstained`, neutral KODAMA membership rendering, a dedicated uncertainty figure, and observation-level abstention GeoJSON. Stage 12 separately publishes a categorical uncertainty raster and H&E overlay, preventing excluded observations from being mistaken for ordinary background.

Additional automatic outputs:

- `04_TMA/<sample>/tma_<sample>/<sample>_tma_spots.geojson` contains TMA spot polygons when the image is detected as a tissue microarray.
- `04_TMA/<sample>/tma_<sample>/<sample>_objects_tma_assigned.csv` contains StarDist objects with appended `tma_spot_*` assignment columns.
- `05_gigatime/<sample>/gigatime_<sample>_ometiff/gigatime_probs.ome.tif` contains the crop-aligned GigaTIME virtual mIF prediction stack. The sibling `gigatime_<sample>/gigatime_probs.zarr` remains the chunked inference store.
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_mean_intensity.csv` and `..._cyto_gigatime_mean_intensity.csv` contain per-object mean uncalibrated virtual-marker scores; `intensity` is retained in the filename for compatibility.
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_intensity_stats.csv` and `..._cyto_gigatime_intensity_stats.csv` also include per-marker score sums and maxima.
- `05_gigatime/<sample>/gigatime_<sample>/gigatime_marker_score_qc.{json,tsv,png}` records sampled score distributions, technical warnings, tissue/background contrast and channel correlations without implying biological calibration.
- `05_gigatime/<sample>/gigatime_<sample>/gigatime_seam_qc.json` and `.png` quantify and visualize every persisted channel at each inference-block boundary, including zero-fill transition fractions.
- If an input ROI GeoJSON was provided, the pipeline rasterizes the crop-aligned ROI into a labeled mask at `06_roi/<sample>/<sample>_input_roi_mask.tif`, writes a preview overlay at `06_roi/<sample>/<sample>_input_roi_mask_preview.png`, and records the value-to-label mapping at `06_roi/<sample>/<sample>_input_roi_mask_labels.json`.
- `11_clustering/<sample>/` contains clustering CSVs, summaries, KODAMA membership plots, stability tables, descriptive interpretation evidence, and blinded-review packets for every configured variant.
- `10_kodama/<sample>/uni2_route_comparison_<sample>/` contains matched-grid cell-versus-grid representation metrics, virtual-marker proxy endpoints, a visual comparison, and an interpretation-limited report when `uni2_sampling_mode=both`.
- `14_medsam_refine_tissue/<sample>/` contains the variant-specific KODAMA membership PNG copied next to each MedSAM-refined mask as `<sample>_<variant>_medsam_kodama_membership.png`.
# Multi-Model Cell Identification

`cell_detection_mode=consensus` enables GPU-only multi-detector instance fusion; `cell_detection_mode=stardist` explicitly selects the single-detector route. The deprecated `cell_consensus_enable` parameter is read only as a compatibility fallback when `cell_detection_mode` is absent. A consensus request on CPU fails rather than silently changing the analyzed cell population. StarDist, HoVer-Net MoNuSAC, and CellViT++ have no inter-detector channel dependency and can run in parallel from the same shared crop and GrandQC mask; fusion waits for all three. `cell_consensus_fusion_acceptance_policy=broad_pair` requires the broad-scope StarDist and CellViT++ pair for a canonical instance; HoVer-Net remains scoped support. `any_two` reproduces the legacy policy and must be treated as a sensitivity analysis. `cell_consensus_min_support` remains an additional minimum-source gate (default `2`). `cell_consensus_match_radius_um` is the direct broad-pair maximum centroid distance in physical units (default `4.0` microns); it is converted to pixels from `shift.json`. `cell_consensus_geometry_priority` deterministically selects the preferred available contour for each accepted component. `cell_consensus_min_agreement_score` can add a stricter broad-detector geometric gate. Instance geometry, broad support, scoped support and detector-specific phenotype evidence are reported separately: the pipeline does not vote incompatible detector taxonomies into a synthetic cell type. `cell_consensus_count_ratio_warning` defaults to `2.0` and reports count imbalance separately for all detectors and for the broad-scope StarDist/CellViT++ pair. Because detector contours can overlap completely, the raster writer reserves one unique centroid-near seed pixel per canonical cell and verifies that every ID in `objects.csv` is present in `labels.tif` before publishing the stage. Pairwise match fractions, centroid distances, polygon IoU and Hausdorff distances are detector-agreement benchmarks, not reference-standard accuracy or calibrated confidence estimates.

HoVer-Net uses `hovernet_target_mpp=0.25`, the official `fast` MoNuSAC checkpoint, and the resource parameters prefixed by `hovernet_`. The MoNuSAC positive classes do not cover all nuclei, so its count is not expected to match StarDist or CellViT++. The wrapper passes the shared GrandQC clean-tissue mask into HoVer-Net WSI inference, records checkpoint/upstream provenance and the model-scope limitation, then applies the same mask again as a centroid-level output invariant. `hovernet_postproc_workers=0` selects a memory-aware worker count (one worker per 6 GB of allocated memory); an explicit value is still capped by that safety limit. `hovernet_prediction_cache` can resume post-processing from a completed upstream `pred_map.npy` after validating it against the regenerated slide geometry, avoiding repeated GPU inference after a post-processing-only failure. CellViT++ uses `cellvit_model` (`HIPT` by default), `cellvit_taxonomy`, mixed precision via `cellvit_amp`, and resource parameters prefixed by `cellvit_`. `cellvit_ray_workers` defaults to `1` because CellViT++ 1.0.9 fails with its upstream zero-worker default; `cellvit_ray_worker_cpus=0` divides the allocated task CPUs automatically. The HIPT weights and classifiers are baked once into `cellvit_cache_dir` inside the versioned GPU image, avoiding duplicate downloads across tasks and runs.

The stage name is `cell_consensus`; aliases `hovernet`, `cellvit`, and `consensus` select the same aggregate stage because both inference branches are required to build the result.

`cellvit_export_embeddings=true` requests optional source-bound numeric graph
tokens. The workflow passes its verified source-resolution report automatically;
standalone `run_cellvitpp.py --export-embeddings` requires `--resolution-json`.
`cellvit_amp=false` disables forced AMP, not the checkpoint's own mixed-precision
default. Physical preprocessing, source population and actual configured CLI
runtime are bound in the completion receipt; see
[the CellViT feature contract](docs/CELL_ATLAS_USAGE.md#source-bound-cellvit-features).

## Neoplastic Section, TITAN, and PathoFMPred

| Parameter | Default | Meaning |
|---|---|---|
| `titan_enable` | `false` | Enable deterministic neoplastic-section selection followed by TITAN embedding. |
| `neoplastic_section_names` | `neoplastic` | Comma-separated named CellViT++ classes counted as neoplastic. Matching is case-insensitive. |
| `neoplastic_section_require_cells` | `true` | Fail if no connected final section contains a named neoplastic cell. |
| `neoplastic_section_padding_um` | `128` | Physical padding around the selected section export. |
| `neoplastic_section_default_mpp` | `0.5` | Fallback MPP only when image metadata are unavailable. |
| `neoplastic_section_spatial_bin_size` | `1024` | Spatial-index bin size in level-0 pixels for streaming cell-to-section assignment. |
| `neoplastic_section_tile_size` | `512` | Tile size for the selected-section mask and OME-TIFF writer. |
| `titan_model` | `MahmoodLab/TITAN` | Official gated Hugging Face model ID or an authorized local snapshot directory. |
| `titan_revision` | `main` | Model revision; production runs should use a pinned commit or a checksum-verified local snapshot. |
| `titan_offline` | `false` | Restrict TITAN to local model files/cache. |
| `titan_cache_dir` | `${baseDir}/../.cache/titan` | Shared external TITAN cache; keep model files outside output/work directories. |
| `titan_target_mpp` | `0.5` | Physical resolution used for CONCH v1.5 patches. |
| `titan_patch_size` | `512` | Patch size in the regularized 20x/0.5-MPP coordinate system. |
| `titan_min_tissue_coverage` | `0.2` | Minimum selected-section mask coverage for a patch. |
| `titan_batch_size` | `0` | CONCH batch size; `0` selects a GPU-memory-aware value. |
| `titan_gpu` | `0` | CUDA device index. TITAN is GPU-only in this pipeline. |
| `pathofmpred_enable` | `false` | Run PathoFMPred after TITAN. This also enables stages 16 and 17. |
| `pathofmpred_cancer` | empty | Required TCGA cancer code, for example `BRCA`. No cancer type is inferred silently. |
| `pathofmpred_library_dir` | `${baseDir}/../.cache/pathofmpred/R_library` | Protected external R library containing PathoFMPred and its private fitted registry. |
| `pathofmpred_rscript` | `/opt/micromamba/envs/kodama-r/bin/Rscript` | R 4.6 executable used for PathoFMPred and the pinned `fastPLS` dependency. |
| `pathofmpred_report_format` | `html` | Report output format. |
| `pathofmpred_include_limited_evidence` | `false` | Include endpoints marked as limited evidence by the package. |

TITAN uses the official model's `return_conch()` patch encoder and `encode_slide_from_patch_features()` aggregator. It writes exactly 768 named features (`titan_000` through `titan_767`) for PathoFMPred. PathoFMPred outputs are research estimates derived from TCGA discovery/internal models; they are not calibrated probabilities or clinical predictions.
