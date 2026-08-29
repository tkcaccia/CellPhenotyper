# Parameters

All parameters are defined in `nextflow.config` inside `params { ... }`.

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
| `folder_input` | `null` | Input folder for multi-sample mode. Supported image extensions: `.ome.tif`, `.ome.tiff`, `.btf`, `.czi`, `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. |
| `image_input` | `null` | Single-sample input image. Supports `.ome.tif`, `.ome.tiff`, `.btf`, `.czi`, `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.vms`, `.vmu`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. Ignored when `folder_input` is set. |
| `roi_geojson` | `null` | Single-sample ROI GeoJSON path. Ignored when `folder_input` is set. |
| `outdir_base` | `results` | Base output directory. |
| `run_full_pipeline` | `true` | Run complete workflow including UNI-2 and KODAMA. |
| `start_point` | `convert` | Stage where execution starts. |
| `end_point` | `auto` | Stage where execution stops (`auto` = `cluster_geojson` if full pipeline, else `tissue_mask`). |
| `tissue_mask_from_input` | `false` | Build tissue mask from input image directly. |
| `compute_device` | `cpu` | `cpu`, `gpu`, or `auto`. |
| `host_arch` | `auto` | `auto`, `amd64`, or `arm64` host architecture selector/override. |
| `enable_gpu_on_arm64` | `false` | Allow GPU selection on arm64 hosts when a compatible GPU container exists. |
| `enable_stardist_gpu_on_arm64` | `false` | On arm64, keep StarDist on CPU container by default; set `true` to force StarDist into GPU container. |
| `runtime_image_mode` | `auto` | `auto` uses architecture/device-aware image selection; `manual` uses `singularity_image`/`docker_image`. |
| `uni2_device_auto` | `cpu` | UNI-2 device when `compute_device=auto`. |
| `container_repo` | `ghcr.io/tkcaccia/cellphenotyper` | Base GHCR repository used by auto image selection. |
| `container_gpu_repo` | `ghcr.io/tkcaccia/cellphenotyper-runtime` | GHCR repository used for the current amd64 GPU runtime. |
| `container_cpu_tag` | `2.6` | Legacy generic fallback; multi-arch CPU manifest tag. Architecture-specific CPU tags below remain authoritative. |
| `container_cpu_tag_amd64` | `2.6-amd64` | CPU tag for amd64 hosts. |
| `container_cpu_tag_arm64` | `2.6-arm64` | CPU tag for arm64 hosts. |
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
| `gpu_debug_diagnostics` | `false` | When true, GPU-capable processes print `nvidia-smi` and framework CUDA diagnostics. |
| `max_cpus` | `4` | Global CPU cap. |
| `max_memory_gb` | `8` | Global RAM cap in GB. |
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

GPU run notes:
- amd64: use `--compute_device gpu --host_arch amd64`.
- arm64: set `--compute_device gpu --host_arch arm64 --enable_gpu_on_arm64 true` and provide an arm64 GPU container (`singularity_gpu_asset_arm64` or `gpu_container_image`).
- If no arm64 GPU container is available, GPU-capable processes fall back to CPU containers with warnings.
- On arm64, StarDist defaults to CPU container unless `--enable_stardist_gpu_on_arm64 true`.
- On GB10 (`sm_121`), use an arm64 GPU SIF built with nightly `cu130` PyTorch (the `v2.2` arm64 GPU asset may fail with `no kernel image is available`).

`start_point` / `end_point` allowed values:
`convert`, `grandqc`, `stardist`, `cell_consensus`, `tma`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `gigatime`, `marker_quantification`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `cluster_geojson`, `neoplastic_section`, `titan`, `pathofmpred`.

## Input Resolution

The `convert` stage validates physical pixel size before doing expensive conversion and validates it again on the generated OME-TIFF. Source and converted reports are published in `01_input/<sample>/`. Upsampling is still permitted within the accepted range, but the report states explicitly that resampling cannot create additional spatial detail.

| Parameter | Default | Meaning |
|---|---|---|
| `input_resolution_check` | `true` | Run source and post-conversion MPP validation. |
| `input_resolution_strict` | `true` | Stop before cell analysis when resolution metadata is missing or outside the configured contract. |
| `input_resolution_min_mpp` | `0.05` | Lower metadata sanity bound in µm/px. |
| `input_resolution_max_mpp` | `0.50` | Coarsest accepted native resolution in µm/px. This limits linear upsampling to 2× for 0.25-µm cell models. |
| `input_resolution_cell_target_mpp` | `0.25` | Reference cell-model MPP used to report the required upsampling factor. |
| `input_resolution_max_anisotropy_fraction` | `0.05` | Maximum relative difference between X and Y MPP. |
| `input_resolution_max_conversion_drift_fraction` | `0.02` | Maximum MPP change permitted during conversion. |
| `input_resolution_override_mpp` | `0.0` | Explicit isotropic source MPP override; `0` disables the override. Use only for independently verified metadata errors. |

## Conversion (`.btf` -> `.ome.tif`)

| Parameter | Default | Meaning |
|---|---|---|
| `convert_compression` | `LZW` | Compression mode. |
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
| `grandqc_artifact_mpp_model` | `auto` | Select the official 1.0, 1.5, or 2.0 MPP artifact checkpoint from source-image MPP. |
| `grandqc_tissue_mpp_model` | `10.0` | Physical scale requested by the tissue detector. |
| `grandqc_patch_size` | `512` | Official tissue-detector patch size. It does not control artifact context tiling. |
| `grandqc_artifact_tile_size` | `0` | Artifact context tile size. `0` selects 1024 on CUDA GPUs with at least 8 GiB VRAM and 512 otherwise. Larger fully convolutional context suppresses tile-position bands without changing model MPP. Explicit values must be divisible by 32. |
| `grandqc_artifact_overlap_fraction` | `0.5` | Fractional artifact-tile overlap used for probability blending. |
| `grandqc_preview_max_side` | `4096` | Maximum long side of GrandQC preview assets. |

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
| `gigatime_skip_background_blocks` | `true` | Skip coarse-mask-confirmed blank blocks during chunked inference. Blocks containing nucleus or cytoplasm labels are always processed even if the coarse tissue estimate marks them as background. |
| `gigatime_strict_target_mpp` | `true` | Enforce the requested GigaTIME physical scale from image metadata with exact floating-point resampling. This includes upsampling when the source MPP is coarser than the model target; GigaTIME fails if MPP cannot be resolved and does not coarsen the image to satisfy the output-size budget. |
| `gigatime_max_output_gib` | `8.0` | Maximum estimated uncompressed persisted GigaTIME image size before automatic extra downsampling is applied when strict target MPP is disabled. |
| `gigatime_output_format` | `zarr` | Persist the GigaTIME marker image as chunked `gigatime_probs.zarr` by default. When `ome_tiff` is requested, the saved `gigatime_probs.ome.tif` is pyramidal by default for QuPath/WSI viewing. |
| `gigatime_output_channels` | `DAPI,PD-1,CD3,CD8,PD-L1` | Marker channels persisted in the GigaTIME image store. Integrated single-cell quantification is still computed from all GigaTIME model channels. |
| `gigatime_export_ometiff` | `true` | Export the GigaTIME store to a pyramidal multichannel `gigatime_probs.ome.tif` after prediction. This keeps inference reliable with Zarr while still producing a QuPath-ready OME-TIFF. |
| `gigatime_integrated_quantification` | `true` | Quantify all 23 GigaTIME model markers over nuclei and cytoplasm during the same tiled inference pass. The persisted image may remain a smaller selected channel subset; full-marker quantification does not require storing a 23-channel WSI. |
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

## Tissue mask

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_work_downsample` | `8` | Internal downsample for memory control. |
| `tissue_preview_factor` | `10` | Preview downsample factor. |
| `tissue_close_radius` | `12` | Morphological closing radius. |
| `tissue_min_obj_area` | `30000` | Remove objects below this area. |
| `tissue_hole_area` | `30000` | Fill holes below this area. |
| `tissue_keep_largest` | `false` | Keep only the largest connected component when explicitly enabled. Leave disabled for TMA and other multi-fragment tissue. |
| `tissue_tile` | `512` | Output tile size. |
| `tissue_compression` | `deflate` | TIFF compression. |
| `tissue_bigtiff` | `true` | Use BigTIFF output. |
| `tissue_mask_cpus` | `8` | CPU allocation. |
| `tissue_mask_memory_gb` | `16` | RAM allocation. |
| `tissue_mask_time` | `6h` | Time allocation. |

## Final cluster GeoJSON

| Parameter | Default | Meaning |
|---|---|---|
| `cluster_geojson_page` | `0` | TIFF page index. |
| `cluster_geojson_dissolve_by_value` | `true` | Dissolve polygons by label value. |
| `cluster_geojson_min_area` | `500` | Minimum polygon area. |
| `cluster_geojson_smooth_buffer` | `10.0` | Buffer-based smoothing radius. |
| `cluster_geojson_smooth_passes` | `3` | Number of smoothing passes. |
| `cluster_geojson_simplify` | `6.0` | Simplification tolerance. |
| `cluster_geojson_preserve_topology` | `true` | Preserve topology on simplify. |
| `cluster_geojson_fill_holes` | `true` | Remove interior holes. |
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
| UNI-2 embeddings | `uni2_*`, `hf_*`, `hf_token_env_var_name` |
| KODAMA R step | `r_*` |
| KODAMA clustering | `cluster_*`, `cluster_primary_variant`, `cluster_secondary_variant`, `cluster_secondary_profile`, `cluster_fine_resolution_multiplier`, `cluster_fine_score_margin` |
| Cluster mask build | `cluster_mask_*` |
| Grow clusters to tissue | `grow_*` |
| MedSAM border refinement | `medsam_refine_*`, `medsam_*` |
| Final cluster GeoJSON | `cluster_geojson_*` |

UNI-2 behavior:

- `uni2_fuse_tile_inner_square=true` is the optimized default: one Python/model session loads UNI2-h once, prepares StarDist-centered tile crops and fixed centered inner-square crops in memory, and runs both image streams through UNI2-h in the same batched call. The `inner_square` family does not use the cytoplasm mask; it uses `uni2_inner_square_fixed_px=90` in the 224 x 224 UNI2 input space. Set this to `false` for comparison runs that use the slower two-pass behavior: one centered tile pass plus one separate `inner_square` masked-image forward pass.
- `uni2_target_mpp=0.25` makes the source crop physically calibrated before the 224 x 224 UNI2 input transform. If a StarDist crop lost OME pixel-size metadata, UNI2 recovers the MPP from the crop `shift.json` and the original input image when possible; otherwise it uses `uni2_default_source_mpp`.
- For a source image at `0.08706 µm/px`, the default 224-pixel UNI2 input is extracted from a 643-pixel source crop, giving an effective resolution of about `0.25 µm/px`.
- `uni2_save_tiles=false` is the production default. KODAMA reads the embedding CSVs, not the per-cell PNG crops, so saving tiles is mainly for debugging/QC and can dominate disk I/O on large runs.
- `uni2_reuse_existing=false` is the fresh-run default. Set `--uni2_reuse_existing true` for recovery runs where `09_embeddings/<sample>/embeddings_<sample>_*` already exists and should be consumed by KODAMA instead of scheduling UNI-2 again, even if the requested stage window includes `uni2`.
- `kodama_r_library_dir` and `cluster_r_library_dir` isolate R package lookup to the bundled R 4.6 and StarDist R libraries, respectively. The pipeline also disables host `.Renviron` and `.Rprofile` files so Singularity cannot accidentally load ABI-incompatible packages from the user's home directory.

MedSAM behavior:

- MedSAM refinement runs per cluster label. For large WSI crops, each cluster border is split into overlapping tiles controlled by `medsam_cluster_tile_size` and `medsam_cluster_tile_overlap`, so a tile may legitimately contain only one cluster.
- Large masks use downsampled morphology for protected-core and editable-band construction, and full-resolution work is limited to the cluster-border tile passed to MedSAM. CUDA is still required when `medsam_device=cuda`.

Dual clustering behavior:

- `cluster_primary_variant` defaults to `standard` and preserves the existing clustering choice.
- `cluster_secondary_variant` defaults to `fine` and runs a second clustering branch with slightly higher cluster granularity.
- `cluster_secondary_profile=fine` tells `bin/Rcode_Clustering.R` to keep the same KODAMA embedding input but prefer a nearby higher-cluster solution when the score remains close to the standard branch.
- `cluster_fine_resolution_multiplier` is used when `cluster_resolution` is fixed instead of `auto`.
- `cluster_fine_score_margin` controls how far the fine branch is allowed to deviate from the best standard auto-clustering score.

Additional automatic outputs:

- `04_TMA/<sample>/tma_<sample>/<sample>_tma_spots.geojson` contains TMA spot polygons when the image is detected as a tissue microarray.
- `04_TMA/<sample>/tma_<sample>/<sample>_objects_tma_assigned.csv` contains StarDist objects with appended `tma_spot_*` assignment columns.
- `05_gigatime/<sample>/gigatime_<sample>_ometiff/gigatime_probs.ome.tif` contains the crop-aligned GigaTIME virtual mIF prediction stack. The sibling `gigatime_<sample>/gigatime_probs.zarr` remains the chunked inference store.
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_mean_intensity.csv` and `..._cyto_gigatime_mean_intensity.csv` contain per-object mean marker intensities.
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_intensity_stats.csv` and `..._cyto_gigatime_intensity_stats.csv` also include per-marker sums and maxima.
- If an input ROI GeoJSON was provided, the pipeline rasterizes the crop-aligned ROI into a labeled mask at `06_roi/<sample>/<sample>_input_roi_mask.tif`, writes a preview overlay at `06_roi/<sample>/<sample>_input_roi_mask_preview.png`, and records the value-to-label mapping at `06_roi/<sample>/<sample>_input_roi_mask_labels.json`.
- `11_clustering/<sample>/` now contains both `standard` and `fine` clustering CSVs, summaries, and KODAMA membership plots as both PDF and PNG.
- `14_medsam_refine_tissue/<sample>/` now contains the variant-specific KODAMA membership PNG copied next to the MedSAM-refined mask outputs for both clustering branches as `<sample>_<variant>_medsam_kodama_membership.png`.
# Multi-Model Cell Identification

`cell_consensus_enable` enables the GPU-only post-StarDist ensemble. On a resolved CPU run the pipeline logs a warning and retains StarDist outputs rather than launching unavailable GPU models. HoVer-Net MoNuSAC and CellViT++ run serially to cap peak VRAM, then `cell_consensus_min_support` controls how many methods must identify a cell (default `2`). `cell_consensus_match_radius_um` is the maximum centroid distance in physical units (default `4.0` microns); it is converted to pixels from `shift.json`. `cell_consensus_geometry_priority` deterministically selects the preferred available contour for each accepted component. Because detector contours can overlap completely, the raster writer reserves one unique centroid-near seed pixel per canonical cell and verifies that every ID in `objects.csv` is present in `labels.tif` before publishing the stage.

HoVer-Net uses `hovernet_target_mpp=0.25`, the official `fast` MoNuSAC checkpoint, and the resource parameters prefixed by `hovernet_`. `hovernet_postproc_workers=0` selects a memory-aware worker count (one worker per 6 GB of allocated memory); an explicit value is still capped by that safety limit. `hovernet_prediction_cache` can resume post-processing from a completed upstream `pred_map.npy` after validating it against the regenerated slide geometry, avoiding repeated GPU inference after a post-processing-only failure. CellViT++ uses `cellvit_model` (`HIPT` by default), `cellvit_taxonomy`, mixed precision via `cellvit_amp`, and resource parameters prefixed by `cellvit_`. `cellvit_ray_workers` defaults to `1` because CellViT++ 1.0.9 fails with its upstream zero-worker default; `cellvit_ray_worker_cpus=0` divides the allocated task CPUs automatically. The HIPT weights and classifiers are baked once into `cellvit_cache_dir` inside the versioned GPU image, avoiding duplicate downloads across tasks and runs.

The stage name is `cell_consensus`; aliases `hovernet`, `cellvit`, and `consensus` select the same aggregate stage because both inference branches are required to build the result.

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
