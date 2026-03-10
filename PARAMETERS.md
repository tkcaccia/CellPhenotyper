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
| `folder_input` | `null` | Input folder for multi-sample mode. Supported image extensions: `.ome.tif`, `.ome.tiff`, `.btf`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. |
| `image_input` | `null` | Single-sample input image (`.ome.tif` or `.btf`). Ignored when `folder_input` is set. |
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
| `container_cpu_tag` | `0.2.0` | Legacy generic fallback; architecture-specific CPU tags below are authoritative. |
| `container_cpu_tag_amd64` | `0.2.0-amd64` | CPU tag for amd64 hosts. |
| `container_cpu_tag_arm64` | `0.2.0` | CPU tag for arm64 hosts. |
| `container_gpu_tag` | `0.2.0-gpu-amd64` | GPU tag used when `compute_device` resolves to GPU. |
| `singularity_image_source` | `auto` | `auto` and `release` try release/local `.sif` first. On arm64 GPU runs, missing GPU assets fall back to CPU (not amd64 docker GPU). |
| `singularity_release_repo` | `tkcaccia/CellPhenotyper` | GitHub repo used to resolve release-hosted `.sif` assets. |
| `singularity_release_tag` | `v0.2.0` | GitHub release tag containing `.sif` assets. |
| `singularity_cpu_asset_amd64` | `cellphenotyper-0.2.0-amd64.sif` | CPU Singularity asset name for amd64 hosts. |
| `singularity_cpu_asset_arm64` | `cellphenotyper-0.2.0-arm64.sif` | CPU Singularity asset name for arm64 hosts. |
| `singularity_gpu_asset_amd64` | `cellphenotyper-0.2.0-gpu-amd64.sif` | GPU Singularity asset name for amd64 hosts. |
| `singularity_gpu_asset_arm64` | `cellphenotyper-0.2.0-gpu-arm64.sif` | GPU Singularity asset name for arm64 hosts. |
| `singularity_local_dir` | `''` | Optional local directory with prebuilt `.sif`; checked before release/docker fallback. |
| `singularity_cache_dir` | `''` | Optional Apptainer/Singularity cache path; default is `<repo>/.apptainer_cache`. |
| `cpu_container_image` | `''` | Optional explicit CPU container URI/path. |
| `gpu_container_image` | `''` | Optional explicit GPU container URI/path. |
| `singularity_image` | `''` | Manual container URI/path for `-profile singularity` (`runtime_image_mode: manual`). |
| `docker_image` | `''` | Manual image for `-profile docker` (`runtime_image_mode: manual`). |
| `gpu_debug_diagnostics` | `false` | When true, GPU-capable processes print `nvidia-smi` and framework CUDA diagnostics. |
| `max_cpus` | `4` | Global CPU cap. |
| `max_memory_gb` | `8` | Global RAM cap in GB. |
| `hf_home` | `${baseDir}/.hf_cache` | Hugging Face cache root for UNI-2 model files. |
| `hf_hub_cache` | `${baseDir}/.hf_cache/hub` | Hugging Face Hub cache directory for UNI-2 model files. |
| `hf_hub_offline` | `false` | If `true`, UNI-2 runs in strict offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and only uses local cache. |
| `hf_token_env_var_name` | `HF_UNI2` | Env var name used to read the HuggingFace token for UNI-2. |
| `hf_token_env_file` | `tokens.env` | Env file path sourced at runtime before UNI-2 starts (recommended in repo root for Docker profile). |

ROI resolution rules:
- In `folder_input` mode, each image `<sample>.<supported_extension>` uses `<sample>.geojson` if present in the same folder.
- If `<sample>.geojson` is missing, the ROI defaults to the full image.
- In single-sample mode, `--roi_geojson` is optional; if omitted, `<image_root>.geojson` is searched next to `image_input`, otherwise full-image ROI is generated.

GPU run notes:
- amd64: use `--compute_device gpu --host_arch amd64`.
- arm64: set `--compute_device gpu --host_arch arm64 --enable_gpu_on_arm64 true` and provide an arm64 GPU container (`singularity_gpu_asset_arm64` or `gpu_container_image`).
- If no arm64 GPU container is available, GPU-capable processes fall back to CPU containers with warnings.
- On arm64, StarDist defaults to CPU container unless `--enable_stardist_gpu_on_arm64 true`.
- On GB10 (`sm_121`), use an arm64 GPU SIF built with nightly `cu130` PyTorch (the `v0.2.0` arm64 GPU asset may fail with `no kernel image is available`).

`start_point` / `end_point` allowed values:
`convert`, `stardist`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `cluster_geojson`.

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

## StarDist

| Parameter | Default | Meaning |
|---|---|---|
| `stardist_model` | `2D_versatile_he` | StarDist model preset. |
| `stardist_prob` | `0.52` | Detection probability threshold. |
| `stardist_nms` | `0.28` | NMS threshold. |
| `stardist_keras_home` | `''` | Optional persistent Keras cache directory for StarDist pretrained model files. |
| `stardist_pretrained_zip` | `''` | Optional local zip path (for example `python_2D_versatile_he.zip`) copied into StarDist cache before execution. |
| `stardist_autoinstall_runtime` | `true` | If StarDist/TensorFlow runtime is missing in container, auto-install required Python deps into task-local `.pydeps`. |
| `stardist_tensorflow_version` | `2.16.2` | TensorFlow version used by StarDist runtime auto-install fallback. |
| `stardist_tiles_x` | `32` | Tiles in X. |
| `stardist_tiles_y` | `32` | Tiles in Y. |
| `stardist_pythonpath` | `''` | Optional extra `PYTHONPATH` for StarDist runtime dependencies (e.g. external TensorFlow path on M1). |
| `write_full_labels` | `true` | Write full labels TIFF. |
| `full_format` | `tif` | Full label file format. |
| `allow_huge_tif` | `true` | Allow huge TIFF writes. |
| `stardist_cpus` | `16` | CPU allocation. |
| `stardist_memory_gb` | `48` | RAM allocation. |
| `stardist_time` | `24h` | Time allocation. |

## Tissue mask

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_work_downsample` | `8` | Internal downsample for memory control. |
| `tissue_preview_factor` | `10` | Preview downsample factor. |
| `tissue_close_radius` | `12` | Morphological closing radius. |
| `tissue_min_obj_area` | `30000` | Remove objects below this area. |
| `tissue_hole_area` | `30000` | Fill holes below this area. |
| `tissue_keep_largest` | `true` | Keep largest connected component only. |
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
| KODAMA clustering | `cluster_*` |
| Cluster mask build | `cluster_mask_*` |
| Grow clusters to tissue | `grow_*` |
| Final cluster GeoJSON | `cluster_geojson_*` |
