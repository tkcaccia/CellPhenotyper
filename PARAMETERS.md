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
  --max_cpus 12 \
  --max_memory_gb 48 \
  --uni2_batch 32
```

## Core

| Parameter | Default | Meaning |
|---|---|---|
| `image_input` | `null` | Input image (`.ome.tif` or `.btf`). |
| `roi_geojson` | `null` | ROI GeoJSON path. |
| `outdir_base` | `results` | Base output directory. |
| `run_full_pipeline` | `true` | Run complete workflow including UNI-2 and KODAMA. |
| `start_point` | `convert` | Stage where execution starts. |
| `end_point` | `auto` | Stage where execution stops (`auto` = `cluster_geojson` if full pipeline, else `tissue_mask`). |
| `tissue_mask_from_input` | `false` | Build tissue mask from input image directly. |
| `compute_device` | `cpu` | `cpu`, `gpu`, or `auto` (auto picks GPU only on amd64 + NVIDIA). |
| `host_arch` | `auto` | `auto`, `amd64`, or `arm64` host architecture selector/override. |
| `runtime_image_mode` | `auto` | `auto` uses architecture/device-aware image selection; `manual` uses `singularity_image`/`docker_image`. |
| `uni2_device_auto` | `cpu` | UNI-2 device when `compute_device=auto`. |
| `container_repo` | `ghcr.io/tkcaccia/cellphenotyper` | Base GHCR repository used by auto image selection. |
| `container_cpu_tag` | `0.2.0` | Generic CPU tag fallback. |
| `container_cpu_tag_amd64` | `0.2.0-amd64` | CPU tag for amd64 hosts. |
| `container_cpu_tag_arm64` | `0.2.0` | CPU tag for arm64 hosts. |
| `container_gpu_tag` | `0.2.0-gpu` | GPU tag used when `compute_device` resolves to GPU. |
| `singularity_image_source` | `auto` | `auto` tries release `.sif` first and falls back to `docker://`; `release` is strict release-only; `docker` forces `docker://`. |
| `singularity_release_repo` | `tkcaccia/CellPhenotyper` | GitHub repo used to resolve release-hosted `.sif` assets. |
| `singularity_release_tag` | `v0.2.0` | GitHub release tag containing `.sif` assets. |
| `singularity_cpu_asset_amd64` | `cellphenotyper-0.2.0-amd64.sif` | CPU Singularity asset name for amd64 hosts. |
| `singularity_cpu_asset_arm64` | `cellphenotyper-0.2.0-arm64.sif` | CPU Singularity asset name for arm64 hosts. |
| `singularity_gpu_asset_amd64` | `cellphenotyper-0.2.0-gpu-amd64.sif` | GPU Singularity asset name for amd64 hosts. |
| `singularity_image` | `''` | Manual container URI/path for `-profile singularity` (`runtime_image_mode: manual`). |
| `docker_image` | `''` | Manual image for `-profile docker` (`runtime_image_mode: manual`). |
| `max_cpus` | `Runtime.runtime.availableProcessors()` | Global CPU cap. |
| `max_memory_gb` | `64` | Global RAM cap in GB. |
| `hf_token_env_var_name` | `HF_TOKEN` | Env var name used to read the HuggingFace token for UNI-2. |
| `hf_token_env_file` | `''` | Optional env file path (for example `tokens.env`) sourced at runtime before UNI-2 starts. |

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
| `stardist_prob` | `0.48` | Detection probability threshold. |
| `stardist_nms` | `0.30` | NMS threshold. |
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
