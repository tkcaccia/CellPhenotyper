# CellPhenotyper

CellPhenotyper runs with **Nextflow** as the main command (`nextflow run main.nf`).

## Installation

Use the dedicated installation page:

- `INSTALL.md`

That page includes:

- Ubuntu, macOS, and Windows setup
- both Singularity and Docker runtime paths
- start/end step options so users can begin from the right point

## Complete ROI example (full pipeline + UNI-2)

After completing installation through `INSTALL.md` (including token setup), run:

```bash
nextflow run main.nf \
  -profile singularity \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --singularity_image singularity/cellphenotyper_full_cpu.sif \
  --run_full_pipeline true \
  --tissue_mask_from_input false \
  --compute_device cpu \
  --outdir_base results_full \
  --max_cpus 8 \
  --max_memory_gb 32 \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

For Docker runtime, use the equivalent command in `INSTALL.md` at **Step 8-D**.

Main outputs:

- `results_full/04_tissue_geojson/`
- `results_full/05_cell_assignments/`
- `results_full/06_cytoplasm/`
- `results_full/07_embeddings/`
- `results_full/08_kodama/`

Validated tissue GeoJSON example committed in this repository:

- `examples/validated_outputs/ROI_tissue_mask.geojson`

## Where to modify parameters

Edit defaults in:

- `nextflow.config` (inside `params { ... }`)

Or override per run from CLI with `--parameter value`.

Example override:

```bash
nextflow run main.nf -profile singularity \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --singularity_image singularity/cellphenotyper_full_cpu.sif \
  --max_cpus 12 \
  --max_memory_gb 48 \
  --uni2_batch 32
```
## Complete parameter reference

Every parameter below can be modified in `nextflow.config` (`params`) or overridden on the CLI with `--<name>`.

## Core I/O and execution

| Parameter | Default | Meaning |
|---|---|---|
| `image_input` | `null` | Input image path (`.ome.tif` or `.btf`). |
| `roi_geojson` | `null` | ROI GeoJSON path used by ROI-aware steps. |
| `outdir_base` | `results` | Base output directory. |
| `run_full_pipeline` | `true` | `true` runs full workflow (UNI-2 + KODAMA included). |
| `tissue_mask_from_input` | `false` | `true` builds tissue mask directly from input image; `false` uses StarDist crop output. |
| `compute_device` | `cpu` | Main device mode: `cpu`, `gpu`, or `auto`. |
| `uni2_device_auto` | `cpu` | Device used by UNI-2 only when `compute_device=auto`. |
| `singularity_image` | `''` | Singularity image path for `-profile singularity`. |
| `docker_image` | `''` | Docker image for `-profile docker`. |
| `max_cpus` | `Runtime.runtime.availableProcessors()` | Global CPU cap for all process-level CPU requests. |
| `max_memory_gb` | `64` | Global RAM cap (GB) for all process-level memory requests. |

## BTF -> OME-TIFF conversion

| Parameter | Default | Meaning |
|---|---|---|
| `btf_converter_script` | `bin/convert_btf_to_ometiff.sh` | Converter script path. |
| `convert_compression` | `LZW` | OME-TIFF compression mode. |
| `convert_downsample` | `GAUSSIAN` | Pyramid downsample algorithm. |
| `convert_rgb` | `true` | Convert to RGB output when possible. |
| `convert_overwrite` | `true` | Overwrite output if it already exists. |
| `convert_cpus` | `8` | CPU request for conversion step. |
| `convert_memory_gb` | `16` | RAM request (GB) for conversion step. |
| `convert_time` | `6h` | Wall time request for conversion step. |

## StarDist segmentation

| Parameter | Default | Meaning |
|---|---|---|
| `stardist_script` | `bin/run_stardist_roi_segmentation.py` | StarDist runner script path. |
| `stardist_model` | `2D_versatile_he` | Pretrained StarDist model name. |
| `stardist_prob` | `0.48` | Probability threshold for detections. |
| `stardist_nms` | `0.30` | Non-max suppression threshold. |
| `stardist_tiles_x` | `32` | Number of StarDist tiles in X. |
| `stardist_tiles_y` | `32` | Number of StarDist tiles in Y. |
| `write_full_labels` | `true` | Write full-image labels output. |
| `full_format` | `tif` | Output format for full labels. |
| `allow_huge_tif` | `true` | Allow huge TIFF writing in StarDist script. |
| `stardist_cpus` | `16` | CPU request for StarDist step. |
| `stardist_memory_gb` | `48` | RAM request (GB) for StarDist step. |
| `stardist_time` | `24h` | Wall time request for StarDist step. |

## Tissue mask generation

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_mask_script` | `bin/build_tissue_mask.py` | Tissue mask script path. |
| `tissue_work_downsample` | `8` | Downsample factor for mask computation (higher = lower RAM). |
| `tissue_preview_factor` | `10` | Downsample factor for preview image. |
| `tissue_close_radius` | `12` | Morphological closing radius (pixels). |
| `tissue_min_obj_area` | `30000` | Remove connected components smaller than this area. |
| `tissue_hole_area` | `30000` | Fill holes smaller than this area. |
| `tissue_keep_largest` | `true` | Keep only largest connected tissue component. |
| `tissue_tile` | `512` | TIFF tile size for output mask. |
| `tissue_compression` | `deflate` | Compression for output tissue mask TIFF. |
| `tissue_bigtiff` | `true` | Write BigTIFF for large outputs. |
| `tissue_mask_cpus` | `8` | CPU request for tissue-mask step. |
| `tissue_mask_memory_gb` | `16` | RAM request (GB) for tissue-mask step. |
| `tissue_mask_time` | `6h` | Wall time request for tissue-mask step. |

## Tissue mask -> GeoJSON

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_geojson_script` | `bin/convert_tissue_mask_to_geojson.py` | GeoJSON conversion script path. |
| `tissue_geojson_page` | `0` | TIFF page index to read (usually full-res page 0). |
| `tissue_geojson_binary` | `true` | Treat mask as binary foreground. |
| `tissue_geojson_dissolve` | `true` | Merge binary polygons into one geometry. |
| `tissue_geojson_dissolve_by_value` | `false` | For labeled masks: dissolve geometries by value. |
| `tissue_geojson_min_area` | `2500` | Drop polygons below this area. |
| `tissue_geojson_smooth_buffer` | `6` | Smoothing buffer radius (pixels). |
| `tissue_geojson_smooth_passes` | `2` | Number of smoothing passes. |
| `tissue_geojson_simplify` | `2` | Simplification tolerance (pixels). |
| `tissue_geojson_preserve_topology` | `true` | Preserve topology during simplification. |
| `tissue_geojson_fill_holes` | `true` | Remove interior holes in polygons. |
| `tissue_geojson_cpus` | `8` | CPU request for GeoJSON conversion step. |
| `tissue_geojson_memory_gb` | `16` | RAM request (GB) for GeoJSON conversion step. |
| `tissue_geojson_time` | `6h` | Wall time request for GeoJSON conversion step. |

## Cell-to-ROI assignment

| Parameter | Default | Meaning |
|---|---|---|
| `assign_script` | `bin/map_cells_to_roi_polygons.py` | Assignment script path. |
| `assign_label_prop` | `name` | ROI property used as label. |
| `assign_out_col` | `polygon_label` | Output column name for assigned label. |
| `assign_choose` | `smallest` | Rule when multiple polygons contain a cell. |
| `assign_chunk_rows` | `20000` | CSV chunk size during assignment. |
| `assign_xcol` | `null` | Optional custom X coordinate column in objects CSV. |
| `assign_ycol` | `null` | Optional custom Y coordinate column in objects CSV. |
| `assign_cpus` | `12` | CPU request for assignment step. |
| `assign_memory_gb` | `24` | RAM request (GB) for assignment step. |
| `assign_time` | `6h` | Wall time request for assignment step. |

## Cytoplasm expansion

| Parameter | Default | Meaning |
|---|---|---|
| `expand_script` | `bin/expand_labels_to_cytoplasm.py` | Cytoplasm expansion script path. |
| `expand_px` | `12` | Expansion distance in pixels. |
| `expand_compression` | `zlib` | Compression mode for expanded label TIFF output. |
| `expand_cpus` | `8` | CPU request for expansion step. |
| `expand_memory_gb` | `16` | RAM request (GB) for expansion step. |
| `expand_time` | `4h` | Wall time request for expansion step. |

## UNI / UNI-2 embeddings

| Parameter | Default | Meaning |
|---|---|---|
| `uni2_script` | `bin/extract_uni2_embeddings.py` | UNI embedding script path. |
| `uni2_image_level` | `0` | Image pyramid level used for tiles. |
| `uni2_force_full_image` | `true` | Force full image load before sampling tiles. |
| `uni2_grid` | `10x10` | Spatial grid for batching cells/tiles. |
| `uni2_tile_size` | `224` | Tile size in pixels. |
| `uni2_save_tiles` | `true` | Save extracted tiles to disk. |
| `uni2_tiles_root` | `tiles` | Root folder name for saved tiles. |
| `uni2_bucket_size` | `5000` | Number of cells per tile subfolder bucket. |
| `uni2_min_area` | `0` | Minimum cell area filter before embeddings. |
| `uni2_encoder` | `uni2-h` | Encoder preset/name (UNI2 default). |
| `uni2_backend` | `auto` | Backend choice (`auto`, `timm_hf`, `hf_transformers`, `dinov2_hub`). |
| `uni2_pooling` | `auto` | Token pooling strategy. |
| `uni2_img_size` | `224` | Model input image size. |
| `uni2_batch` | `64` | Embedding batch size. |
| `uni2_torch_threads` | `16` | Torch thread count. |
| `uni2_rows_per_csv` | `10000` | Rows written per embedding CSV shard. |
| `uni2_mask_block` | `4096` | Block size for mask streaming. |
| `uni2_cpus` | `16` | CPU request for UNI step. |
| `uni2_memory_gb` | `48` | RAM request (GB) for UNI step. |
| `uni2_time` | `24h` | Wall time request for UNI step. |

## Hugging Face cache/token wiring

| Parameter | Default | Meaning |
|---|---|---|
| `hf_home` | `${baseDir}/.hf_cache` | Hugging Face home/cache root path. |
| `hf_hub_cache` | `${baseDir}/.hf_cache/hub` | Hugging Face hub cache path. |
| `hf_token_env_var_name` | `HF_TOKEN` | Environment variable name read for token. |

## R / KODAMA step

| Parameter | Default | Meaning |
|---|---|---|
| `r_script` | `bin/run_kodama_analysis.R` | R script path for KODAMA analysis. |
| `r_cpus` | `8` | CPU request for R/KODAMA step. |
| `r_memory_gb` | `24` | RAM request (GB) for R/KODAMA step. |
| `r_time` | `8h` | Wall time request for R/KODAMA step. |
