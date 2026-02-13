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
| `end_point` | `auto` | Stage where execution stops (`auto` = `kodama` if full pipeline, else `tissue_geojson`). |
| `tissue_mask_from_input` | `false` | Build tissue mask from input image directly. |
| `compute_device` | `cpu` | `cpu`, `gpu`, or `auto`. |
| `uni2_device_auto` | `cpu` | UNI-2 device when `compute_device=auto`. |
| `singularity_image` | `''` | Container path for `-profile singularity`. |
| `docker_image` | `''` | Image name for `-profile docker`. |
| `max_cpus` | `Runtime.runtime.availableProcessors()` | Global CPU cap. |
| `max_memory_gb` | `64` | Global RAM cap in GB. |
| `hf_token_env_var_name` | `HF_TOKEN` | Env var name used to read the HuggingFace token for UNI-2. |

`start_point` / `end_point` allowed values:
`convert`, `stardist`, `tissue_mask`, `tissue_geojson`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`.

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

## Tissue GeoJSON

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_geojson_page` | `0` | TIFF page index. |
| `tissue_geojson_binary` | `true` | Treat mask as binary. |
| `tissue_geojson_dissolve` | `true` | Dissolve binary polygons. |
| `tissue_geojson_dissolve_by_value` | `false` | Dissolve by value for labeled masks. |
| `tissue_geojson_min_area` | `2500` | Minimum polygon area. |
| `tissue_geojson_smooth_buffer` | `6` | Buffer-based smoothing radius. |
| `tissue_geojson_smooth_passes` | `2` | Number of smoothing passes. |
| `tissue_geojson_simplify` | `2` | Simplification tolerance. |
| `tissue_geojson_preserve_topology` | `true` | Preserve topology on simplify. |
| `tissue_geojson_fill_holes` | `true` | Remove interior holes. |
| `tissue_geojson_cpus` | `8` | CPU allocation. |
| `tissue_geojson_memory_gb` | `16` | RAM allocation. |
| `tissue_geojson_time` | `6h` | Time allocation. |

## Cell assignment + cytoplasm + UNI-2 + R

| Parameter group | Key controls |
|---|---|
| Cell assignment | `assign_*` |
| Cytoplasm expansion | `expand_*` (`expand_full_labels=false` skips full-image cytoplasm expansion) |
| UNI-2 embeddings | `uni2_*`, `hf_*`, `hf_token_env_var_name` |
| KODAMA R step | `r_*` |
