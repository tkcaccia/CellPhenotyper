# Parameters

All defaults are in `nextflow.config` (`params { ... }`).

Use overrides with:

```bash
nextflow run main.nf -profile singularity -params-file pipeline_paramers.yml --max_cpus 12
```

## Core runtime parameters

| Parameter | Default | Meaning |
|---|---|---|
| `compute_device` | `auto` | `cpu`, `gpu`, or `auto`. |
| `host_arch` | `auto` | `auto`, `amd64`, or `arm64`. |
| `container_repo` | `ghcr.io/tkcaccia/cellphenotyper` | Base image repo. |
| `container_cpu_tag` | `0.2.0` | Generic CPU tag fallback. |
| `container_cpu_tag_amd64` | `0.2.0` | CPU tag for amd64 hosts. |
| `container_cpu_tag_arm64` | `0.2.0` | CPU tag for arm64 hosts. |
| `container_gpu_tag` | `0.2.0-gpu` | GPU tag (amd64 + NVIDIA). |
| `singularity_image` | `""` | If empty, auto-resolved from repo/tag/arch/device. |
| `docker_image` | `""` | If empty, auto-resolved from repo/tag/arch/device. |
| `max_cpus` | `Runtime.runtime.availableProcessors()` | Global CPU cap. |
| `max_memory_gb` | `64` | Global RAM cap. |

## Input/output parameters

| Parameter | Default | Meaning |
|---|---|---|
| `image_input` | `null` | Input image (`.ome.tif` or `.btf`). |
| `roi_geojson` | `null` | ROI GeoJSON. |
| `outdir_base` | `results` | Output root directory. |
| `run_full_pipeline` | `true` | Run full workflow. |
| `start_point` | `convert` | First stage to run. |
| `end_point` | `auto` | Last stage to run. |

## Token parameters

| Parameter | Default | Meaning |
|---|---|---|
| `hf_token_env_file` | `''` | Env file containing UNI-2 token. |
| `hf_token_env_var_name` | `HF_TOKEN` | Token variable name in env file. |

## Stage names

Use these for `start_point` and `end_point`:

`convert`, `stardist`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `cluster_geojson`.
