# CellPhenotyper

CellPhenotyper is a Nextflow DSL2 pipeline for H&E image phenotyping from ROI segmentation to final cluster GeoJSON.

The pipeline is architecture-aware at runtime:

- detects host architecture (`amd64` or `arm64`)
- resolves compute mode (`cpu`/`gpu` from `compute_device`)
- selects the proper container tag automatically

Main command:

```bash
nextflow run main.nf
```

## Quick start

Inputs already included in this repository:

- `Data/ROI.ome.tif`
- `Data/ROI.geojson`

Run one profile only (`singularity` or `docker`):

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper

nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

Expected final output:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

## Runtime auto-selection

Default runtime behavior is fully automatic (see `nextflow.config` + `pipeline_paramers.yml`):

- `compute_device: auto`
- `host_arch: auto`
- `singularity_image: ""`
- `docker_image: ""`

Container tags used by auto-selection:

- `container_cpu_tag_arm64: 0.2.0`
- `container_cpu_tag_amd64: 0.2.0-amd64`
- `container_gpu_tag: 0.2.0-gpu`

You can force architecture if needed:

```bash
--host_arch amd64
```

You can still override images directly:

```bash
--singularity_image docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
# or
--docker_image ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
```

## UNI-2 token

UNI-2 requires a Hugging Face token with access to `MahmoodLab/UNI2-h`.

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
```

Pipeline defaults:

- `hf_token_env_file: tokens.env`
- `hf_token_env_var_name: HF_UNI2`

## Singularity definitions in repository

The repository includes updated definition files:

- `singularity/cellphenotyper_full_cpu.def`
- `singularity/cellphenotyper_full_gpu.def`

Users do not need to build `.sif` manually for standard runs.
With `-profile singularity`, Nextflow pulls from GHCR automatically.

## Documentation

- [Installation](INSTALL.md)
- [How to use](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Release](RELEASE.md)
