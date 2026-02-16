# CellPhenotyper

CellPhenotyper is a fully reproducible, modular computational pathology workflow that takes raw H&E whole-slide images to quantitative cell-level phenotypes.  
The pipeline performs image conversion/ROI handling, StarDist-based ROI-informed segmentation, UNI-2 foundation-model embeddings, unsupervised KODAMA analysis, and post-KODAMA spatial region construction (`Rcode_Clustering`, `labels_to_cluster_mask`, `grow_to_tissue`, `mask_to_geojson`) to produce final cluster-level GeoJSON.

The workflow is implemented in Nextflow DSL2 with containerized execution for portability, scalability, and auditability across datasets and compute environments.  
Select the runtime image in `pipeline_paramers.yml` using `singularity_image` or `docker_image`.
For full UNI-2 execution, the Hugging Face token must have access to `MahmoodLab/UNI2-h`.
Token loading is configured via `hf_token_env_file` and `hf_token_env_var_name` in `pipeline_paramers.yml`.

The main entrypoint is always:

```bash
nextflow run main.nf
```

## Start here: clone, pull container, run first example

Use this first if you want to run immediately with the repository example input files:

- `Data/ROI.ome.tif`
- `Data/ROI.geojson`

Current runtime image version:

- CPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

1. Clone repository:

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

2. Pull runtime image (choose one):

```bash
# Singularity/Apptainer image
./docker/pull_runtime_image.sh --tag 0.2.0 --variant cpu --no-docker

# Docker image
./docker/pull_runtime_image.sh --tag 0.2.0 --variant cpu --no-singularity
```

3. Add UNI-2 token file:

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
```

4. Run first full example (choose one profile):

```bash
# Singularity
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example

# Docker
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

5. Final result:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

## Documentation

- [Installation](INSTALL.md)
- [How to use](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Release](RELEASE.md)

## Quick start

After completing installation and UNI-2 token setup, edit `pipeline_paramers.yml` (`start_point` / `end_point` included) and run:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -resume
```

## Runtime images (GHCR)

Default runtime tags are hosted on GHCR:

- CPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- GPU: `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

Pull helpers:

```bash
./docker/pull_runtime_image.sh --tag 0.2.0 --variant cpu
```

This pulls Docker and, if available, also materializes a Singularity/Apptainer SIF in `singularity/`.

Maintainer publish helper:

```bash
./docker/publish_ghcr.sh --tag 0.2.0 --variant cpu --latest
```

Validated tissue GeoJSON example committed in this repository:

- `examples/validated_outputs/ROI_tissue_mask.geojson`
