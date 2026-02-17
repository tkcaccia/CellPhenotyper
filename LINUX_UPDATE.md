# Linux Update Playbook (after Dockerfile or `.def` changes)

Use this page when pipeline container definitions were updated and you need to run the new release on a Linux machine.

## 1) Pull latest pipeline code

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
# or, if already cloned:
git pull
```

## 2) Set UNI-2 token

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="$HF_UNI2"
```

## 3) Choose only one runtime

Do not use both profiles in the same run.

## 4A) Docker path (recommended if Docker is available)

Pull fresh image tag(s):

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

Run:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

## 4B) Singularity/Apptainer path

Clean old local singularity cache if needed:

```bash
rm -rf work/singularity
apptainer cache clean -f || true
```

Run (auto-selects matching architecture image):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

If release `.sif` asset is missing, force docker source for singularity conversion:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --singularity_image_source docker \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

## 5) Verify run output

Final output:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

Execution report:

- `results_example/00_execution/final_report.md`
- `results_example/00_execution/final_report.json`

## 6) If host has low resources

The defaults target 4 CPU / 8 GB RAM. On smaller systems, reduce at runtime:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --max_cpus 2 \
  --max_memory_gb 3 \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example_lowmem
```
