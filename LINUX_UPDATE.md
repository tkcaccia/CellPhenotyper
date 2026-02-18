# Linux Update Playbook (after Dockerfile or `.def` changes)

Use this page when pipeline container definitions were updated and you need to run the new release on a Linux machine.

## 1) Pull latest pipeline code

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
# or, if already cloned:
git fetch origin
git checkout main
git pull --ff-only
git rev-parse --short HEAD
git rev-parse --short origin/main
git status --short
```

`HEAD` and `origin/main` must match. `git status --short` should be empty before running.

## 1.1) Linux prerequisites

```bash
java -version
nextflow -version
```

If `nextflow` is not in PATH, use `/home/<user>/.local/bin/nextflow` in commands below.

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
docker pull ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu-arm64
```

Use `0.2.0-gpu` on amd64 and `0.2.0-gpu-arm64` on arm64/aarch64 Spark.

Run:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
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
  --folder_input Data \
  --outdir_base results_example
```

If release `.sif` asset is missing, force docker source for singularity conversion:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --singularity_image_source docker \
  --folder_input Data \
  --outdir_base results_example
```

## 5) Verify run output

Final output:

- `results_example/12_cluster_geojson/<sample_root>/<sample_root>_grown_mask_smooth_class.geojson`

Execution report:

- `results_example/00_execution/final_report.md`
- `results_example/00_execution/final_report.json`

## 6) If host has low resources

The defaults target 1 CPU / 8 GB RAM. Increase or tune at runtime if needed:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --max_cpus 2 \
  --max_memory_gb 3 \
  --folder_input Data \
  --outdir_base results_example_lowmem
```

## 7) Rerun only `10_cluster_mask` and `11_grown_tissue`

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example \
  --start_point cluster_mask \
  --end_point grow_tissue
```
