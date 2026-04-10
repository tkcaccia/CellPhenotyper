# Linux Update Playbook (after Dockerfile or `.def` changes)

Use this page when pipeline container definitions were updated and you need to run the new release on a Linux machine.

For full error catalog and fixes, see [Troubleshooting](TROUBLESHOOTING.md).

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

Validate token and model access:

```bash
singularity exec <image.sif> python - <<'PY'
import os
from huggingface_hub import whoami
print(whoami(token=os.environ["HF_TOKEN"].strip()))
PY
```

Pre-cache UNI-2 once (recommended):

```bash
export HF_HOME=/scratch/<project>/CellPhenotyper/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE"
singularity exec <image.sif> python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("MahmoodLab/UNI2-h", token=os.environ["HF_TOKEN"].strip())
print("UNI2 cache ready")
PY
```

## 3) Choose only one runtime

Do not use both profiles in the same run.

## 4A) Docker path (recommended if Docker is available)

Pull fresh image tag(s):

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-arm64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64
```

Use `2.2-gpu-amd64` on amd64 and `2.2-gpu-arm64` on arm64/aarch64 Spark.

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

On HPC, always set writable large tmp/cache:

```bash
mkdir -p /scratch/<project>/{tmp,cache,singularity}
export APPTAINER_TMPDIR=/scratch/<project>/tmp
export APPTAINER_CACHEDIR=/scratch/<project>/cache
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export TMPDIR=$APPTAINER_TMPDIR
export NXF_SINGULARITY_CACHEDIR=/scratch/<project>/cache
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

If you want to see pull progress explicitly, pre-pull image once:

```bash
singularity pull /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
```

Then run with manual image path:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --runtime_image_mode manual \
  --singularity_image /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  --folder_input Data \
  --outdir_base results_example
```

If compute nodes have restricted internet, export offline HF cache env before Nextflow:

```bash
export APPTAINERENV_HF_HOME=/scratch/<project>/CellPhenotyper/.hf_cache
export APPTAINERENV_HF_HUB_CACHE=/scratch/<project>/CellPhenotyper/.hf_cache/hub
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export SINGULARITYENV_HF_HOME=$APPTAINERENV_HF_HOME
export SINGULARITYENV_HF_HUB_CACHE=$APPTAINERENV_HF_HUB_CACHE
export SINGULARITYENV_HF_HUB_OFFLINE=$APPTAINERENV_HF_HUB_OFFLINE
export SINGULARITYENV_HF_TOKEN=$APPTAINERENV_HF_TOKEN
```

## 4C) Optional: pre-cache StarDist model for offline/slow nodes

```bash
mkdir -p /scratch/<project>/keras/models
curl -L --retry 5 --connect-timeout 30 \
  -o /scratch/<project>/keras/models/python_2D_versatile_he.zip \
  https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip
```

Add to run command:

```bash
--stardist_keras_home /scratch/<project>/keras \
--stardist_pretrained_zip /scratch/<project>/keras/models/python_2D_versatile_he.zip
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
