# Troubleshooting

This page lists the real issues encountered while validating CellPhenotyper on macOS, Linux, HPC, Docker, and Singularity.

## 1) Singularity architecture mismatch

Error:

```text
the image's architecture (arm64) could not run on the host's (amd64)
```

Cause: wrong image architecture was pulled or cached.

Fix:

```bash
uname -m
rm -rf work/singularity
singularity cache clean -f || true
```

Then run with explicit architecture (example amd64 GPU):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --compute_device gpu \
  --host_arch amd64
```

## 2) Singularity pull/build fails with `No space left on device`

Cause: default `/tmp` or cache location is too small.

Fix (HPC-safe):

```bash
mkdir -p /scratch/<project>/{tmp,cache,singularity}
export APPTAINER_TMPDIR=/scratch/<project>/tmp
export APPTAINER_CACHEDIR=/scratch/<project>/cache
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export TMPDIR=$APPTAINER_TMPDIR
export NXF_SINGULARITY_CACHEDIR=/scratch/<project>/cache
```

## 3) Nextflow fails with `.nextflow/history.lock`

Error:

```text
ERROR ~ .nextflow/history.lock (No such file or directory)
```

Cause: running from read-only mount (common in Lima under host-shared folders).

Fix:

```bash
mkdir -p ~/CellPhenotyper
rsync -a --delete <host_project_dir>/ ~/CellPhenotyper/
cd ~/CellPhenotyper
```

Optional:

```bash
export NXF_HOME=/var/tmp/nextflow_home
mkdir -p "$NXF_HOME"
```

## 3b) Nextflow fails after a restart because Java comes from conda

Symptoms:

- `Nextflow` starts with the wrong `JAVA_HOME`
- Java 11 from `miniconda3` is picked up
- resume attempts fail even though the container itself is correct

Cause: `Nextflow` runs on the host, not inside the Singularity image. A conda `base`
environment can override `JAVA_HOME` and `PATH` with an incompatible host Java.

Fix: launch `Nextflow` with a system Java 17+ explicitly.

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
unset JAVA_CMD JAVA_LD_LIBRARY_PATH
java -version
nextflow -version
```

If a previous run was killed abruptly, also verify the session lock is stale before deleting it:

```bash
lsof .nextflow/cache/*/db/LOCK
ps aux | grep -E 'nextflow|java' | grep -v grep
```

Only remove the lock if no live `Nextflow`/`java` process is still using it.

## 4) StarDist model download times out

Error:

```text
URL fetch failure ... python_2D_versatile_he.zip ... Connection timed out
```

Cause: compute node cannot reliably access GitHub during task runtime.

Fix: pre-download the model and pass persistent cache parameters.

```bash
mkdir -p /scratch/<project>/keras/models
curl -L --retry 5 --connect-timeout 30 \
  -o /scratch/<project>/keras/models/python_2D_versatile_he.zip \
  https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip
```

Run pipeline with:

```bash
--stardist_keras_home /scratch/<project>/keras \
--stardist_pretrained_zip /scratch/<project>/keras/models/python_2D_versatile_he.zip
```

## 4b) `GROW_TO_TISSUE` fails with `tifffile.TiffFileError: missing data offset`

Error:

```text
TiffFileError: missing data offset
```

Cause: an internal tissue mask written as a pyramidal TIFF can be unreadable for
later `tifffile`-based steps, especially on small crops.

Fix:

- Use the current pipeline version, where `bin/build_tissue_mask.py` writes a
  standard compressed TIFF for the intermediate tissue mask.
- If recovering a partially completed run, regenerate the failed tissue mask in
  its work directory and then rerun with `-resume`.

Why this is safe:

- the tissue mask is an internal pipeline artifact
- downstream reliability matters more than pyramidal storage for this file

## 5) UNI-2 fails with 401 / gated model

Error:

```text
401 Unauthorized
GatedRepoError: Access to model MahmoodLab/UNI2-h is restricted
```

Fix:

1. Request access to `MahmoodLab/UNI2-h` on Hugging Face.
2. Save token in `tokens.env`.
3. Source token before run.

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Quick validation:

```bash
source tokens.env
export HF_TOKEN="${HF_UNI2}"
singularity exec <image.sif> python - <<'PY'
import os
from huggingface_hub import whoami
print(whoami(token=os.environ["HF_TOKEN"].strip()))
PY
```

If this fails, your token is wrong or the account is not approved for `MahmoodLab/UNI2-h`.

## 6) UNI-2 transient HF client/network errors

Error:

```text
RuntimeError: Cannot send a request, as the client has been closed.
```

Fix:

- Current pipeline retries UNI-2 model loading automatically.
- You can tune:
  - `uni2_hf_load_retries`
  - `uni2_hf_load_retry_delay_sec`
  - `hf_hub_download_timeout`
  - `hf_hub_etag_timeout`

## 6b) UNI-2 works manually but fails inside Nextflow on HPC

Cause: compute tasks run in restricted network context; downloading inside each task is fragile.

Fix:

1. Pre-cache once:

```bash
export HF_HOME=/scratch/<project>/CellPhenotyper/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE"
singularity exec <image.sif> python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
token = os.environ["HF_TOKEN"].strip()
repo = "MahmoodLab/UNI2-h"
snapshot_download(repo, token=token, local_files_only=False)
try:
    hf_hub_download(repo_id=repo, filename="model.safetensors", token=token, local_files_only=False)
except Exception:
    hf_hub_download(repo_id=repo, filename="pytorch_model.bin", token=token, local_files_only=False)
print("UNI2 cache ready")
PY
```

2. Run with offline cache env:

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

3. Ensure pipeline uses the same cache path and strict offline mode:

```bash
nextflow run main.nf ... \
  --hf_home /scratch/<project>/CellPhenotyper/.hf_cache \
  --hf_hub_cache /scratch/<project>/CellPhenotyper/.hf_cache/hub \
  --hf_hub_offline true
```

## 7) `nextflow: command not found` on HPC GPU nodes

Cause: module not loaded on compute node.

Fix:

```bash
module load software/nextflow-25.10.2
nextflow -version
```

## 8) Singularity pull appears stuck in Nextflow

Cause: Nextflow suppresses detailed pull progress while resolving tasks.

Fix: pre-pull manually once, then run in manual mode.

```bash
singularity pull /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
```

## 9) Image builds "successfully" but KODAMA is still missing at runtime

Symptom:

```text
Error in library(KODAMA) : there is no package called ‘KODAMA’
```

Cause:

- The container dependency layer did not actually contain the full R stack.
- A previous cached Docker layer can hide this problem and make the build appear successful.

Required rule:

- Do not rely on host `R_LIBS_USER`, host Python environments, or task-local package bootstrapping for normal runs.
- The Docker/Singularity image itself must contain `KODAMA`, `KODAMAextra`, `SPARK`, `umap`, and the rest of the runtime stack.

Fix:

1. Ensure the container recipes install and validate the full R runtime:

```bash
micromamba install -y -n stardist \
  -c conda-forge -c bioconda --channel-priority strict \
  r-base r-devtools r-data.table r-irlba r-biocmanager r-remotes r-r.utils r-igraph r-umap \
  bioconductor-bluster

micromamba run -n stardist Rscript -e 'devtools::install_github("xzhoulab/SPARK")'
micromamba run -n stardist Rscript -e 'devtools::install_github("tkcaccia/KODAMA")'
micromamba run -n stardist Rscript -e 'devtools::install_github("tkcaccia/KODAMAextra")'
micromamba run -n stardist Rscript -e 'library(KODAMA); library(KODAMAextra); library(SPARK); library(umap); cat("R runtime package check passed\n")'
```

2. Keep the Nextflow KODAMA step as a validator, not an installer. It should fail clearly if the container is incomplete.

Quick check on a finished image:

```bash
docker run --rm <image> \
  Rscript -e 'library(KODAMA); library(KODAMAextra); library(SPARK); library(umap); cat("R packages OK\n")'
```

## 10) Docker dependency changes do not show up because the build reused an old layer

Symptom:

- You edit `docker/Dockerfile.full.cpu` or `docker/Dockerfile.full.gpu`.
- The next image build finishes quickly.
- Runtime behavior still matches the older image.

Cause:

- Docker reused a previously cached dependency layer.

Fix:

Use a forced rebuild when changing the dependency/install layer:

```bash
docker build --no-cache --pull \
  -f docker/Dockerfile.full.gpu \
  -t cellphenotyper:2.2-gpu-amd64-fresh .
```

If needed, verify the image immediately after build:

```bash
docker run --rm cellphenotyper:2.2-gpu-amd64-fresh \
  Rscript -e 'library(KODAMA); library(KODAMAextra); library(SPARK); library(umap); cat("R packages OK\n")'
```

Do not assume a "successful" build is valid until these runtime checks pass.

If that direct pull is too slow, convert from the already-local Docker image instead:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker save -o /scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar \
  ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
singularity pull /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  docker-archive:///scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar
rm -f /scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar
```

Then:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --runtime_image_mode manual \
  --singularity_image /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif
```

## 9) Check if runtime has TensorFlow installed

```bash
singularity exec <image.sif> \
  python -c "import tensorflow as tf; print(tf.__version__)"
```

If this command fails, rebuild/pull the correct image tag.

## 10) UNI-2 fails on RTX 50xx with `no kernel image is available for execution on the device`

Error:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Cause: PyTorch build does not include kernels for Blackwell (`sm_120`).

Fix:

- Use `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64` (CUDA-enabled PyTorch cu128 build).
- Keep `tokens.env` inside repository root and pass:

```bash
--hf_token_env_file tokens.env \
--hf_token_env_var_name HF_UNI2
```

Quick check:

```bash
docker run --rm --gpus all ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64 \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```
