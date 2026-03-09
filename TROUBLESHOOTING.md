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
singularity pull /scratch/<project>/singularity/cellphenotyper-0.2.0-gpu-amd64.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu
```

Then:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --runtime_image_mode manual \
  --singularity_image /scratch/<project>/singularity/cellphenotyper-0.2.0-gpu-amd64.sif
```

## 9) Check if runtime has TensorFlow installed

```bash
singularity exec <image.sif> \
  python -c "import tensorflow as tf; print(tf.__version__)"
```

If this command fails, rebuild/pull the correct image tag.
