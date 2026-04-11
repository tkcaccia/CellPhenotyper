# Installation (Ubuntu, macOS, Windows)

This page is step-based so you can choose where to start and where to stop.

## Step map (choose your start and end)

| Your situation | Start | End |
|---|---|---|
| Fresh machine, run with Singularity profile | `Step 0` | `Step 8-S` |
| Fresh machine, run with Docker profile | `Step 0` | `Step 8-D` |
| Java + Nextflow already installed | `Step 3` | `Step 8-S` or `Step 8-D` |
| Runtime already installed and working | `Step 5` | `Step 8-S` or `Step 8-D` |
| Repo already cloned and runtime configured | `Step 7` | `Step 8-S` or `Step 8-D` |
| Only need UNI-2 token refresh | `Step 7` | `Step 7` |

## Step 0: Choose your shell environment

Why: Nextflow and container tools must run in a Linux-compatible shell.

- Ubuntu: use normal terminal.
- macOS:
  - Docker path: normal terminal.
  - Singularity path: use Lima VM terminal.
- Windows:
  - Docker path: Docker Desktop + PowerShell or WSL2.
  - Singularity path: WSL2 Ubuntu terminal.

For Windows WSL2 install (recommended):

```powershell
wsl --install -d Ubuntu-24.04
```

References:

- [WSL install docs](https://learn.microsoft.com/windows/wsl/install)

## Step 1: Install Java 17

Why: Nextflow requires Java.

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless
java -version
```

macOS (Homebrew):

```bash
brew install --cask temurin@17
java -version
```

Windows (PowerShell):

```powershell
winget install EclipseAdoptium.Temurin.17.JRE
java -version
```

## Step 2: Install Nextflow

Why: `nextflow run main.nf` is the main command.

Linux/macOS/WSL:

```bash
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
nextflow -version
```

Reference:

- [Nextflow installation](https://www.nextflow.io/docs/latest/install.html)

## Step 3: Install Docker (optional runtime)

Do this step only if you plan to use `-profile docker`.

## Step 3-Ubuntu (Docker Engine)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

source /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Log out/in, then verify:

```bash
docker run hello-world
```

Reference:

- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

## Step 3-macOS / Step 3-Windows (Docker Desktop)

Install Docker Desktop and verify:

```bash
docker --version
docker run hello-world
```

References:

- [Docker Desktop for macOS](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

## Step 4: Install Singularity/Apptainer (optional runtime)

Do this step only if you plan to use `-profile singularity`.

Important: Singularity/Apptainer is Linux-native.

- macOS: use Lima VM.
- Windows: use WSL2 Ubuntu.

Reference:

- [Apptainer installation docs](https://apptainer.org/docs/admin/main/installation.html)

## Step 4-Ubuntu (native or inside Lima/WSL Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y apptainer squashfuse fuse2fs gocryptfs
```

Verify:

```bash
apptainer --version || singularity --version
```

## Step 4-macOS (Lima path for Singularity)

Install Lima:

```bash
brew install lima
limactl start
limactl shell default
```

Inside Lima shell, run Ubuntu installation from **Step 4-Ubuntu**.

Reference:

- [Lima documentation](https://lima-vm.io/docs/)

## Step 5: Clone repository

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

## Step 6: Configure runtime image source

Use published runtime images. For Singularity, the default is release-hosted `.sif` auto-selection.

Reference OCI tags:

- CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-amd64`
- CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:2.2-arm64`
- GPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64`
- GPU arm64: `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64`

Reference Singularity assets (GitHub release `v2.2`):

- `cellphenotyper-2.2-amd64.sif`
- `cellphenotyper-2.2-arm64.sif`
- `cellphenotyper-2.2-gpu-amd64.sif`
- `cellphenotyper-2.2-gpu-arm64.sif` (optional)

## Step 6-S (Singularity/Apptainer via Nextflow)

No manual pull command is required.
When you run with `-profile singularity`, Nextflow resolves and pulls `params.singularity_image` automatically.
Default behavior is `runtime_image_mode: auto` and `singularity_image_source: auto`
(local `.sif` first, then GHCR `oras://` SIF tags, then legacy release assets, then `docker://`; on arm64 GPU runs, missing GPU assets fall back to CPU containers).

Default automatic runtime settings in `pipeline_paramers.yml`:

- `runtime_image_mode: auto`
- `singularity_image_source: auto`
- `singularity_image: ""`

To force a specific manual image, set:

- `runtime_image_mode: manual`
- `singularity_image: https://github.com/tkcaccia/CellPhenotyper/releases/download/v2.2/cellphenotyper-2.2-amd64.sif` (example)

To force the prebuilt GHCR SIF directly, set:

- `runtime_image_mode: manual`
- `singularity_image: oras://ghcr.io/tkcaccia/cellphenotyper:2.2-sif-amd64` (example)

To force docker:// fallback instead of ORAS/release assets in auto mode:

- `runtime_image_mode: auto`
- `singularity_image_source: docker`

If the package is private, authenticate first with a GitHub token that has `read:packages`:

```bash
export SINGULARITY_DOCKER_USERNAME="<github_username>"
export SINGULARITY_DOCKER_PASSWORD="<github_token_with_read_packages>"
```

## Step 6-D (Docker image from GHCR)

Pull images with Docker:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-arm64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-arm64
```

Use:
- `2.2-amd64` on Linux x86_64/amd64
- `2.2-arm64` on arm64
- `2.2-gpu-amd64` on Linux amd64 with NVIDIA
- `2.2-gpu-arm64` on Linux arm64 with NVIDIA

## Step 6-M (Maintainer image publish to GHCR)

Use this only when you need to publish a new container version.

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="2.2"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
docker build -f docker/Dockerfile.full.cpu -t "${IMAGE}" .
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push "${IMAGE}"
```

GPU publish:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="2.2-gpu-amd64"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
docker build -f docker/Dockerfile.full.gpu -t "${IMAGE}" .
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push "${IMAGE}"
```

## Step 7: Configure UNI-2 token (required for full UNI-2 run)

Why: UNI-2 model access is authenticated through Hugging Face.

1. Sign in at [Hugging Face](https://huggingface.co)
2. Request access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
3. Create a read token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Save token in `tokens.env` (recommended, file is git-ignored):

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_read_token>" > tokens.env
```

By default the pipeline reads token file/variable from:

- `hf_token_env_file: tokens.env`
- `hf_token_env_var_name: HF_UNI2`

Important for Docker profile:
- Keep `tokens.env` in the repository root or another path inside the bind-mounted project directory.
- If the token file is outside the mounted project path, tasks may fail with `401 Unauthorized` for UNI-2.

## Step 8-S: Run complete pipeline with Singularity (end step)

First, edit `pipeline_paramers.yml` with your runtime choices (`image_input`, `roi_geojson`, `compute_device`, `runtime_image_mode`, resource caps).
To control where the pipeline starts/stops, set `start_point` and `end_point` in the same file.
For automatic architecture-aware CPU/GPU selection, keep:

- `runtime_image_mode: auto`
- `compute_device: auto`

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

If you see:

`ERROR ~ .nextflow/history.lock (No such file or directory)`

it usually means you are running in a read-only mount (common in Lima under `/Users/...`).
Fix by running from a writable Linux path (for example `~/CellPhenotyper`), or set:

```bash
export NXF_HOME=/var/tmp/nextflow_home
mkdir -p "$NXF_HOME"
```

For GPU runs (Linux amd64 + NVIDIA), use:

- `compute_device: gpu`
- `host_arch: amd64`
- keep `runtime_image_mode: auto`

For GPU runs (Linux arm64 + NVIDIA), use:

- `compute_device: gpu`
- `host_arch: arm64`
- `enable_gpu_on_arm64: true`
- either provide `singularity_gpu_asset_arm64` (release/local) or set `gpu_container_image` explicitly
- optional: `enable_stardist_gpu_on_arm64: true` to force StarDist into the GPU container on arm64

On GB10-class arm64 GPUs (`sm_121`), build the GPU SIF locally from `singularity/cellphenotyper_full_gpu.def` (nightly `cu130` PyTorch path). Older arm64 GPU assets can report CUDA visible but fail at kernel launch.

## Step 8-D: Run complete pipeline with Docker (end step)

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

GPU run (Linux amd64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --compute_device gpu \
  --host_arch amd64 \
  -with-report results_full_gpu/report.html \
  -with-trace results_full_gpu/trace.txt \
  -with-timeline results_full_gpu/timeline.html \
  -resume
```

For Linux-side update commands after container/definition changes, see `LINUX_UPDATE.md`.

## Step 9: Check status and copy results (optional)

Inside Linux/Lima:

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
tail -n 50 -f .nextflow.log
```

Copy results to macOS host (run on macOS terminal):

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_full \
  /Users/<your-user>/Documents/test/CellPhenotyper/
```

For GPU run, set:

- `compute_device: gpu` in `pipeline_paramers.yml`
- `runtime_image_mode: manual` in `pipeline_paramers.yml`
- `gpu_container_image: ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64` in `pipeline_paramers.yml`
