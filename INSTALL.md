# Installation (Ubuntu, macOS, Windows)

This page is step-based so you can choose where to start and where to stop.

## Step map (choose your start and end)

| Your situation | Start | End |
|---|---|---|
| Fresh machine, run with Singularity profile | `Step 0` | `Step 8-S` |
| Fresh machine, run with Docker profile | `Step 0` | `Step 8-D` |
| Java + Nextflow already installed | `Step 3` | `Step 8-S` or `Step 8-D` |
| Runtime already installed and working | `Step 5` | `Step 8-S` or `Step 8-D` |
| Repo already cloned and image already built | `Step 7` | `Step 8-S` or `Step 8-D` |
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

## Step 6: Prepare container image

## Step 6-S (Singularity image from GHCR, CPU)

Pull the prebuilt CPU image:

```bash
singularity pull singularity-cellphenotyper-0.1.0.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:0.1.0
```

If the package is private, authenticate first with a GitHub token that has `read:packages`:

```bash
export SINGULARITY_DOCKER_USERNAME="<github_username>"
export SINGULARITY_DOCKER_PASSWORD="<github_token_with_read_packages>"
singularity pull singularity-cellphenotyper-0.1.0.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:0.1.0
```

Verify:

```bash
ls -lh singularity-cellphenotyper-0.1.0.sif
```

## Step 6-D (Docker images)

Full CPU image:

```bash
docker build -f docker/Dockerfile.full.cpu -t cellphenotyper:full-cpu .
```

Full GPU image (Linux x86_64 + NVIDIA hosts):

```bash
docker build -f docker/Dockerfile.full.gpu -t cellphenotyper:full-gpu .
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

## Step 8-S: Run complete pipeline with Singularity (end step)

First, edit `pipeline_paramers.yml` with your runtime choices (`image_input`, `roi_geojson`, `compute_device`, `singularity_image`, resource caps).
To control where the pipeline starts/stops, set `start_point` and `end_point` in the same file.
For the provided project image, keep:

- `singularity_image: singularity-cellphenotyper-0.1.0.sif`

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

For GPU run, set:

- `compute_device: gpu` in `pipeline_paramers.yml`
- `singularity_image: singularity/cellphenotyper_full_gpu.sif` in `pipeline_paramers.yml`

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

For GPU run, set:

- `compute_device: gpu` in `pipeline_paramers.yml`
- `docker_image: cellphenotyper:full-gpu` in `pipeline_paramers.yml`
