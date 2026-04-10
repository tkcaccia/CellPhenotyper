# Singularity Assets: Maintainer Guide

This folder contains:

- `cellphenotyper_full_cpu.def`: CPU definition
- `cellphenotyper_full_gpu.def`: GPU definition (amd64 + NVIDIA)
- `publish_sif_release_asset.sh`: build/upload helper

Important:

- Do **not** commit `.sif` files to git.
- Upload `.sif` files as **GitHub Release assets**.
- Pipeline users pull automatically with `-profile singularity`.

## Naming Convention

For version `X.Y.Z`, expected asset names are:

- CPU arm64: `cellphenotyper-X.Y.Z-arm64.sif`
- CPU amd64: `cellphenotyper-X.Y.Z-amd64.sif`
- GPU amd64 (optional): `cellphenotyper-X.Y.Z-gpu-amd64.sif`
- GPU arm64 (optional): `cellphenotyper-X.Y.Z-gpu-arm64.sif`

Release tag: `vX.Y.Z`

## Prerequisites

- `singularity` or `apptainer`
- `gh` (GitHub CLI) authenticated
- Repository access to `tkcaccia/CellPhenotyper`

If you are on macOS and using Singularity via Lima, enter the VM first:

```bash
limactl shell default
```

Headless `gh` auth inside Lima:

```bash
BROWSER=false gh auth login
gh auth status
```

## Standard Release Workflow

### 1) Build/upload arm64 CPU asset (Mac M1/M2 or Linux arm64)

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --upload
```

If `/tmp` is small tmpfs in Lima, force temp/cache explicitly:

```bash
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --tmpdir /var/tmp/apptainer/tmp \
  --cachedir /var/tmp/apptainer/cache \
  --outdir /var/tmp/apptainer/out \
  --upload
```

### 2) Build/upload amd64 CPU asset (Linux amd64)

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --upload
```

### 3) Optional: build/upload amd64 GPU asset

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device gpu \
  --upload
```

### 4) Optional: build/upload arm64 GPU asset

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device gpu \
  --upload
```

Notes for arm64 GPU builds:

- The GPU definition now enforces CUDA-enabled PyTorch (`torch.backends.cuda.is_built()==True`) and fails the build if only CPU wheels are installed.
- For GB10-class GPUs (`sm_121`), the arm64 GPU definition installs nightly `cu130` PyTorch wheels (stable `cu126` arm64 wheels are not sufficient).
- TensorFlow GPU availability on generic Linux arm64 may still be limited; UNI-2 (PyTorch) is the primary GPU path.

## Verify Release Assets

```bash
gh release view v2.2 \
  --repo tkcaccia/CellPhenotyper \
  --json assets \
  --jq '.assets[].name'
```

You should see the correct asset names for your release.

## If You Need Docker-Source Build Instead of `.def`

```bash
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --source docker \
  --device cpu \
  --upload
```

## Troubleshooting: `No space left on device`

If a Singularity build fails with errors like:

- `Could not write to file ... No space left on device`
- `Failure writing output to destination`
- `Error when extracting package`

clean temporary/cache directories inside your Linux/Singularity runtime (for macOS this is usually inside Lima), then force temporary and cache directories to a larger writable location.

```bash
# enter Linux VM if using Lima on macOS
limactl shell default

# inspect usage
df -h
df -h /tmp
du -sh /tmp /var/tmp ~/.singularity ~/.apptainer ~/.cache 2>/dev/null

# clean common temp/cache leftovers
sudo rm -rf /tmp/build-temp-* /tmp/bundle-temp-* /tmp/nxf-* /tmp/pip-*
sudo rm -rf /var/tmp/build-temp-* /var/tmp/bundle-temp-* /var/tmp/Rtmp* /var/tmp/pip-*
rm -rf ~/.singularity/cache ~/.apptainer/cache ~/.cache/pip

# use larger on-disk dirs for Apptainer/Singularity temp/cache
sudo mkdir -p /var/tmp/apptainer/tmp /var/tmp/apptainer/cache
sudo chown -R "$USER":"$USER" /var/tmp/apptainer
export APPTAINER_TMPDIR=/var/tmp/apptainer/tmp
export APPTAINER_CACHEDIR=/var/tmp/apptainer/cache
export SINGULARITY_TMPDIR=/var/tmp/apptainer/tmp
export SINGULARITY_CACHEDIR=/var/tmp/apptainer/cache
export TMPDIR=/var/tmp/apptainer/tmp
```

Then rerun:

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 2.2 \
  --device cpu \
  --outdir /var/tmp/apptainer/out \
  --upload
```

If `/Users/...` mount is read-only in Lima, always write SIF to `/var/tmp/apptainer/out` and copy back to macOS:

```bash
limactl copy default:/var/tmp/apptainer/out/cellphenotyper-2.2-arm64.sif \
  /Users/<your-user>/Documents/CellPhenotyper/
```

## Update Pipeline Defaults for a New Release

When publishing a new release (example `0.3.0`), update:

- `nextflow.config`
- `pipeline_paramers.yml`
- `README.md`
- `INSTALL.md`
- `PARAMETERS.md`
- `RELEASE.md`

Fields to update include:

- `singularity_release_tag`
- `singularity_cpu_asset_amd64`
- `singularity_cpu_asset_arm64`
- `singularity_gpu_asset_amd64`
- `singularity_gpu_asset_arm64`
- container tags (`container_cpu_tag*`, `container_gpu_tag`) if changed

Then commit and push.
