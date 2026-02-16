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

Release tag: `vX.Y.Z`

## Prerequisites

- `singularity` or `apptainer`
- `gh` (GitHub CLI) authenticated (`gh auth login`)
- Repository access to `tkcaccia/CellPhenotyper`

If you are on macOS and using Singularity via Lima, enter the VM first:

```bash
limactl shell default
```

## Standard Release Workflow

### 1) Build/upload arm64 CPU asset (Mac M1/M2 or Linux arm64)

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device cpu \
  --upload
```

### 2) Build/upload amd64 CPU asset (Linux amd64)

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device cpu \
  --upload
```

### 3) Optional: build/upload amd64 GPU asset

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
  --device gpu \
  --upload
```

## Verify Release Assets

```bash
gh release view v0.2.0 \
  --repo tkcaccia/CellPhenotyper \
  --json assets \
  --jq '.assets[].name'
```

You should see the correct asset names for your release.

## If You Need Docker-Source Build Instead of `.def`

```bash
./singularity/publish_sif_release_asset.sh \
  --version 0.2.0 \
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
du -sh /tmp /var/tmp ~/.singularity ~/.apptainer ~/.cache 2>/dev/null

# clean common temp/cache leftovers
sudo rm -rf /tmp/build-temp-* /tmp/bundle-temp-* /tmp/nxf-* /tmp/pip-*
sudo rm -rf /var/tmp/build-temp-* /var/tmp/bundle-temp-* /var/tmp/Rtmp* /var/tmp/pip-*
rm -rf ~/.singularity/cache ~/.apptainer/cache ~/.cache/pip

# use larger host-mounted dirs for Singularity temp/cache
mkdir -p /Users/$USER/.singularity-tmp /Users/$USER/.singularity-cache
export SINGULARITY_TMPDIR=/Users/$USER/.singularity-tmp
export APPTAINER_TMPDIR=/Users/$USER/.singularity-tmp
export SINGULARITY_CACHEDIR=/Users/$USER/.singularity-cache
export APPTAINER_CACHEDIR=/Users/$USER/.singularity-cache
```

Then rerun:

```bash
cd /path/to/CellPhenotyper
./singularity/publish_sif_release_asset.sh --version 0.2.0 --device cpu --upload
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
- container tags (`container_cpu_tag*`, `container_gpu_tag`) if changed

Then commit and push.
