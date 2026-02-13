# Installation Guide

This repository is designed for local execution with Nextflow + Singularity.

## 1. Required software

- Nextflow `25.10.2` or newer
- Java 17+
- Singularity/Apptainer
- macOS Apple Silicon users: Lima VM (`limactl`)

## 2. macOS M1/M2 workflow (Lima)

Run all pipeline/build commands inside Lima:

```bash
limactl shell default
```

Inside the VM, verify tools:

```bash
nextflow -version
singularity --version
java -version
```

If Nextflow is missing, install:

```bash
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

## 3. Build Singularity images

From project root (inside Lima):

- Tissue-only image (fast, for tissue mask + GeoJSON):

```bash
./scripts/build_singularity_tissue.sh singularity/cellphenotyper_tissue.sif
```

- Full CPU image (all pipeline tools):

```bash
./scripts/build_singularity_full_cpu.sh singularity/cellphenotyper_full_cpu.sif
```

- Full GPU image (Linux x86_64 + NVIDIA hosts):

```bash
./scripts/build_singularity_full_gpu.sh singularity/cellphenotyper_full_gpu.sif
```

## 4. Container definitions in this repo

- `singularity/cellphenotyper_tissue.def`
- `singularity/cellphenotyper_full_cpu.def`
- `singularity/cellphenotyper_full_gpu.def`
- `stardist.yml`

`stardist.yml` is the shared environment spec used by full container builds.
