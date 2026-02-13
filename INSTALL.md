# Installation

Primary documentation is in `README.md`.

This workflow is executed with `nextflow run main.nf`.

## Requirements

- Nextflow `25.10.2+`
- Java 17+
- Singularity/Apptainer

On macOS M1/M2, start in Lima:

```bash
limactl shell default
```

Install runtime helpers:

```bash
sudo apt-get update
sudo apt-get install -y apptainer squashfuse fuse2fs gocryptfs
```

Build full CPU image:

```bash
sudo singularity build --force \
  singularity/cellphenotyper_full_cpu.sif \
  singularity/cellphenotyper_full_cpu.def
```

Configure UNI-2 token:

```bash
export HF_TOKEN="<your_hf_read_token>"
```
