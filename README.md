# CellPhenotyper

CellPhenotyper is a Nextflow DSL2 pipeline for tissue phenotyping from whole-slide style microscopy images.

## What this pipeline does

Given an input image (`.ome.tif` or `.btf`) and an ROI GeoJSON, the pipeline can run:

1. Input preparation (`.btf` -> `.ome.tif` when needed)
2. ROI StarDist segmentation
3. Tissue mask extraction
4. Tissue mask conversion to GeoJSON
5. Optional downstream analysis:
   - cell-to-polygon assignment
   - cytoplasm expansion
   - UNI/UNI2 embeddings
   - KODAMA analysis

Main workflow files:

- `main.nf`
- `nextflow.config`
- `modules/prepare_input_ometiff.nf`
- `modules/run_stardist_roi_segmentation.nf`
- `modules/build_tissue_mask.nf`
- `modules/convert_tissue_mask_to_geojson.nf`
- `modules/map_cells_to_roi_polygons.nf`
- `modules/expand_labels_to_cytoplasm.nf`
- `modules/extract_uni2_embeddings.nf`
- `modules/run_kodama_analysis.nf`

Pipeline implementation code layout:

- `modules/`: Nextflow DSL2 process modules
- `bin/`: pipeline executable code (Python, R, and shell scripts used by Nextflow)

Profiles supported:

- `singularity`
- `docker`

No SLURM profile is used.

## Installation

## 1) Prerequisites

- Nextflow `25.10.2+`
- Java 17+
- Singularity/Apptainer

For macOS Apple Silicon, run inside Lima:

```bash
limactl shell default
```

Recommended runtime packages in Lima:

```bash
sudo apt-get update
sudo apt-get install -y apptainer squashfuse fuse2fs gocryptfs
```

Install Nextflow in Lima if needed:

```bash
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

## 2) Build containers

From repository root:

- Tissue-only container (fastest to validate tissue GeoJSON):

```bash
./scripts/build_singularity_tissue.sh singularity/cellphenotyper_tissue.sif
```

- Full CPU container (includes `.btf` conversion + full pipeline stack):

```bash
./scripts/build_singularity_full_cpu.sh singularity/cellphenotyper_full_cpu.sif
```

- Full GPU container (NVIDIA Linux hosts):

```bash
./scripts/build_singularity_full_gpu.sh singularity/cellphenotyper_full_gpu.sif
```

## UNI-2 token setup (Hugging Face)

UNI2 model loading in `bin/extract_uni2_embeddings.py` uses Hugging Face auth (parameter `--hf-token`, wired by Nextflow via `HF_TOKEN`).

1. Create/sign in to Hugging Face: [huggingface.co](https://huggingface.co)
2. Request access to model repo: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
3. Create a **Read** access token: [HF Tokens](https://huggingface.co/settings/tokens)
4. Export token in your shell before running full pipeline:

```bash
export HF_TOKEN="<your_hf_read_token>"
```

The pipeline default token env var name is `HF_TOKEN` (see `nextflow.config` -> `hf_token_env_var_name`).

## Practical examples

## Example A: validated tissue GeoJSON run (local)

This command path has been validated locally and produces:

- `results_singularity_local/04_tissue_geojson/ROI_tissue_mask.geojson`

Run:

```bash
./scripts/run_tissue_geojson_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_tissue.sif \
  results_singularity_local
```

A copied validated artifact is also included in:

- `examples/validated_outputs/ROI_tissue_mask.geojson`

## Example B: public Visium HD image download + tissue GeoJSON

Dataset page: [10x Genomics Visium HD CRC](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc)

Download image:

```bash
mkdir -p Data/visium_hd
cd Data/visium_hd
curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_tissue_image.btf
cd ../..
```

Create a placeholder ROI file (required by CLI, not used in tissue-only mode):

```bash
cat > Data/visium_hd/placeholder_roi.geojson <<'JSON'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "placeholder"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]
      }
    }
  ]
}
JSON
```

Run tissue-only workflow from `.btf` input (use full CPU image because `.btf` conversion tools are included there):

```bash
./scripts/run_tissue_geojson_local_singularity.sh \
  Data/visium_hd/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
  Data/visium_hd/placeholder_roi.geojson \
  singularity/cellphenotyper_full_cpu.sif \
  results_visium_hd
```

## Example C: full pipeline with UNI2 embeddings

CPU:

```bash
export HF_TOKEN="<your_hf_read_token>"
MAX_CPUS=8 MAX_MEM_GB=32 COMPUTE_DEVICE=cpu \
./scripts/run_full_pipeline_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_full_cpu.sif \
  results_full_cpu
```

GPU (NVIDIA Linux host):

```bash
export HF_TOKEN="<your_hf_read_token>"
MAX_CPUS=16 MAX_MEM_GB=64 COMPUTE_DEVICE=gpu \
./scripts/run_full_pipeline_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_full_gpu.sif \
  results_full_gpu
```

## Resource management knobs

Global resource caps:

- `--max_cpus`
- `--max_memory_gb`
- `--compute_device` (`cpu|gpu|auto`)

For tissue mask memory/performance tradeoff:

- `--tissue_work_downsample` (default `8`)

Higher downsample reduces RAM and runtime for large images.
