# Tutorial

## A. Tissue segmentation to GeoJSON (recommended first run)

Run inside Lima (`limactl shell default`) from project root:

```bash
./scripts/run_tissue_geojson_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_tissue.sif \
  results_singularity_local
```

Expected output:

- `results_singularity_local/03_tissue_mask/ROI_tissue_mask.tif`
- `results_singularity_local/03_tissue_mask/ROI_tissue_mask_preview.png`
- `results_singularity_local/04_tissue_geojson/ROI_tissue_mask.geojson`

## B. Full pipeline run

```bash
./scripts/run_full_pipeline_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_full_cpu.sif \
  results_full
```

For GPU-enabled run (on NVIDIA host):

```bash
COMPUTE_DEVICE=gpu \
./scripts/run_full_pipeline_local_singularity.sh \
  Data/ROI.ome.tif \
  Data/ROI.geojson \
  singularity/cellphenotyper_full_gpu.sif \
  results_full_gpu
```

## C. Resource tuning

Use environment variables before running scripts:

```bash
MAX_CPUS=4 MAX_MEM_GB=12 TISSUE_WORK_DOWNSAMPLE=8 \
./scripts/run_tissue_geojson_local_singularity.sh \
  Data/ROI.ome.tif Data/ROI.geojson singularity/cellphenotyper_tissue.sif
```

- `MAX_CPUS`: upper CPU cap used by Nextflow tasks
- `MAX_MEM_GB`: upper RAM cap (GB)
- `TISSUE_WORK_DOWNSAMPLE`: memory/performance tradeoff for tissue-mask computation

## D. Notes for Lima users

The host-mounted project path can be read-only from inside Lima.
If needed, run from a writable VM folder (for example `/var/tmp/CellPhenotyper_vm`) and copy results back:

```bash
limactl copy default:/var/tmp/CellPhenotyper_vm/results_singularity_local/04_tissue_geojson/ROI_tissue_mask.geojson \
  /Users/stefano/Documents/CellPhenotyper/results_singularity_local/04_tissue_geojson/ROI_tissue_mask.geojson
```
