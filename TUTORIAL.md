# Tutorial

Primary tutorial is in `README.md`.

Run the complete pipeline (including UNI-2) with:

```bash
nextflow run main.nf \
  -profile singularity \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --singularity_image singularity/cellphenotyper_full_cpu.sif \
  --run_full_pipeline true \
  --tissue_mask_from_input false \
  --compute_device cpu \
  --outdir_base results_full \
  --max_cpus 8 \
  --max_memory_gb 32 \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```
