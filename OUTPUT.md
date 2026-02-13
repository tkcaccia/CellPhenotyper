# Output

With `--outdir_base results_full`, the pipeline writes:

- `results_full/01_input/`
- `results_full/02_stardist/`
- `results_full/03_tissue_mask/`
- `results_full/04_tissue_geojson/`
- `results_full/05_cell_assignments/`
- `results_full/06_cytoplasm/`
- `results_full/07_embeddings/`
- `results_full/08_kodama/`
- `results_full/08_kodama_logs/`

## Key files to inspect

- Tissue GeoJSON:
  - `results_full/04_tissue_geojson/*_tissue_mask.geojson`
- Assigned cells:
  - `results_full/05_cell_assignments/*_objects_assigned.csv`
- Embeddings:
  - `results_full/07_embeddings/`
- KODAMA logs:
  - `results_full/08_kodama_logs/*.Rout`

## Runtime reports

If run with reporting flags:

- `results_full/report.html`
- `results_full/trace.txt`
- `results_full/timeline.html`
