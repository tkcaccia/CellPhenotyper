# CellPhenotyper

CellPhenotyper is a Nextflow DSL2 pipeline for tissue segmentation, cell-level feature extraction, UNI-2 embeddings, and KODAMA downstream analysis.
The default Singularity runtime image used by the parameter file is `singularity-stardist_UNI-2.sif`.

The main entrypoint is always:

```bash
nextflow run main.nf
```

## Documentation

- [Installation](INSTALL.md)
- [How to use](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Release](RELEASE.md)

## Quick start

After completing installation and UNI-2 token setup, edit `pipeline_paramers.yml` (`start_point` / `end_point` included) and run:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

Validated tissue GeoJSON example committed in this repository:

- `examples/validated_outputs/ROI_tissue_mask.geojson`
