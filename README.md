# CellPhenotyper

CellPhenotyper is a fully reproducible, modular computational pathology workflow that takes raw H&E whole-slide images to quantitative cell-level phenotypes.  
The pipeline performs image conversion/ROI handling, StarDist-based ROI-informed segmentation, UNI-2 foundation-model embeddings, and unsupervised KODAMA analysis to stratify cells into coherent phenotypic groups for exploratory spatial discovery.

The workflow is implemented in Nextflow DSL2 with containerized execution for portability, scalability, and auditability across datasets and compute environments.  
Select the runtime image in `pipeline_paramers.yml` using `singularity_image` or `docker_image`.
For full UNI-2 execution, the Hugging Face token must have access to `MahmoodLab/UNI2-h`.
Token loading is configured via `hf_token_env_file` and `hf_token_env_var_name` in `pipeline_paramers.yml`.

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
