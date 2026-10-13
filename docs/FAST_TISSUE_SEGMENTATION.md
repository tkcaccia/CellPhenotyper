# Fast tissue segmentation without cell identification

This route creates unsupervised tissue domains from the H&E image without
running StarDist, HoVer-Net, CellViT++, GigaTIME, cell assignment, cytoplasm
expansion, or cell profiling. It still runs the stages required for tissue
segmentation:

1. lossless pyramidal OME-TIFF conversion and physical-resolution validation;
2. GrandQC tissue/artifact annotation;
3. shared pyramidal analysis-crop preparation;
4. regular tissue-grid sampling and Virchow2 feature extraction;
5. KODAMA (`ncomp=50`) and graph-derived clustering;
6. dense cluster-mask construction;
7. Bora boundary refinement and original-image-coordinate GeoJSON export.

`cell_detection_mode=none` is the important switch. Crop preparation is shared
infrastructure and still occupies the historical `stardist` stage boundary for
restart compatibility, but no StarDist process or cell-label raster is created.

## Single image on an NVIDIA HPC node

Virchow2 is gated. Accept its Hugging Face terms, authenticate once on the
execution host, and make the approved cache visible to Singularity. Then run:

```bash
nextflow run tkcaccia/CellPhenotyper \
  -profile singularity \
  -params-file examples/fast_tissue_segmentation.yml \
  --image_input /data/sample.btf \
  --outdir_base /results/sample \
  --compute_device gpu \
  -resume
```

The Singularity profile pulls the versioned native GPU SIF configured by the
release. To use a pre-pulled image, add:

```bash
--runtime_image_mode manual \
--singularity_image /shared/containers/cellphenotyper-<version>-gpu-amd64.sif
```

Use `--foundation_encoder uni2-h` when UNI2-h is preferred and that gated model
has been staged. The rest of the tissue-only API is unchanged.

## Cohort folder

For one top-level output directory per specimen:

```bash
python bin/run_sample_batch.py \
  --input-dir /data/cohort \
  --output-root /results/cohort \
  --profile singularity \
  --params-file examples/fast_tissue_segmentation.yml \
  -- --compute_device gpu
```

Successful samples discard their isolated Nextflow work directory by default;
failed or interrupted samples retain it for `-resume`.

## Outputs to review

- `10_kodama/<sample>/kodama_output/`: KODAMA coordinates and diagnostics.
- `11_clustering/<sample>/`: cluster table and membership plot.
- `12_cluster_mask/<sample>/`: dense categorical mask and uncertainty mask.
- `14_medsam_refine_tissue/<sample>/`: Bora-refined pyramidal mask and QC.
- `15_cluster_geojson/<sample>/`: simplified tissue-domain GeoJSON in original
  level-zero image coordinates.

GrandQC candidate artifacts are annotated, not automatically removed from
Virchow2/KODAMA. They become hard negatives only when the post-KODAMA outlier
policy confirms them. The clusters are unsupervised tissue domains and require
biological interpretation; their numeric IDs are not cell-type labels.
