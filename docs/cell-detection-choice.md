# Choose one cell detector

Set one value in your run's YAML, or override it on the command line:

| Setting | Runs | Cell-type output |
| --- | --- | --- |
| `cell_detection_mode: cellvitpp` | CellViT++ only | The requested pretrained CellViT++ classifiers; full-slide mode retains all eight |
| `cell_detection_mode: stardist` | StarDist only | Nuclear instances; no pretrained CellViT++ cell-type labels |

CellViT++ is the default. The pipeline rejects `consensus`, an invalid detector,
and a StarDist choice combined with `cell_identification_enable: true`. It does
not run HoVer-Net or automatically run a second detector for review.

```bash
nextflow run main.nf -profile singularity -params-file my-run.yml \
  --cell_detection_mode cellvitpp

# A separate run using StarDist:
nextflow run main.nf -profile singularity -params-file my-run.yml \
  --cell_detection_mode stardist --cell_identification_enable false
```

Keep a different output directory for each run. The selected engine is recorded
in `00_execution/analysis_contract.json`. The choice changes nuclear detection;
it does not change the requested Virchow2/KODAMA/Bora tissue algorithm.

## CellViT++ setup and scope

CellViT++ requires a resolved GPU runtime. Its default `cellvit_input_scope:
full_slide` uses the existing bounded, pinned Singularity WSI campaign. Supply
the persistent image, clean mask, full-image shift, model cache, GPU UUID and
runtime paths/hashes described in [full-slide CellViT](cellvit-fullslide.md).
The choice never falls back to a small ROI or to StarDist when inputs are
missing. All planned cores must finish before downstream cell outputs are used.

The `roi` scope is an explicit alternative; the small-ROI example is
[`params/cell_identification.yml`](../params/cell_identification.yml). Its SAM
backbone supports all eight requested heads. This is a different scope from the
full-slide campaign, and must be selected intentionally.

## Shared outputs

Both detectors provide a calibrated analysis-crop `labels.tif` and `objects.csv`
for cell assignment, cytoplasm/marker processing and other downstream consumers.
Shared ROI preparation and tissue/QC support run independently of the detector.

CellViT++ additionally writes:

- `03b_cellvitpp/<sample>/cellvit_primary_<sample>/cell_instances.geojson`
- `03b_cellvitpp/<sample>/cellvit_primary_<sample>/segmentation_metadata.json`

The CPU export adapter verifies completed source hashes and native frames,
translates full-slide coordinates using the saved crop offset, excludes cells
outside the shared support and checks that every retained cell has a raster ID.
Stable CellViT IDs and all retained classifier evidence remain separate from
numeric label-image IDs. This adapter performs no inference or detector fusion.
Downstream CellViT mask/image consumers require `uni2_use_roi_crop: true`; the
pipeline does not invent a full-image label raster by resizing the crop.

StarDist outputs remain under `03_stardist/<sample>/stardist_out`. The shared
crop's historical `03_stardist/.../prepared_crop` location is also retained for
compatibility when CellViT++ is chosen; its directory name does not mean that
StarDist inference ran.

Changing the detector requires rerunning dependent cell stages. Existing tissue
and cell predictions are preserved; changing a setting does not retrospectively
relabel a completed experiment. Runtime deployment must include the updated
Nextflow modules/scripts; an active SIF/campaign is never replaced in place.
