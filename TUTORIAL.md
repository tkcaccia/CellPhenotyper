# How To Use The Pipeline

This page shows practical usage of CellPhenotyper with Nextflow.

For `-profile docker`, you can pre-pull image from GHCR.
For `-profile singularity`, Nextflow pulls automatically from `singularity_image`.
Do not rebuild `.sif` on every test run.

For frequent runtime issues and fixes, see [Troubleshooting](TROUBLESHOOTING.md).

## Current Virchow2 → Bora tissue route

Use a measured-resolution RGB WSI, preferably OME-TIFF, with its tissue ROI if available. CellPhenotyper keeps the prepared crop's physical-resolution sidecar and checks it against the Virchow2 grid before Bora starts. The complete coarse mask comes from unsupervised KODAMA tile clusters; the separate uncertainty map does not delete those raw tile labels. The classifier is a previously trained Bora Neighbor-context fastPLS `.npz` model, with class names from its training annotations. No expert annotation of the test WSI is passed to Bora.

The current source retains `uni2` and `medsam_refine` as **stage-name compatibility aliases**. In grid mode they execute Virchow2 feature extraction and Bora refinement respectively. `cluster_geojson` is written by the same Bora process. Set `uni2_sampling_mode=grid`, `uni2_encoder=virchow2`, `uni2_target_mpp=0.5`, `kodama_embedding_mode=tile`, and disable all inner-square/fused modes. Each top-left-anchored tile is physically 112 µm wide and has no overlap.

Provide these paths in `pipeline_paramers.yml` or on the command line:

```bash
nextflow run main.nf --folder_input /path/to/WSI-folder \
  --virchow2_checkpoint /path/to/pinned/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72 \
  --bora_classifier_model /path/to/neighbor-context.npz \
  --bora_source /path/to/Bora \
  --uni2_sampling_mode grid --uni2_encoder virchow2 \
  --uni2_target_mpp 0.5 --end_point cluster_geojson
```

The Python environment used by the two GPU stages must have PyTorch, `timm`, `safetensors`, `pyvips`, and the installed `bora-segmentation==0.2.1` package **including its compiled native fastPLS extension**. Staging an unbuilt Bora source tree is not a substitute for that installation. The pipeline checks the pinned checkpoint revision and SHA-256 and stops if the runtime or geometry contract is missing. The currently published CellPhenotyper GPU container has not yet been rebuilt with this Bora wheel; use a verified local runtime for a real WSI execution.

Output is `15_cluster_geojson/<sample>/bora_<sample>_<variant>/mask.ome.tif` plus `mask.geojson`, `PIPELINE_COMPLETE.json`, and window/checkpoint audits. The GeoJSON contains Bora's predicted tissue classes; islands and enclosed holes strictly below 20 native pixels are removed or filled by the final vector pass.

At the right or bottom WSI edge, a non-overlapping grid tile can be only a few pixels wide. Such a partial tile may acquire a cluster ID but cannot supply Bora's required 90-pixel observation. The bridge assigns that edge-only ID to its nearest observed neighboring group, records the reassignment in the window audit, and fails if any unsupported group occurs away from the partial outer edge. No tiles overlap to hide this case.

## Earlier repository examples

Use the input files already present in this repo:

- `Data/ROI_A.ome.tif`
- `Data/ROI_A.geojson`
- `Data/ROI_B.ome.tif`
- `Data/ROI_B.geojson`

Clone and enter repository:

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
```

Prepare runtime:

```bash
# Singularity/Apptainer:
# no manual pull command is required.
# Nextflow pulls `singularity_image` automatically on first run.

# Docker (optional pre-pull, amd64 CPU example)
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
```

Earlier UNI-2 examples below describe a superseded route and are retained only for historical comparison. They are not valid settings for the current Virchow2 → Bora tissue pipeline.

Create UNI-2 token file (historical route only):

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Recommended HPC preparation (avoid UNI-2 download failures during jobs):

```bash
export HF_HOME=/scratch/<project>/CellPhenotyper/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE"

singularity exec <image.sif> python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("MahmoodLab/UNI2-h", token=os.environ["HF_TOKEN"].strip())
print("UNI2 cache ready")
PY
```

Run full example:

```bash
# Singularity
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example

# Docker
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Final results:

- `results_example/15_cluster_geojson/ROI_A/ROI_A_standard_grown_mask_smooth_class.geojson`
- `results_example/15_cluster_geojson/ROI_B/ROI_B_standard_grown_mask_smooth_class.geojson`

## Lima users: run location matters

If your project is under `/Users/...` inside Lima and that mount is read-only, Nextflow may fail creating
`.nextflow/history.lock`.

Use a writable Linux path:

```bash
cd ~
rsync -a --delete <host_project_dir>/CellPhenotyper/ ~/CellPhenotyper/
cd ~/CellPhenotyper
```

Then run the pipeline from `~/CellPhenotyper`.

## Check run progress

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
tail -n 50 -f .nextflow.log
```

Quick recent activity:

```bash
ls -lt work | head
```

## Copy outputs back to macOS

From macOS terminal:

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_example \
  <host_project_dir>/
```

## Starting point

- Multi-image input folder: `folder_input` in `pipeline_paramers.yml` (example: `Data`)
- Single-image mode (optional): `image_input` + optional `roi_geojson`
- If a `<sample>.geojson` file is missing, the full image is used as ROI.
- For CZI with multiple scan regions, place `<image>.czi` beside region-specific ROI files such as `<image>.czi - ScanRegion0.geojson`; the pipeline will create one sample per matching region.

## Endpoint

- Main output base folder: `outdir_base` in `pipeline_paramers.yml` (example: `results_full`)
- Final segmentation endpoint: `results_full/15_cluster_geojson/<sample_root>/<sample_root>_<variant>_grown_mask_smooth_class.geojson`

## Stage window (where to start/stop)

Use `start_point` and `end_point` in `pipeline_paramers.yml` to select the execution window.
Allowed stages:
`convert`, `grandqc`, `stardist`, `cell_consensus`, `tma`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `gigatime`, `marker_quantification`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `medsam_refine`, `cluster_geojson`, `neoplastic_section`, `titan`, `pathofmpred`.

On the current Virchow2 grid route, `--start_point medsam_refine` reruns the Bora bridge from published coarse masks, crop, resolution and grid metadata. `cluster_geojson` is emitted by the same Bora process. The MedSAM restart description below applies only to archived cell-centred runs.

Example:

```yaml
start_point: stardist
end_point: uni2
```

This runs from StarDist and stops after UNI-2 embeddings.

## 1) Run the complete pipeline (Singularity + UNI-2)

UNI-2 requires a valid Hugging Face token with access to `MahmoodLab/UNI2-h`.
By default the pipeline loads the token from `tokens.env` using `hf_token_env_file` + `hf_token_env_var_name` in `pipeline_paramers.yml`.
Default runtime image mode in `pipeline_paramers.yml` is automatic (`runtime_image_mode: auto`), so `singularity_image` stays empty unless you want manual override.

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 2) Run the complete pipeline (Docker + UNI-2)

Default image selection is automatic (`runtime_image_mode: auto`), so no manual `docker_image` is required.

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  -with-report results_full/report.html \
  -with-trace results_full/trace.txt \
  -with-timeline results_full/timeline.html \
  -resume
```

## 3) Runtime image behavior

- `-profile singularity`: Nextflow resolves image automatically (local `.sif` -> GHCR `oras://` SIF tag -> legacy release asset -> `docker://` fallback).
- `-profile docker`: pull image with Docker if needed:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-arm64
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
```

## 4) Run only tissue mask stage (fast validation)

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --start_point convert \
  --end_point tissue_mask \
  -resume
```

Expected output:

- `results_full/04_tissue_mask/ROI_A/ROI_A_tissue_mask.tif`
- `results_full/04_tissue_mask/ROI_B/ROI_B_tissue_mask.tif`

## 5) GPU execution

For GPU-capable Linux hosts (amd64/arm64 + NVIDIA), run:

```bash
# Docker profile
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_gpu \
  --compute_device gpu \
  --host_arch amd64 \
  --runtime_image_mode manual \
  --gpu_container_image ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64 \
  --hf_token_env_file tokens.env \
  --hf_token_env_var_name HF_UNI2

# Singularity profile
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_gpu \
  --compute_device gpu \
  --host_arch amd64
```

For Linux arm64/aarch64 Spark, use `--host_arch arm64` (same commands).

If singularity pull is slow or appears stuck, pre-pull once to monitor progress:

```bash
singularity pull /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  docker://ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
```

If Docker already has the image locally, converting from a local archive avoids a second large registry download:

```bash
docker pull ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
docker save -o /scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar \
  ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64
singularity pull /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif \
  docker-archive:///scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar
rm -f /scratch/<project>/cellphenotyper-2.2-gpu-amd64.tar
```

Then run in manual image mode:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_gpu \
  --compute_device gpu \
  --host_arch amd64 \
  --runtime_image_mode manual \
  --singularity_image /scratch/<project>/singularity/cellphenotyper-2.2-gpu-amd64.sif
```

For offline/restricted compute nodes, export HF cache env before running:

```bash
export APPTAINERENV_HF_HOME=/scratch/<project>/CellPhenotyper/.hf_cache
export APPTAINERENV_HF_HUB_CACHE=/scratch/<project>/CellPhenotyper/.hf_cache/hub
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export SINGULARITYENV_HF_HOME=$APPTAINERENV_HF_HOME
export SINGULARITYENV_HF_HUB_CACHE=$APPTAINERENV_HF_HUB_CACHE
export SINGULARITYENV_HF_HUB_OFFLINE=$APPTAINERENV_HF_HUB_OFFLINE
export SINGULARITYENV_HF_TOKEN=$APPTAINERENV_HF_TOKEN
```

## 6) TensorFlow in runtime images

StarDist runtime dependencies are already preinstalled in the official Docker and Singularity images/assets (including TensorFlow and `imagecodecs`).

You should not install TensorFlow manually before running the pipeline.

## 7) Resume and rerun

To continue from previous successful steps, keep `-resume`.

To force a clean run with a new output folder:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --outdir_base results_full_fresh
```
