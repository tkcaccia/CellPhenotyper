# CellPhenotyper

CellPhenotyper is a Nextflow DSL2 pipeline for H&E tissue image analysis. It runs GrandQC artifact QC; StarDist, HoVer-Net and CellViT++ consensus segmentation; optional tissue microarray (TMA) analysis; GigaTIME virtual mIF inference; marker quantification; paired UNI-2 embeddings; KODAMA and Leiden clustering; MedSAM border refinement; and final tissue-cluster GeoJSON export.

Main command:

```bash
nextflow run main.nf
```

## Documentation

- [Installation](INSTALL.md)
- [How to run](TUTORIAL.md)
- [Parameters](PARAMETERS.md)
- [Output](OUTPUT.md)
- [Pipeline step differences vs upstream tools](UPSTREAM_DIFFS.md)
- [Release](RELEASE.md)
- [Linux update playbook](LINUX_UPDATE.md)
- [Singularity maintainer guide](singularity/README.md)

## Example input in this repository

- `Data/ROI_A.ome.tif`
- `Data/ROI_A.geojson`
- `Data/ROI_B.ome.tif`
- `Data/ROI_B.geojson`

If `Data/<sample>.geojson` is missing, CellPhenotyper automatically uses the full image as ROI for that sample.

For CZI inputs with multiple scan regions, place the `.czi` file and the region-specific GeoJSON files in the same folder using the naming pattern:

- `<image>.czi`
- `<image>.czi - ScanRegion0.geojson`
- `<image>.czi - ScanRegion1.geojson`

CellPhenotyper resolves one pipeline sample per matching `ScanRegionN` GeoJSON, converts only that CZI region to TIFF, and keeps the region-specific ROI paired with the derived TIFF.

## Runtime behavior (automatic container selection)

Use one profile per run:

- `-profile docker`
- `-profile singularity`

Do not use both profiles in the same run.

Default image selection is automatic (`runtime_image_mode: auto`):

- Docker profile uses GHCR images.
- Singularity profile auto-resolves architecture-specific `.sif` assets when available.
- Singularity/Apptainer GPU roles use `singularity_gpu_image_source: docker` by default, selecting the verified `2.7-gpu-amd64` OCI runtime instead of legacy GPU SIF metadata.
- In GPU mode, GPU-capable stages including GrandQC, StarDist, HoVer-Net, CellViT++, GigaTIME, UNI-2, MedSAM and TITAN use the GPU runtime; other stages use the CPU runtime.
- On arm64 GPU runs, missing GPU assets fall back to CPU containers (no amd64 GPU image fallback).
- On arm64, StarDist defaults to CPU container unless `--enable_stardist_gpu_on_arm64 true`.

Currently verified and published:

- Docker GPU amd64 (`v2.7`): `ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64`
- Docker CPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-amd64`
- Docker CPU arm64: `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- Legacy Docker GPU amd64: `ghcr.io/tkcaccia/cellphenotyper:2.2-gpu-amd64`
- Published GHCR/ORAS SIF artifacts: `2.2-sif-amd64`, `2.2-sif-arm64`, `2.2-sif-gpu-amd64`, and `2.2-sif-gpu-arm64` under `ghcr.io/tkcaccia/cellphenotyper`.

The legacy arm64 GPU artifact is published but is not suitable for every GPU generation. In particular, GB10-class (`sm_121`) systems require a rebuilt arm64 GPU SIF with a compatible CUDA/PyTorch stack.
The currently built amd64 SIF files are too large for ordinary GitHub release assets, so the stable amd64 Singularity workflow is still local `apptainer pull` / `singularity pull` from the verified Docker tags.

Verify actual published Docker tags before instructing users to pull them:

```bash
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper:2.2-amd64
docker buildx imagetools inspect ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64
```

## UNI-2 token setup (required)

1. Create/sign in at [Hugging Face](https://huggingface.co).
2. Request access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).
3. Create a read token at [Hugging Face tokens](https://huggingface.co/settings/tokens).
4. In the project root:

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Run these `source/export` commands in every new shell before starting Nextflow.
For Docker profile runs, keep `tokens.env` in the repository root (default bind-mounted working directory) and pass `--hf_token_env_file tokens.env`.

If you get `401 Unauthorized` during UNI-2 download, check token validity and model access approval.

## Project-local model caches (default)

StarDist and Hugging Face model downloads now default to project-local cache paths:

- StarDist: `${REPO}/.keras`
- UNI-2 / GigaTIME: `${REPO}/.hf_cache`

This matters for Docker reruns: because both caches live inside the repository, they are visible inside containerized Nextflow tasks and can be reused offline.

One-time StarDist predownload into the default project cache:

```bash
mkdir -p .keras/models
curl -L --retry 5 --connect-timeout 30 \
  -o .keras/models/python_2D_versatile_he.zip \
  https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip
```

On the first StarDist run, CellPhenotyper will normalize and extract that zip into `.keras/models/StarDist2D/...` automatically. Later reruns reuse the local cache and do not re-download the pretrained model.

One-time Hugging Face predownload into the default project cache:

```bash
source tokens.env
export HF_TOKEN="${HF_UNI2}"
export HF_HOME="${PWD}/.hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download("MahmoodLab/UNI2-h", token=os.environ["HF_TOKEN"].strip())
snapshot_download("prov-gigatime/GigaTIME", token=os.environ["HF_TOKEN"].strip())
print("HF caches ready")
PY
```

Later Docker reruns can set `--hf_hub_offline true` and reuse the same mounted cache for both UNI-2 and GigaTIME.

## Input ROI mask

When an input ROI GeoJSON is present, the pipeline writes a crop-aligned mask to `06_roi/<sample>/` using the same cropped ROI coordinates generated for StarDist. The mask preserves distinct annotation classes from the input GeoJSON whenever those labels are present in properties such as `classification.name`, and it also writes:

- a colored preview overlay PNG
- a JSON value-to-label map recording the class IDs used in the rasterized mask

## TMA Detection

After StarDist, the optional `04_TMA` step detects whether the crop behaves like a tissue microarray by segmenting compact separated tissue cores on a thumbnail and checking spot count, spot-size consistency, and grid-like layout. When a TMA is detected it writes `04_TMA/<sample>/tma_<sample>/<sample>_tma_spots.geojson`; in all cases it writes `04_TMA/<sample>/tma_<sample>/<sample>_objects_tma_assigned.csv`, preserving the StarDist object rows and appending `tma_spot_*` columns.

## GigaTIME marker quantification

GigaTIME is enabled by default, so the pipeline also quantifies the crop-aligned GigaTIME marker stack over:

- StarDist nuclei labels
- expanded cytoplasm labels

The source MPP is recovered from TIFF metadata or the StarDist coordinate metadata. With strict target-MPP mode enabled, CellPhenotyper resamples by the exact floating-point MPP ratio before tiling, including lazy pyvips upsampling when the source is coarser than the requested model resolution. The persisted Zarr and pyramidal OME-TIFF record the resulting physical pixel size and keep markers as separate channels.

Outputs are written per sample to:

- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_quantification.csv`
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_mean_intensity.csv`
- `05_gigatime/<sample>/quantification_<sample>/<sample>_nuclei_gigatime_intensity_stats.csv`
- `05_gigatime/<sample>/quantification_<sample>/<sample>_cyto_gigatime_quantification.csv`
- `05_gigatime/<sample>/quantification_<sample>/<sample>_cyto_gigatime_mean_intensity.csv`
- `05_gigatime/<sample>/quantification_<sample>/<sample>_cyto_gigatime_intensity_stats.csv`

The new `*_gigatime_quantification.csv` file is a wide per-object table in the same spirit as mcMicro-style single-cell quantification outputs: one row per label with area, centroid, bounding box, and per-marker mean/sum/max columns. The mean-intensity CSV remains convenient for lightweight downstream modeling, while the stats CSV preserves the explicit summary fields.

## Optional dual clustering outputs

After KODAMA, CellPhenotyper produces the `standard` clustering variant by default. Set `--cluster_secondary_variant fine` to add a second branch:

- `standard`: the current/default clustering behavior
- `fine`: a slightly higher-resolution clustering that prefers a few more clusters when the KODAMA clustering score stays close to the standard solution

Downstream stages run independently for every configured variant:

- `11_clustering`
- `12_cluster_mask`
- `13_grown_tissue`
- `14_medsam_refine_tissue`
- `15_cluster_geojson`

Variant-specific filenames are written inside the per-sample folders, for example:

- `<sample>_standard_cluster.csv`
- `<sample>_fine_cluster.csv`
- `<sample>_standard_grown_mask_refined.ome.tif`
- `<sample>_fine_grown_mask_refined.ome.tif`
- `<sample>_standard_grown_mask_smooth_class.geojson`
- `<sample>_fine_grown_mask_smooth_class.geojson`

Step `14_medsam_refine_tissue` also copies the corresponding KODAMA membership PNG for each variant into the MedSAM output folder as:

- `<sample>_standard_medsam_kodama_membership.png`
- `<sample>_fine_medsam_kodama_membership.png`

That keeps the refined tissue result side by side with the clustering visualization that produced it.

UNI-2 embeddings are generated after StarDist. The default configuration uses the optimized paired `tile` + `inner_square` pass: one Python process loads UNI2-h once, prepares StarDist-centered tile crops and fixed centered 90-pixel inner-square crops in memory, and encodes both image streams in the same batched model session. The `inner_square` output is not derived from the cytoplasm mask; it is the centered square defined by `uni2_inner_square_fixed_px` in the 224 x 224 UNI2 input space. Set `--uni2_fuse_tile_inner_square false` to force the slower two-pass comparison mode. The source crop is calibrated to `uni2_target_mpp` before the 224 x 224 UNI2 input transform. Per-cell PNG tile export is disabled by default because KODAMA consumes the embedding CSVs; enable `--uni2_save_tiles true` only when tile-level QC/debug crops are needed. MedSAM refinement is applied per cluster label using overlapping cluster-border tiles, so a tile can contain only one cluster while still preserving cluster-specific border refinement.

## Linux quick run

Before every run, sync and verify you are on the latest `main`:

```bash
git fetch origin
git checkout main
git pull --ff-only
git rev-parse --short HEAD
git rev-parse --short origin/main
git status --short
```

`HEAD` and `origin/main` must match, and `git status --short` should be empty.

Linux runtime precheck:

```bash
java -version
nextflow -version
```

If `nextflow` is not in PATH in your shell, use `/home/<user>/.local/bin/nextflow`.

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
```

Docker:

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Docker (GPU, Linux amd64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu \
  --compute_device gpu \
  --host_arch amd64 \
  --gpu_container_image ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64 \
  --hf_token_env_file tokens.env \
  --hf_token_env_var_name HF_UNI2
```

Singularity/Apptainer:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Singularity/Apptainer (GPU, Linux amd64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu \
  --compute_device gpu \
  --host_arch amd64
```

For HPC clusters without outbound internet from compute nodes, use this offline-ready flow (CPU or GPU):

```bash
# 0) paths
export REPO=/scratch/<project>/CellPhenotyper
export BASE=/scratch/<project>/cellphenotyper_cache
export SIF_DIR=$BASE/singularity
export KERAS_HOME=$BASE/keras
export HF_HOME=$BASE/hf
export HF_HUB_CACHE=$HF_HOME/hub
export SIF=$SIF_DIR/cellphenotyper-2.2-amd64.sif
mkdir -p "$SIF_DIR" "$KERAS_HOME/models/StarDist2D" "$HF_HUB_CACHE"

# 1) token file must define HF_TOKEN=...
source /scratch/<project>/tokens.env

# 2) pull prebuilt SIF once (on a node with internet)
apptainer pull -F "$SIF" docker://ghcr.io/tkcaccia/cellphenotyper:2.2-amd64

# 3) predownload StarDist model and normalize to expected local folder
curl -L -o "$KERAS_HOME/models/StarDist2D/python_2D_versatile_he.zip" \
  https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip
mkdir -p /tmp/stardist_unpack
unzip -o "$KERAS_HOME/models/StarDist2D/python_2D_versatile_he.zip" -d /tmp/stardist_unpack
mkdir -p "$KERAS_HOME/models/StarDist2D/2D_versatile_he"
if [ -d /tmp/stardist_unpack/python_2D_versatile_he ]; then
  cp -a /tmp/stardist_unpack/python_2D_versatile_he/. "$KERAS_HOME/models/StarDist2D/2D_versatile_he/"
else
  cp -a /tmp/stardist_unpack/. "$KERAS_HOME/models/StarDist2D/2D_versatile_he/"
fi

# 4) predownload UNI2 model once
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
apptainer exec "$SIF" python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download

repo = "MahmoodLab/UNI2-h"
snapshot_download(repo, token=os.environ.get("HF_TOKEN", "").strip() or None, local_files_only=False)

# Force actual weight file in cache for strict offline runs.
try:
    p = hf_hub_download(repo_id=repo, filename="model.safetensors", token=os.environ.get("HF_TOKEN", "").strip() or None, local_files_only=False)
    print("UNI2 weight cached:", p)
except Exception:
    p = hf_hub_download(repo_id=repo, filename="pytorch_model.bin", token=os.environ.get("HF_TOKEN", "").strip() or None, local_files_only=False)
    print("UNI2 weight cached:", p)

print("UNI2 cache ready")
PY

# 5) offline cache env for all Nextflow tasks
export APPTAINERENV_KERAS_HOME="$KERAS_HOME"
export APPTAINERENV_XDG_CACHE_HOME="$KERAS_HOME"
export SINGULARITYENV_KERAS_HOME="$KERAS_HOME"
export SINGULARITYENV_XDG_CACHE_HOME="$KERAS_HOME"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_HF_HOME="$APPTAINERENV_HF_HOME"
export SINGULARITYENV_HF_HUB_CACHE="$APPTAINERENV_HF_HUB_CACHE"
export SINGULARITYENV_HF_TOKEN="$APPTAINERENV_HF_TOKEN"
export SINGULARITYENV_HF_HUB_OFFLINE="$APPTAINERENV_HF_HUB_OFFLINE"

# 6) run (CPU example; compute_device can be cpu or gpu)
nextflow run "$REPO/main.nf" \
  -profile singularity \
  -params-file "$REPO/pipeline_paramers.yml" \
  --folder_input "$REPO/Data" \
  --outdir_base "$REPO/results_hpc_offline" \
  --compute_device cpu \
  --host_arch amd64 \
  --runtime_image_mode manual \
  --singularity_image "$SIF" \
  --stardist_keras_home "$KERAS_HOME" \
  --hf_home "$HF_HOME" \
  --hf_hub_cache "$HF_HUB_CACHE" \
  --hf_hub_offline true \
  --max_cpus "${SLURM_CPUS_PER_TASK:-8}" \
  -resume
```

Important:
- Run inside a scheduler allocation (`srun`, `sbatch`, etc.) so Nextflow sees the allocated CPUs.
- Do not pass `--stardist_pretrained_zip` when the extracted folder already exists under `.../StarDist2D/2D_versatile_he`.
- In `-profile singularity`, automatic resolution may still probe multiple sources, but the verified manual path today is a local `.sif` created from `docker://ghcr.io/tkcaccia/cellphenotyper:2.2-amd64` or `docker://ghcr.io/tkcaccia/cellphenotyper-runtime:2.7-gpu-amd64`.

Singularity/Apptainer (GPU, Linux arm64 + NVIDIA):

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example_gpu_arm64 \
  --compute_device gpu \
  --host_arch arm64 \
  --enable_gpu_on_arm64 true
```

Note: on GB10-class arm64 GPUs (`sm_121`), use a locally rebuilt arm64 GPU SIF from `singularity/cellphenotyper_full_gpu.def` (nightly `cu130` PyTorch). The legacy `2.2-gpu-arm64` asset may expose CUDA but still fail at runtime with `no kernel image is available`.

Rerun only `cluster_mask` and `grow_tissue`:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example \
  --start_point cluster_mask \
  --end_point grow_tissue
```

## macOS quick run

Docker (native with Docker Desktop):

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
nextflow run main.nf \
  -profile docker \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Singularity (via Lima Linux VM):

```bash
limactl shell default
mkdir -p ~/CellPhenotyper
rsync -a --delete /Users/<your-user>/Documents/CellPhenotyper/ ~/CellPhenotyper/
cd ~/CellPhenotyper
source tokens.env
export HF_TOKEN="${HF_UNI2}"
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --folder_input Data \
  --outdir_base results_example
```

Important for Lima: run from a writable Linux path (for example `~/CellPhenotyper`), not from `/Users/...` mount paths.
On Apple Silicon/Linux arm64, GPU mode requires an arm64-compatible GPU container asset (`singularity_gpu_asset_arm64` or `gpu_container_image`).

## Check status and outputs

Run status:

```bash
ps aux | grep -E 'nextflow|java' | grep -v grep
tail -n 50 -f .nextflow.log
```

Final output:

- `results_example/15_cluster_geojson/ROI_A/ROI_A_standard_grown_mask_smooth_class.geojson`
- `results_example/15_cluster_geojson/ROI_B/ROI_B_standard_grown_mask_smooth_class.geojson`
- `results_example/05_gigatime/ROI_A/gigatime_ROI_A_ometiff/gigatime_probs.ome.tif`
- `results_example/05_gigatime/ROI_B/gigatime_ROI_B_ometiff/gigatime_probs.ome.tif`
- `results_example/06_roi/ROI_A/ROI_A_input_roi_mask.tif` if `ROI_A.geojson` was supplied
- `results_example/06_roi/ROI_B/ROI_B_input_roi_mask.tif` if `ROI_B.geojson` was supplied
- `results_example/06_roi/ROI_A/ROI_A_input_roi_mask_preview.png` if `ROI_A.geojson` was supplied
- `results_example/06_roi/ROI_B/ROI_B_input_roi_mask_preview.png` if `ROI_B.geojson` was supplied
- `results_example/06_roi/ROI_A/ROI_A_input_roi_mask_labels.json` if `ROI_A.geojson` was supplied
- `results_example/06_roi/ROI_B/ROI_B_input_roi_mask_labels.json` if `ROI_B.geojson` was supplied
- `results_example/05_gigatime/ROI_A/quantification_ROI_A/ROI_A_nuclei_gigatime_quantification.csv`
- `results_example/05_gigatime/ROI_A/quantification_ROI_A/ROI_A_cyto_gigatime_quantification.csv`

Execution report:

- `results_example/00_execution/final_report.md`
- `results_example/00_execution/final_report.json`

Copy results from Lima to macOS host:

```bash
limactl copy default:/home/<lima-user>/CellPhenotyper/results_example \
  /Users/<your-user>/Documents/CellPhenotyper/
```

## Maintainer: publish updated containers

Docker and Singularity build/publish workflows are documented here:

- [Release](RELEASE.md)
- [Singularity maintainer guide](singularity/README.md)
- [Linux update playbook](LINUX_UPDATE.md)

Minimal Docker publish example:

```bash
export GHCR_USER="tkcaccia"
source GHCRtoken.env
export TAG="<new-version>-amd64"
export IMAGE="ghcr.io/${GHCR_USER}/cellphenotyper:${TAG}"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker build -f docker/Dockerfile.full.cpu -t "${IMAGE}" .
docker push "${IMAGE}"
```
Cell identification can combine three independent instance-segmentation procedures: StarDist, HoVer-Net with the MoNuSAC checkpoint, and CellViT++. The two additional GPU models run serially after StarDist on the same MPP-aware crop to avoid VRAM oversubscription. A 2-of-3 spatial consensus creates canonical cell IDs and becomes the nuclei mask/object table for all downstream analysis while retaining per-model provenance. CPU runs skip the two-model GPU extension and continue from StarDist instead of failing.

Normalized HoVer-Net and CellViT++ outputs expose both the upstream numeric `type_id` and a readable `type` name. MoNuSAC names are `background`, `epithelial`, `lymphocyte`, `macrophage`, and `neutrophil`; CellViT++ PanNuke names are `neoplastic`, `inflammatory`, `connective`, `dead`, and `epithelial`.

## Neoplastic Section to PathoFMPred

When `--titan_enable true` is set, CellPhenotyper treats each connected polygon in the final tissue GeoJSON as a section, streams the consensus cell table through a spatial index, and counts named CellViT++ `neoplastic` cells in each polygon. It selects the section with the largest neoplastic count, with total cells, area, and stable section ID used only as deterministic ties. Stage 16 exports the selected full-resolution masked section, mask, original/crop GeoJSON, coordinate shift, count table, summary, and preview.

Stage 17 samples the selected section at the physical equivalent of 512 pixels at 0.5 microns per pixel. It uses the official gated TITAN implementation and its CONCH v1.5 patch encoder, then writes one 768-dimensional section vector with columns `titan_000` through `titan_767`. The TITAN model/cache remains external to the run and work directories to avoid duplication.

With `--pathofmpred_enable true --pathofmpred_cancer BRCA`, stage 18 passes the named TITAN vector to the protected PathoFMPred R package. The package is loaded from `pathofmpred_library_dir` and is not baked into the public runtime image. PathoFMPred scores are TCGA-derived research estimates without independent external clinical validation; binary outputs are not calibrated probabilities.

Example GPU run using a checksum-verified local TITAN snapshot:

```bash
nextflow run main.nf -profile docker \
  --image_input /data/breast.tif \
  --roi_geojson /data/breast.geojson \
  --outdir_base /results/breast \
  --compute_device gpu \
  --cell_consensus_enable true \
  --titan_enable true \
  --titan_model /models/titan/dac6773d \
  --titan_offline true \
  --pathofmpred_enable true \
  --pathofmpred_cancer BRCA \
  --pathofmpred_library_dir /models/pathofmpred/R_library
```

See `OUTPUT.md` for the complete stage 16-18 artifacts and `PARAMETERS.md` for restart points. A restart at `titan` reuses stage 16; a restart at `pathofmpred` reuses the existing TITAN CSV.
