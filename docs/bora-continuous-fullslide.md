# Continuous Bora whole-slide review route

`bora_continuous_fullslide.nf` is a separate **CPU-only local** continuation
for one slide with a finished, source-bound global RGB9/RGB19 replay. It does
not run Virchow2, KODAMA, CellViT, a biological tissue classifier, or any GPU
job. It does not replace the historical hard-group Bora route or `main.nf`.

The result is an **anonymous exploratory boundary** for review, not a
tumor/stroma annotation, an expert ground truth, or a PLS training label. The
entire native slide is projected through bounded, deterministic halo shards;
the shards are checked for complete coverage and exact halo agreement, then
assembled and vectorized globally. Parts and holes strictly under 100 µm²
are removed. The smoothed review GeoJSON and exact-mask QA provenance enter
one new CellPhenotyper project file as a locked review artifact.

## Runtime and inputs

The Nextflow process uses `executor 'local'` and the **running host Python**.
It deliberately rejects Docker and Singularity profiles. A supplied `.sif`
path is hashed as extra evidence only; it does **not** mean that the process
executed inside the SIF. A qualified container route has not been established.
The running Python environment needs the dependencies used by staged Bora and
CellPhenotyper, including NumPy, SciPy, rasterio, Shapely, scikit-image and
Pillow. The identity command records their versions and module-file hashes,
the interpreter binary hash, and an optional SIF hash. It also hashes every
Python file under staged `src/bora` plus `pyproject.toml`.

All four inputs must belong to the same image: a completed frozen
`TOKEN_REPLAY_COMPLETE.json` with p9/p19/combined `.npy` files; a verified
dual-MPP registration; its native GrandQC support TIFF; and an existing
CellPhenotyper base project whose image is the native 0.25 µm/pixel frame.
The registration's full-image/frame/hash/MPP and support SHA-256 must match.
The global replay is for the 0.5 µm/pixel working image. The action never
creates or repairs any missing upstream evidence.

First, in the exact Python environment that will execute the action, obtain
the two requested pins:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 bin/bora_continuous_fullslide_action.py identity \
  --bora-source /absolute/path/to/staged-bora-source
```

Then put the returned `bora_source_sha256` and `runtime_sha256` in the command
below. The run output path must **not** exist, and its parent must exist.

```bash
nextflow run bora_continuous_fullslide.nf --compute_device cpu \
  --bora_continuous_replay /absolute/path/to/frozen-replay \
  --bora_continuous_registration /absolute/path/to/registration.json \
  --bora_continuous_support_tiff /absolute/path/to/registered-grandqc.tif \
  --bora_continuous_base_project /absolute/path/to/base.cellphenotyper.json \
  --bora_continuous_source /absolute/path/to/staged-bora-source \
  --bora_continuous_source_sha256 SOURCE_SHA256 \
  --bora_continuous_runtime_sha256 RUNTIME_SHA256 \
  --bora_continuous_output /absolute/new/slide-result
```

Defaults are 2048-pixel native cores with 128-pixel halos, a 4 GiB
declared memory bound in the Nextflow wrapper (512 MiB when the Python
action is called directly), a 16 GiB total disk admission, two CPUs, one process and
zero retries. The action additionally requires free space for its full
estimated shard and global-mask footprint, including a 2 GiB reserve. It
streams the registered support into a memory-mapped `inputs/support.npy`;
the Bora global chain independently rematerializes and verifies its support.
The maximum is 4096 shards, 16M pixels per padded shard, 1 billion native
pixels, 32 anonymous classes, and 25 hours in the Nextflow wrapper. The
operator can change core/halo/memory/disk parameters only within the
wrapper's bounded range. These are resource limits, not a claim that a
23,904 × 14,762 slide has been run or visually qualified.

The single new output folder contains `PREFLIGHT.json`, `inputs/support.npy`,
numbered immutable `shards/`, `global/` with exact/review vectors and QA
receipts, `experiment.cellphenotyper.json`, and `RUN_COMPLETE.json`. If a
shard or later stage fails, the folder stays intact with `RUN_FAILED.json`
when writable; the project is not published and this action will not resume
or overwrite the folder. A hard kill or full disk can prevent the failure
receipt, so absence of `RUN_COMPLETE.json` is always incomplete. The
Nextflow `-stub-run` emits `complete:false` and creates no scientific result.

The real PM016 corrected boundary remains **unqualified** until paired
real RGB9/RGB19 data at the disputed macro-border are processed and compared
visually with H&E and any available expert annotation. Full-slide CellViT
coverage is an independent unfinished task. Do not use this local route to
claim the complete four-slide pipeline is finished.
