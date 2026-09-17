# Optional isolated atlas container runtime

The atlas overlay adds a separate Python 3.12 environment for cell profiles,
neighborhoods, unsupervised hierarchy discovery, reference mapping, and
SpatialData export. It does **not** install the NumPy 2 / Zarr 3 stack into the
existing model environment, change default images, or change pipeline settings.
UNI-2 feature extraction continues to use the base image's model environment.

**Verification status:** the source-controlled wrappers and a functional local
CPU probe pass 11 tests. The probe executes native PCA/KMeans, morphology, and
an actual canonical-profile → SpatialData → exact readback round trip. No atlas
Docker/SIF image has been built or published as part of this work. Linux wheel
availability, isolated built-container execution, real model weights, GPU
execution, and biological accuracy remain unverified. The local combined CPU
test uses existing packages from two development environments; a separate test
confirms that the build's strict-isolation check rejects that mixed setup.

## Runtime boundaries

| Work | Interpreter selected by the optional configuration |
| --- | --- |
| Cell morphology, profiles, hierarchy links, reference mapping | `cellphenotyper-atlas-python` → isolated atlas Python |
| SpatialData export and its Python preflight | `cellphenotyper-atlas-python` → isolated atlas Python |
| `prepare_hierarchy_features.py` | `cellphenotyper-hierarchy-python` → existing model Python |
| `discover_tissue_hierarchy.py` | `cellphenotyper-hierarchy-python` → isolated atlas Python |

Both hierarchy processes currently share `tissue_hierarchy_python`; the small
dispatcher routes these two explicit script names and rejects every other
entry point. It is not a general `python -c` replacement. Default locations are
`/opt/micromamba/envs/cellphenotyper-atlas/bin/python` and
`/opt/micromamba/envs/stardist/bin/python`. The optional overrides
`CELLPHENOTYPER_ATLAS_PREFIX` and `CELLPHENOTYPER_ML_PYTHON` must resolve to an
absolute executable location; a missing environment fails without falling back
to the host interpreter.

The atlas wrapper removes inherited Python/user-site and model-library search
paths. It preserves script-local imports needed by the pipeline. The model
route preserves the base image's CUDA/library configuration. Neither wrapper
installs packages at task execution time.

### Source-code bytecode isolation

The eight atlas Nextflow stages source `bin/activate_source_python.sh` before
Python. A fresh empty task-private bytecode prefix plus disabled writes prevents
timestamp/size-based stale `.pyc` from defeating a correct Nextflow source hash.
The helper does not replace task/GPU cleanup traps or delete shared caches.
The empty private directory remains in the task work directory.

Both packaged launchers keep `-E -s` isolation. Since `-E` ignores `PYTHON*`
environment variables, the launchers verify the explicit
`CELLPHENOTYPER_SOURCE_PYCACHEPREFIX` task marker and matching no-write settings,
then forward literal `-B -X pycache_prefix=...` arguments. Missing, mismatched,
nonempty or symlinked prefixes fail closed. A lone inherited
`PYTHONPYCACHEPREFIX` is not accepted as this contract; invoke the task helper
first or remove that unrelated setting for an ordinary standalone invocation.

Local direct/launcher stale-bytecode and negative tests passed; see the
[array-backed/cache audit](../audits/spatial_atlas_20260904/array_backed_neighborhoods.md).
This is not a built-container/GPU verification. Disabling bytecode-cache reads
can add import startup time; no whole-pipeline speed gain is inferred.

## Build an explicitly selected candidate

Run these commands only when a container build is intended. Builds download
dependencies and need the normal Docker/Apptainer build permissions. Start in
the repository root. Replace the example base reference with a verified,
architecture-matched **full CellPhenotyper runtime** containing micromamba and
the existing model environment. A generic Python image is not sufficient.
Choose a CPU or GPU base appropriate to the intended task; the overlay does not
add GPU support to a CPU base.

```bash
docker build \
  --build-arg BASE_IMAGE='registry.example/cellphenotyper@sha256:REPLACE_WITH_PLATFORM_DIGEST' \
  -f docker/Dockerfile.atlas \
  -t cellphenotyper-local:atlas-candidate .
```

`BASE_IMAGE` deliberately has no default. For releases, use the verified
platform-specific manifest digest, not a mutable tag or an unexamined
multi-platform index. The overlay refuses to replace an existing atlas prefix;
use a base without this overlay when rebuilding it.

The equivalent source-build SIF definition requires Apptainer definition-file
templating (documented in Apptainer 1.2 and later). An omitted required build
argument fails. See the [official Apptainer definition-file documentation](https://apptainer.org/docs/user/1.2/definition_files.html).

```bash
apptainer build \
  --build-arg BASE_IMAGE='registry.example/cellphenotyper@sha256:REPLACE_WITH_PLATFORM_DIGEST' \
  atlas-candidate.sif singularity/cellphenotyper_atlas.def
```

Alternatively, convert an already built, verified atlas OCI image into SIF:

```bash
apptainer build atlas-candidate.sif \
  docker://registry.example/cellphenotyper-atlas@sha256:REPLACE_WITH_PLATFORM_DIGEST
```

The conversion reuses the installed environment rather than independently
resolving dependencies again. An ordinary base image converted to SIF does not
gain the atlas runtime. These examples do not publish anything, and the existing
release helper does not automatically select the optional atlas definition.

## Explicit pipeline invocation

The optional `docker/atlas-runtime.config` changes only the three interpreter
parameters above. Select an atlas-enabled image separately and retain your
existing feature flags and scientific parameters. For the model-free rebuild
from [existing sample artifacts](CELL_ATLAS_USAGE.md):

```bash
nextflow run cell_profiles.nf -profile docker -c docker/atlas-runtime.config \
  --cell_profile_samples /absolute/path/samples.json \
  --cell_profiles_spatialdata true \
  --cpu_container_image cellphenotyper-local:atlas-candidate
```

For Apptainer/Singularity use `-profile singularity` and
`--cpu_container_image /absolute/path/atlas-candidate.sif` instead. A full
pipeline run can use the same configuration, but must explicitly select an
atlas-enabled CPU image **and** an appropriate GPU image when both are used:
`--cpu_container_image ... --gpu_container_image ...`. Keep the existing
params file and enable only the requested atlas/hierarchy stages. Selecting the
configuration alone does not enable those stages or supply model weights.

The four existing Docker full/source-refresh/runtime-update recipes now copy
the current atlas source, supporting `lib` and `resources`, and standalone
workflow alongside the existing pipeline. They still do not install the atlas
environment unless the optional overlay is applied.

## Build checks and evidence

Both overlay recipes call the same build-time installer. It creates the atlas
prefix without system site packages, installs `requirements-atlas.txt` (which
includes `requirements-spatialdata.txt`), runs `pip check`, then runs two probes:

- Atlas: checks direct version pins, actual source imports, native numerical
  operations, compressed TIFF I/O, and an actual SpatialData export/readback.
  Readback checks exact cell labels, cell IDs, feature values with missingness,
  and float64 predicted-marker values. Critical package origins must be inside
  the atlas prefix; visible Torch, TensorFlow, or timm packages are rejected.
- Model: imports `prepare_hierarchy_features` and the actual UNI-2 encoder
  loader from `extract_uni2_embeddings`, exercising their top-level
  Torch/timm/torchvision dependencies, then checks the NumPy↔Torch CPU bridge.
  This does not load weights, construct the UNI-2 model, or execute a GPU kernel.
  A later local probe passed using the existing Miniforge Python 3.10.15,
  Torch 2.4.0, timm 1.0.17 and NumPy 2.0.2 environment (5.55 seconds). Both
  real module imports and the NumPy↔Torch bridge passed; weights were not
  accessed and no models were loaded. This local, non-isolated macOS import
  check does not verify a container build or CUDA execution; the actual Linux
  image must independently pass its build-time probes.

  That Torch 2.4.0 import success does not satisfy CellViT's restricted graph
  decoder: it lacks `torch.serialization.safe_globals`. The new serialized-graph
  producer tests passed with existing Torch 2.14.0 in `.venv-marker-tests` using
  tiny synthetic tensors, not learned weights. The CellViT wrapper must retain
  its explicit decoder preflight; no unsafe-pickle fallback is permitted.

Successful builds write `atlas_runtime_build.json` and
`atlas_model_runtime_build.json` under `/opt/cellphenotyper`. They record the
observed interpreter, architecture, package versions/origins, probe results, and
limitations. The atlas report additionally binds the SHA-256 of every
recursively included requirements file, not just the top-level file. Reports
are evidence of the checks actually executed, not biological certification or
a transitive dependency lock. The verifier refuses to overwrite a prior report.

To repeat the CPU check inside an already built image:

```bash
docker run --rm cellphenotyper-local:atlas-candidate \
  /usr/local/bin/cellphenotyper-atlas-python \
  /opt/cellphenotyper/docker/verify_atlas_runtime.py \
  --source-root /opt/cellphenotyper \
  --requirements /opt/cellphenotyper/requirements-atlas.txt --strict-isolation
```

Use `apptainer exec /absolute/path/atlas-candidate.sif` followed by the same
interpreter and arguments for SIF. The SIF definition also runs this check in
its `%test` section. These checks use only small synthetic data and do not need
patient slides, model weights, or network access.

## Task resources and GPU admission

The migrated GrandQC, StarDist, HoVer-Net, UNI2, CellViT, KODAMA, GigaTIME,
MedSAM, TITAN and seven atlas stages receive a
resolved `TaskRuntime` value input. The main workflow publishes that map inside
`00_execution/hardware_plan.json`; CPU and RAM settings are capped independently
for each stage. The artifact-only entry point also preserves explicit host and
executor limits, including a 1-GB cap. Unit-bearing hierarchy limits such as
`1536 MB` are not rounded up to whole GB.

Docker and Singularity profiles retain the same host-side admission hook and
resolve GPU image, exposure and admission from the same task device decision.
Hierarchy, GrandQC, MedSAM and KODAMA device/backend overrides take precedence
where supported. KODAMA's GPU backend is named `cuda`, not `gpu`. MPS/Metal are
native-only requests and are rejected under these Linux container profiles.
Dynamic admission exposes a single device; nonzero task-local device ordinals
are rejected rather than silently changed. Existing TITAN model binds,
HoVer-Net container options, manual image choices and global extra options
are retained.

The whole-GPU allocator intersects inherited CUDA and NVIDIA device restrictions
before creating reservation locks. Unset host-side variables impose no extra
restriction; an explicitly empty/disabled allocation is not permission to use
all GPUs. Numeric IDs and uniquely identifying UUID prefixes are supported;
malformed lists, ambiguous IDs and MIG allocations fail closed. The selected
full UUID is exported when the inventory provides it; NVML index-based lock
names remain compatible with existing reservations. These rules follow the
[CUDA visibility contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
and [NVIDIA container device controls](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html),
with stricter rejection of CUDA's invalid-token/truncated-list syntax.

The current local executor runs the hook before container launch. Tests inspect
generated wrappers and use fake engines/leases to check that the selected UUID
and index reach Docker and Singularity, including Singularity's sanitized
environment. These are command/handoff tests, not real CUDA inference or
proof of driver/image compatibility. GrandQC `auto` now resolves to an explicit
CPU/CUDA request from the task plan, independently of performance auto-tuning;
an explicit stage-level MPS request remains supported on a native host.
The arm64 StarDist CPU default is applied before container exposure/admission
as well as in the task command. MedSAM retains its planned worker cap, and
HoVer-Net postprocessing is capped by task CPU and RAM allocations.
Other legacy CPU-stage resource handoffs have not all been migrated.
GPU reservation itself requires Linux-compatible Bash >=4.1 and
`flock`; full reservation/concurrency tests are not validated on this Mac's
Bash 3.2 runtime. Explicit site replacement of `containerOptions` or the
Singularity environment whitelist can override the supplied forwarding rules.

## Compatibility limits

- Direct dependencies are pinned, including scikit-learn and scikit-image;
  transitive dependencies and the Python 3.12 patch release are not fully
  hash-locked. The build report captures the resolved environment, but identical
  future resolution is not guaranteed. Release reproducibility needs an
  immutable built-image digest and an architecture-specific verified receipt.
- Installation requires binary wheels for every pip dependency. Unsupported
  Linux/Python/architecture combinations fail instead of silently compiling a
  different native stack. Neither amd64 nor arm64 overlay builds have been
  performed here, and macOS functional success does not establish Linux ABI
  compatibility.
- The model environment remains unchanged. A base with incompatible
  Torch/timm/torchvision or NumPy fails its source-import/bridge probe; the
  overlay does not repair that environment. CUDA driver support and real UNI-2
  embedding output require separate validation on the intended host.
- The runtime probe is intentionally tiny. It does not establish whole-slide
  peak memory, throughput, hierarchy stability, segmentation quality, or
  predicted-marker accuracy. Those remain separate validation tasks.
