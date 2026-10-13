# Tool and model access

CellPhenotyper fails closed when a requested executable, checkpoint, model approval,
or private package is unavailable. It does not replace a requested model with a
different model, switch from consensus to StarDist, or download an unreviewed
checkpoint. Access must therefore be prepared for the selected route before a
production or offline HPC run.

## Access modes

CellPhenotyper uses five explicit access modes:

1. **Bundled in the published runtime**: code and checkpoints are already inside
   the versioned Docker/Singularity image. No task-time credential is required.
2. **Public upstream artifact**: a custom image build downloads a public release,
   Zenodo record, Google Drive artifact, or public package. Review its terms before
   use; public download does not imply unrestricted checkpoint redistribution.
3. **Gated Hugging Face model**: every user must create their own Hugging Face
   account, accept that model's terms, receive access, and use a read token.
4. **Run-supplied local checkpoint**: the pipeline never acquires the asset. The
   user supplies a pinned source checkout, configuration, checkpoint, and license
   evidence through explicit parameters.
5. **Private package/model archive**: access must be obtained from its provider.
   CellPhenotyper records the installed files but cannot grant or redistribute it.

## Complete access matrix

| Component | When required | Access route | What CellPhenotyper does |
|---|---|---|---|
| CellPhenotyper source, Nextflow and Java | Every run | Public GitHub source plus public Nextflow/Java distributions | The user installs the host workflow runtime or uses an institutional installation. |
| Docker/Singularity runtime | Every containerized run | Published GHCR image, locally built image, or an explicitly supplied local SIF | Public images normally pull anonymously. If a registry policy requires authentication, use a GitHub token with `read:packages` only. |
| Bio-Formats, OME conversion, OpenSlide and pyvips | Input conversion | Bundled in the published runtime | No model account or token is required. |
| GrandQC tissue and artifact models | Every full route | Bundled in published runtimes; custom builds use the official public Zenodo records linked in `resources/model_registry.json` | Runtime downloading is disabled by default (`grandqc_download_models=false`). Missing bundled files fail rather than selecting a heuristic replacement. |
| StarDist `2D_versatile_he` | Default cell segmentation | Bundled/cached in published runtimes, or public `stardist-models` v0.1 download | First-run cache installation is supported; checkpoint licensing must still be reviewed separately from the BSD software license. |
| HoVer-Net fast MoNuSAC | Optional `cell_detection_mode=consensus` | Bundled in the full GPU runtime; custom builds acquire the upstream public Google Drive checkpoint | The route requires CUDA and the exact readable local checkpoint. It is not downloaded by a pipeline task. |
| CellViT++ HIPT/PanNuke | Optional `cell_detection_mode=consensus` | Bundled cache in the full GPU runtime; custom builds use CellViT++'s checkpoint/cache utility and upstream checkpoint terms | The route requires CUDA. Some upstream checkpoints cannot be redistributed because of component-specific licenses. |
| KODAMA, fastEmbedR, fastPLS and clustering dependencies | Tissue-domain analysis | Pinned public source packages built into the runtime | No model token is required. The exact package revisions are fixed in the runtime definition. |
| Bora | Default final refinement/vectorization | Pinned source bundled in the runtime, or `bora_source_dir` pointing to the exact checkout | No token is required. The expected Git revision is checked. Absence of explicit upstream license terms remains a release-readiness blocker; see `MODEL_PROVENANCE.md`. |
| MedSAM ViT-B | Explicit legacy sensitivity route only | Bundled in the runtime; custom builds use the linked public Zenodo checkpoint | The production default is Bora. Checkpoint terms remain distinct from the Apache-2.0 software license. |
| UNI2-h | `foundation_encoder=uni2-h` and optional hierarchy snapshot preparation | **Gated Hugging Face**: request access at [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), accept the terms, then use a read token | The pipeline uses `hf_token_env_file` and `hf_token_env_var_name` (default `HF_UNI2`). Each user must obtain individual approval; weights must not be committed or redistributed. |
| Virchow | `foundation_encoder=virchow` | **Gated Hugging Face**: accept/request access at [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow), then use a read token | Uses the same encoder-neutral foundation API and cache as UNI2-h. Access to UNI2-h does not grant Virchow access. |
| Virchow2 | `foundation_encoder=virchow2` | **Gated Hugging Face**: accept/request access at [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2), then use a read token | Uses the same token parameter and cache. The pipeline records the resolved snapshot and excludes Virchow2's four register tokens from spatial pooling. |
| Phikon-v2 | `foundation_encoder=phikon-v2` | Public Hugging Face repository [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2); no approval gate was present in the reviewed model page | A token is normally unnecessary for public download, but the Owkin non-commercial license must be accepted by the user and recorded for the run. |
| GigaTIME | `gigatime_enable=true` | **Gated Hugging Face**: accept access at [prov-gigatime/GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME), then use a read token | Uses `gigatime_hf_token_env_var_name` (default `HF_GIGATIME`), falling back to `HF_TOKEN`/the foundation token. The upstream license-file/model-card conflict remains a release blocker. |
| PathSegmentor | `pathsegmentor_enable=true` | Public [source repository](https://github.com/zhi-xuan-chen/PathSegmentor) plus the upstream Google Drive checkpoint, or a checkpoint trained by the user | CellPhenotyper never downloads it. Supply `pathsegmentor_repo`, `pathsegmentor_config`, and `pathsegmentor_checkpoint`; all must be pinned and visible inside the container. |
| TITAN/CONCH | `titan_enable=true` or `pathofmpred_enable=true` | **Gated Hugging Face**: request access at [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN), accept the terms, then use a read token | Online runs use `HF_TOKEN`; offline runs use `titan_cache_dir` or an absolute local snapshot in `titan_model`. TITAN requires CUDA. |
| PathoFMPred endpoint models | `pathofmpred_enable=true` | Private installed R package/model archive supplied by its provider | CellPhenotyper cannot download or grant access. Set `pathofmpred_library_dir`, declare the intended cancer code, and retain source/license evidence. |
| Tissue hierarchy encoder | `tissue_hierarchy_enable=true` | Explicit immutable local UNI2 snapshot | No automatic download is attempted. Supply `tissue_hierarchy_model_snapshot` and, when needed, `tissue_hierarchy_weights_filename`. |

## Hugging Face setup

Approval is repository-specific. Being logged in, or having access to UNI2-h,
does not grant access to Virchow, Virchow2, GigaTIME, or TITAN. MahmoodLab and
Paige gated forms require an institutional primary email and complete affiliation
and research-use information. Wait until the model page explicitly says access
has been granted before staging the weights.

Create a read token at <https://huggingface.co/settings/tokens>. Store it in the
git-ignored `tokens.env`, never in YAML, a command history, a manuscript, a
container definition, or GitHub:

```bash
umask 077
printf 'HF_UNI2=%s\nHF_GIGATIME=%s\nHF_TOKEN=%s\n' \
  '<read-token>' '<read-token>' '<read-token>' > tokens.env
chmod 600 tokens.env
```

The same read token may be referenced by all three variables only when that
account has separately been approved for every requested gated repository. Do
not share the token or a gated model cache with an unapproved user.

Foundation-model parameters:

```yaml
hf_token_env_file: tokens.env
hf_token_env_var_name: HF_UNI2
foundation_encoder: virchow2  # uni2-h | virchow | virchow2 | phikon-v2
foundation_inner_strategy: inner28_stride56
```

GigaTIME uses `HF_GIGATIME` by default. TITAN uses `HF_TOKEN` when downloading
online. For Docker/Singularity, the token file and caches must be located on a
path visible inside the task container.

## Online staging and offline HPC execution

Download only approved models on a networked login/staging host. The compute
node can then run with network access disabled:

```bash
source tokens.env
export HF_HOME=/shared/cellphenotyper-models/hf
export HF_HUB_CACHE="$HF_HOME/hub"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = "paige-ai/Virchow2"  # replace with the encoder approved for this run
snapshot_download(repo, token=os.environ["HF_UNI2"], cache_dir=os.environ["HF_HUB_CACHE"])
PY
```

Then set:

```yaml
hf_home: /shared/cellphenotyper-models/hf
hf_hub_cache: /shared/cellphenotyper-models/hf/hub
hf_hub_offline: true
```

The cache must be writable during staging and readable from every compute node.
The pipeline resolves the immutable snapshot and hashes the loaded checkpoint.
It fails if offline mode is requested but the selected model is incomplete.

TITAN uses its separate `titan_cache_dir` and `titan_offline` controls. A local
absolute `titan_model` snapshot is also supported and mounted read-only by the
container profile.

## Container access

Published GHCR images should be preferred because they contain the compatible
software stack and public/bundled checkpoints. If anonymous GHCR pulls are
blocked by site policy, create a GitHub personal access token scoped to
`read:packages` and use it only for the pull:

```bash
export SINGULARITY_DOCKER_USERNAME='<github-user>'
export SINGULARITY_DOCKER_PASSWORD='<read-packages-token>'
```

Maintainer credentials with package-write permission are not required to run the
pipeline. A local image build is an alternative, but it downloads public assets
listed in the Docker/Singularity definitions and must reproduce their pinned
versions and checksums.

## Access diagnostics

- `401 Unauthorized`: token missing, invalid, expired, or not visible inside the container.
- `403`/`GatedRepoError`: the account has not accepted the selected repository's terms or approval is still pending.
- `LocalEntryNotFoundError` with offline mode: the selected snapshot is absent or incomplete in the configured cache.
- Permission denied under `HF_HUB_CACHE/.locks`: stage into a user-writable cache; do not make a shared token world-readable.
- Missing PathSegmentor/HoVer-Net/CellViT/MedSAM checkpoint: supply the exact reviewed local asset or use the matching published runtime; do not substitute another checkpoint.
- Missing PathoFMPred library: obtain the private package from its provider; there is no public or automatic fallback.

Every completed run writes `00_execution/model_inventory.{json,tsv}`. Review it
before publication: successful computation alone does not prove that checkpoint
redistribution, commercial use, or a release-ready scientific claim is permitted.
