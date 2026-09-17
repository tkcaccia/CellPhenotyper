# Optional foundation encoders, with KODAMA retained

Status: opt-in implementation completed and model-free pipeline graph validated,
2026-09-07. No alternative checkpoint has been downloaded or executed by this
work. Real learned-feature, runtime and biological validation remain pending.

## Scientific contract

Stefano has explicitly retained KODAMA as fundamental to tissue segmentation.
An alternative foundation model changes the upstream image representation,
not the tissue-discovery method. Compare each encoder through the same KODAMA
workflow, with `KODAMA.matrix ncomp=50` recorded separately from PCA rank.
Keep the saved two-domain example and original outputs immutable. A change of
encoder must create new, model-named features and downstream results.

Cell/neighbourhood niche labels remain separate from tissue-domain labels.
The optional within-parent hierarchy now uses source-bound native KODAMA graphs
for local/context/combined features, with graph-specific assignment evidence.
It remains disabled by default pending real learned-feature and histological
validation. The old k-means route is an explicitly labelled standalone legacy
comparison, not a production fallback. See [the hierarchy contract](TISSUE_HIERARCHY.md).

## Candidate order and evidence limits

The current [PathoROB repository](https://github.com/bifold-pathomics/PathoROB)
reports average robustness indices of 0.757 for UNI2-h, 0.815 for H0-mini,
0.852 for CONCHv1.5 and 0.861 for Virchow2. Higher is better; these are not
tissue-segmentation accuracies or speed measurements. Its newer Atlas 2 and
GenBio-PathFM entries are attributed to external publications and explicitly
not validated by the repository authors.

| Candidate | Intended first comparison | Important qualification |
|---|---|---|
| H0-mini | Lower-cost encoder alternative | Its 85.7M-parameter model motivates a speed/memory test, not a measured pipeline speedup. |
| CONCHv1.5 | Tissue grouping and cross-slide retrieval | Prioritise the patch encoder, not a slide-level TITAN vector. Its preprocessing and pooling are not interchangeable with UNI2. |
| Virchow2 | Strong robustness comparator | A high robustness index does not establish the best unsupervised tissue partition. |
| UNI2-h | Preserved baseline | Compare the actual existing pooling separately from any paper-reproduction pooling. |

The [PathoROB preprint](https://arxiv.org/html/2507.17845), particularly its
clustering analysis, distinguishes local representation robustness from global
clustering behaviour: CONCH/CONCHv1.5/Atlas performed strongly, whereas
Virchow2's clustering did not simply follow its robustness-index rank. Those
experiments use their own clustering protocol, not CellPhenotyper's KODAMA
pipeline. Their results motivate candidates; they do not select our winner.

The [H0-mini model card](https://huggingface.co/bioptimus/H0-mini) specifies
224-pixel input in its example, 768-dimensional CLS features and optional
1536-dimensional CLS-plus-mean-patch features. It recommends excluding all
prefix/register tokens when taking the patch mean. No speed ratio against
this pipeline's UNI2 implementation has been established.

## Access and checkpoint handling

The public pages currently require account approval/terms acceptance for
[H0-mini](https://huggingface.co/bioptimus/H0-mini),
[CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) and
[Virchow2](https://huggingface.co/paige-ai/Virchow2). Their model pages describe
non-commercial academic-research conditions and institutional-email access
requirements. A user must review/accept the conditions; the pipeline must not
accept them automatically. Existing UNI2 approval does not grant these models.

Use a read-capable Hugging Face credential belonging to an approved account;
a new token is not necessarily needed if an existing token has the required
repository permissions. Never paste credentials into a conversation, commit
them, print them in logs or bake them into a container. Do not redistribute
restricted weights with the pipeline. Acquisition is separate from inference.
The [official TITAN repository](https://github.com/mahmoodlab/TITAN) also
documents a CONCHv1.5 loading route through TITAN; any additional repository
approval depends on the selected, audited adapter and is not assumed granted.

## Implemented encoder adapters

`uni2_encoder` now accepts the fixed registry `uni2-h`, `virchow`, `virchow2`
and `phikon-v2`. Every entry declares the repository, loader backend, pooling,
224-pixel input, number of prefix/register tokens and whether token-subset
inner features are legal. Unknown models, incompatible input sizes and
non-square spatial-token grids fail closed. Virchow and Virchow2 use the
published CLS-plus-mean-patch representation; Virchow2 removes four register
tokens before the patch mean or central token subset. The existing calibrated
90-pixel inner square remains configurable through `uni2_inner_square_fixed_px`
and is derived in the same forward pass. Model-specific initial batch ceilings
(UNI2-h 256, Virchow 16, Virchow2 16, Phikon-v2 64) prevent the generic GPU
auto-tuner from beginning at an implausibly large batch; OOM halving is retained.

This deliberately retains compatibility output directories named
`09_embeddings`; every completion receipt records the selected encoder adapter,
repository revision, preprocessing and loaded-state identity. A model change
must use a fresh result directory. The optional hierarchy loader remains
UNI2-h-specific and is not silently reused for other architectures.

## Implemented PathSegmentor branch

PathSegmentor is not treated as an unsupervised clusterer or a cell-instance
model. The CUDA-only adapter requires a pinned local checkout, configuration,
checkpoint and fixed prompt panel. It tiles the GrandQC-supported analysis crop
at the paper's 1024-pixel input geometry and target 0.25 µm/px, blends overlaps,
and stores a coarse multichannel probability OME-TIFF plus per-tile text
similarities, checkpoint hash, repository revision and all physical settings.

After KODAMA is complete, the branch exports prompt-score means/maxima for every
primary observation, a descriptive per-cluster summary, and—in grid-primary
runs—a separate table for canonical cells. These scores are supervised evidence,
not discovered cell types and not cluster ground truth. An optional sensitivity
refinement learns semantic prototypes from eroded interiors of the already fixed
KODAMA regions. It may alter only a physically declared boundary band, preserves
the cluster count, obeys GrandQC as a hard support mask, emits a change mask and
provenance, and never replaces the default MedSAM result. Filling initially
unassigned pixels is separately opt-in and restricted to GrandQC tissue.

The fixed breast prompt panel contains broad tumor/stroma tissue prompts and
epithelial/neoplastic/lymphocyte/fibroblast nuclear prompts using the paper's
template style. It is intentionally small to limit correlated exploratory
queries. Prompt variants may be compared only as declared sensitivity analyses.

## Ground-truth comparison plan

1. Freeze explicit model/config/processor/code/weights identities, input
   transforms, pooling, batch ceilings and PathSegmentor prompts before exposing
   held-out ground truth. Inference uses pre-approved local checkpoints; there is
   no silent network, architecture, pooling or device fallback.
2. Preserve each observation ID, source-image hash, physical crop and feature
   family separately. Do not concatenate different models under a UNI2 label or
   map new vectors into an old UNI2 reference atlas. References require matching
   encoder and preprocessing definitions; otherwise remain unmatched/incompatible.
3. First run one encoder at a time on identical source observations and fields,
   retaining model-specific input resampling. Record source field width and
   effective model MPP. A separate recommended-scale experiment may change the
   physical field, but must not be called a matched-geometry encoder comparison.
4. Hold KODAMA classifier, `ncomp=50`, clustering policy, seeds and tissue support
   fixed. Fit each model's dimensional reduction on the declared development
   population. Keep tissue K=2 as this example's sensitivity setting, not a
   general claim that exactly two biological domains were discovered.
5. Measure encoder-only and full downstream wall time, throughput, accelerator
   peak memory, CPU memory and output storage under fixed hardware/precision
   conditions. Distinguish cold I/O, warm cache and model loading. Parameter
   count, internal seed stability and attractive overlays cannot establish
   either accuracy or a speedup.
6. Evaluate held-out tissue overlap and boundaries, abstention coverage,
   sparse/pale tissue and site/scanner strata. Use patients as independent
   statistical units. The corrected GeoJSON is development evidence only,
   never an inference input or independent test set. Predicted GigaTIME scores
   are not independent molecular validation of another H&E-derived encoder.
7. Compare raw KODAMA reconstruction, the current MedSAM output and the separate
   PathSegmentor boundary result using the same tissue support. Report tissue
   coverage, label-invariant overlap, boundary F-score/Hausdorff distance,
   false exclusion of annotated tissue and pixels changed from the raw mask.
8. Only after separate-model results, test explicitly weighted multi-encoder
   feature blocks as an optional ensemble. Report the extra inference cost;
   an ensemble is not a free accuracy improvement.

No default encoder, KODAMA parameter, cached feature, reference or previous
result was changed based on the external leaderboard. Model-free unit tests and
a two-specimen full Nextflow stub DAG pass; they verify contracts and
orchestration only. Real checkpoint inference on Chiamaka, end-to-end numerical
acceptance, speed measurement and independent accuracy remain open.
