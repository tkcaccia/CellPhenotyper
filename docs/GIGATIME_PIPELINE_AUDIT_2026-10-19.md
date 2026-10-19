# GigaTIME pipeline audit, 19 October 2026

## Audit question and result

CellPhenotyper now applies the released GigaTIME CNN with its pinned checkpoint
(`a6556030bdb8071857559e12b921620aa1bbac5b`, SHA-256
`d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c`),
RGB/ImageNet input normalization, 256-pixel patches, calibrated 0.25 µm/pixel
sampling and channel 0 as DAPI. Its current raised-cosine overlap-add uses the
actual positive weight sum at every pixel; the earlier border-darkening error
is fixed. The persisted native marker store, rather than a JPEG preview, feeds
the independent StarDist-versus-DAPI QC. In the default `fail` mode, that QC
must pass before downstream marker use.

One fixed held-out patch from the GigaTIME authors' paired H&E/mIF release was
run through **the current CellPhenotyper inference and QC code** on Chiamaka.
The 556-pixel, 128 µm registered field was resampled from 0.230216 to
0.25 µm/pixel, yielding 512 × 512 model pixels. The current stored virtual
DAPI reached full-image AUC 0.861 against released nuclei and 0.901 against the
released thresholded DAPI mask. The nearby-nucleus QC passed: AUC 0.732,
prevalence-matched Dice 0.641, nuclear boundary contrast 0.073 and
edge-to-core nuclear signal ratio 1.006. Full-image AUC and nearby-nucleus AUC
have different negative-pixel definitions and must not be compared directly.
The fixed patch is `0_24464_556_556`; the first five sorted patches were used
for earlier exploratory calibration and this patch was selected before
running the production code. One passing patch does not establish a cohort
pass rate, independent-patient accuracy, or biological validity of other
virtual channels.

The previously calibrated MoNuSAC prostate ROI
`TCGA-G9-6356-01Z-00-DX1_6` failed StarDist nuclear QC after the
overlap-denominator correction: nearby-nucleus AUC 0.668 and boundary contrast
0.016 (70 detected nuclei). That ROI used one of the red/blue-swapped companion
TIFFs identified below. Macenko and tissue-Reinhard experiments on that wrong
input also failed (AUC 0.625 and 0.586); they cannot assess whether stain
normalization would help the corrected scanner-RGB image. The public
[paired control and failure comparison](https://tkcaccia.github.io/Tambua-Analysis/#dapi-qc)
shows those historical images and reports. No stain transfer was enabled in
production.

### Source scale, color format and compression audit

We compared all **120** MoNuSAC staged H&E inputs with both their source file
and the scanner SVS image. All scanner headers record 40× acquisition and
0.19448–0.2527 µm/pixel; staged OME metadata matches those values exactly.
The pipeline's strict physical resampling then targets 0.25 µm/pixel. All
scanner SVS images use JPEG2000 compression, which is part of the released
source and cannot be reversed. The staged OME-TIFFs use lossless Deflate;
there is no new lossy compression in GigaTIME input preparation.

The decisive format error was **channel order**. For 116 ROIs, the companion
TIFF is tagged RGB but is an exact red/blue swap of the canonical image
returned from its matching scanner SVS. The original MoNuSAC preparer used
those TIFFs as RGB and passed their wrong color order to GigaTIME. The four
SVS-only ROIs were decoded correctly. Every staged image exactly matched the
file selected by that preparer, which is why a simple staging/source equality
check missed the error. The corrected preparer now uses the scanner SVS for
all cases, checks any companion TIFF against it, records the discovered
channel order and refuses a difference other than an exact swap. This is a
technical correction to the input, not a trained color adjustment.

For `TCGA-2Z-A9JG-01Z-00-DX1_4`, the scanner records 40× and
0.2524 µm/pixel. Its source companion TIFF is 8-bit RGBA with an opaque
alpha channel; its red and blue channels are exactly reversed relative to the
scanner SVS. The previous staged RGB image therefore reproduced the wrong
order. A CPU replay of the pinned model on that input, with corrected overlap
weighting, gave only **0.508** nearby expert-nucleus AUC, with all nine local
sectors below the provisional 0.60 threshold. The earlier dark outer border
was no longer evident, but the image still failed because nuclear signal was
misplaced.

A second **CPU-only** replay used canonical scanner RGB with the same pinned
model, 0.25 µm/pixel grid, 20 patch positions and expert mask. Native DAPI
nearby-nucleus AUC improved from 0.508 to **0.828**. Every one of the nine
local AUCs rose above 0.785, and the edge/core nuclear mean ratio changed
from 2.059 to **1.012**. Visual comparison shows nuclei better represented
and no evident dark outer shadow. This supports the RGB swap as a major
cause of the observed failure. An exploratory evaluation from the saved 8-bit
display PNG reports boundary contrast 0.038, below the provisional 0.05 gate;
it is not a native-store production QC pass. No other virtual channel or
cell-type accuracy was revalidated by this one-field DAPI replay.

The [all-image channel-order audit](https://tkcaccia.github.io/Tambua-Analysis/assets/dapi-2z-a9jg/input_format_120_audit.json)
and diagnostic images are published with Tambua-Analysis. The prior 120 marker
runs lack persisted native DAPI, so their JPEG preview QC is preliminary; 117
of 120 previews missed the provisional global AUC threshold and 117 had at
least one failing local sector. All historical marker measurements and the
cell classifier trained from them are invalidated by this input error. Both
training and test marker tables must be regenerated from the corrected images
and pass native nuclear/shadow QC before cell-type accuracy is reported.

The overlap-denominator correction removes one *numerical* darkening mechanism;
it does not guarantee shadow-free or biologically correct predictions. The
overlap-corrected but RGB-swapped prostate ROI still had visible spatial
shading. Its edge-to-core ratio of 1.148 is a global statistic over detected
nuclei and cannot exclude a local dark region. The 120 saved MoNuSAC marker previews, and the earlier ORION
gallery, were generated before the correction and still display the old border
artifact. They have not been regenerated or accepted by the nuclear QC gate.
On the released paired patch, a bounded CPU test of 128-pixel reflected image
context worsened outer-24-pixel DAPI-reference AUC from 0.855 to 0.838; it was
rejected. The [edge comparison](https://tkcaccia.github.io/Tambua-Analysis/assets/dapi-shadow-audit/report.json)
records the exact method and scores. This test does not determine the cause of
the previously observed MoNuSAC failure.

## Automatic inference settings and spatial shadow gate

The default `gigatime_auto_parameters: true` profile applies only to the pinned
official CNN revision. It fixes the model's 256-pixel patch, 128-pixel stride,
0.25 µm/pixel grid and strict physical resampling, enables adaptive memory and
batch sizing, runs blockwise, and does not skip tissue blocks. Calibrated source
MPP sets the resampling factor. The report records requested and effective
scientific settings plus the hardware adaptation. Output channels and storage
compression remain user-facing choices; native DAPI must be persisted for QC.

The post-inference CPU gate uses the independent H&E StarDist mask on the same
physical field. Its new 3 × 3 local AUC screen detects sectors with weak
DAPI-versus-nearby-background contrast that the whole-image edge/core ratio can
miss. A region needs at least 500 sampled nuclear and 500 nearby-background
pixels; eligible regions require AUC ≥ 0.60. The threshold was chosen during
this exploratory diagnostic and needs validation on more registered paired
images before it can be regarded as a calibrated error rate. Missing local
coverage is reported as insufficient. The QC emits a 3 × 3 map alongside the
StarDist/DAPI overlay. A local AUC failure detects lost nuclear contrast; it
cannot by itself distinguish a shading artifact from an incorrect StarDist mask
or tissue domain shift. The existing global alignment, boundary
contrast and edge ratio gates remain. The Nextflow QC process fingerprints its
source so a resumed run cannot silently reuse an older QC decision. Pipeline
execution reports now list the nuclear/shadow QC JSON, StarDist overlay and
spatial QC map as first-class quality evidence.

Read-only reanalysis of saved native outputs gave `spatial_shadow_status=pass`
for the released paired control and `fail` for the MoNuSAC ROI; its upper-right
and middle-right regional AUCs were 0.502 and 0.562. The latter had passed the
old global edge ratio of 1.148. No new GigaTIME inference was required for
this comparison, and these two cases are a functional control rather than an
independent validation cohort. The MoNuSAC result is confounded by its
red/blue-swapped input and is superseded by the channel-order audit above.

## Pipeline handoff checked

1. Input-resolution validation requires physical pixel metadata and checks
   source/converted agreement, plausible cell-imaging range, RGB layout and
   X/Y anisotropy before analysis. The shared crop carries a calibrated
   `source_mpp` in its `shift.json`.
2. GigaTIME strict-target mode uses the source MPP to select the 0.25 µm/pixel
   inference grid. It errors when calibration is missing. The original image,
   resampled shape, effective MPP, model revision, checkpoint hash and output
   channel names are recorded.
3. The model stores uncalibrated virtual-marker scores. It does not store
   measured protein concentrations or calibrated probabilities. Quantification
   uses the same model-grid pixels as the segmentation masks, mapping masks
   from the source grid with nearest-neighbor sampling.
4. Native-store nuclear QC compares virtual DAPI with independently generated
   H&E StarDist nuclei. It checks nearby-nucleus AUC, prevalence-matched Dice,
   nuclear boundary contrast and edge/core signal. The default `fail` mode
   stops downstream marker analysis. Already-written integrated quantification
   from a failing run remains an unvalidated diagnostic artifact; it must not
   be interpreted as an accepted result.
5. The seam check and nuclear check answer different questions. A smooth image
   cannot substitute for nuclear or measured-marker agreement. Neither check
   proves immune-marker biology; independent registered mIF validation remains
   necessary under `VIRTUAL_MARKER_VALIDATION_PROTOCOL.md`.

## Audit fixes

- Shared-crop, direct StarDist and GigaTIME resolution readers now use the
  same unit-aware TIFF/OME parser as input validation. An OME pixel size of
  `250 nm` is interpreted as `0.25 µm`, and source X/Y anisotropy over 5% is
  rejected instead of collapsed into a misleading scalar MPP.
- GigaTIME rejects non-8-bit input before inference. Its former per-block
  percentile conversion of 16-bit images could assign different stain colors
  to identical intensities in different WSI blocks. Such sources require one
  explicit, documented whole-image conversion before model inference.
- The Nextflow code fingerprints include the shared resolution parser, so
  changing its behavior invalidates relevant cached crop, StarDist and
  GigaTIME processes. GigaTIME now rejects a configuration with no persisted
  native store or no DAPI channel before launching inference.
- Standalone integrated cell quantification now requires an independent H&E
  nuclei mask and a persisted DAPI channel. The direct runner marks its
  quantification summary `technical_qc_passed=false` until native DAPI passes
  nuclear/shadow QC; a failure raises an error and prohibits downstream cell
  typing. JPEG-compressed OME-TIFF cannot serve as this QC store. Zarr uses
  lossless Zstd, while the public JPEG images are previews only.

These are input-safety fixes, not evidence that virtual DAPI became accurate
on MoNuSAC. The old 85.2% Tambua cell-typing result used pre-fix virtual
markers and remains **historical and unvalidated** until corrected inference
passes technical QC and measured-marker validation across the relevant cohort.

## Verification and limits

The focused CPU-only Chiamaka suite passed 29 tests, covering input calibration,
16-bit rejection, overlap-add, native-store QC and quantification. The final
broad isolated run passed 259 tests with three skips after excluding unrelated
TITAN and Bora test modules; including them in this lightweight audit
environment had produced three dependency failures from missing `transformers`
and an unstaged `bora.wsi`. A CPU preflight of the released paired H&E with the
final input reader recovered 0.230216 µm/pixel and selected the same 512 × 512,
0.25 µm/pixel model grid. `nextflow config -flat` parsed the configuration. The paired and
MoNuSAC GPU cases had bounded execution and verified release of owned
processes and assigned GPU contexts. No UCT HPC GPU work was run.

The paired control used a released thresholded fluorescence DAPI reference;
the MoNuSAC check has **no measured DAPI** and uses H&E-derived StarDist
nuclei instead. Poor StarDist masks can change its score. Thresholds for the
nuclear gate are provisional and must be calibrated on additional paired
reference fields before interpreting a pass as general reliability.
