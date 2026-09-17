# Exact cell-to-tissue hierarchy membership

When both cell profiles and tissue hierarchy are enabled, the workflow now joins
the two products before cell reference mapping and SpatialData export. This is
a deterministic measurement of existing masks, not new biological inference.
No expert annotation is used and neither parent-domain labels nor KODAMA settings
are changed.

## Membership and uncertainty

`cell_hierarchy_overlaps.{parquet,csv}` partitions every canonical nuclear pixel
by parent domain, subdomain, connected region, hierarchy status and original
parent-uncertainty code. When physical compartments are available, the disjoint
perinuclear ring is measured separately. Each row retains `cell_uid`, `cell_id`,
`sample_id`, `compartment`, region identity, integer pixel count, area in square
micrometres and fraction of the **entire** compartment. Background is included
in that denominator. Fractions sum to one for each nonempty compartment.

The canonical table gains `hierarchy_nucleus_*` and optional
`hierarchy_perinuclear_ring_*` columns: accepted/unresolved/background fractions,
parent-zero and unassigned-parent uncertainty fractions, upstream uncertainty,
crossing flags and dominant membership summaries.
Dominant membership does not erase smaller overlaps. A cell can overlap several
regions; parent-only tissue is not background. In particular, a parent label of
zero does **not** by itself establish that tissue is absent:

| Native parent assignment | Original uncertainty | Meaning in the linked profile |
| --- | --- | --- |
| Positive parent, accepted region | Code 0 | `accepted_fraction` |
| Positive parent, no accepted region | Any available code | `unresolved_fraction`; raw codes 1–254 also contribute to `parent_uncertain_fraction` |
| Parent 0 | Codes 1–254, including 253 | Unassigned tissue: `unassigned_parent_uncertain_fraction` and `unresolved_fraction`, not background |
| Parent 0 | Code 0 | `background_fraction`: **parent-map background**, not an independent measurement proving tissue absence |
| Parent 0 | Original uncertainty unavailable | Explicit unavailable-assignment status and `unassigned_parent_uncertainty_unavailable_fraction`; `background_fraction` is NaN |

`parent_zero_fraction` counts all parent-zero pixels regardless of uncertainty.
When the original uncertainty raster is available, accepted, unresolved and
background fractions partition each nonempty compartment and sum to one.
`parent_uncertain_fraction` covers only **positive-parent** pixels with literal
raw codes 1–254; unassigned-parent uncertainty is reported separately. The
fractions must not be added indiscriminately because the uncertainty fractions
are subsets of unresolved membership.

Hierarchy status 10 can cover an entire rejected grid core, including pixels
whose original uncertainty code remains 0. The link preserves those literal
codes: for example, 100% unresolved membership can coexist with only 25% raw
parent uncertainty. It does not expand upstream uncertainty to the whole core.

Empty rings retain zero pixels and NaN fractions. If original uncertainty is
unavailable, the link-table-only sentinel 255 and NaN uncertainty fractions
state that absence. Parent-zero compartments receive
`no_parent_assignment_uncertainty_unavailable` (or the partial-assignment
equivalent), never `background_only`. Sentinel 255 is never written into a
source raster or interpreted as confidence. Accepted/unresolved/background do
not claim a complete partition when part of the parent-zero assignment is
unavailable; the separate unavailable fraction records that remainder.

## Outputs and immutable provenance

The original stage-19 profiles remain unchanged. The derived bundle is at:

```
24_cell_tissue_links/<sample>/cell_profiles/
```

Every existing cell, original column value, feature block and spatial graph is
retained in the original row order. `hierarchy_source/` preserves the exact
original registry/manifest and, when supplied, ring mask and construction
summary. This allows independent checks after moving the full output bundle.
Source image, label, shift, resolution, parent and hierarchy hashes must agree.
Validation recomputes the overlaps from the actual rasters, rather than only
checking self-consistent table totals.

Cell profiles must use the same refined primary parent variant as the hierarchy.
Missing/foreign/duplicate sample joins fail. Linked profiles are published
separately to avoid a race overwriting the original profile directory.

## Existing-artifact workflow

In a `cell_profiles.nf` sample record, add `hierarchy` pointing to the hierarchy
directory containing `hierarchy_summary.json`. Supply `labels` and optional
`compartments` (the directory containing `labels_perinuclear_ring.tif`). The
record may rebuild profiles from existing artifacts or supply `existing_profiles`.
An already linked `existing_profiles` bundle is not linked again; provide its
matching `hierarchy` for export and do not replace its retained compartments.

The standalone linker accepts:

```bash
python bin/link_cell_tissue_hierarchy.py \
  --profile-dir /absolute/path/to/original/cell_profiles \
  --labels /absolute/path/to/canonical/labels.tif \
  --hierarchy-dir /absolute/path/to/hierarchy \
  --outdir /absolute/path/to/new/linked_profiles
```

## SpatialData and inspection

SpatialData includes registered parent/subdomain/region/status/uncertainty
rasters, an optional retained ring raster, actual region feature blocks and a
separate many-to-many `cell_hierarchy_overlaps` table. Region instances are not
misrepresented as cells. Measured assays remain separate. An assay bound to the
original profile registry can be retained only after verifying the exact
additive derivation and unchanged original values; it is not silently rebound.

The cell inspector displays the overlap rows and per-cell summaries. Region
inspection includes every cell with positive nuclear overlap, not only cells
whose centroids lie inside. Region marker means use whole-nucleus overlap
fractions; displayed quantiles remain explicitly unweighted. Picking a location
without an accepted region reports the literal parent, hierarchy-status and
available original uncertainty codes. Parent 0/code 253 is an unresolved
assignment; missing original uncertainty is reported as unavailable, never
confident background. Parent 0/code 0 is labelled `parent_map_background` with
the explicit caveat that this does not independently establish tissue absence.
The portable atlas
chooses the derived profile over its base only with an exact declared parent
manifest hash. Nothing opens or launches automatically.

These are native-mask relationships and engineering checks. They do not validate
cell membranes, predicted markers, tissue biology or independent cohort accuracy.
