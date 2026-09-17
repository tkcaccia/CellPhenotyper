#!/usr/bin/env python3
"""Create revised manuscript copies describing optional encoder experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def insert_before(anchor: Paragraph, text: str, style: str) -> Paragraph:
    element = OxmlElement("w:p")
    anchor._p.addprevious(element)
    paragraph = Paragraph(element, anchor._parent)
    paragraph.style = style
    paragraph.add_run(text)
    return paragraph


def find_anchor(document: Document, text: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Could not find insertion anchor: {text}")


def replace_paragraph_starting(document: Document, prefix: str, text: str) -> None:
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting with {prefix!r}, found {len(matches)}")
    matches[0].text = text


def revise_main(source: Path, destination: Path) -> None:
    document = Document(source)
    replace_paragraph_starting(
        document,
        "Refined class masks were vectorized",
        "Refined categorical masks were vectorized from the native-resolution label raster with four-connected rasterio polygonization while retaining interior rings and every positive-area annotated component. Valid polygon parts kept their original class. Shared edges with unequal collinear noding were planarized once and rebuilt as labelled atomic faces. The configured simplification tolerance was treated as an upper search bound: exact categorical symmetric difference against the unsimplified native coverage was measured at each candidate, and the strongest shared-boundary simplification satisfying the declared annotation-error fraction was selected. One dissolved Polygon or MultiPolygon feature was then emitted per positive label; per-class hole filling and buffer smoothing were disabled. Pairwise positive-area intersections were tested before publication, and the stage failed closed if vector geometries were not mutually exclusive. Connected components were ranked by exact point-in-polygon counts of consensus cells carrying a normalized CellViT++ neoplastic, tumor or tumour name. Ties were resolved by total assigned cells, polygon area and stable section identifier. The selected component was exported with ROI and local coordinates, a mask, a 256-µm padded pyramidal OME-TIFF crop, preview, count table and coordinate transform.",
    )
    anchor = find_anchor(document, "Data availability")
    insert_before(anchor, "Pre MedSAM boundary competition", "Heading 2")
    insert_before(
        anchor,
        "Before MedSAM inference, adjacent KODAMA tissue labels compete within a bounded internal-boundary error band. The optimizer uses deterministic mean-field annealing of a multiclass Potts objective comprising a robustly scaled Lab and optical-density appearance term and an edge-weighted four-neighbour smoothness term. At each cooling iteration, a pixel can retain its current label or receive only a label present on its local four-connected frontier; this prevents non-local label teleportation. The default schedule decreases temperature geometrically from 2.0 to 0.05 over 16 iterations at fourfold working downsampling and a 64-native-pixel boundary radius. The lowest hard-energy state observed is retained, so the declared objective cannot increase. Confident eroded label cores, foreground/background membership and GrandQC clean-tissue support remain invariant. MedSAM receives this optimized categorical baseline together with the original protected seeds. The operation is unsupervised and does not use the expert evaluation annotation.",
        "Normal",
    )
    insert_before(anchor, "Pyramidal ROI crop", "Heading 2")
    insert_before(
        anchor,
        "The shared full-resolution ROI crop is written automatically as a tiled, losslessly compressed pyramidal BigTIFF. The native RGB level remains pixel exact, and successive twofold SubIFD overview levels are generated until the image fits within one 512-pixel tile. The writer preserves the verified physical calibration, validates the native dimensions, tiling, level count, level geometry and resolution tags before publication, and records the resulting contract in crop_summary.json. Both the primary streaming pyvips path and the bounded-memory TIFF fallback implement the same pyramid contract.",
        "Normal",
    )
    insert_before(anchor, "Topology preserving cluster vectors", "Heading 2")
    insert_before(
        anchor,
        "The final multiclass GeoJSON is derived from the native-resolution MedSAM-refined categorical raster using four-connected rasterio polygon topology. Interior rings and every positive-area annotated component are retained. Shared boundaries with unequal collinear vertices are noded once and polygonized into atomic faces that retain their original raster class. Shared-boundary Visvalingam-Whyatt simplification is optimized up to a configured maximum tolerance subject to an exact categorical symmetric-difference limit against the unsimplified coverage; a zero limit requests exact geometry. Per-class hole filling and buffering remain disabled. After vectorization, every pair of cluster geometries is tested for positive-area intersection; the stage fails without publishing a GeoJSON if mutually exclusive raster classes become overlapping vectors. In the K=3 breast-specimen audit, the stronger default 0.5% error ceiling selected 7.734375 native pixels and reduced 7,762,218 segments to 483,708 (93.77%) while categorical change was 0.49694%, external tissue-boundary change was 0.02372%, and pairwise overlap was zero. The native categorical raster remains the authority for exact pixel counts, while the GeoJSON validation receipt records the source dimensions, selected pyramid level, scaling, connectivity, repair diagnostics, requested and selected tolerance, segment reduction, measured fidelity and zero-overlap result.",
        "Normal",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)


def revise_supplement(source: Path, destination: Path) -> None:
    document = Document(source)
    anchor = find_anchor(document, "Supplementary reporting checklist")
    sections = [
        (
            "Supplementary Note 12 | Automatic pyramidal ROI crop",
            "PREPARE_ANALYSIS_CROP writes crop_roi.tif as a tiled pyramidal BigTIFF without changing the downstream filename or native pixel grid. The level-0 RGB pixels use lossless DEFLATE compression in 512 × 512-pixel tiles. Twofold SubIFD overviews continue until the longest image side fits within one tile. The primary path performs streaming crop and pyramid generation with pyvips. If the source TIFF cannot be decoded by that path, a bounded window reader fills a disk-backed RGB buffer and generates each overview through a bounded-memory twofold box average. Before publication, the pipeline requires the expected native dimensions, tiled level 0, exact twofold level geometry, complete level count and physical-resolution tags consistent with the verified source MPP. The validation receipt in crop_summary.json records the level shapes, tiling, compression, SubIFD status and recovered level-0 MPP. A production-container probe confirmed a pixel-exact native level and a discoverable SubIFD hierarchy; this engineering check does not evaluate image-analysis accuracy.",
        ),
        (
            "Supplementary Note 13 | Topology preserving cluster GeoJSON",
            "MASK_TO_GEOJSON converts the final MedSAM-refined categorical mask through four-connected rasterio polygonization at native resolution. Invalid point-touching rings are split into valid positive-area parts without changing their class. When shared boundaries contain unequal collinear vertices, combined boundary linework is noded once and polygonized into atomic faces; each face is assigned to the unique covering source class, while background faces are discarded. Every positive-area annotated component is retained. Visvalingam-Whyatt simplification is applied to the complete polygon coverage, so shared edges are simplified once rather than independently for each class. The configured tolerance is an upper bound: binary search selects the strongest candidate whose exact categorical symmetric-difference area, counting both positive-label swaps and tissue/background changes once, does not exceed the declared fraction of native foreground. A zero fraction requests exact geometry. Interior rings remain present; per-class hole filling and buffer smoothing remain disabled. The vectorizer calculates every pairwise positive-area intersection after geometry construction and fails closed when any overlap exceeds numerical tolerance. Its sibling provenance JSON records native source shape and hashes, label-wise pixel counts, selected pyramid level, scale, connectivity, repair diagnostics, backend, requested and selected tolerance, coordinate and segment reduction, measured fidelity, feature count and mutual exclusivity. Uncertainty and refinement-provenance summaries are included when their source sidecars are available. In the unfiltered K=3 audit, the exact coverage contained 7,762,218 segments. The stronger default 0.5% categorical-change ceiling selected a 7.734375-native-pixel tolerance, retained every positive-area component, reduced the geometry to 483,708 segments (93.77%), changed 0.49694% of categorical foreground and 0.02372% of the tissue/background boundary, and produced three mutually exclusive features with zero overlap. This is raster-to-vector engineering validation, not biological cluster-accuracy evidence.",
        ),
        (
            "Supplementary Note 14 | Annealed multiclass competition before MedSAM",
            "The pre-MedSAM competition step treats the grown KODAMA mask as a categorical partition rather than refining each class independently. An internal boundary is detected where two positive labels meet, dilated to the declared error radius and intersected with the clustering-uncertainty editable mask. Robust Lab and optical-density features are standardized by the tissue median and interquartile range. Label prototypes are estimated from fixed label interiors outside the editable band. The local energy contains squared prototype distance and a four-neighbour Potts disagreement penalty whose coupling decreases across strong image-feature gradients. Mean-field label probabilities are updated with a Boltzmann temperature that decreases geometrically from 2.0 to 0.05 over 16 iterations. Candidate labels are restricted to the current pixel label and labels on its four-connected frontier. Protected cores are clamped at every iteration, zero-labelled pixels are not created or filled, and all pixels outside GrandQC clean tissue remain zero. The lowest hard-energy state encountered is supplied to MedSAM as its multiclass baseline; the original confident seeds remain the prompt source. The default fourfold working scale bounds WSI memory, and overlapping tile interiors are committed once. The summary records exact committed changes, schedule and energy settings; the streaming preview is marked diagnostic because it is reconstructed at display scale. This algorithm is deterministic engineering functionality. Its boundary radius, appearance weight, smoothness weight, edge decay and temperature schedule require prespecified held-out evaluation before accuracy claims.",
        ),
    ]
    for heading, body in sections:
        insert_before(anchor, heading, "Heading 1")
        insert_before(anchor, body, "Normal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-source", type=Path, required=True)
    parser.add_argument("--supplement-source", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    revise_main(
        args.main_source,
        args.outdir / "CellPhenotyper_Nature_Methods_foundation_models_20260908.docx",
    )
    revise_supplement(
        args.supplement_source,
        args.outdir / "CellPhenotyper_Supplementary_Methods_foundation_models_20260908.docx",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
