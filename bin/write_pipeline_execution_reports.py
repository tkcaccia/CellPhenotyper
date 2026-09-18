#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from specimen_atlas import write_specimen_atlas
from build_model_inventory import write_inventory as write_model_inventory


STAGE_DEFS = [
    {"id": "input", "folder": "01_input", "title": "Input Conversion", "expected": [".ome.tif", ".source_resolution.json", ".converted_resolution.json"]},
    {"id": "grandqc", "folder": "02_grandqc", "title": "GrandQC Artifact QC", "expected": ["_grandqc_summary.json", "_grandqc_artifact_mask.tif", "_grandqc_clean_tissue_mask.tif"]},
    {"id": "stardist", "folder": "03_stardist", "title": "StarDist Segmentation", "expected": ["labels.tif", "objects.csv"]},
    {"id": "hovernet_monusac", "folder": "03b_hovernet_monusac", "title": "HoVer-Net MoNuSAC", "expected": ["hovernet_cells.json.gz"]},
    {"id": "cellvitpp", "folder": "03c_cellvitpp", "title": "CellViT++", "expected": ["cellvit_cells.json"]},
    {"id": "cell_consensus", "folder": "03d_cell_consensus", "title": "Role-Aware Multi-Detector Instance Fusion", "expected": ["labels.tif", "objects.csv", "alignment.csv", "detector_agreement_benchmark.csv", "detector_agreement_benchmark.json", "consensus_cells.geojson", "consensus_summary.json", "consensus_preview.png"]},
    {"id": "tma", "folder": "04_TMA", "title": "TMA Detection and Cell-to-Spot Assignment", "expected": ["_tma_summary.json", "_tma_spots.geojson", "_objects_tma_assigned.csv"]},
    {"id": "tissue_mask", "folder": "04_tissue_mask", "title": "GrandQC Clean-Tissue Mask", "expected": ["_tissue_mask.tif", "_grandqc_crop_mask_summary.json"]},
    {"id": "gigatime", "folder": "05_gigatime", "title": "GigaTIME Virtual mIF + Marker Quantification", "expected": ["gigatime_probs.ome.tif", "gigatime_probs.zarr", "gigatime_seam_qc.json", "gigatime_seam_qc.png", "gigatime_marker_score_qc.json", "gigatime_marker_score_qc.tsv", "gigatime_marker_score_qc.png", "_gigatime_quantification.csv", "_gigatime_mean_intensity.csv", "_gigatime_intensity_stats.csv", "_gigatime_intensity_summary.json"]},
    {"id": "roi", "folder": "06_roi", "title": "ROI GeoJSON and Input ROI Mask", "expected": [".roi.geojson", ".roi_qc.json", "_input_roi_mask.tif", "_input_roi_mask_preview.png", "_input_roi_mask_labels.json"]},
    {"id": "cell_assignment", "folder": "07_cell_assignments", "title": "Cell Assignment", "expected": ["_objects_assigned.csv"]},
    {"id": "cytoplasm", "folder": "08_cytoplasm", "title": "Cytoplasm Expansion", "expected": ["_labels_cyto.tif"]},
    {"id": "grid_tiles", "folder": "09_grid_tiles", "title": "UNI-2 Spatial Grid", "expected": ["_uni2_grid_objects.csv", "_uni2_grid_metadata.json", "_uni2_grid_preview.png"]},
    {"id": "embeddings", "folder": "09_embeddings", "title": "UNI-2 Embeddings", "expected": ["embeddings_"]},
    {"id": "kodama", "folder": "10_kodama", "title": "UNI-2 and GigaTIME KODAMA", "expected": ["kodama_output", "gigatime_kodama_output", "_uni2_route_comparison.csv", "_uni2_route_marker_endpoints.csv", ".Rout"]},
    {"id": "clustering", "folder": "11_clustering", "title": "Clustering and Interpretation Evidence", "expected": ["_cluster.csv", "_cluster_summary.csv", "_cluster_stability.csv", "_cluster_kodama_uncertainty.png", "cluster_spatial_coherence.csv", "cluster_marker_enrichment.csv", "cluster_spatial_uncertainty.png", "cluster_abstentions.geojson", "blinded_review_form.csv", "cluster_interpretation_summary.json", ".Rout"]},
    {"id": "cluster_mask", "folder": "12_cluster_mask", "title": "Cluster and Abstention Masks", "expected": ["_cluster_mask.tif", "_cluster_uncertainty_mask.tif", "_cluster_uncertainty_preview.png", "_cluster_mask_summary.json"]},
    {"id": "grown_tissue", "folder": "13_grown_tissue", "title": "Cell-Mask Tissue Growth (cell route only)", "expected": ["_grown_mask.ome.tif"]},
    {"id": "medsam_refine_tissue", "folder": "14_medsam_refine_tissue", "title": "MedSAM + Full-Resolution Boundary Refinement", "expected": ["_grown_mask_refined.ome.tif", "_grown_mask_refined_uncertainty.tif", "_grown_mask_refined_provenance.tif", ".ome.tif.provenance.json", "_medsam_editable_band.png", "_medsam_raw_vs_final_panel.png", "_medsam_tissue_support.png", "_medsam_grandqc_empty_exclusion.png", "_medsam_kodama_membership.png"]},
    {"id": "cluster_geojson", "folder": "15_cluster_geojson", "title": "Cluster GeoJSON", "expected": [".geojson", ".geojson.provenance.json"]},
    {"id": "neoplastic_section", "folder": "16_neoplastic_section", "title": "Neoplastic-Enriched Tissue Section", "expected": ["selected_section.ome.tif", "section_neoplastic_counts.csv", "selected_section_summary.json", "selected_section_preview.png"]},
    {"id": "titan", "folder": "17_titan", "title": "TITAN Section Representation", "expected": ["titan_embedding.csv", "titan_patch_features.h5", "titan_metadata.json"]},
    {"id": "pathofmpred", "folder": "18_pathofmpred", "title": "PathoFMPred Research Predictions", "expected": ["pathofmpred_predictions.csv", "pathofmpred_research_report.html", "pathofmpred_continuous_radar.png", "pathofmpred_binary_predictions.png"]},
    {"id": "cell_profiles", "folder": "19_cell_profiles", "title": "Canonical Cell Profiles and Spatial Niches", "expected": ["cell_profiles_manifest.json", "cell_profiles.parquet", "cell_profiles.csv", "feature_rows.csv", "neighborhood_summary.json", "cell_morphology.csv"]},
    {"id": "spatialdata", "folder": "20_spatialdata", "title": "Linked SpatialData Export", "expected": ["zarr.json"]},
    {"id": "reference_mapping", "folder": "21_reference_mapping", "title": "Versioned Reference Assignment", "expected": ["reference_assignments.csv", "reference_assignments.atlas.json", "reference_assignments.mapping.json"]},
    {"id": "tissue_hierarchy", "folder": "22_tissue_hierarchy", "title": "Multiscale Tissue Hierarchy and Region Profiles", "expected": ["hierarchy_summary.json", "hierarchy_features_summary.json", "region_profiles_manifest.json", "region_mask.ome.tif", "hierarchy_status.ome.tif", "parent_uncertainty.ome.tif"]},
    {"id": "region_reference_mapping", "folder": "23_region_reference_mapping", "title": "Versioned Region Reference Assignment", "expected": ["reference_assignments.csv", "reference_assignments.atlas.json", "reference_assignments.mapping.json"]},
    {"id": "cell_tissue_links", "folder": "24_cell_tissue_links", "title": "Cell-to-Tissue Hierarchy Membership", "expected": ["cell_profiles_manifest.json", "cell_profiles.parquet", "cell_hierarchy_overlaps.csv"]},
    {"id": "cohort_niches", "folder": "25_cohort_niches", "title": "Shared Exploratory Cellular Niches", "expected": ["cohort_niche_assignments.parquet", "cohort_niche_model.json", "cohort_niche_summary.json", "cohort_niches_completion.json"]},
    {"id": "execution", "folder": "00_execution", "title": "Execution Metadata", "expected": ["index.html", "specimen_atlas.html", "specimen_atlas.json", "analysis_contract.json", "validation_readiness.json", "uncertainty_register.json", "uncertainty_register.tsv", "model_inventory.json", "model_inventory.tsv", "human_review.json", "hardware_plan.json", "storage_preflight.json", "trace.tsv", "timeline.html", "dag.html"]},
]

UNCERTAINTY_SPECS = {
    "cohort_niches": ("descriptive_pooled_stability", "shared self-plus-neighbourhood feature clusters", "isolated and unsupported cells remain unassigned", "not_a_probability", "Shared labels are exploratory, not validated biological identities; missingness and specimen-size imbalance can influence the fit."),
    "cell_tissue_links": ("propagated_categorical_overlap", "exact cell compartment overlap with parent/subdomain/region and uncertain tissue", "mixed and unresolved membership retained", "not_a_probability", "Raster area fractions describe tissue membership and inherited uncertainty, not phenotype confidence or validated cell membranes."),
    "tissue_hierarchy": ("descriptive_stability_with_abstention", "seed and independent physical-scale subdivision stability plus inherited parent uncertainty", "retain parent-only and unresolved regions", "not_a_probability", "Stable subdivisions are exploratory; parent unknown reasons remain separate and unchanged."),
    "region_reference_mapping": ("descriptive_reference_distance", "region feature compatibility and resemblance", "unmatched outside compatible reference support", "not_a_probability", "Reference resemblance is separate from tissue discovery and is not validated biological classification."),
    "input": ("technical_qc", "metadata and image-layout conflicts", "hard fail or explicit override", "not_applicable", "Technical intake QC does not establish biological validity."),
    "grandqc": ("not_quantified", "artifact/tissue model decision", "hard exclusion from downstream support", "not_calibrated", "Binary exclusion is not a calibrated probability of artifact or biological irrelevance."),
    "stardist": ("not_quantified", "single-model nuclear instance prediction", "thresholded instance output", "not_calibrated", "Per-instance segmentation uncertainty is not propagated."),
    "hovernet_monusac": ("not_quantified", "scoped nuclear instance and phenotype prediction", "excluded unless broad-detector support exists", "not_calibrated", "MoNuSAC coverage is non-exhaustive and its taxonomy is detector-specific."),
    "cellvitpp": ("not_quantified", "broad nuclear instance and phenotype prediction", "participates in broad-pair fusion", "not_calibrated", "Class labels and contours are model predictions, not reference-standard observations."),
    "cell_consensus": ("descriptive_agreement_with_abstention", "broad-detector geometry agreement", "abstain without required broad support", "not_a_probability", "Agreement measures corroboration, not instance accuracy or phenotype certainty."),
    "tma": ("not_quantified", "TMA/non-TMA and core assignment decision", "no confidence-based abstention", "not_calibrated", "Grid regularity and core-detection uncertainty are not yet quantified."),
    "tissue_mask": ("deterministic_transform", "cropped GrandQC support", "hard support mask", "not_applicable", "This inherits GrandQC errors and adds no independent uncertainty estimate."),
    "gigatime": ("technical_qc_only", "tile-seam continuity and sampled marker-score distributions", "fail or review on technical discontinuity/collapse", "biological_scores_uncalibrated", "Distribution QC does not quantify marker-prediction uncertainty or measured-protein concordance."),
    "roi": ("technical_qc", "geometry validity and image association", "hard fail or explicit legacy warning", "not_applicable", "Valid geometry does not establish that the selected region is biologically representative."),
    "cell_assignment": ("deterministic_transform", "point-in-polygon assignment", "unassigned when outside polygons", "not_applicable", "Upstream coordinate and detector errors are inherited rather than re-estimated."),
    "cytoplasm": ("not_quantified", "morphological label expansion", "no uncertainty-based abstention", "not_calibrated", "Expanded regions are approximations and are not validated cell boundaries."),
    "grid_tiles": ("technical_qc", "tissue occupancy and tile geometry", "exclude below tissue threshold", "not_calibrated", "Tissue occupancy is not uncertainty in the UNI-2 representation."),
    "embeddings": ("not_quantified", "UNI-2 latent representation", "no per-observation abstention beyond input filtering", "not_calibrated", "Embedding uncertainty and out-of-domain status are not estimated."),
    "kodama": ("descriptive_only", "latent-coordinate and paired-route diagnostics", "no KODAMA-level abstention", "not_calibrated", "Coordinate agreement cannot identify a biologically superior route."),
    "clustering": ("quantified_with_abstention", "landmark-vote margin and repeated-seed stability", "abstain on ambiguity or instability", "not_calibrated", "Stability and spatial coherence do not prove biological identity."),
    "cluster_mask": ("propagated_categorical", "observation-level clustering uncertainty", "uncertain observations map to zero with a separate reason mask", "not_calibrated", "Raster uncertainty inherits clustering thresholds and is not boundary confidence."),
    "grown_tissue": ("not_quantified", "deterministic sparse-mask growth", "constrained to GrandQC tissue support", "not_calibrated", "Growth uncertainty and biological boundary accuracy are not quantified."),
    "medsam_refine_tissue": ("propagated_categorical_with_technical_qc", "original uncertainty, inferred/modified/unresolved pixel provenance and edit metrics", "constrained to tissue support and protected cores; unresolved reasons retained", "not_calibrated", "Categorical provenance and edit metrics are not calibrated boundary uncertainty or expert agreement. Historical outputs may lack this evidence."),
    "cluster_geojson": ("propagated_categorical_native_label", "native-label counts and uncertainty/provenance linked to vector features", "unresolved label-zero counts retained; missing legacy evidence explicit", "not_a_probability", "Fractions cover all native pixels of a label, not each smoothed polygon. Vectorization adds no independent biological confidence."),
    "neoplastic_section": ("not_quantified", "deterministic ranking from predicted neoplastic labels", "fail when required cells are absent", "not_calibrated", "Upstream phenotype uncertainty is not propagated into section selection."),
    "titan": ("not_quantified", "foundation-model section embedding", "no predictive abstention", "not_calibrated", "A latent vector has no direct biological or clinical confidence interpretation."),
    "pathofmpred": ("not_quantified", "research endpoint prediction", "optional limited-evidence exclusion", "uncalibrated_score", "TCGA-derived scores are not calibrated clinical probabilities."),
    "cell_profiles": ("descriptive_only", "missing feature blocks, detector evidence, support-constrained neighbourhoods and niche stability", "missing features and unknown niches retained", "not_calibrated", "Canonical joining and spatial stability do not validate cell phenotypes or virtual marker biology."),
    "spatialdata": ("deterministic_transform", "linked image, labels, polygons, tables and physical transforms", "fail on coordinate or canonical identity mismatch", "not_applicable", "A readable spatial package does not establish biological accuracy."),
    "reference_mapping": ("descriptive_distance_with_abstention", "distance to compatible immutable reference distributions", "unknown for missing, unfamiliar or ambiguous observations", "not_calibrated", "Reference resemblance is separate from discovery and is not a calibrated cell-type probability."),
    "execution": ("evidence_gate", "study declaration, claim ceiling and human review", "warn or fail according to evidence gate", "not_applicable", "A successful execution cannot raise the claim ceiling without external evidence."),
}

UNCERTAINTY_SIGNAL_KINDS = {
    "input": {"input_image_qc"},
    "grandqc": {"grandqc_artifact_qc"},
    "cell_consensus": {"cell_instance_fusion"},
    "gigatime": {"gigatime_seam_qc", "gigatime_marker_score_qc"},
    "roi": {"roi_geometry_qc"},
    "kodama": {"uni2_route_comparison"},
    "clustering": {"clustering_stability", "forced_cluster_count", "cluster_interpretation"},
    "medsam_refine_tissue": {"medsam_refinement", "refinement_provenance"},
    "cluster_geojson": {"vectorized_refinement_provenance"},
    "execution": {"validation_readiness", "human_review", "storage_capacity", "model_provenance"},
}


def bytes_to_human(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def seconds_to_human(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_duration_seconds(raw: str) -> float:
    text = (raw or "").strip()
    if not text:
        return 0.0
    unit_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(d|h|m|s|ms)\b", text)
    if unit_matches:
        total = 0.0
        for value, unit in unit_matches:
            amount = float(value)
            if unit == "d":
                total += amount * 86400.0
            elif unit == "h":
                total += amount * 3600.0
            elif unit == "m":
                total += amount * 60.0
            elif unit == "s":
                total += amount
            elif unit == "ms":
                total += amount / 1000.0
        return total
    if text.endswith("ms"):
        try:
            return float(text[:-2]) / 1000.0
        except ValueError:
            return 0.0
    if text.endswith("s") and text.count(":") == 0:
        try:
            return float(text[:-1])
        except ValueError:
            return 0.0
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            parts_f = [float(p) for p in parts]
        except ValueError:
            return 0.0
        if len(parts_f) == 2:
            return parts_f[0] * 60.0 + parts_f[1]
        return parts_f[0] * 3600.0 + parts_f[1] * 60.0 + parts_f[2]
    return 0.0


def parse_memory_bytes(raw: str) -> int:
    text = (raw or "").strip().upper().replace("IB", "B")
    if not text:
        return 0
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def summarize_trace(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    with trace_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    idx_process = header.index("process") if "process" in header else (header.index("name") if "name" in header else -1)
    idx_realtime = header.index("realtime") if "realtime" in header else -1
    idx_peak_rss = header.index("peak_rss") if "peak_rss" in header else -1
    idx_status = header.index("status") if "status" in header else -1
    grouped: dict[str, dict] = {}
    for cols in rows[1:]:
        if idx_process < 0 or idx_process >= len(cols):
            continue
        process_name = cols[idx_process]
        bucket = grouped.setdefault(process_name, {"tasks": 0, "realtime_s": 0.0, "peak_bytes": 0, "failed": 0})
        bucket["tasks"] += 1
        if idx_realtime >= 0 and idx_realtime < len(cols):
            bucket["realtime_s"] += parse_duration_seconds(cols[idx_realtime])
        if idx_peak_rss >= 0 and idx_peak_rss < len(cols):
            bucket["peak_bytes"] = max(bucket["peak_bytes"], parse_memory_bytes(cols[idx_peak_rss]))
        if idx_status >= 0 and idx_status < len(cols) and cols[idx_status] not in {"COMPLETED", "CACHED"}:
            bucket["failed"] += 1
    return sorted(
        [
            {
                "process": name,
                **vals,
                "realtime_human": seconds_to_human(vals["realtime_s"]),
                "peak_human": bytes_to_human(vals["peak_bytes"]),
            }
            for name, vals in grouped.items()
        ],
        key=lambda x: x["realtime_s"],
        reverse=True,
    )


def list_files(stage_dir: Path) -> list[dict]:
    if not stage_dir.exists():
        return []
    files = []
    seen_dirs: set[str] = set()
    for root, dirs, names in os.walk(stage_dir, followlinks=True):
        try:
            real_root = os.path.realpath(root)
        except OSError:
            real_root = root
        if real_root in seen_dirs:
            dirs[:] = []
            continue
        seen_dirs.add(real_root)
        for name in sorted(names):
            f = Path(root) / name
            if not f.is_file():
                continue
            try:
                size = f.stat().st_size
                published = str(f.absolute())
                resolved_target = str(f.resolve())
                rel = f.relative_to(stage_dir).as_posix()
            except OSError:
                continue
            files.append(
                {
                    "relative_path": rel,
                    "absolute_path": published,
                    "resolved_target_path": resolved_target,
                    "size_bytes": size,
                }
            )
    files.sort(key=lambda item: item["relative_path"])
    return files


def collect_quality_signals(outdir: Path) -> list[dict]:
    signals: list[dict] = []
    for path in sorted((outdir / "01_input").rglob("*.converted_resolution.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        layout = payload.get("image_layout") or {}
        digest = str(payload.get("file_sha256") or "")
        signals.append(
            {
                "kind": "input_image_qc",
                "status": str(payload.get("status", "unknown")),
                "path": str(path.absolute()),
                "details": (
                    f"mpp={payload.get('effective_mpp', 'NA')}; "
                    f"shape={layout.get('shape', 'NA')}; "
                    f"dtype={layout.get('dtype', 'NA')}; "
                    f"channels={layout.get('channel_count', 'NA')}; "
                    f"pyramid_levels={layout.get('pyramid_level_count', 'NA')}; "
                    f"mpp_conflicts={len(payload.get('mpp_conflicts') or [])}; "
                    f"sha256={digest[:12] if digest else 'disabled'}"
                ),
            }
        )
    for path in sorted((outdir / "06_roi").rglob("*.roi_qc.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        digest = str(payload.get("roi_sha256") or "")
        signals.append(
            {
                "kind": "roi_geometry_qc",
                "status": str(payload.get("status", "unknown")),
                "path": str(path.absolute()),
                "details": (
                    f"source={payload.get('roi_source_kind', 'NA')}; "
                    f"valid_features={payload.get('valid_in_bounds_feature_count', 'NA')}/"
                    f"{payload.get('feature_count', 'NA')}; "
                    f"holes={payload.get('hole_count', 'NA')}; "
                    f"outside_px2={payload.get('outside_image_area_px2', 'NA')}; "
                    f"modified={payload.get('geometry_modified', 'NA')}; "
                    f"issues={len(payload.get('failures') or [])}; "
                    f"sha256={digest[:12] if digest else 'NA'}"
                ),
            }
        )
    readiness_path = outdir / "00_execution" / "validation_readiness.json"
    if readiness_path.exists():
        try:
            payload = json.loads(readiness_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if payload:
            signals.append(
                {
                    "kind": "validation_readiness",
                    "status": str(payload.get("status", "unknown")),
                    "path": str(readiness_path.absolute()),
                    "details": (
                        f"phase={payload.get('study_phase', 'NA')}; "
                        f"gate_passed={payload.get('gate_passed', False)}; "
                        f"claim_ceiling={payload.get('claim_ceiling', 'NA')}; "
                        f"issues={len(payload.get('issues') or [])}"
                    ),
                }
            )
    storage_path = outdir / "00_execution" / "storage_preflight.json"
    if storage_path.exists():
        payload = read_json_mapping(storage_path)
        filesystems = payload.get("filesystems") or []
        expected = sum(int(row.get("expected_increment_bytes") or 0) for row in filesystems)
        worst = sum(int(row.get("worst_case_increment_bytes") or 0) for row in filesystems)
        minimum_headroom = min(
            (int(row.get("expected_headroom_bytes") or 0) for row in filesystems),
            default=0,
        )
        signals.append(
            {
                "kind": "storage_capacity",
                "status": str(payload.get("status", "unknown")),
                "path": str(storage_path.absolute()),
                "details": (
                    f"expected_increment={bytes_to_human(expected)}; "
                    f"restart_worst_case={bytes_to_human(worst)}; "
                    f"minimum_expected_headroom={bytes_to_human(max(0, minimum_headroom))}; "
                    f"filesystems={len(filesystems)}; mode={payload.get('policy_mode', 'NA')}"
                ),
            }
        )
    model_inventory_path = outdir / "00_execution" / "model_inventory.json"
    if model_inventory_path.exists():
        payload = read_json_mapping(model_inventory_path)
        blockers = payload.get("release_blockers") or []
        signals.append(
            {
                "kind": "model_provenance",
                "status": str(payload.get("status", "unknown")),
                "path": str(model_inventory_path.absolute()),
                "details": (
                    f"used_models={payload.get('used_model_count', 0)}; "
                    f"release_ready={payload.get('release_ready', False)}; "
                    f"blocked_models={len(blockers)}"
                ),
            }
        )
    for path in sorted((outdir / "03d_cell_consensus").rglob("consensus_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count_qc = payload.get("detector_count_qc") or {}
        signals.append(
            {
                "kind": "cell_instance_fusion",
                "status": str(count_qc.get("status", "unknown")),
                "path": str(path.absolute()),
                "details": (
                    f"policy={payload.get('fusion_acceptance_policy', 'NA')}; "
                    f"consensus_cells={payload.get('consensus_count', 'NA')}; "
                    f"broad_ratio={count_qc.get('broad_scope_max_min_ratio', 'NA')}; "
                    f"all_detector_ratio={count_qc.get('all_detector_max_min_ratio', 'NA')}"
                ),
            }
        )
    for path in sorted((outdir / "05_gigatime").rglob("gigatime_seam_qc.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        signals.append(
            {
                "kind": "gigatime_seam_qc",
                "status": str(payload.get("status", "unknown")),
                "path": str(path.absolute()),
                "details": (
                    f"failed_channel_boundaries={payload.get('failed_channel_boundaries', 'NA')}; "
                    f"tests={payload.get('channel_boundary_tests', 'NA')}"
                ),
            }
        )
    for path in sorted((outdir / "05_gigatime").rglob("gigatime_marker_score_qc.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        channels = payload.get("channels") or []
        warnings = payload.get("warnings") or []
        semantics = payload.get("value_semantics") or {}
        if not isinstance(semantics, dict):
            semantics = {"value_type": semantics}
        signals.append(
            {
                "kind": "gigatime_marker_score_qc",
                "status": str(payload.get("status", "unknown")),
                "path": str(path.absolute()),
                "details": (
                    f"channels={len(channels)}; warnings={len(warnings)}; "
                    f"value_type={semantics.get('value_type', 'NA')}; "
                    f"calibrated_probability={semantics.get('calibrated_probability', False)}"
                ),
            }
        )
    for path in sorted((outdir / "11_clustering").rglob("*_cluster_summary.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle), None)
        except OSError:
            row = None
        if not row:
            continue
        abstained = int(float(row.get("abstained_observation_count") or 0))
        unstable = int(float(row.get("unstable_observation_count") or 0))
        forced_count = str(row.get("forced_cluster_count_requested") or "").strip().lower() in {
            "true", "1", "yes",
        }
        analysis_role = row.get("cluster_analysis_role") or (
            "sensitivity_forced_cluster_count" if forced_count else "legacy_unrecorded"
        )
        signals.append(
            {
                "kind": "clustering_stability",
                "status": "review" if abstained or unstable else "pass",
                "path": str(path.absolute()),
                "details": (
                    f"mean_ari={row.get('stability_mean_adjusted_rand_index', 'NA')}; "
                    f"min_ari={row.get('stability_min_adjusted_rand_index', 'NA')}; "
                    f"vote_margin_threshold={row.get('assignment_min_vote_margin', 'NA')}; "
                    f"stability_threshold={row.get('stability_min_fraction', 'NA')}; "
                    f"accepted_fraction={row.get('accepted_observation_fraction', 'NA')}; "
                    f"unstable={unstable}; abstained={abstained}; role={analysis_role}"
                ),
            }
        )
        if forced_count:
            signals.append(
                {
                    "kind": "forced_cluster_count",
                    "status": "sensitivity_only",
                    "path": str(path.absolute()),
                    "details": (
                        f"target={row.get('target_clusters', 'NA')}; "
                        f"applied={row.get('forced_cluster_count_applied', row.get('target_applied', 'NA'))}; "
                        "do not present this partition as the graph-derived primary result"
                    ),
                }
            )
    for path in sorted((outdir / "11_clustering").rglob("cluster_interpretation_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        spatial = payload.get("spatial_coherence") or {}
        markers = payload.get("marker_enrichment") or {}
        signals.append(
            {
                "kind": "cluster_interpretation",
                "status": str(payload.get("status", "independent_review_required")),
                "path": str(path.absolute()),
                "details": (
                    f"interpretable={payload.get('interpretable_observations', 'NA')}; "
                    f"abstained={payload.get('abstained_observations', 'NA')}; "
                    f"spatial={spatial.get('status', 'NA')}; "
                    f"spatial_excess={spatial.get('excess_over_prevalence_chance', 'NA')}; "
                    f"markers={markers.get('status', 'NA')}"
                ),
            }
        )
    for path in sorted((outdir / "10_kodama").rglob("*_uni2_route_comparison.csv")):
        signals.append(
            {
                "kind": "uni2_route_comparison",
                "status": "descriptive_only",
                "path": str(path.absolute()),
                "details": "No route is selected automatically; review study-specific biological endpoints.",
            }
        )
    refinement_records = list_files(outdir / "14_medsam_refine_tissue")
    for path in (Path(row["absolute_path"]) for row in refinement_records if row["relative_path"].endswith(".ome.tif.provenance.json")):
        payload = read_json_mapping(path)
        binding = payload.get("output_binding_status", "legacy_unbound_output_files")
        # This is a report of producer evidence, not a second raster/hash audit.
        signals.append({"kind": "refinement_provenance", "status": "review" if binding != "complete" or payload.get("stub") else "technical_provenance_recorded",
                        "path": str(path.absolute()),
                        "details": f"producer_binding={binding}; original_uncertainty_provided={payload.get('input_uncertainty_provided', 'unavailable')}; categorical reasons only; source hashes not reverified by this report"})
    vector_records = list_files(outdir / "15_cluster_geojson")
    for path in (Path(row["absolute_path"]) for row in vector_records if row["relative_path"].endswith(".geojson.provenance.json")):
        payload = read_json_mapping(path)
        binding = payload.get("producer_output_binding", "unavailable")
        valid_schema = payload.get("schema_version") == "cellphenotyper.vectorized_refinement_provenance.v1"
        signals.append({"kind": "vectorized_refinement_provenance", "status": "technical_provenance_recorded" if valid_schema and binding == "exact_output_sha256_verified" and not payload.get("stub") else "review",
                        "path": str(path.absolute()),
                        "details": f"producer_binding={binding}; source_label_zero_included={payload.get('source_label_zero_included', False)}; native-label scope, not per-polygon confidence; source hashes not reverified by this report"})
    review_path = outdir / "00_execution" / "human_review.json"
    if review_path.exists():
        payload = read_json_mapping(review_path)
        review = payload.get("reviewer") or {}
        overall = payload.get("overall") or {}
        stages = payload.get("stage_decisions") or []
        if payload:
            signals.append(
                {
                    "kind": "human_review",
                    "status": str(overall.get("decision", "pending")),
                    "path": str(review_path.absolute()),
                    "details": (
                        f"role={review.get('role', 'NA')}; "
                        f"independent={review.get('independent_of_pipeline_development', False)}; "
                        f"blinded={review.get('blinded_to_outcomes', False)}; "
                        f"stage_decisions={len(stages)}; "
                        f"claim_ceiling={overall.get('approved_claim_ceiling', 'NA')}"
                    ),
                }
            )
    return signals


def stage_summary(outdir: Path) -> list[dict]:
    rows = []
    for spec in STAGE_DEFS:
        stage_dir = outdir / spec["folder"]
        files = list_files(stage_dir)
        size_bytes = sum(f["size_bytes"] for f in files)
        key_files = []
        for needle in spec["expected"]:
            match = next((f["absolute_path"] for f in files if needle in f["relative_path"]), None)
            if match:
                key_files.append(match)
        rows.append(
            {
                "id": spec["id"],
                "folder": spec["folder"],
                "title": spec["title"],
                "present": bool(files),
                "file_count": len(files),
                "size_bytes": size_bytes,
                "size_human": bytes_to_human(size_bytes),
                "key_files": key_files,
                "files": files,
            }
        )
    return rows


def build_uncertainty_register(
    stages: list[dict], quality_signals: list[dict]
) -> list[dict]:
    """Make quantified and missing uncertainty equally visible for every stage."""
    signal_paths: dict[str, list[str]] = {}
    for signal in quality_signals:
        signal_paths.setdefault(str(signal.get("kind", "")), []).append(
            str(signal.get("path", ""))
        )

    rows = []
    for stage in stages:
        stage_id = stage["id"]
        if stage_id not in UNCERTAINTY_SPECS:
            raise ValueError(f"Missing uncertainty specification for stage: {stage_id}")
        implementation, source, decision, calibration, limitation = UNCERTAINTY_SPECS[stage_id]
        paths = []
        for kind in sorted(UNCERTAINTY_SIGNAL_KINDS.get(stage_id, set())):
            paths.extend(path for path in signal_paths.get(kind, []) if path)
        if not paths:
            paths.extend(str(path) for path in stage.get("key_files", [])[:3])
        rows.append(
            {
                "stage_id": stage_id,
                "stage_title": stage["title"],
                "stage_present": bool(stage["present"]),
                "run_status": "present" if stage["present"] else "not_run_or_not_published",
                "uncertainty_implementation": implementation,
                "uncertainty_source": source,
                "decision_or_abstention": decision,
                "calibration_status": calibration,
                "interpretation_limit": limitation,
                "evidence_paths": sorted(set(paths)),
            }
        )
    return rows


def write_uncertainty_register(
    execution_dir: Path, rows: list[dict]
) -> tuple[Path, Path]:
    json_path = execution_dir / "uncertainty_register.json"
    tsv_path = execution_dir / "uncertainty_register.tsv"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "principle": (
                    "Technical QC, model agreement, stability and calibrated predictive "
                    "uncertainty are distinct. Missing estimates are reported explicitly."
                ),
                "stages": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "stage_id", "stage_title", "stage_present", "run_status",
        "uncertainty_implementation", "uncertainty_source", "decision_or_abstention",
        "calibration_status", "interpretation_limit", "evidence_paths",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "evidence_paths": ";".join(row["evidence_paths"])})
    return json_path, tsv_path


def make_output_id(namespace: str, stage_id: str, relative_path: str) -> str:
    seed = f"{namespace}\t{stage_id}\t{relative_path}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()[:16]
    return f"{stage_id}_{digest}"


def build_project_records(outdir: Path, stages: list[dict]) -> list[dict]:
    records = []
    for stage in stages:
        for file_record in stage["files"]:
            records.append(
                {
                    "output_id": make_output_id(
                        str(outdir), stage["id"], file_record["relative_path"]
                    ),
                    "stage_id": stage["id"],
                    "stage_title": stage["title"],
                    "stage_folder": stage["folder"],
                    **file_record,
                }
            )
    return records


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def preserve_trace(
    execution_dir: Path,
    run_name: str,
    success: bool,
    start_point: str,
    end_point: str,
) -> tuple[Path, dict | None]:
    """Snapshot each run trace and retain the latest successful full-pipeline trace."""
    trace_path = execution_dir / "trace.tsv"
    runs_dir = execution_dir / "run_traces"
    runs_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = runs_dir / f"{safe_name(run_name)}_{safe_name(start_point)}_to_{safe_name(end_point)}.tsv"
    if trace_path.exists():
        shutil.copy2(trace_path, snapshot_path)

    full_trace_path = execution_dir / "full_pipeline_trace.tsv"
    full_meta_path = execution_dir / "full_pipeline_run.json"
    final_stages = {"cluster_geojson", "neoplastic_section", "titan", "pathofmpred"}
    is_full_run = success and start_point == "convert" and end_point in final_stages
    if is_full_run and trace_path.exists():
        shutil.copy2(trace_path, full_trace_path)
        full_meta_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "success": success,
                    "start_point": start_point,
                    "end_point": end_point,
                    "trace_file": str(full_trace_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    full_meta = None
    if full_trace_path.exists() and full_meta_path.exists():
        try:
            full_meta = json.loads(full_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            full_meta = None
    return (full_trace_path if full_trace_path.exists() else trace_path), full_meta


def status_class(status: str) -> str:
    value = (status or "unknown").strip().lower()
    if any(token in value for token in ("fail", "error", "invalid", "reject")):
        return "fail"
    if value in {"pass", "passed", "ok", "completed", "success", "accepted_for_declared_research_use"}:
        return "pass"
    if value in {"descriptive_only", "not_applicable"}:
        return "info"
    return "review"


def relative_href(path: str | Path, execution_dir: Path) -> str:
    target = Path(path)
    try:
        relative = Path(os.path.relpath(target, execution_dir)).as_posix()
    except (OSError, ValueError):
        relative = str(target)
    return quote(relative, safe="/._-")


def review_assets(project_records: list[dict], limit: int = 12) -> list[dict]:
    preferred = (
        "grandqc_artifact_overlay",
        "consensus_preview",
        "tma_spots_preview",
        "input_roi_mask_preview",
        "gigatime_seam_qc.png",
        "gigatime_marker_score_qc.png",
        "uni2_grid_preview",
        "cluster_uncertainty_preview",
        "kodama_uncertainty",
        "cluster_spatial_uncertainty",
        "kodama_membership",
        "cluster_interpretation",
        "medsam_raw_vs_final_panel",
        "medsam_random_fullres_qc",
        "medsam_grandqc_empty_exclusion",
        "selected_section_preview",
    )
    candidates = []
    for record in project_records:
        name = record["relative_path"].lower()
        if Path(name).suffix not in {".png", ".jpg", ".jpeg"}:
            continue
        rank = next((idx for idx, token in enumerate(preferred) if token in name), None)
        if rank is not None:
            candidates.append((rank, record["stage_folder"], name, record))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]


def read_json_mapping(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def render_landing_page(
    *,
    execution_dir: Path,
    run_name: str,
    success: bool,
    start_point: str,
    end_point: str,
    analysis_intent: str,
    uni2_sampling_mode: str,
    cell_detection_mode: str,
    stages: list[dict],
    quality_signals: list[dict],
    trace_rows: list[dict],
    project_records: list[dict],
    uncertainty_register: list[dict] | None = None,
) -> str:
    readiness = read_json_mapping(execution_dir / "validation_readiness.json")
    claim_ceiling = str(
        readiness.get("claim_ceiling")
        or "not recorded; treat outputs as engineering feasibility only"
    )
    review_signals = sorted(
        quality_signals,
        key=lambda signal: (
            {"fail": 0, "review": 1, "info": 2, "pass": 3}[status_class(signal["status"])],
            signal["kind"],
        ),
    )
    present_stages = [stage for stage in stages if stage["present"] and stage["id"] != "execution"]
    total_bytes = sum(stage["size_bytes"] for stage in present_stages)
    slowest = trace_rows[:8]
    assets = review_assets(project_records)
    uncertainty_register = uncertainty_register or []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    signal_html = []
    for signal in review_signals:
        css_class = status_class(signal["status"])
        signal_html.append(
            "<article class='signal {css}'><div><strong>{kind}</strong>"
            "<span class='badge {css}'>{status}</span></div><p>{details}</p>"
            "<a href='{href}'>Open evidence</a></article>".format(
                css=css_class,
                kind=esc(signal["kind"].replace("_", " ").title()),
                status=esc(signal["status"]),
                details=esc(signal["details"]),
                href=relative_href(signal["path"], execution_dir),
            )
        )
    if not signal_html:
        signal_html.append(
            "<article class='signal review'><strong>No structured QC signals found</strong>"
            "<p>Inspect the requested stage window and execution logs manually.</p></article>"
        )

    stage_html = []
    for stage in stages:
        if stage["id"] == "execution":
            continue
        folder_href = relative_href(execution_dir.parent / stage["folder"], execution_dir)
        state = "present" if stage["present"] else "not present"
        stage_html.append(
            f"<tr><td><a href='{folder_href}'>{esc(stage['folder'])}</a></td>"
            f"<td>{esc(stage['title'])}</td><td>{state}</td>"
            f"<td>{stage['file_count']}</td><td>{esc(stage['size_human'])}</td></tr>"
        )

    runtime_html = []
    for row in slowest:
        runtime_html.append(
            f"<tr><td>{esc(row['process'])}</td><td>{row['tasks']}</td>"
            f"<td>{esc(row['realtime_human'])}</td><td>{esc(row['peak_human'])}</td>"
            f"<td>{row['failed']}</td></tr>"
        )
    if not runtime_html:
        runtime_html.append("<tr><td colspan='5'>Trace data are not available.</td></tr>")

    asset_html = []
    for record in assets:
        href = relative_href(record["absolute_path"], execution_dir)
        asset_html.append(
            "<figure><a href='{href}'><img loading='lazy' src='{href}' alt='{alt}'></a>"
            "<figcaption><span>{stage}</span>{name}<code>{output_id}</code></figcaption></figure>".format(
                href=href,
                alt=esc(record["relative_path"]),
                stage=esc(record["stage_folder"]),
                name=esc(Path(record["relative_path"]).name),
                output_id=esc(record["output_id"]),
            )
        )
    if not asset_html:
        asset_html.append("<p>No bounded-size QC preview assets were found.</p>")

    uncertainty_html = []
    for row in uncertainty_register:
        if not row["stage_present"] and row["stage_id"] != "execution":
            continue
        uncertainty_html.append(
            "<tr><td>{stage}</td><td>{implementation}</td><td>{decision}</td>"
            "<td>{calibration}</td><td>{limit}</td></tr>".format(
                stage=esc(row["stage_title"]),
                implementation=esc(row["uncertainty_implementation"]),
                decision=esc(row["decision_or_abstention"]),
                calibration=esc(row["calibration_status"]),
                limit=esc(row["interpretation_limit"]),
            )
        )
    if not uncertainty_html:
        uncertainty_html.append(
            "<tr><td colspan='5'>No analytical-stage uncertainty records are available.</td></tr>"
        )

    overall_class = "pass" if success else "fail"
    overall_text = "Pipeline completed" if success else "Pipeline did not complete"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CellPhenotyper run review: {esc(run_name)}</title>
<style>
:root{{--ink:#132a33;--paper:#f4f0e7;--card:#fffdf7;--line:#c8c1b2;--accent:#b84a2b;--teal:#16706a;--amber:#a66a10;--muted:#617078}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:linear-gradient(145deg,#e9e2d2 0,#f8f5ed 52%,#dfe8e4 100%);font-family:Georgia,'Times New Roman',serif}}
header,main,footer{{width:min(1180px,calc(100% - 32px));margin:auto}}header{{padding:54px 0 28px;border-bottom:3px solid var(--ink)}}
.eyebrow,th,.badge,code,.metric small,figcaption span{{font-family:'Avenir Next','Gill Sans',sans-serif;text-transform:uppercase;letter-spacing:.08em}}
h1{{font-size:clamp(2.2rem,6vw,5.2rem);line-height:.92;margin:.18em 0}}h2{{margin-top:42px;font-size:1.65rem}}p{{line-height:1.55}}
.summary{{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:12px;margin:24px 0}}.metric,.signal,figure,.notice{{background:rgba(255,253,247,.9);border:1px solid var(--line);border-radius:4px;padding:16px;box-shadow:4px 4px 0 rgba(19,42,51,.09)}}
.metric strong{{display:block;font-size:1.35rem;margin-top:6px;overflow-wrap:anywhere}}.badge{{display:inline-block;font-size:.7rem;padding:5px 8px;margin-left:8px;border-radius:999px;color:white;background:var(--muted)}}
.badge.pass{{background:var(--teal)}}.badge.fail{{background:var(--accent)}}.badge.review{{background:var(--amber)}}.badge.info{{background:var(--muted)}}
.notice{{border-left:8px solid var(--accent);font-size:1.08rem}}.signals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}.signal p{{font-family:'Avenir Next','Gill Sans',sans-serif;color:#354a52}}.signal.fail{{border-top:5px solid var(--accent)}}.signal.review{{border-top:5px solid var(--amber)}}.signal.pass{{border-top:5px solid var(--teal)}}
a{{color:#0b5c65;text-decoration-thickness:2px;text-underline-offset:3px}}table{{width:100%;border-collapse:collapse;background:rgba(255,253,247,.88);font-family:'Avenir Next','Gill Sans',sans-serif;font-size:.9rem}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--ink);color:white;font-size:.72rem}}tr:hover td{{background:#fff}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}figure{{margin:0;padding:10px}}figure img{{display:block;width:100%;height:190px;object-fit:contain;background:#e8e3d9}}figcaption{{display:grid;gap:4px;padding-top:9px;font-family:'Avenir Next','Gill Sans',sans-serif;font-size:.82rem;overflow-wrap:anywhere}}figcaption span{{color:var(--accent);font-size:.68rem}}code{{font-size:.68rem;color:var(--muted);text-transform:none}}
.links{{display:flex;gap:18px;flex-wrap:wrap;font-family:'Avenir Next','Gill Sans',sans-serif}}footer{{padding:36px 0 60px;color:var(--muted)}}
@media(max-width:760px){{.summary{{grid-template-columns:1fr 1fr}}table{{display:block;overflow-x:auto}}header{{padding-top:34px}}}}
</style>
</head>
<body>
<header><div class="eyebrow">CellPhenotyper / review before interpretation</div><h1>{esc(run_name)}</h1><p>Generated {generated}. Stage window: <strong>{esc(start_point)} to {esc(end_point)}</strong>.</p></header>
<main>
<section class="summary"><div class="metric"><small>Execution</small><strong>{overall_text} <span class="badge {overall_class}">{str(success).lower()}</span></strong></div><div class="metric"><small>Intent</small><strong>{esc(analysis_intent)}</strong></div><div class="metric"><small>Observation route</small><strong>{esc(uni2_sampling_mode)}</strong></div><div class="metric"><small>Published data</small><strong>{esc(bytes_to_human(total_bytes))}</strong></div></section>
<section class="notice"><strong>Claim ceiling: {esc(claim_ceiling)}</strong><p>Technical completion is not evidence of biological accuracy, clinical validity, fairness or external generalization. Predicted markers are not measured abundance, detector agreement is not a reference standard, and unsupervised clusters require independent interpretation.</p></section>
<h2>Quality and uncertainty first</h2><section class="signals">{''.join(signal_html)}</section>
<h2>Uncertainty coverage</h2><p>Absence of an uncertainty estimate is reported explicitly and must not be read as high confidence. The complete register also lists stages not run.</p><table><thead><tr><th>Stage</th><th>Implementation</th><th>Decision / abstention</th><th>Calibration</th><th>Interpretation limit</th></tr></thead><tbody>{''.join(uncertainty_html)}</tbody></table>
<h2>Key review images</h2><section class="gallery">{''.join(asset_html)}</section>
<h2>Stage outputs</h2><table><thead><tr><th>Folder</th><th>Stage</th><th>Status</th><th>Files</th><th>Size</th></tr></thead><tbody>{''.join(stage_html)}</tbody></table>
<h2>Slowest processes</h2><table><thead><tr><th>Process</th><th>Tasks</th><th>Total runtime</th><th>Peak RSS</th><th>Failed</th></tr></thead><tbody>{''.join(runtime_html)}</tbody></table>
<h2>Complete records</h2><p class="links"><a href="specimen_atlas.html">Specimen atlas</a><a href="final_report.md">Final report</a><a href="final_report.json">Structured report</a><a href="uncertainty_register.tsv">Uncertainty register</a><a href="model_inventory.tsv">Learned-model inventory</a><a href="project_outputs.tsv">All output IDs</a><a href="project_outputs.json">Output manifest JSON</a><a href="trace.tsv">Nextflow trace</a><a href="timeline.html">Timeline</a><a href="dag.html">DAG</a></p>
</main><footer>CellPhenotyper research-use output. Cell detection: {esc(cell_detection_mode or 'not recorded')}.</footer>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--success", required=True)
    ap.add_argument("--start-point", required=True)
    ap.add_argument("--end-point", required=True)
    ap.add_argument("--image-input", default="")
    ap.add_argument("--roi-geojson", default="")
    ap.add_argument("--cell-detection-mode", default="")
    ap.add_argument("--analysis-intent", default="exploratory")
    ap.add_argument("--uni2-sampling-mode", default="cells")
    ap.add_argument("--analysis-contract", default="")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    execution_dir = (outdir / "00_execution").resolve()
    execution_dir.mkdir(parents=True, exist_ok=True)

    success = args.success.strip().lower() == "true"
    trace_source, full_run_meta = preserve_trace(
        execution_dir,
        args.run_name,
        success,
        args.start_point,
        args.end_point,
    )

    registry_path = Path(__file__).resolve().parents[1] / "resources" / "model_registry.json"
    model_inventory_json, model_inventory_tsv, model_inventory = write_model_inventory(
        outdir,
        registry_path,
        execution_dir,
    )

    stages = stage_summary(outdir)
    quality_signals = collect_quality_signals(outdir)
    uncertainty_register = build_uncertainty_register(stages, quality_signals)
    uncertainty_json_path, uncertainty_tsv_path = write_uncertainty_register(
        execution_dir, uncertainty_register
    )
    # Refresh after creating the register so it receives stable output IDs too.
    stages = stage_summary(outdir)
    uncertainty_register = build_uncertainty_register(stages, quality_signals)
    write_uncertainty_register(execution_dir, uncertainty_register)
    trace_rows = summarize_trace(trace_source)
    trace_history = []
    for run_trace in sorted((execution_dir / "run_traces").glob("*.tsv")):
        for row in summarize_trace(run_trace):
            trace_history.append({"trace": run_trace.name, **row})
    report_run_name = str(full_run_meta.get("run_name")) if full_run_meta else args.run_name
    report_start = str(full_run_meta.get("start_point")) if full_run_meta else args.start_point
    report_end = str(full_run_meta.get("end_point")) if full_run_meta else args.end_point
    report_success = bool(full_run_meta.get("success")) if full_run_meta else success

    readiness = read_json_mapping(execution_dir / "validation_readiness.json")
    claim_ceiling = str(
        readiness.get("claim_ceiling")
        or "not recorded; treat outputs as engineering feasibility only"
    )
    preliminary_records = build_project_records(outdir, stages)
    atlas_html_path, atlas_json_path, _ = write_specimen_atlas(
        execution_dir=execution_dir,
        run_name=report_run_name,
        success=report_success,
        analysis_intent=args.analysis_intent,
        uni2_sampling_mode=args.uni2_sampling_mode,
        cell_detection_mode=args.cell_detection_mode,
        claim_ceiling=claim_ceiling,
        project_records=preliminary_records,
        quality_signals=quality_signals,
        uncertainty_register=uncertainty_register,
    )
    # Refresh so the atlas itself is assigned stable output IDs and counted.
    stages = stage_summary(outdir)
    uncertainty_register = build_uncertainty_register(stages, quality_signals)
    uncertainty_json_path, uncertainty_tsv_path = write_uncertainty_register(
        execution_dir, uncertainty_register
    )

    manifest_path = execution_dir / "outputs_manifest.txt"
    manifest_lines = [
        "CellPhenotyper Output Manifest",
        f"Run name: {report_run_name}",
        f"Success: {str(report_success).lower()}",
        f"Stage window: {report_start} -> {report_end}",
        f"Timing trace: {trace_source}",
        "",
        f"Stage folders under: {outdir}",
    ]
    for s in stages:
        status = "PRESENT" if s["present"] else "MISSING"
        manifest_lines.append(f"{status}\t{outdir / s['folder']}\tfiles={s['file_count']}\tsize={s['size_human']}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    project_records = build_project_records(outdir, stages)

    project_json_path = execution_dir / "project_outputs.json"
    project_tsv_path = execution_dir / "project_outputs.tsv"
    project_payload = {
        "run_name": report_run_name,
        "success": report_success,
        "output_root": str(outdir),
        "stage_window": {"start": report_start, "end": report_end},
        "cell_detection_mode": args.cell_detection_mode or None,
        "analysis_intent": args.analysis_intent,
        "uni2_sampling_mode": args.uni2_sampling_mode,
        "analysis_contract": args.analysis_contract or None,
        "landing_page": str(execution_dir / "index.html"),
        "specimen_atlas_html": str(atlas_html_path),
        "specimen_atlas_json": str(atlas_json_path),
        "quality_signals": quality_signals,
        "uncertainty_register_json": str(uncertainty_json_path),
        "uncertainty_register_tsv": str(uncertainty_tsv_path),
        "model_inventory_json": str(model_inventory_json),
        "model_inventory_tsv": str(model_inventory_tsv),
        "model_release_ready": bool(model_inventory.get("release_ready")),
        "input_context": {
            "image_input": str(Path(args.image_input).resolve()) if args.image_input else None,
            "roi_geojson": str(Path(args.roi_geojson).resolve()) if args.roi_geojson else None,
        },
        "records": project_records,
    }
    project_json_path.write_text(json.dumps(project_payload, indent=2), encoding="utf-8")
    with project_tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "output_id", "stage_id", "stage_title", "stage_folder", "relative_path",
                "absolute_path", "resolved_target_path", "size_bytes",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(project_records)

    report_md_path = execution_dir / "final_report.md"
    report_lines = [
        "# CellPhenotyper Final Report",
        "",
        f"- Run name: `{report_run_name}`",
        f"- Success: `{str(report_success).lower()}`",
        f"- Stage window: `{report_start} -> {report_end}`",
        f"- Cell detection mode: `{args.cell_detection_mode or 'not recorded'}`",
        f"- Analysis intent: `{args.analysis_intent}`",
        f"- UNI-2 sampling mode: `{args.uni2_sampling_mode}`",
        f"- Analysis contract: `{args.analysis_contract or 'not recorded'}`",
        f"- Timing trace: `{trace_source}`",
        f"- Output root: `{outdir}`",
        f"- Project outputs JSON: `{project_json_path}`",
        f"- Project outputs TSV: `{project_tsv_path}`",
        f"- Review landing page: `{execution_dir / 'index.html'}`",
        f"- Specimen atlas: `{atlas_html_path}`",
        f"- Uncertainty register: `{uncertainty_tsv_path}`",
        f"- Learned-model inventory: `{model_inventory_tsv}`",
        f"- Learned-model release-ready: `{str(bool(model_inventory.get('release_ready'))).lower()}`",
        "",
        "## Stage Outputs",
        "",
        "| Folder | Stage | Status | Files | Size |",
        "|---|---|---:|---:|---:|",
    ]
    for s in stages:
        report_lines.append(f"| `{s['folder']}` | {s['title']} | {'PRESENT' if s['present'] else 'MISSING'} | {s['file_count']} | {s['size_human']} |")
    report_lines.extend(["", "## Key Result Files", ""])
    for s in stages:
        if s["key_files"]:
            report_lines.append(f"- `{s['folder']}`")
            for path in s["key_files"]:
                report_lines.append(f"  - `{path}`")
    report_lines.extend(["", "## Quality And Uncertainty Signals", ""])
    if quality_signals:
        report_lines.extend([
            "| Signal | Status | Details | Report |",
            "|---|---|---|---|",
        ])
        for signal in quality_signals:
            report_lines.append(
                f"| `{signal['kind']}` | `{signal['status']}` | {signal['details']} | `{signal['path']}` |"
            )
    else:
        report_lines.append("No structured quality signals were found for this stage window.")
    report_lines.extend([
        "", "## Uncertainty Coverage", "",
        "| Stage | Present | Implementation | Decision / Abstention | Calibration | Interpretation Limit |",
        "|---|---:|---|---|---|---|",
    ])
    for row in uncertainty_register:
        report_lines.append(
            f"| {row['stage_title']} | {str(row['stage_present']).lower()} | "
            f"`{row['uncertainty_implementation']}` | {row['decision_or_abstention']} | "
            f"`{row['calibration_status']}` | {row['interpretation_limit']} |"
        )
    report_lines.extend(["", "## Runtime And Memory (From trace.tsv)", ""])
    if trace_rows:
        report_lines.extend([
            "| Process | Tasks | Total Realtime | Total Realtime (s) | Peak RSS | Failed Tasks |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in trace_rows:
            report_lines.append(
                f"| `{row['process']}` | {row['tasks']} | {row['realtime_human']} | {row['realtime_s']:.1f} | {row['peak_human']} | {row['failed']} |"
            )
    else:
        report_lines.append("Trace file not available.")
    report_lines.extend(["", "## Run Timing History", ""])
    if trace_history:
        report_lines.extend([
            "| Run trace | Process | Tasks | Total Realtime | Peak RSS | Failed Tasks |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in trace_history:
            report_lines.append(
                f"| `{row['trace']}` | `{row['process']}` | {row['tasks']} | {row['realtime_human']} | {row['peak_human']} | {row['failed']} |"
            )
    else:
        report_lines.append("No preserved per-run traces are available.")
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    final_report_json_path = execution_dir / "final_report.json"
    final_report_json_path.write_text(
        json.dumps(
            {
                "run_name": report_run_name,
                "success": report_success,
                "stage_window": {"start": report_start, "end": report_end},
                "cell_detection_mode": args.cell_detection_mode or None,
                "analysis_intent": args.analysis_intent,
                "uni2_sampling_mode": args.uni2_sampling_mode,
                "analysis_contract": args.analysis_contract or None,
                "quality_signals": quality_signals,
                "uncertainty_register": uncertainty_register,
                "uncertainty_register_json": str(uncertainty_json_path),
                "uncertainty_register_tsv": str(uncertainty_tsv_path),
                "model_inventory": model_inventory,
                "model_inventory_json": str(model_inventory_json),
                "model_inventory_tsv": str(model_inventory_tsv),
                "timing_trace": str(trace_source),
                "output_root": str(outdir),
                "stages": [
                    {
                        "id": s["id"],
                        "folder": s["folder"],
                        "stage": s["title"],
                        "present": s["present"],
                        "file_count": s["file_count"],
                        "size_bytes": s["size_bytes"],
                        "key_files": s["key_files"],
                    }
                    for s in stages
                ],
                "process_trace_summary": trace_rows,
                "run_trace_history": trace_history,
                "project_outputs_json": str(project_json_path),
                "project_outputs_tsv": str(project_tsv_path),
                "landing_page": str(execution_dir / "index.html"),
                "specimen_atlas_html": str(atlas_html_path),
                "specimen_atlas_json": str(atlas_json_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    landing_page_path = execution_dir / "index.html"
    landing_page_path.write_text(
        render_landing_page(
            execution_dir=execution_dir,
            run_name=report_run_name,
            success=report_success,
            start_point=report_start,
            end_point=report_end,
            analysis_intent=args.analysis_intent,
            uni2_sampling_mode=args.uni2_sampling_mode,
            cell_detection_mode=args.cell_detection_mode,
            stages=stages,
            quality_signals=quality_signals,
            trace_rows=trace_rows,
            project_records=project_records,
            uncertainty_register=uncertainty_register,
        ),
        encoding="utf-8",
    )

    print(f"Execution reports dir: {execution_dir}")
    print(f"Output manifest: {manifest_path}")
    print(f"Project outputs JSON: {project_json_path}")
    print(f"Project outputs TSV: {project_tsv_path}")
    print(f"Final report: {report_md_path}")
    print(f"Final report JSON: {final_report_json_path}")
    print(f"Review landing page: {landing_page_path}")
    print(f"Specimen atlas HTML: {atlas_html_path}")
    print(f"Specimen atlas JSON: {atlas_json_path}")
    print(f"Uncertainty register JSON: {uncertainty_json_path}")
    print(f"Uncertainty register TSV: {uncertainty_tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
