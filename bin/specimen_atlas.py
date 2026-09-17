#!/usr/bin/env python3
"""Build a review-first, specimen-level atlas from published pipeline outputs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote


ATLAS_LAYERS = [
    {
        "id": "context",
        "title": "H&E Context",
        "purpose": "Orient all derived outputs to the source morphology.",
        "semantic": "source image context; not a prediction",
        "observation_unit": "whole slide or selected crop",
        "tokens": ("grandqc_tissue_thumbnail", "input_thumbnail"),
        "limit": 2,
    },
    {
        "id": "support",
        "title": "Tissue, Artifact, ROI And TMA Support",
        "purpose": "Review what was included, excluded, or assigned before biological interpretation.",
        "semantic": "technical QC or deterministic support; not a reference standard",
        "observation_unit": "image region",
        "tokens": (
            "grandqc_artifact_overlay", "grandqc_overlay", "grandqc_qc_mask",
            "tissue_mask_preview", "input_roi_mask_preview", "tma_spots_preview",
        ),
        "limit": 6,
    },
    {
        "id": "cells",
        "title": "Cell Detection And Instance Fusion",
        "purpose": "Compare detector output and the role-aware canonical instance set.",
        "semantic": "predicted nuclear instances; detector agreement is not accuracy",
        "observation_unit": "nuclear instance",
        "tokens": (
            "consensus_preview", "stardist_out/overlay", "stardist_out/boundaries",
            "hovernet", "cellvit", "labels_cyto_preview",
        ),
        "limit": 6,
    },
    {
        "id": "virtual_markers",
        "title": "Virtual Markers",
        "purpose": "Review predicted marker fields and their technical score diagnostics.",
        "semantic": "uncalibrated virtual-marker score; not measured protein abundance",
        "observation_unit": "image pixel and segmentation compartment",
        "tokens": ("/jpg_image/", "gigatime_marker_score_qc", "gigatime_seam_qc"),
        "limit": 8,
    },
    {
        "id": "uncertainty",
        "title": "Uncertainty And Abstention",
        "purpose": "Expose ambiguous or seed-unstable assignments instead of hiding them as background.",
        "semantic": "descriptive stability and abstention; not calibrated predictive uncertainty",
        "observation_unit": "grid core or contextual cell representation",
        "tokens": (
            "cluster_uncertainty", "kodama_uncertainty", "spatial_uncertainty", "abstention",
        ),
        "limit": 6,
    },
    {
        "id": "domains",
        "title": "Representations And Domains",
        "purpose": "Relate sampling geometry, latent representation, and unsupervised spatial domains.",
        "semantic": "derived representation or unsupervised assignment; not a biological label",
        "observation_unit": "grid core or contextual cell representation",
        "tokens": (
            "uni2_grid_preview", "kodama_membership", "cluster_kodama",
            "cluster_tissue_overlay", "cluster_mask_preview", "route_comparison",
        ),
        "limit": 8,
    },
    {
        "id": "refinement",
        "title": "Boundary Refinement",
        "purpose": "Inspect the constrained MedSAM/watershed edit rather than only the final mask.",
        "semantic": "derived boundary refinement; not expert ground truth",
        "observation_unit": "tissue-domain boundary",
        "tokens": (
            "medsam_raw_vs_final_panel", "medsam_random_fullres_qc",
            "medsam_grandqc_empty_exclusion", "medsam_tissue_support",
            "medsam_boundary_compare", "medsam_streaming_change_map",
        ),
        "limit": 8,
    },
    {
        "id": "research_endpoints",
        "title": "Optional Research Endpoints",
        "purpose": "Keep section selection and outcome-model estimates visibly downstream and optional.",
        "semantic": "research prediction; not diagnosis, prognosis, or calibrated clinical probability",
        "observation_unit": "selected tissue section or patient-level study record",
        "tokens": ("selected_section_preview", "pathofmpred_"),
        "limit": 5,
    },
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MARKER_ORDER = {name: index for index, name in enumerate(("DAPI", "PD-1", "CD3", "CD8", "PD-L1"))}
SHARED_PREPROCESSING_STAGES = {"input", "grandqc", "roi", "tissue_mask"}
CELL_CHARACTERIZATION_STAGES = {
    "stardist", "hovernet_monusac", "cellvitpp", "cell_consensus", "tma",
    "gigatime", "cell_assignment", "cytoplasm",
}


def _read_json(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: str | Path) -> list[dict]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _record_parts(record: dict) -> tuple[str, ...]:
    return PurePosixPath(str(record.get("relative_path") or "")).parts


def discover_sample_ids(project_records: list[dict]) -> list[str]:
    sample_ids: set[str] = set()
    for record in project_records:
        if record.get("stage_id") != "input":
            continue
        parts = _record_parts(record)
        if len(parts) > 1:
            sample_ids.add(parts[0].removesuffix("__cells"))
    if not sample_ids:
        for record in project_records:
            if record.get("stage_id") in ("execution", "cohort_niches"):
                continue
            parts = _record_parts(record)
            if len(parts) > 1:
                sample_ids.add(parts[0].removesuffix("__cells"))
    return sorted(sample_ids)


def record_sample_and_route(record: dict, sample_ids: list[str], default_route: str) -> tuple[str | None, str]:
    parts = _record_parts(record)
    if not parts:
        return None, default_route
    top = parts[0]
    for sample_id in sorted(sample_ids, key=len, reverse=True):
        if top == sample_id:
            stage_id = str(record.get("stage_id") or "")
            if stage_id in SHARED_PREPROCESSING_STAGES:
                return sample_id, "shared preprocessing"
            if stage_id in CELL_CHARACTERIZATION_STAGES:
                return sample_id, "cell identification and marker phenotyping"
            return sample_id, default_route
        if top == f"{sample_id}__cells":
            return sample_id, "cell-centred"
    return None, default_route


def _signal_sample_id(signal_path: str, sample_ids: list[str], execution_dir: Path) -> str | None:
    """Scope published evidence conservatively; ambiguous cache paths remain run-level."""
    try:
        parts = Path(signal_path).relative_to(execution_dir.parent).parts
    except (ValueError, OSError):
        return None
    if len(parts) < 2:
        return None
    top = parts[1]
    for sample_id in sorted(sample_ids, key=len, reverse=True):
        if top in {sample_id, f"{sample_id}__cells"}:
            return sample_id
    return None


def _layer_for_record(record: dict) -> dict | None:
    name = f"/{str(record.get('relative_path') or '').lower()}"
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    for layer in ATLAS_LAYERS:
        if any(token in name for token in layer["tokens"]):
            return layer
    return None


def _asset_priority(record: dict, layer_id: str) -> tuple:
    name = str(record.get("relative_path") or "")
    basename = Path(name).stem
    if layer_id == "virtual_markers" and "/jpg_image/" in f"/{name.lower()}":
        return (0, MARKER_ORDER.get(basename, 100), basename.lower())
    token_order = {
        token: index
        for index, token in enumerate(next(layer["tokens"] for layer in ATLAS_LAYERS if layer["id"] == layer_id))
    }
    rank = min((index for token, index in token_order.items() if token in name.lower()), default=999)
    return (rank, name.lower())


def _relative_href(path: str | Path, execution_dir: Path) -> str:
    try:
        relative = Path(os.path.relpath(Path(path), execution_dir)).as_posix()
    except (OSError, ValueError):
        relative = str(path)
    return quote(relative, safe="/._-")


def _metric(label: str, value, interpretation: str = "") -> dict | None:
    if value is None or value == "":
        return None
    return {"label": label, "value": value, "interpretation": interpretation}


def _first_record(records: list[dict], token: str) -> dict | None:
    return next((record for record in records if token in str(record.get("relative_path") or "").lower()), None)


def discover_cell_inspector(records: list[dict], execution_dir: Path, sample_id: str,
                            *, cohort_niches=None, cohort_records=None) -> dict:
    """Resolve only indexed, provenance-matched artifacts; never start a server.

    The launch description is embedded in specimen_atlas.json. Its paths are
    relative to that JSON file, so copying the complete results tree preserves
    the launch contract, including standard published relative symlinks.
    """
    unavailable = {"status": "unavailable", "read_only": True, "auto_launch": False}
    manifests = [record for record in records
                 if Path(str(record.get("absolute_path", ""))).name == "cell_profiles_manifest.json"
                 and "hierarchy_source" not in Path(str(record.get("relative_path", ""))).parts]
    manifests = [record for record in manifests
                 if _read_json(record["absolute_path"]).get("sample_id") == sample_id]
    if len(manifests) == 2 and len({record['absolute_path'] for record in manifests}) == 2:
        # A linked derivative and its immutable base are intentional separate
        # outputs. Prefer only an exact, hash-declared additive descendant.
        candidates = []
        for index, candidate in enumerate(manifests):
            other = manifests[1 - index]
            payload = _read_json(candidate['absolute_path'])
            source_sha = payload.get('cell_hierarchy', {}).get('source_profile_manifest_sha256')
            try:
                actual_sha = hashlib.sha256(Path(other['absolute_path']).read_bytes()).hexdigest()
            except OSError:
                continue
            if source_sha and source_sha == actual_sha and payload.get('inputs') == _read_json(other['absolute_path']).get('inputs'):
                candidates.append(candidate)
        if len(candidates) == 1:
            manifests = candidates
    if len(manifests) != 1:
        return {**unavailable, "reason": "No canonical cell profiles found" if not manifests else "Multiple cell profiles found; choose explicit startup inputs"}
    manifest_record = manifests[0]
    manifest_path = Path(manifest_record["absolute_path"]).absolute()
    manifest = _read_json(manifest_path)
    source_hashes = manifest.get("inputs", {})
    if not isinstance(source_hashes, dict):
        return {**unavailable, "reason": "Profile input provenance is not an object"}
    for key in ("labels_sha256", "image_sha256"):
        if key in source_hashes and (not isinstance(source_hashes[key], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hashes[key])):
            return {**unavailable, "reason": f"Profile {key} is not a valid source hash"}
    digest_cache = {}

    def digest(record):
        try:
            path = Path(record["absolute_path"]).resolve()
            if path in digest_cache:
                return digest_cache[path]
            result = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    result.update(block)
            digest_cache[path] = result.hexdigest()
            return digest_cache[path]
        except OSError:
            return None

    def matching(name, expected):
        if not expected:
            return []
        return [record for record in records if Path(record["absolute_path"]).name == name and digest(record) == expected]

    objects = matching("objects.csv", source_hashes.get("objects_sha256"))
    shifts = matching("shift.json", source_hashes.get("shift_sha256"))
    by_path = {Path(record["absolute_path"]).absolute(): record for record in records}
    labels = {Path(record["absolute_path"]).absolute().parent / "labels.tif" for record in objects}
    labels = sorted(path for path in labels if path in by_path and path.is_file())
    if "labels_sha256" in source_hashes:
        labels = [path for path in labels if digest(by_path[path]) == source_hashes["labels_sha256"].lower()]
        if not labels:
            return {**unavailable, "reason": "Recorded canonical-label SHA256 does not match any indexed candidate"}
    pairs = []
    for record in shifts:
        shift_path = Path(record["absolute_path"]).absolute()
        # A native crop and shift are published together by the crop/cell stage.
        image_path = shift_path.parent / (sample_id + ".tif")
        if image_path in by_path and image_path.is_file():
            pairs.append((image_path, shift_path))
    if "image_sha256" in source_hashes:
        pairs = [pair for pair in pairs if digest(by_path[pair[0]]) == source_hashes["image_sha256"].lower()]
        if not pairs:
            return {**unavailable, "reason": "Recorded H&E-image SHA256 does not match any indexed candidate"}
    if not labels or not pairs:
        return {**unavailable, "reason": "Profiles exist but canonical labels/native crop could not be matched to profile provenance",
                "profile_manifest_href": _relative_href(manifest_path, execution_dir)}
    if len(labels) > 1:
        return {**unavailable, "reason": "Multiple canonical label candidates; choose explicit startup inputs"}
    image_path, shift_path = sorted(pairs)[0]
    datasets = {"profile_dir": manifest_path.parent, "image": image_path,
                "labels": labels[0], "shift": shift_path}
    resolution_hash = source_hashes.get("resolution_json_sha256")
    if resolution_hash:
        resolutions = [record for record in records if str(record["absolute_path"]).endswith(".converted_resolution.json")
                       and digest(record) == resolution_hash]
        if not resolutions:
            return {**unavailable, "reason": "The verified physical-resolution report required by this profile is missing"}
        datasets["resolution_json"] = Path(sorted(resolutions, key=lambda row: row["absolute_path"])[0]["absolute_path"])
    else:
        if not _read_json(shift_path).get("source_mpp") and not _read_json(shift_path).get("microns_per_pixel"):
            return {**unavailable, "reason": "No explicit physical calibration is available for the inspector"}
    region_inspector = {"status": "not_provided", "reason": "No matching hierarchy summary was indexed"}
    summaries = [record for record in records if Path(record["absolute_path"]).name == "hierarchy_summary.json"
                 and _read_json(record["absolute_path"]).get("sample_id") == sample_id]
    if summaries:
        region_inspector = {"status": "unavailable", "reason": "Hierarchy provenance is incomplete, ambiguous or differs from canonical profiles"}
    if len(summaries) == 1 and "resolution_json" in datasets and all(name + "_sha256" in source_hashes for name in ("image", "labels")):
        summary_path = Path(summaries[0]["absolute_path"]).absolute()
        summary = _read_json(summary_path)
        hierarchy_root = summary_path.parent
        required = ("region_mask.ome.tif", "hierarchy_status.ome.tif", "parent_domains.ome.tif",
                    "grid_subdomains.csv", "region_profiles/region_profiles_manifest.json")
        inputs, outputs = summary.get("inputs", {}), summary.get("outputs", {})
        inputs = inputs if isinstance(inputs, dict) else {}
        outputs = outputs if isinstance(outputs, dict) else {}
        def source_hash(name):
            value = inputs.get(name)
            return value.get("sha256") if isinstance(value, dict) else None
        def output_matches(name):
            record = outputs.get(name)
            expected = record.get("sha256") if isinstance(record, dict) else record
            path = hierarchy_root / name
            return bool(expected and path in by_path and digest(by_path[path]) == expected)
        sources_match = (source_hashes.get("domain_mask_sha256") and
            source_hash("image") == source_hashes.get("image_sha256") and
            source_hash("parent_mask") == source_hashes["domain_mask_sha256"] and
            source_hash("shift_json") == source_hashes.get("shift_sha256") and
            source_hash("resolution_json") == source_hashes.get("resolution_json_sha256"))
        if source_hashes.get("domain_uncertainty_sha256"):
            sources_match = sources_match and source_hash("parent_uncertainty") == source_hashes["domain_uncertainty_sha256"]
        if "parent_uncertainty" in inputs:
            required += ("parent_uncertainty.ome.tif",)
        if sources_match and all(output_matches(name) for name in required):
            region_manifest = _read_json(hierarchy_root / "region_profiles/region_profiles_manifest.json")
            files = region_manifest.get("files", {})
            files = files if isinstance(files, dict) else {}
            blocks = region_manifest.get("feature_blocks", {})
            valid_blocks = isinstance(blocks, dict) and bool(blocks) and all(
                isinstance(record, dict) and isinstance(record.get("path"), str) and record["path"] for record in blocks.values())
            expected_files = {"region_profiles.csv", "feature_rows.csv"} | ({record["path"] for record in blocks.values()} if valid_blocks else set())
            def profile_file_matches(name):
                path = Path(os.path.abspath(hierarchy_root / "region_profiles" / name))
                return (path.is_relative_to(hierarchy_root / "region_profiles") and path in by_path and
                        isinstance(files.get(name), str) and digest(by_path[path]) == files[name])
            if (valid_blocks and region_manifest.get("observation_unit") == "tissue_region" and
                region_manifest.get("hierarchy_id") == summary.get("hierarchy_id") and summary.get("hierarchy_id") and
                all(profile_file_matches(name) for name in expected_files) and
                digest(by_path[hierarchy_root / "parent_domains.ome.tif"]) == source_hashes["domain_mask_sha256"]):
                datasets["hierarchy_dir"] = hierarchy_root
                region_inspector = {"status": "available_runtime_validation_required", "region_count": region_manifest.get("region_count"),
                    "hierarchy_id": summary["hierarchy_id"], "reason": "Source and output hashes match; startup additionally verifies native raster geometry, labels, parent membership and row identities"}
    cohort_info = {"status": "not_provided", "reason": "No shared cohort niche bundle was explicitly selected"}
    indexed_cohort = [record for record in (cohort_records or records)
                      if record.get("stage_id") == "cohort_niches"
                      and record.get("stage_folder") == "25_cohort_niches"]
    if indexed_cohort and cohort_niches is None:
        cohort_info = {"status": "not_attached_current_run_unverified", "reason":
            "Stage-25 files are indexed, but the index cannot establish their producing run. Select a source-bound bundle explicitly with --cohort-niches; no cohort labels were inferred or attached."}
    if cohort_niches is not None:
        try:
            from cohort_niche_io import load_cohort_bundle, verify_cohort_sources
            bundle = Path(cohort_niches).absolute()
            if bundle.is_file():
                if bundle.name != "cohort_niches_completion.json":
                    raise ValueError("Select the cohort directory or its exact completion receipt")
                bundle = bundle.parent
            expected_root = execution_dir.absolute().parent / "25_cohort_niches" / "cohort_niches"
            if bundle != expected_root:
                raise ValueError("Portable attachment requires the explicitly selected current stage-25 cohort directory")
            required_cohort = ("cohort_niche_model.json", "cohort_niche_summary.json",
                               "cohort_niche_assignments.parquet", "cohort_niches_completion.json")
            for name in required_cohort:
                matches = [record for record in indexed_cohort
                           if Path(record.get("absolute_path", "")).absolute() == bundle / name]
                if len(matches) != 1:
                    raise ValueError("The explicit cohort bundle must have exactly one indexed entry for each required artifact")
            frame, record = load_cohort_bundle(bundle, profile_dir=manifest_path.parent, sample_id=sample_id)
            verify_cohort_sources(bundle, manifest_path.parent, record)
            datasets["cohort_niches"] = bundle
            cohort_info = {"status": "verified_explicit_source_bound", "selection": "explicit_api_input",
                "cell_count": len(frame), "cohort_niche_model_id": record["model"]["cohort_niche_model_id"],
                "reason": "Explicit bundle matches the exact selected canonical profile. Per-slide niches and frozen reference assignments remain separate; no producing run was inferred from filenames."}
        except (ValueError, OSError, KeyError, ImportError) as exc:
            cohort_info = {"status": "unavailable", "reason": "Explicit cohort attachment rejected: " + str(exc)}
    # Do not advertise portable launch metadata for sources outside this result
    # tree. Published symlinks are allowed; the lexical artifact path is portable.
    results_root = execution_dir.absolute().parent
    if any(not Path(os.path.abspath(path)).is_relative_to(results_root) for path in datasets.values()):
        return {**unavailable, "reason": "Inspector sources are outside the portable results tree"}
    command = ('python "$CELLPHENOTYPER_REPO/bin/cell_inspector.py" '
               '--atlas-manifest specimen_atlas.json --sample-id ' + shlex.quote(sample_id))
    raster_provenance = {name: {"status": "verified_sha256" if name + "_sha256" in source_hashes else "legacy_missing_sha256",
                                "sha256": source_hashes.get(name + "_sha256", "").lower() or None}
                         for name in ("image", "labels")}
    raster_identity_status = ("verified_sha256" if all(record["status"] == "verified_sha256"
        for record in raster_provenance.values()) else "legacy_partial_or_unverified")
    return {"schema_version": "1.0.0", "status": "available", "read_only": True, "auto_launch": False,
            "raster_identity_status": raster_identity_status, "raster_provenance": raster_provenance,
            "region_inspector": region_inspector,
            "cohort_niches": cohort_info,
            "cell_count": manifest.get("cell_count"),
            "datasets": {key: os.path.relpath(path, execution_dir) for key, path in datasets.items()},
            "path_base": "specimen_atlas_json_directory", "profile_manifest_href": _relative_href(manifest_path, execution_dir),
            "launch_command": command, "launch_working_directory": "00_execution",
            "launch_instructions": "Set CELLPHENOTYPER_REPO to your repository path. Run this command from 00_execution. The report never starts a server.",
            "local_url_after_manual_launch": "http://127.0.0.1:8765",
            "interpretation": "Canonical nuclear review; marker predictions, descriptive distances and expert notes remain separate."}


def collect_sample_metrics(records: list[dict]) -> dict:
    metrics: dict[str, list[dict]] = {
        "input_and_support": [],
        "cell_instances": [],
        "virtual_markers": [],
        "clustering": [],
        "refinement": [],
    }

    converted = _first_record(records, ".converted_resolution.json")
    if converted:
        payload = _read_json(converted["absolute_path"])
        for item in (
            _metric("Input QC", payload.get("status"), "technical acceptance only"),
            _metric("Dimensions", f"{payload.get('width_px')} x {payload.get('height_px')} px" if payload.get("width_px") and payload.get("height_px") else None),
            _metric("Effective MPP", payload.get("effective_mpp"), "physical sampling; upsampling adds no detail"),
            _metric("MPP source", payload.get("metadata_source")),
        ):
            if item:
                metrics["input_and_support"].append(item)

    grandqc = _first_record(records, "grandqc_summary.json")
    if grandqc:
        payload = _read_json(grandqc["absolute_path"])
        for item in (
            _metric("GrandQC clean-tissue fraction", payload.get("clean_tissue_fraction"), "model-defined analysis support"),
            _metric("GrandQC artifact fraction", payload.get("artifact_fraction"), "not expert-reference artifact burden"),
            _metric("Artifact checkpoint MPP", payload.get("artifact_mpp_model")),
            _metric("GrandQC device", payload.get("device")),
        ):
            if item:
                metrics["input_and_support"].append(item)

    consensus = _first_record(records, "consensus_summary.json")
    if consensus:
        payload = _read_json(consensus["absolute_path"])
        input_counts = payload.get("input_counts") or {}
        for detector in ("stardist", "cellvitpp", "hovernet"):
            item = _metric(f"{detector} detections", input_counts.get(detector), "predicted instances")
            if item:
                metrics["cell_instances"].append(item)
        for item in (
            _metric("Canonical instances", payload.get("consensus_count"), "role-aware geometry fusion; not true-cell count"),
            _metric("Fusion policy", payload.get("fusion_acceptance_policy") or f"minimum support {payload.get('minimum_support')}"),
        ):
            if item:
                metrics["cell_instances"].append(item)

    marker_qc = _first_record(records, "gigatime_marker_score_qc.json")
    if marker_qc:
        payload = _read_json(marker_qc["absolute_path"])
        semantics = payload.get("value_semantics") or {}
        if not isinstance(semantics, dict):
            semantics = {"value_type": semantics}
        metrics["virtual_markers"].append(
            _metric("Value semantics", semantics.get("value_type") or "uncalibrated_virtual_marker_score", "not measured protein abundance")
        )
        metrics["virtual_markers"].append(_metric("Technical score QC", payload.get("status")))
        metrics["virtual_markers"] = [item for item in metrics["virtual_markers"] if item]
        preferred = {"DAPI", "PD-1", "CD3", "CD8", "PD-L1"}
        for channel in payload.get("channels") or []:
            if channel.get("channel") not in preferred:
                continue
            p05, p50, p95 = channel.get("p05"), channel.get("p50"), channel.get("p95")
            value = f"p05={p05:.3f}, median={p50:.3f}, p95={p95:.3f}" if all(isinstance(v, (int, float)) for v in (p05, p50, p95)) else "not available"
            metrics["virtual_markers"].append(
                {"label": str(channel.get("channel")), "value": value, "interpretation": "sampled uncalibrated score distribution"}
            )

    for record in records:
        if not str(record.get("relative_path") or "").lower().endswith("_cluster_summary.csv"):
            continue
        for row in _read_csv_rows(record["absolute_path"]):
            route = "cell-centred" if "__cells" in str(record.get("relative_path")) else "configured primary route"
            variant = row.get("cluster_profile") or "cluster"
            metrics["clustering"].append(
                {
                    "label": f"{route} / {variant}",
                    "value": f"clusters={row.get('final_cluster_count', 'NA')}; accepted={row.get('accepted_observation_fraction', 'NA')}; mean ARI={row.get('stability_mean_adjusted_rand_index', 'NA')}",
                    "interpretation": row.get("claim_status") or row.get("cluster_analysis_role") or "unsupervised descriptive partition",
                }
            )

    medsam = _first_record(records, "medsam_summary.json")
    if medsam:
        payload = _read_json(medsam["absolute_path"])
        for item in (
            _metric("Refinement mode", payload.get("mode"), "constrained edit, not first-pass segmentation"),
            _metric("MedSAM device", payload.get("medsam_device")),
            _metric("Runtime", f"{payload.get('medsam_runtime_sec')} s" if payload.get("medsam_runtime_sec") is not None else None),
            _metric("Changed internal pixels", payload.get("image_guided_changed_pixels"), "requires boundary review"),
            _metric("Excluded background leakage", payload.get("final_background_clamp_pixels"), "must remain zero"),
        ):
            if item:
                metrics["refinement"].append(item)

    return metrics


def build_specimen_atlas(
    *,
    execution_dir: Path,
    run_name: str,
    success: bool,
    analysis_intent: str,
    uni2_sampling_mode: str,
    cell_detection_mode: str,
    claim_ceiling: str,
    project_records: list[dict],
    quality_signals: list[dict],
    uncertainty_register: list[dict],
    cohort_niches=None,
) -> dict:
    sample_ids = discover_sample_ids(project_records)
    default_route = {
        "cells": "cell-centred",
        "grid": "grid tissue-domain",
        "both": "grid tissue-domain primary; cell-centred auxiliary",
    }.get(str(uni2_sampling_mode), str(uni2_sampling_mode or "not recorded"))
    specimens = []
    for sample_id in sample_ids:
        sample_records = []
        for record in project_records:
            assigned_sample, route = record_sample_and_route(record, sample_ids, default_route)
            if assigned_sample == sample_id:
                sample_records.append({**record, "atlas_route": route})

        layers = []
        for layer in ATLAS_LAYERS:
            candidates = [record for record in sample_records if (_layer_for_record(record) or {}).get("id") == layer["id"]]
            candidates.sort(key=lambda record: _asset_priority(record, layer["id"]))
            assets = []
            for record in candidates[: int(layer["limit"])]:
                assets.append(
                    {
                        "output_id": record.get("output_id"),
                        "stage_id": record.get("stage_id"),
                        "stage_title": record.get("stage_title"),
                        "route": record.get("atlas_route"),
                        "name": Path(str(record.get("relative_path"))).name,
                        "href": _relative_href(record.get("absolute_path"), execution_dir),
                        "semantic": layer["semantic"],
                        "observation_unit": layer["observation_unit"],
                    }
                )
            layers.append({**{key: layer[key] for key in ("id", "title", "purpose", "semantic", "observation_unit")}, "assets": assets})

        sample_signal_rows = []
        for signal in quality_signals:
            signal_path = str(signal.get("path") or "")
            signal_sample = _signal_sample_id(signal_path, sample_ids, execution_dir)
            if signal_sample in {None, sample_id}:
                sample_signal_rows.append(
                    {
                        "kind": signal.get("kind"),
                        "status": signal.get("status"),
                        "details": signal.get("details"),
                        "evidence_href": _relative_href(signal_path, execution_dir) if signal_path else None,
                    }
                )
        specimens.append(
            {
                "sample_id": sample_id,
                "metrics": collect_sample_metrics(sample_records),
                "quality_signals": sample_signal_rows,
                "layers": layers,
                "cell_inspector": discover_cell_inspector(sample_records, execution_dir, sample_id,
                    cohort_niches=cohort_niches, cohort_records=project_records),
            }
        )

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "success": bool(success),
        "analysis_intent": analysis_intent,
        "uni2_sampling_mode": uni2_sampling_mode,
        "cell_detection_mode": cell_detection_mode or None,
        "claim_ceiling": claim_ceiling,
        "interpretation_contract": (
            "The atlas supports expert review. It does not convert technical completion, model agreement, "
            "visual plausibility, virtual-marker scores, or cluster stability into biological validation."
        ),
        "uncertainty_register": uncertainty_register,
        "specimens": specimens,
    }


def render_specimen_atlas(payload: dict) -> str:
    def esc(value) -> str:
        return html.escape(str(value), quote=True)

    specimen_nav = "".join(
        f"<a href='#{quote(str(specimen['sample_id']), safe='._-')}'>{esc(specimen['sample_id'])}</a>"
        for specimen in payload.get("specimens") or []
    ) or "<span>No specimens discovered</span>"
    uncertainty_rows = []
    for row in payload.get("uncertainty_register") or []:
        if not row.get("stage_present") and row.get("stage_id") != "execution":
            continue
        uncertainty_rows.append(
            "<tr><td>{stage}</td><td>{implementation}</td><td>{decision}</td>"
            "<td>{calibration}</td><td>{limit}</td></tr>".format(
                stage=esc(row.get("stage_title")),
                implementation=esc(row.get("uncertainty_implementation")),
                decision=esc(row.get("decision_or_abstention")),
                calibration=esc(row.get("calibration_status")),
                limit=esc(row.get("interpretation_limit")),
            )
        )
    if not uncertainty_rows:
        uncertainty_rows.append(
            "<tr><td colspan='5'>No analytical-stage uncertainty records are available.</td></tr>"
        )
    specimen_html = []
    metric_titles = {
        "input_and_support": "Input And Support",
        "cell_instances": "Cell Instances",
        "virtual_markers": "Virtual Markers",
        "clustering": "Clustering",
        "refinement": "Refinement",
    }
    for specimen in payload.get("specimens") or []:
        metric_sections = []
        for metric_id, title in metric_titles.items():
            rows = specimen.get("metrics", {}).get(metric_id) or []
            if not rows:
                continue
            cards = "".join(
                "<article class='metric'><small>{label}</small><strong>{value}</strong><p>{note}</p></article>".format(
                    label=esc(row.get("label")), value=esc(row.get("value")), note=esc(row.get("interpretation") or "")
                )
                for row in rows
            )
            metric_sections.append(f"<h3>{esc(title)}</h3><div class='metrics'>{cards}</div>")

        signals = specimen.get("quality_signals") or []
        signal_rows = "".join(
            "<tr><td>{kind}</td><td><span class='status {status_class}'>{status}</span></td>"
            "<td>{details}</td><td>{evidence}</td></tr>".format(
                kind=esc(signal.get("kind")),
                status=esc(signal.get("status")),
                status_class=(
                    "fail"
                    if any(word in str(signal.get("status", "")).lower() for word in ("fail", "error", "reject"))
                    else "pass"
                    if str(signal.get("status", "")).lower() in {"pass", "passed", "ok", "success", "completed"}
                    else "review"
                ),
                details=esc(signal.get("details")),
                evidence=(
                    f"<a href='{esc(signal['evidence_href'])}'>open</a>"
                    if signal.get("evidence_href") else "not linked"
                ),
            )
            for signal in signals
        ) or "<tr><td colspan='4'>No sample-scoped structured QC signal was found.</td></tr>"

        layer_sections = []
        for layer in specimen.get("layers") or []:
            figures = []
            for asset in layer.get("assets") or []:
                figures.append(
                    "<figure><a href='{href}'><img loading='lazy' src='{href}' alt='{alt}'></a>"
                    "<figcaption><b>{name}</b><span>{stage}</span><span>Route: {route}</span>"
                    "<span>Unit: {unit}</span><em>{semantic}</em><code>{output_id}</code></figcaption></figure>".format(
                        href=asset["href"], alt=esc(asset["name"]), name=esc(asset["name"]),
                        stage=esc(asset.get("stage_title")), route=esc(asset.get("route")),
                        unit=esc(asset.get("observation_unit")), semantic=esc(asset.get("semantic")),
                        output_id=esc(asset.get("output_id")),
                    )
                )
            if figures:
                content = f"<div class='gallery'>{''.join(figures)}</div>"
            else:
                content = "<p class='missing'>Not produced in this stage window. Absence is not a negative biological result.</p>"
            layer_sections.append(
                f"<section class='layer'><div class='layer-head'><h3>{esc(layer['title'])}</h3>"
                f"<p>{esc(layer['purpose'])}</p><small>{esc(layer['semantic'])}</small></div>{content}</section>"
            )

        inspector = specimen.get("cell_inspector", {})
        inspector_html = "<h3>Interactive Cell Review</h3>"
        if inspector.get("status") == "available":
            identity_note = ("Native H&E and canonical labels match the profile's recorded SHA256 hashes."
                             if inspector.get("raster_identity_status") == "verified_sha256"
                             else "Legacy / incomplete raster provenance: missing source hashes are not verified by ID or dimension agreement.")
            inspector_html += (
                f"<p>{esc(inspector.get('cell_count'))} canonical cells. {esc(inspector['interpretation'])}</p>"
                f"<p>{esc(identity_note)}</p>"
                f"<p>{esc(inspector['launch_instructions'])}</p><pre><code>{esc(inspector['launch_command'])}</code></pre>"
                f"<p><a href='{esc(inspector['profile_manifest_href'])}'>Cell-profile manifest</a> · "
                f"<a href='specimen_atlas.json'>Portable launch manifest</a> · "
                f"<a href='http://127.0.0.1:8765'>Local inspector after manual launch</a></p>"
                "<p class='missing'>Read-only: search or click cells, review native H&amp;E and nuclear boundaries, "
                "compare separate features and inspect flagged cells. Expert notes never alter inference.</p>"
            )
            region = inspector.get("region_inspector", {})
            inspector_html += (f"<p>Tissue-region inspection: {esc(region.get('status', 'not_provided'))}. "
                               f"{esc(region.get('reason', 'No verified hierarchy supplied'))}</p>")
            cohort = inspector.get("cohort_niches", {})
            inspector_html += (f"<p>Shared cohort niches: {esc(cohort.get('status', 'not_provided'))}. "
                               f"{esc(cohort.get('reason', 'No explicit cohort bundle supplied'))}</p>")
        else:
            inspector_html += f"<p class='missing'>{esc(inspector.get('reason', 'Inspector inputs were not discovered'))}</p>"
        anchor = quote(str(specimen["sample_id"]), safe="._-")
        specimen_html.append(
            f"<article class='specimen' id='{anchor}'><header><p class='eyebrow'>Specimen</p>"
            f"<h2>{esc(specimen['sample_id'])}</h2></header>{''.join(metric_sections)}"
            f"<h3>Quality Signals</h3><table><thead><tr><th>Signal</th><th>Status</th><th>Evidence summary</th><th>Evidence</th></tr>"
            f"</thead><tbody>{signal_rows}</tbody></table>{inspector_html}{''.join(layer_sections)}</article>"
        )

    if not specimen_html:
        specimen_html.append("<article class='specimen'><h2>No specimen outputs found</h2><p>The atlas remains empty for this stage window.</p></article>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CellPhenotyper specimen atlas: {esc(payload.get('run_name'))}</title>
<style>
:root{{--ink:#162a31;--paper:#f4efe3;--card:#fffdf8;--rust:#ae452b;--blue:#1d6373;--gold:#b58327;--line:#c9c0ae;--muted:#647278}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 85% 5%,#d5e4df 0,transparent 34%),linear-gradient(145deg,#eee5d4,#f9f6ef 60%,#e0e8e5);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}
body>header,nav,main,footer{{width:min(1480px,calc(100% - 32px));margin:auto}}body>header{{padding:54px 0 26px;border-bottom:4px solid var(--ink)}}
.eyebrow,small,th,.status,code,figcaption span{{font-family:'Avenir Next','Gill Sans',sans-serif;text-transform:uppercase;letter-spacing:.08em}}h1{{font-size:clamp(2.5rem,7vw,6.2rem);line-height:.88;margin:.15em 0}}h2{{font-size:clamp(2rem,4vw,4rem);margin:.1em 0}}h3{{margin:32px 0 12px}}p{{line-height:1.5}}
nav{{display:flex;gap:14px;flex-wrap:wrap;padding:16px 0;position:sticky;top:0;z-index:4;background:rgba(244,239,227,.94);border-bottom:1px solid var(--line)}}nav a{{color:var(--blue);font-family:'Avenir Next','Gill Sans',sans-serif;font-weight:700}}
.notice{{margin:26px 0;padding:18px 22px;background:#fff1e9;border-left:9px solid var(--rust);box-shadow:5px 5px 0 rgba(22,42,49,.1)}}.specimen{{margin:34px 0 70px;padding:30px;background:rgba(255,253,248,.78);border:1px solid var(--line);box-shadow:8px 8px 0 rgba(22,42,49,.09)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}.metric{{background:var(--card);border-top:5px solid var(--blue);padding:14px;min-height:120px}}.metric strong{{display:block;font-size:1.1rem;margin:8px 0;overflow-wrap:anywhere}}.metric p{{font-family:'Avenir Next','Gill Sans',sans-serif;color:var(--muted);font-size:.82rem;margin:0}}
table{{width:100%;border-collapse:collapse;background:var(--card);font-family:'Avenir Next','Gill Sans',sans-serif;font-size:.88rem}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--ink);color:white;font-size:.7rem}}.status{{padding:4px 7px;border-radius:999px;color:white;background:var(--gold);font-size:.65rem}}.status.fail{{background:var(--rust)}}.status.pass{{background:var(--blue)}}
.layer{{margin:42px 0}}.layer-head{{display:grid;grid-template-columns:minmax(240px,1fr) 2fr;gap:12px;border-bottom:2px solid var(--ink);align-items:end}}.layer-head h3{{margin:0}}.layer-head small{{grid-column:1/-1;color:var(--rust);padding-bottom:9px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:16px}}figure{{margin:0;background:var(--card);border:1px solid var(--line);padding:10px}}figure img{{display:block;width:100%;height:270px;object-fit:contain;background:#e8e2d6}}figcaption{{display:grid;gap:4px;padding:10px 3px 2px;font-family:'Avenir Next','Gill Sans',sans-serif;font-size:.78rem;overflow-wrap:anywhere}}figcaption b{{font-family:Georgia,'Times New Roman',serif;font-size:.95rem}}figcaption span{{font-size:.62rem;color:var(--blue)}}figcaption em{{color:var(--rust);font-weight:700}}code{{font-size:.62rem;color:var(--muted);text-transform:none}}.missing{{padding:18px;background:#eee9df;color:var(--muted);font-family:'Avenir Next','Gill Sans',sans-serif}}
a{{color:var(--blue);text-decoration-thickness:2px;text-underline-offset:3px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee9df;padding:14px}}pre code{{font:13px/1.5 ui-monospace,monospace;letter-spacing:normal}}footer{{padding:0 0 60px;color:var(--muted)}}
@media(max-width:760px){{.specimen{{padding:18px}}.layer-head{{grid-template-columns:1fr}}figure img{{height:220px}}table{{display:block;overflow-x:auto}}}}
</style></head><body>
<header><p class="eyebrow">CellPhenotyper / specimen-level review atlas</p><h1>{esc(payload.get('run_name'))}</h1>
<p>Intent: <strong>{esc(payload.get('analysis_intent'))}</strong> · UNI-2 route: <strong>{esc(payload.get('uni2_sampling_mode'))}</strong> · Cell detection: <strong>{esc(payload.get('cell_detection_mode') or 'not recorded')}</strong></p></header>
<nav><a href="index.html">Run review</a><a href="uncertainty_register.tsv">Uncertainty register</a><a href="project_outputs.tsv">All outputs</a>{specimen_nav}</nav>
<main><section class="notice"><strong>Claim ceiling: {esc(payload.get('claim_ceiling'))}</strong><p>{esc(payload.get('interpretation_contract'))}</p></section>
<section><h2>Uncertainty Coverage</h2><p>Missing estimates are explicit. Technical QC and repeated-seed stability are not calibrated biological uncertainty.</p>
<table><thead><tr><th>Stage</th><th>Implementation</th><th>Decision / abstention</th><th>Calibration</th><th>Interpretation limit</th></tr></thead><tbody>{''.join(uncertainty_rows)}</tbody></table></section>
{''.join(specimen_html)}</main>
<footer>Research-use atlas generated {esc(payload.get('generated_utc'))}. Predicted and measured quantities are never treated as interchangeable.</footer>
</body></html>"""


def write_specimen_atlas(
    *,
    execution_dir: Path,
    run_name: str,
    success: bool,
    analysis_intent: str,
    uni2_sampling_mode: str,
    cell_detection_mode: str,
    claim_ceiling: str,
    project_records: list[dict],
    quality_signals: list[dict],
    uncertainty_register: list[dict],
    cohort_niches=None,
) -> tuple[Path, Path, dict]:
    payload = build_specimen_atlas(
        execution_dir=execution_dir,
        run_name=run_name,
        success=success,
        analysis_intent=analysis_intent,
        uni2_sampling_mode=uni2_sampling_mode,
        cell_detection_mode=cell_detection_mode,
        claim_ceiling=claim_ceiling,
        project_records=project_records,
        quality_signals=quality_signals,
        uncertainty_register=uncertainty_register,
        cohort_niches=cohort_niches,
    )
    json_path = execution_dir / "specimen_atlas.json"
    html_path = execution_dir / "specimen_atlas.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(render_specimen_atlas(payload), encoding="utf-8")
    return html_path, json_path, payload
