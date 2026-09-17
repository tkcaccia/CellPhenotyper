"""Bind categorical detector labels to the same verified source as its vectors.

No detector/model imports, biological remapping, label mutation or recorded-path
following. The explicit bundle argument is the already validated CellViT input.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


UNKNOWN_LABELS = frozenset(("", "unknown", "__unknown__"))


def _unknown(value):
    return pd.isna(value) or (isinstance(value, str) and value in UNKNOWN_LABELS)


def _literal(value, field):
    if not isinstance(value, str) or not value or value.strip() != value or re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError(f"Invalid literal CellViT phenotype {field}")
    return value


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate CellViT phenotype JSON field: {key}")
            result[key] = value
        return result
    def finite(value):
        raise ValueError(f"Nonfinite CellViT phenotype JSON value: {value}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite)


def phenotype_definition(cells, feature_record=None, *, source=None):
    """Return portable taxonomy semantics plus separate specimen-source hashes.

    Incomplete legacy taxonomy metadata is explicitly unverified. Contradictions
    in an otherwise source-verified normalized population fail closed, rather
    than quietly attaching a verified label definition to unrelated phenotypes.
    """
    unverified = lambda reason: {"verified": False, "definition": {},
        "source_binding": {"status": "unverified", "reason": reason}}
    if not feature_record:
        return unverified("No source-bound CellViT feature bundle was attached")
    binding = feature_record.get("source_binding", {})
    if (feature_record.get("reference_compatible") is not True
            or binding.get("reference_compatible") is not True
            or binding.get("status") != "verified_exact_inputs_and_population"
            or feature_record.get("canonical_correspondence", {}).get("status")
                != "verified_source_ids_detector_centroids_and_population"):
        return unverified("CellViT model/source/canonical correspondence is not fully verified")
    if source is None:
        raise ValueError("Verified CellViT phenotypes require the explicit source bundle")
    source_definition = binding.get("feature_definition")
    if not isinstance(source_definition, dict) or feature_record.get("feature_definition") != source_definition:
        raise ValueError("CellViT phenotype and feature definitions disagree")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(source_definition.get("checkpoint_sha256", "")))
            or not isinstance(source_definition.get("runtime"), dict) or not source_definition["runtime"]
            or not source_definition.get("model")):
        raise ValueError("Incomplete verified CellViT phenotype model identity")
    root = Path(source).resolve()
    path = root / "cellvit_cells.json"
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping CellViT retained phenotype population")
    # Never follow feature_record.sources[*].path. Bind the exact bytes parsed.
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    sources = [entry for entry in feature_record.get("sources", []) if entry.get("role") == "retained_population"]
    receipt = binding.get("receipt", {})
    expected = receipt.get("files", {}).get("retained_population", {})
    if (len(sources) != 1 or sources[0].get("sha256") != actual
            or binding.get("retained_population_sha256") != actual
            or expected.get("sha256") != actual or expected.get("size_bytes") != len(raw)
            or expected.get("path") != "cellvit_cells.json"):
        raise ValueError("CellViT retained phenotype population SHA256/size binding mismatch")
    payload = _json(raw)
    if not isinstance(payload, dict) or not payload.get("taxonomy") or not payload.get("type_map"):
        return unverified("Retained CellViT population lacks explicit normalized taxonomy/type_map")
    taxonomy = _literal(payload["taxonomy"], "taxonomy").lower()
    if taxonomy != str(source_definition.get("taxonomy", "")).strip().lower():
        raise ValueError("Retained CellViT phenotype taxonomy differs from the verified model definition")
    raw_map = payload["type_map"]
    if not isinstance(raw_map, dict):
        raise ValueError("CellViT phenotype type_map must be an explicit dictionary")
    type_map = {}
    for key, value in raw_map.items():
        key = _literal(key, "type code")
        # These are the producer's normalized emitted strings, not inferred
        # biological names or an observed-only vocabulary from this specimen.
        value = _literal(value, "type name")
        if value != value.lower():
            raise ValueError("CellViT phenotype type_map is not producer-normalized")
        type_map[key] = value
    members = payload.get("cells")
    if not isinstance(members, (list, dict)):
        raise ValueError("CellViT phenotype population must contain cells")
    lookup = {}
    for fallback, member in (members.items() if isinstance(members, dict) else enumerate(members)):
        if not isinstance(member, dict):
            raise ValueError("Invalid CellViT phenotype population member")
        raw_id = member.get("id", fallback)
        if raw_id is None:
            raise ValueError("Missing CellViT phenotype source ID")
        identity = _literal(str(raw_id), "source ID")
        if identity in lookup:
            raise ValueError("Duplicate CellViT phenotype source ID")
        label = member.get("type_name", member.get("type"))
        if label is None:
            return unverified("Retained CellViT population lacks normalized categorical labels")
        label = _literal(label, "source label")
        if "type_name" in member and "type" in member and member["type"] != label:
            raise ValueError("CellViT type_name and normalized type disagree")
        if "type_id" in member:
            code = str(member["type_id"])
            if code in type_map:
                if type_map[code] != label:
                    raise ValueError("CellViT source phenotype differs from declared type_map code")
            elif label != f"unknown_{code}":
                raise ValueError("CellViT unmapped type code must remain explicitly unknown")
        elif label not in type_map.values() and label not in UNKNOWN_LABELS:
            raise ValueError("CellViT source phenotype is absent from its declared taxonomy")
        lookup[identity] = label
    if not {"cellvitpp_id", "phenotype"} <= set(cells):
        raise ValueError("Canonical profile lacks CellViT phenotype correspondence columns")
    seen, matched, missing = set(), 0, 0
    for identity, observed in zip(cells.cellvitpp_id, cells.phenotype):
        if identity == "":
            if not _unknown(observed):
                raise ValueError("Canonical phenotype without a CellViT source ID must be unknown")
            missing += 1
            continue
        identity = _literal(identity, "canonical source ID")
        if identity in seen:
            raise ValueError("Duplicate canonical CellViT phenotype source ID")
        seen.add(identity)
        if identity not in lookup:
            raise ValueError("Canonical CellViT phenotype ID is absent from retained population")
        if not isinstance(observed, str) or observed != lookup[identity]:
            raise ValueError("Canonical phenotype differs from its exact retained CellViT source label")
        matched += 1
    definition = copy.deepcopy(source_definition)
    definition.update({"representation": "CellViT predicted categorical nucleus type", "taxonomy": taxonomy,
                       "type_map": dict(sorted(type_map.items())),
                       "unknown_label_semantics": "unknown, __unknown__, and producer unknown_<unmapped_type_id> are missing categorical assignments",
                       "label_semantics": "detector taxonomy, not adjudicated biological cell identity"})
    return {"verified": True, "definition": definition,
            "source_binding": {"status": "verified_exact_retained_population_and_canonical_labels",
                "retained_population_sha256": actual, "retained_population_size_bytes": len(raw),
                "source_inputs_sha256": {name: value["sha256"] for name, value in
                    receipt.get("execution", {}).get("inputs", {}).items()},
                "canonical_mapped_cells": matched, "canonical_unknown_without_source": missing,
                "canonical_source_labels_mutated": False}}
