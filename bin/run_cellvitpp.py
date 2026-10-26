#!/usr/bin/env python3
"""Run official CellViT++ inference and normalize its WSI cell output."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_runtime import auto_batch_from_free_vram
from model_provenance import checkpoint_record, package_version
from cellvit_models import (digest_file, export_graph_features, classify_feature_manifest, read_graph,
                            load_catalog, preflight_models, segmentation_descriptor, write_json,
                            require_bounded_native_outputs, MAX_BOUNDED_CELLS)


PANNUKE_TYPE_MAP = {
    "1": "neoplastic",
    "2": "inflammatory",
    "3": "connective",
    "4": "dead",
    "5": "epithelial",
}

RAY_UNIX_SOCKET_MAX_BYTES = 107
NATIVE_LOG_LIMIT_BYTES = 8 * 1024 * 1024
NATIVE_GEOMETRY_SOURCES = {
    'cellvit': {
        'cellvit/inference/inference.py': '19a2961cd18d1258b5ad84dbf71bba182b2d1d9fcc787470611b86bf382df827',
        'cellvit/inference/postprocessing_numpy.py': 'fb5bacdf88a8093467d6035a45291364e8c93b8b0d3f3e75054e7ff4e83c53bf',
        'cellvit/inference/postprocessing_cupy.py': '3409f5cd5b65b429b33c26d81a02e229bfa44c17c0269776baecb12440cf31c8',
        'cellvit/data/dataclass/wsi_meta.py': 'fd8da1aa8a9754c6595a7bd4269bf1a56211994d22dcd1ad1450a86b8c9c7a7b'},
    'pathopatch': {
        'pathopatch/patch_extraction/dataset.py': '17d1e1796cd454962adc410c76b1faaae6f0962073805f923c4ea37814d9f63f',
        'pathopatch/utils/patch_util.py': '4501aaf466305031eac0d9b8ae81941b81a07e2d647e7b8dbb6326d7db54cad1'}}


def native_runtime_contract(enforce_amp=False):
    """Resolve installed preprocessing, without importing/probing CUDA or Ray.

    CellViT's dependency range allows newer PathoPatch implementations. Those
    are not interchangeable with the geometry audited here; fail explicitly.
    """
    from importlib.metadata import distribution, version
    sources = {}
    for package, expected in NATIVE_GEOMETRY_SOURCES.items():
        dist = distribution(package)
        allowed = ('1.0.9',) if package == 'cellvit' else ('1.0.9', '1.0.10')
        if dist.version not in allowed:
            raise ValueError(f'Native cell export requires verified {package} version {allowed}; found {dist.version}')
        for relative, expected_hash in expected.items():
            path = dist.locate_file(relative)
            if not Path(path).is_file() or digest_file(path) != expected_hash:
                raise ValueError(f'Native geometry source differs from the verified wheel: {relative}')
            sources[relative] = expected_hash
    packages = {name: version(name) for name in ('cellvit', 'pathopatch', 'torch', 'torchvision',
                'numpy', 'Pillow', 'scikit-image', 'openslide-python', 'pyvips')}
    return dict(contract_id='cellphenotyper-native-wsi-lossless-v1',
                prepared_pixel_policy='lossless-RGB8-white-right-bottom-minimum1024-native-px',
                coordinate_policy='source-upper-left-origin; stable original cell IDs; checked identical graph/JSON rows',
                validity_policy='source-centroid-in-half-open-original-extent; clean-tissue mask remains independent',
                runtime_geometry_sources_verified=True, runtime_geometry_source_sha256=sources,
                package_versions=packages, enforce_amp=bool(enforce_amp),
                precision_policy='pinned native CUDA autocast float16 when enforced or enabled by checkpoint; otherwise float32',
                native_wsi_parity_qualified=False)


def validate_cli_interpreter(executable):
    """The inspected packages must be those used by the launched console script."""
    path = Path(executable)
    text = path.read_text()
    first = text.splitlines()[0] if text else ''
    if not first.startswith('#!'):
        raise ValueError('CellViT executable must be its installed Python console script')
    command = shlex.split(first[2:])
    lines = text.splitlines()
    if (command == ['/bin/sh'] and len(lines) >= 3 and
            lines[1].startswith("'''exec' ") and lines[2].strip() == "' '''"):
        # pip/distlib's fixed shell/Python trampoline supports environments
        # whose path contains spaces or exceeds the kernel shebang limit.
        trampoline = shlex.split(lines[1][len("'''exec' "):])
        if len(trampoline) != 3 or trampoline[1:] != ['$0', '$@']:
            raise ValueError('Unsupported CellViT console-script shell trampoline')
        command = [trampoline[0]]
    if command and Path(command[0]).name == 'env':
        if len(command) != 2 or command[1].startswith('-'):
            raise ValueError('Unsupported CellViT console-script interpreter; use its installed Python entry point')
        command = [shutil.which(command[1]) or '']
    if (len(command) != 1 or not command[0] or
            Path(command[0]).resolve() != Path(sys.executable).resolve() or
            Path(command[0]).parent.resolve() != Path(sys.executable).parent.resolve() or
            'from cellvit.detect_cells import main' not in text):
        raise ValueError('Run this wrapper with the same Python environment as the configured CellViT executable')
    return dict(executable=str(path.resolve()), executable_sha256=digest_file(path),
                python_executable=sys.executable, interpreter_matches_wrapper=True)


def runtime_preflight(executable, enforce_amp=False):
    """CPU-only import/source gate; never construct a model, Ray service or output."""
    import importlib
    contract = native_runtime_contract(enforce_amp)
    executable = shutil.which(executable) if '/' not in executable else executable
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError('CellViT executable not found in the selected runtime')
    cli = validate_cli_interpreter(executable)
    failures = []
    for name in ('torch', 'torchvision', 'numpy', 'PIL', 'tifffile', 'pyvips', 'skimage', 'openslide'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('CellViT runtime dependencies are unavailable: ' + '; '.join(failures))
    import pyvips
    return dict(status='ready', cpu_imports_only=True, encoder_loaded=False, cuda_probed=False,
                ray_started=False, output_created=False, cli=cli, extraction_contract=contract,
                libvips_version='.'.join(str(pyvips.version(i)) for i in range(3)))


def requested_native_sampling(mpp):
    """Execute the pinned pure metadata calculation before any native model run."""
    if .20 <= mpp <= .30:
        target = mpp
    elif .40 <= mpp <= .60:
        from importlib.metadata import distribution
        source = Path(distribution('cellvit').locate_file('cellvit/data/dataclass/wsi_meta.py'))
        if digest_file(source) != NATIVE_GEOMETRY_SOURCES['cellvit']['cellvit/data/dataclass/wsi_meta.py']:
            raise ValueError('Native sampling source differs from the verified wheel')
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'find_target_mpp')
        namespace = dict(math=math)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        try:
            # Here .25 < baseMPP, so the exact PathoPatch calculation uses downsample1.
            target = namespace['find_target_mpp'](mpp, 1)
        except (AssertionError, ValueError) as exc:
            raise ValueError(f'Pinned CellViT cannot exactly calculate sampling for source MPP {mpp}; '
                             'use a separately resampled, calibrated crop before inference') from exc
    else:
        raise ValueError('Prepared CellViT TIFF requires calibrated source MPP in 0.20–0.30 or 0.40–0.60; '
                         'no scanner magnification is invented')
    return dict(source_mpp=mpp, expected_target_mpp=target,
                contract='pinned-load_wsi_meta-default-magnification-and-find_target_mpp')


def source_mpp(shift_path: Path, fallback: float) -> float:
    from cellvit_shards import rounded_mpp_matches
    data = json.loads(shift_path.read_text())
    value = data.get("source_mpp", data.get("microns_per_pixel", fallback))
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise RuntimeError("No valid MPP in shift.json; set --default-mpp explicitly")
    for axis in ("mpp_x", "mpp_y"):
        pixels = data.get('crop_size', {}).get('width' if axis == 'mpp_x' else 'height')
        if axis in data and (not math.isfinite(float(data[axis])) or
                             not (rounded_mpp_matches(data[axis], value, pixels) if pixels is not None else
                                  math.isclose(float(data[axis]), float(value), rel_tol=1e-6))):
            raise RuntimeError("CellViT requires identical positive X/Y/source MPP within output-pixel rounding; resample explicitly before inference")
    return float(value)


def auto_batch_size(requested: int, gpu: int) -> int:
    # CellViT++ 1.0.9 enforces a minimum batch size of two.
    return auto_batch_from_free_vram(
        requested,
        gpu,
        tiers=((68 * 1024, 32), (38 * 1024, 16), (20 * 1024, 8)),
        fallback=2,
        minimum=2,
        maximum=48,
    )


def resolve_runtime_paths(image: str, shift: str, outdir: str) -> tuple[Path, Path, Path]:
    """Give CellViT and its Ray workers stable paths independent of their CWD."""
    return tuple(Path(value).resolve() for value in (image, shift, outdir))


def required_model_path(cache_dir: Path, model: str) -> Path:
    filename = "CellViT-256-x40-AMP.pth" if model == "HIPT" else "CellViT-SAM-H-x40-AMP.pth"
    return cache_dir / filename


def require_readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable by uid {os.geteuid()}: {path}")


def validate_ray_temp_dir(temp_dir: Path) -> None:
    representative_socket = (
        temp_dir / "ray" / "session_2000-01-01_00-00-00_000000_0000000"
        / "sockets" / "plasma_store"
    )
    socket_bytes = len(os.fsencode(str(representative_socket)))
    if socket_bytes > RAY_UNIX_SOCKET_MAX_BYTES:
        raise RuntimeError(
            f"Ray temporary path is too long for an AF_UNIX socket ({socket_bytes} bytes): "
            f"{temp_dir}"
        )


def prepare_runtime_env(outdir: Path) -> tuple[dict[str, str], Path, Path]:
    runtime_dir = outdir / "runtime"
    for child in (runtime_dir / "cache", runtime_dir / "matplotlib"):
        child.mkdir(parents=True, exist_ok=True)
    temp_root = Path(os.environ.get("CELLVIT_TMP_ROOT", "/tmp"))
    temp_root.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = Path(tempfile.mkdtemp(prefix="cellvit_", dir=str(temp_root)))
    try:
        validate_ray_temp_dir(ray_temp_dir)
    except Exception:
        shutil.rmtree(ray_temp_dir, ignore_errors=True)
        raise
    env = os.environ.copy()
    env["TMPDIR"] = str(ray_temp_dir)
    env["RAY_TMPDIR"] = str(ray_temp_dir)
    env["XDG_CACHE_HOME"] = str(runtime_dir / "cache")
    env["MPLCONFIGDIR"] = str(runtime_dir / "matplotlib")
    return env, runtime_dir, ray_temp_dir


def empty_clean_core_evidence(mask_path: Path, full_shift: Path, shard: dict,
                              image_width: int, image_height: int) -> dict | None:
    """Prove that no core centroid can pass the existing GrandQC mask filter.

    Round the mask coverage outward. A nonzero border pixel makes the core
    eligible for native inference, even when most of its area is blank.
    """
    from grandqc_mask import CropCleanTissueMask
    clean = CropCleanTissueMask(mask_path, full_shift)
    if min(image_width, image_height, clean.mask_width, clean.mask_height) <= 0:
        raise RuntimeError("GrandQC mask and image must have positive dimensions")
    if (clean.crop_width, clean.crop_height) != (image_width, image_height):
        raise RuntimeError("GrandQC mask frame differs from the full CellViT image")
    x0, y0, x1, y1 = shard['core']
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        raise ValueError("CellViT core is outside the GrandQC mask frame")
    mx0 = (x0 * clean.mask_width) // image_width
    my0 = (y0 * clean.mask_height) // image_height
    mx1 = min(clean.mask_width, (x1 * clean.mask_width + image_width - 1) // image_width)
    my1 = min(clean.mask_height, (y1 * clean.mask_height + image_height - 1) // image_height)
    region = clean.mask[my0:my1, mx0:mx1]
    if region.size == 0 or (region != 0).any():
        return None
    return {'policy': 'grandqc-zero-clean-core-conservative-mask-cover-v1',
            'core_xyxy': list(shard['core']), 'mask_xyxy': [mx0, my0, mx1, my1],
            'mask_shape_yx': [clean.mask_height, clean.mask_width],
            'covered_mask_pixels': int(region.size), 'nonzero_mask_pixels': 0,
            'clean_tissue_mask_sha256': digest_file(mask_path),
            'full_shift_sha256': digest_file(full_shift),
            'conclusion': 'no cell centroid in this core passes the declared clean-tissue mask'}


def run_native_with_bounded_log(cmd: list[str], env: dict, log_path: Path) -> dict:
    """Retain bounded, exact native output while preserving live case logging."""
    total = retained = 0
    with log_path.open('xb') as log:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0)
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                total += len(chunk)
                room = max(0, NATIVE_LOG_LIMIT_BYTES - retained)
                if room:
                    saved = chunk[:room]
                    log.write(saved)
                    sys.stdout.buffer.write(saved)
                    sys.stdout.buffer.flush()
                    retained += len(saved)
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        log.flush()
        os.fsync(log.fileno())
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)
    if total != retained:
        print(f"CellViT native log exceeded {NATIVE_LOG_LIMIT_BYTES} bytes; "
              "retained prefix is marked truncated and cannot prove a zero-patch result", flush=True)
    return {'path': str(log_path.resolve()), 'sha256': digest_file(log_path),
            'bytes': retained, 'total_bytes': total, 'truncated': total != retained,
            'native_exit_code': returncode}


def no_patch_native_evidence(log_path: Path, log_record: dict,
                             prepared: Path, raw_dir: Path) -> dict | None:
    """Accept only the pinned upstream's complete, successful zero-patch trace."""
    if (log_record.get('native_exit_code') != 0 or log_record.get('truncated') is not False or
            log_record.get('sha256') != digest_file(log_path)):
        return None
    trace = log_path.read_text(encoding='utf-8', errors='replace')
    no_patches = re.search(r'(?m)^.*No patches sampled from '
                           + re.escape(str(prepared)) + r'\s*$', trace)
    no_cells = re.search(r'(?m)^.*No cells have been extracted\s*$', trace)
    finished = re.search(r'(?m)^.*Finished processing\s*$', trace)
    contradictory = re.search(r'(?m)Total patches sampled(?: after Filtering)?\s*:\s*[1-9][0-9]*\b|'
                              r'Detected cells after cleaning\s*:\s*[1-9][0-9]*\b', trace)
    if (not (no_patches and no_cells and finished) or contradictory or
            any(path.is_file() for path in raw_dir.rglob('*'))):
        return None
    return {'policy': 'pinned-cellvit-native-zero-patches-and-cells-v1',
            'native_log_sha256': log_record['sha256'],
            'native_log_bytes': log_record['bytes'], 'native_exit_code': 0,
            'prepared_image_sha256': digest_file(prepared),
            'raw_output_files': 0,
            'conclusion': 'official process_wsi sampled zero patches and emitted no cells'}


def materialize_empty_native_source(outdir: Path, mpp: float, target_mpp: float,
                                    feature_count: int, evidence: dict) -> tuple[Path, Path]:
    """Create explicit zero-row inputs for the ordinary checked export path."""
    from cellvit_models import _torch
    torch = _torch()
    cells_path = outdir / 'cellvit_verified_empty_source_cells.json'
    graph_path = outdir / 'cellvit_verified_empty_source_graph.pt'
    write_json(cells_path, {
        'wsi_metadata': {'patch_size': 1024, 'patch_overlap': 64,
                         'base_mpp': mpp, 'target_patch_mpp': target_mpp},
        'type_map': PANNUKE_TYPE_MAP, 'cells': [],
        'empty_core_evidence': evidence,
    })
    torch.save({'x': torch.empty((0, feature_count), dtype=torch.float32),
                'positions': torch.empty((0, 2), dtype=torch.float64)}, graph_path)
    return cells_path, graph_path


def make_pyramid(src: Path, dst: Path) -> dict:
    """Preserve every level-zero RGB value; model input must not be JPEG recompressed."""
    import pyvips

    image = pyvips.Image.new_from_file(str(src), access="sequential")
    if image.bands != 3 or image.format != "uchar":
        raise ValueError("CellViT input must be explicitly converted 8-bit, three-channel RGB")
    if image.interpretation not in ('srgb', 'rgb'):
        raise ValueError("CellViT requires RGB color interpretation; convert color space explicitly")
    original_size = (image.width, image.height)
    # Upstream 1.0.9 applies a faulty small-WSI correction after interpolation,
    # and also moves the long axis when only the other axis is short. Padding
    # on right/bottom preserves the source origin and makes both axes multi-tile.
    prepared = image.embed(0, 0, max(1024, image.width), max(1024, image.height), extend="white")
    prepared.tiffsave(
        str(dst), tile=True, tile_width=512, tile_height=512, pyramid=True,
        compression="lzw", bigtiff=True,
    )
    return {"method": "libvips-tiled-pyramidal-TIFF-LZW-RGB8",
            "level_zero_pixel_policy": "lossless-identical-RGB-values",
            "source_sha256": digest_file(src), "prepared_sha256": digest_file(dst),
            "width": original_size[0], "height": original_size[1],
            "prepared_width": prepared.width, "prepared_height": prepared.height,
            "padding_native_yx": [[0, prepared.height-original_size[1]], [0, prepared.width-original_size[0]]],
            "coordinate_policy": "source origin and RGB unchanged; white right/bottom padding excluded by source centroid",
            "libvips_version": ".".join(str(pyvips.version(i)) for i in range(3))}


def normalize_type_map(raw_map: object, taxonomy: str) -> dict[str, str]:
    if isinstance(raw_map, dict) and raw_map:
        return {str(key): str(value).strip().lower() for key, value in raw_map.items()}
    if taxonomy.strip().lower() == "pannuke":
        return dict(PANNUKE_TYPE_MAP)
    return {}


def normalize_output(raw_json: Path, output_json: Path, taxonomy: str, metadata: dict) -> None:
    """Emit one stable schema with named classes and retained upstream IDs."""
    payload = json.loads(raw_json.read_text())
    cells = payload.get("cells", payload) if isinstance(payload, dict) else payload
    type_map = normalize_type_map(payload.get("type_map") if isinstance(payload, dict) else None, taxonomy)
    iterable = cells.items() if isinstance(cells, dict) else enumerate(cells)
    normalized = {
        "model": "CellViT++",
        "taxonomy": taxonomy,
        "metadata": metadata,
        "wsi_metadata": payload.get("wsi_metadata", {}) if isinstance(payload, dict) else {},
        "type_map": type_map,
        "cells": [],
    }
    for fallback_id, item in iterable:
        if not isinstance(item, dict) or "centroid" not in item:
            continue
        type_id = item.get("type_id", item.get("type"))
        key = str(type_id)
        cell = dict(item)
        cell["id"] = str(item.get("source_id", item.get("id", fallback_id)))
        cell["type_id"] = type_id
        cell["type"] = type_map.get(key, f"unknown_{key}")
        normalized["cells"].append(cell)
    output_json.write_text(json.dumps(normalized))


def validate_native_geometry(payload: dict, expected_source_mpp: float) -> dict:
    """Reject known unsupported 1.0.9 small-WSI alignment before publishing cells.

    The pinned wheel removes a 1024-based padding displacement on *both* axes,
    after rescaling to native pixels. This is correct only when both axes have
    one tile and no resampling was applied, or neither axis needs that step.
    """
    metadata = payload.get("wsi_metadata", {})
    required = ("patch_size", "patch_overlap", "base_mpp", "target_patch_mpp",
                "orig_n_tiles_cols", "orig_n_tiles_rows")
    if any(key not in metadata for key in required):
        raise ValueError("Native CellViT geometry requires complete actual WSI sampling metadata")
    if metadata["patch_size"] != 1024 or metadata["patch_overlap"] != 64:
        raise ValueError("Pinned CellViT runtime requires 1024-pixel patches and 64-pixel overlap")
    base, target = float(metadata["base_mpp"]), float(metadata["target_patch_mpp"])
    if (not all(math.isfinite(v) and v > 0 for v in (base, target)) or
            not math.isclose(base, expected_source_mpp, rel_tol=1e-6) or not .20 <= target <= .30):
        raise ValueError("Native CellViT sampling does not match the calibrated input")
    counts = [metadata["orig_n_tiles_cols"], metadata["orig_n_tiles_rows"]]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in counts):
        raise ValueError("Native CellViT tile counts must be positive integers")
    short_axes = [v == 1 for v in counts]
    if any(short_axes) and (not all(short_axes) or abs(base - target) > .035):
        raise ValueError("CellViT 1.0.9 native small-WSI padding/rescaling is unsupported for this geometry; "
                         "use a separately declared CPU sampler or a larger calibrated crop, and do not publish native-aligned features")
    return {"alignment_contract": "CellViT-1.0.9-native-realignment-guard-v1",
            "single_tile_axes": short_axes, "base_mpp": base, "target_patch_mpp": target,
            "full_native_runtime_parity_tested": False}


def select_native_source_outputs(cells_path, graph_path, outdir, width, height, core_rect=None):
    """Keep raw native evidence, then select identical graph/cell rows in source extent.

    Source identifiers are assigned *before* filtering, so exclusion of padding
    never renumbers retained nuclei. Polygons are preserved, including boundary
    cells; only centroids outside the original source rectangle are excluded.
    """
    from cellvit_models import _source_cells, _torch
    require_bounded_native_outputs(cells_path, graph_path)
    payload = json.loads(Path(cells_path).read_text())
    cells = _source_cells(payload)
    x, positions = read_graph(graph_path)
    if x.shape[0] > MAX_BOUNDED_CELLS:
        raise ValueError('CellViT raw partition exceeds bounded-memory row limit; use smaller shard cores')
    if x.shape[0] != len(cells):
        raise ValueError("Native graph/cells row counts differ before source-extent selection")
    keep, selected = [], []
    if core_rect is None:
        core_rect = (0, 0, width, height)
    if (len(core_rect) != 4 or not (0 <= core_rect[0] < core_rect[2] <= width) or
            not (0 <= core_rect[1] < core_rect[3] <= height)):
        raise ValueError('CellViT core rectangle is outside its source crop')
    for index, (source_id, original) in enumerate(cells):
        cx, cy = map(float, original['centroid'])
        if any(abs(float(positions[index, j]) - value) > max(1e-4, abs(value)*2**-22)
               for j, value in enumerate((cx, cy))):
            raise ValueError("Native graph/cells coordinates differ before source-extent selection")
        if core_rect[0] <= cx < core_rect[2] and core_rect[1] <= cy < core_rect[3]:
            keep.append(index)
            cell = dict(original)
            cell['source_id'] = source_id
            cell['id'] = source_id
            selected.append(cell)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected_cells = outdir / 'cellvit_source_extent_cells.json'
    selected_graph = outdir / 'cellvit_source_extent_graph.pt'
    payload['cells'] = selected
    write_json(selected_cells, payload)
    torch = _torch()
    torch.save(dict(x=x[keep], positions=positions[keep]), selected_graph)
    evidence = dict(policy='source-centroid-in-half-open-core-extent', width=width, height=height,
                    core_rect=list(core_rect),
                    original_cells=len(cells), retained_cells=len(keep), excluded_cells=len(cells)-len(keep),
                    raw_cells=dict(path=str(Path(cells_path).resolve()), sha256=digest_file(cells_path)),
                    raw_graph=dict(path=str(Path(graph_path).resolve()), sha256=digest_file(graph_path)),
                    identity_policy='original-source-id-assigned-before-selection; graph and cells selected by same checked rows')
    return selected_cells, selected_graph, evidence


def finalize_normalized_lineage(normalized_path, manifest_path, classification_path, raw_cells_path):
    """Bind the final published segmentation JSON without changing raw lineage.

    The normalized JSON contains output names only, so these hashes form an
    acyclic chain: normalized cells -> feature manifest -> classification.
    All validation happens before either dependent artifact is replaced.
    """
    from export_experiment import atomic_json
    normalized_path, manifest_path, classification_path, raw_cells_path = (
        Path(path) for path in (normalized_path, manifest_path, classification_path, raw_cells_path))
    manifest = json.loads(manifest_path.read_text())
    classification = json.loads(classification_path.read_text())
    normalized = json.loads(normalized_path.read_text())
    if manifest.get("source", {}).get("cells_sha256") != digest_file(raw_cells_path):
        raise ValueError("Raw CellViT source hash differs from the feature manifest")
    if classification.get("provenance", {}).get("feature_manifest_sha256") != digest_file(manifest_path):
        raise ValueError("Classification is not bound to the current feature manifest")
    if classification.get("image") != manifest.get("image"):
        raise ValueError("Normalized lineage requires identical classification/manifest image frames")
    image, metadata = manifest["image"], normalized.get("metadata", {})
    expected_frame = dict(source_image_sha256=image["sha256"], sha256=image["sha256"], **{
        key: image[key] for key in ("width", "height", "mpp_x", "mpp_y", "origin_x", "origin_y", "coordinate_space", "coordinate_convention", "frame_id")})
    if any(metadata.get(key) != value for key, value in expected_frame.items()):
        raise ValueError("Final normalized segmentation frame differs from the feature image")
    expected = {cell["cell_id"]: cell for cell in classification.get("cells", [])}
    seen, source_ids = set(), set()
    for cell in normalized.get("cells", []):
        cid, sid = cell.get("cell_id"), cell.get("source_id")
        predicted = expected.get(cid)
        if (cid in seen or sid in source_ids or predicted is None or sid != predicted["source_id"] or
                str(cell.get("id")) != sid or cell.get("centroid") != [predicted["x"], predicted["y"]]):
            raise ValueError("Final normalized segmentation is not an exact canonical/source-ID and coordinate join")
        for field in ("contour", "bbox", "label"):
            if field in predicted and cell.get(field) != predicted[field]:
                raise ValueError("Final normalized segmentation geometry differs from cached feature geometry")
        seen.add(cid)
        source_ids.add(sid)
    if seen != set(expected):
        raise ValueError("Final normalized segmentation has missing or extra cells")
    reference = dict(path=Path(os.path.relpath(normalized_path, manifest_path.parent)).as_posix(),
                     sha256=digest_file(normalized_path))
    manifest["source"]["normalized_segmentation"] = reference
    atomic_json(manifest_path, manifest)
    classification["provenance"]["feature_manifest_sha256"] = digest_file(manifest_path)
    classification["provenance"]["normalized_segmentation_sha256"] = reference["sha256"]
    atomic_json(classification_path, classification)
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--shift")
    parser.add_argument("--outdir")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check the exact runtime and CPU imports; no image, model, CUDA or Ray execution")
    parser.add_argument("--executable", default=os.environ.get("CELLVIT_EXECUTABLE", "cellvit-inference"))
    parser.add_argument("--model", choices=("SAM", "HIPT"), default="HIPT")
    parser.add_argument("--taxonomy", default="pannuke")
    parser.add_argument("--taxonomies", nargs="+", default=None,
                        help="Cached heads to evaluate after one PanNuke segmentation (default: all)")
    parser.add_argument("--classifier-cache", default="",
                        help="Directory containing vit256/ and sam-h/; never downloaded")
    parser.add_argument("--classification-device", default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=32768)
    parser.add_argument("--ray-workers", type=int, default=1)
    parser.add_argument("--ray-worker-cpus", type=int, default=0)
    parser.add_argument("--default-mpp", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clean-tissue-mask", default="")
    parser.add_argument("--shard-plan", action="store_true",
                        help="Write a deterministic full-image plan; does not start CellViT or a GPU")
    parser.add_argument("--shard-id", default="",
                        help="Process exactly one planned shard; never advances to another shard")
    parser.add_argument("--probe-empty-clean-core", action="store_true",
                        help="CPU-only: report whether the entire planned core has zero GrandQC clean pixels")
    parser.add_argument("--cpu-empty-core", action="store_true",
                        help="CPU-only: publish a checked zero-cell shard only for a proven empty GrandQC core")
    parser.add_argument("--finalize-shards", action="store_true",
                        help="Verify completed shard checkpoints and publish a partition manifest; no GPU")
    parser.add_argument("--compact-shard-id", default="",
                        help="Losslessly archive one completed shard after verified crop release; CPU only")
    parser.add_argument("--pack-features-shard-id", default="",
                        help="CPU-only: retain every checked float32 feature in a versioned matrix bundle")
    parser.add_argument("--shard-core-size", type=int, default=1920)
    parser.add_argument("--shard-halo", type=int, default=960)
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(runtime_preflight(args.executable, args.amp), indent=2))
        return
    if not all((args.image, args.shift, args.outdir)):
        parser.error('--image, --shift and --outdir are required unless --preflight-only is selected')

    image, shift, outdir = resolve_runtime_paths(args.image, args.shift, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    default_classifier_cache = Path(os.environ.get('CELLVIT_CACHE', str(Path.home() / '.cache' / 'cellvit'))) / 'classifier'
    classifier_cache_contract = Path(args.classifier_cache).resolve() if args.classifier_cache else default_classifier_cache.resolve()
    run_contract = {
        'model': args.model,
        'requested_taxonomies': args.taxonomies or ([args.taxonomy] if args.taxonomy != 'pannuke' else ['all']),
        'classifier_cache': str(classifier_cache_contract),
        'classification_device': args.classification_device,
        'clean_tissue_mask': (str(Path(args.clean_tissue_mask).resolve()) if args.clean_tissue_mask else ''),
        'clean_tissue_mask_sha256': (digest_file(args.clean_tissue_mask) if args.clean_tissue_mask else ''),
        'default_mpp': args.default_mpp,
        'amp': args.amp,
    }
    if sum((args.shard_plan, bool(args.shard_id), args.finalize_shards,
            bool(args.compact_shard_id), bool(args.pack_features_shard_id))) > 1:
        parser.error('Choose only one shard plan, execution, compaction or finalization mode')
    if args.probe_empty_clean_core and args.cpu_empty_core:
        parser.error('Choose only one empty-core mode')
    if (args.shard_plan or args.shard_id) and args.classification_device != 'cpu':
        parser.error('Sharded CellViT requires CPU-only cached classification heads')
    empty_evidence = None
    if args.shard_plan or args.shard_id or args.finalize_shards or args.compact_shard_id or args.pack_features_shard_id:
        from cellvit_shards import (atomic_json, complete_shard, finalize_shards, get_shard,
                                    plan_shards, stage_shard, validate_plan)
        plan_path = outdir / 'cellvit_shard_plan.json'
        if args.shard_plan:
            if plan_path.exists():
                raise FileExistsError(f'Existing shard plan must be preserved: {plan_path}')
            plan = plan_shards(image, shift, args.shard_core_size, args.shard_halo, run_contract)
            atomic_json(plan_path, plan)
            print(json.dumps({'plan': str(plan_path), 'shards': len(plan['shards']),
                              'image_sha256': plan['image']['sha256']}))
            return
        plan = validate_plan(json.loads(plan_path.read_text()), image, shift)
        if args.probe_empty_clean_core or args.cpu_empty_core:
            if not args.shard_id or not args.clean_tissue_mask:
                parser.error('Empty-core modes require --shard-id and --clean-tissue-mask')
            selected = get_shard(plan, args.shard_id)
            empty_evidence = empty_clean_core_evidence(
                Path(args.clean_tissue_mask), shift, selected,
                plan['image']['width'], plan['image']['height'])
            if empty_evidence is not None:
                empty_evidence.update(plan_sha256=plan['identity_sha256'],
                                      image_sha256=plan['image']['sha256'])
            if args.probe_empty_clean_core:
                print(json.dumps({'shard': selected['id'], 'empty_clean_core': empty_evidence is not None,
                                  'evidence': empty_evidence}))
                return
            if empty_evidence is None:
                raise RuntimeError('CPU empty-core export refused: GrandQC core contains clean tissue')
        if args.finalize_shards:
            result = finalize_shards(plan, outdir)
            print(json.dumps({'manifest': str(outdir / 'cellvit_partition_manifest.json'),
                              'partitions': result['partition_count'], 'cells': result['cell_count']}))
            return
        if args.pack_features_shard_id:
            from cellvit_shards import replace_feature_csv
            selected = get_shard(plan, args.pack_features_shard_id)
            receipt = replace_feature_csv(plan, selected, outdir / 'shards' / selected['id'])
            print(json.dumps({'shard': selected['id'], 'feature_storage': receipt['schema'],
                              'source_csv_sha256': receipt['source_csv_sha256']}))
            return
        if args.compact_shard_id:
            from cellvit_shards import compact_shard, get_shard
            selected = get_shard(plan, args.compact_shard_id)
            receipt = compact_shard(plan, selected, outdir / 'shards' / selected['id'])
            print(json.dumps({'shard': selected['id'], 'lossless_archive_sha256': receipt['archive_sha256'],
                              'records': len(receipt['members'])}))
            return
    else:
        plan = None
    shard = None
    full_shift = shift
    if args.shard_id:
        if plan['run_contract'] != run_contract:
            raise ValueError('CellViT shard run settings differ from the immutable plan')
        shard = get_shard(plan, args.shard_id)
        shard_dir = outdir / 'shards' / shard['id']
        if (shard_dir / 'complete.json').exists():
            from cellvit_shards import release_verified_input_crops, verify_complete_shard
            verify_complete_shard(plan, shard, shard_dir)
            release_verified_input_crops(plan, shard, shard_dir, outdir / 'staging' / shard['id'])
            print(json.dumps({'shard': shard['id'], 'status': 'already_verified'}))
            return
        if shard_dir.exists() and any(shard_dir.iterdir()):
            raise FileExistsError(f'Incomplete shard evidence is preserved; review before restart: {shard_dir}')
        staged_image, staged_shift = stage_shard(plan, shard, outdir / 'staging' / shard['id'])
        image, shift = staged_image, staged_shift
        outdir = shard_dir
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        # The legacy exporter materializes the complete graph, JSON and all
        # classifier rows in RAM. Admit only bounded inputs on this route.
        import tifffile
        with tifffile.TiffFile(image) as opened:
            pixels = int(opened.pages[0].imagewidth) * int(opened.pages[0].imagelength)
        if pixels > 100_000_000:
            raise ValueError('Full-slide CellViT input exceeds the bounded legacy route; '
                             'use --shard-plan, then separately guarded --shard-id runs')
    executable = shutil.which(args.executable) if "/" not in args.executable else args.executable
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(f"CellViT++ executable not found: {args.executable}")
    cli_runtime = validate_cli_interpreter(executable)
    cache_dir = Path(os.environ.get("CELLVIT_CACHE", str(Path.home() / ".cache" / "cellvit")))
    model_path = required_model_path(cache_dir, args.model)
    require_readable_file(model_path, f"CellViT++ {args.model} checkpoint")
    cellvit_version = package_version("cellvit")
    segmentation_descriptor(model_path, args.model, cellvit_version)
    extraction_contract = native_runtime_contract(args.amp)
    mpp = source_mpp(shift, args.default_mpp)
    sampling_preflight = requested_native_sampling(mpp)
    if args.classification_device != "cpu":
        import re
        if not re.fullmatch(r"cuda:\d+", args.classification_device):
            raise ValueError("Classification device must be cpu or explicitly indexed cuda:N")
    requested_taxonomies = args.taxonomies or ([args.taxonomy] if args.taxonomy != "pannuke" else ["all"])
    classifier_cache = Path(args.classifier_cache) if args.classifier_cache else cache_dir / "classifier"
    # Preflight is CPU/file-only and records unavailable heads before segmentation.
    models_preflight = preflight_models(args.model, requested_taxonomies, classifier_cache)
    if shard and any(model['status'] != 'available' for model in models_preflight):
        unavailable = [model['model_id'] for model in models_preflight if model['status'] != 'available']
        raise RuntimeError(f'Requested CellViT heads are unavailable; refusing a partial shard: {unavailable}')
    write_json(outdir / "cellvit_model_preflight.json", {
        "models": models_preflight,
        "research_variants": load_catalog()["research_variants"], "network_access": False,
    })
    prepared = outdir / "cellvit_input.tif"
    input_preparation = make_pyramid(image, prepared)
    # The CPU-empty path must never query GPU memory, even when the campaign
    # uses its default automatic batch-size setting.
    batch_size = 2 if args.cpu_empty_core else auto_batch_size(args.batch_size, args.gpu)
    raw_dir = outdir / "raw"
    ray_workers = max(1, args.ray_workers)
    ray_worker_cpus = max(1, args.ray_worker_cpus or (args.cpus // ray_workers))
    cmd = [
        str(executable), "--model", args.model, "--nuclei_taxonomy", "pannuke",
        "--gpu", str(args.gpu), "--batch_size", str(batch_size), "--outdir", str(raw_dir),
        "--geojson", "--graph", "--cpu_count", str(args.cpus), "--memory", str(args.memory_mb),
        "--ray_worker", str(ray_workers), "--ray_remote_cpus", str(ray_worker_cpus),
    ]
    if args.amp:
        cmd.append("--enforce_amp")
    cmd += ["process_wsi", "--wsi_path", str(prepared), "--wsi_mpp", str(mpp)]
    env = runtime_dir = ray_temp_dir = None
    try:
        if args.cpu_empty_core:
            source_cells = source_graph = None
            empty_source_evidence = empty_evidence
        else:
            env, runtime_dir, ray_temp_dir = prepare_runtime_env(outdir)
            native_log = outdir / 'cellvit_native_process_wsi.log'
            log_record = run_native_with_bounded_log(cmd, env, native_log)
            candidates = sorted(raw_dir.rglob("cells.json"))
            if len(candidates) == 1:
                source_cells = candidates[0]
                source_graph = source_cells.with_name("cells.pt")
                require_readable_file(source_graph, "CellViT++ graph tokens")
                geometry_validation = validate_native_geometry(json.loads(source_cells.read_text()), mpp)
                empty_source_evidence = None
            elif len(candidates) == 0 and shard:
                empty_source_evidence = no_patch_native_evidence(
                    native_log, log_record, prepared, raw_dir)
                if empty_source_evidence is None:
                    raise RuntimeError(f"Expected one CellViT++ cells.json under {raw_dir}; found 0; "
                                       "no verified zero-patch evidence")
                if args.clean_tissue_mask:
                    clean_empty = empty_clean_core_evidence(
                        Path(args.clean_tissue_mask), full_shift, shard,
                        plan['image']['width'], plan['image']['height'])
                    if clean_empty is None:
                        raise RuntimeError("Official CellViT sampled no patches despite clean tissue in this core; "
                                           "preserve the case for review")
                    empty_source_evidence['grandqc_zero_clean_core'] = clean_empty
                source_cells = source_graph = None
            else:
                raise RuntimeError(f"Expected one CellViT++ cells.json under {raw_dir}; found {len(candidates)}")
        if empty_source_evidence is not None:
            descriptor = segmentation_descriptor(model_path, args.model, cellvit_version)
            source_cells, source_graph = materialize_empty_native_source(
                outdir, mpp, sampling_preflight['expected_target_mpp'],
                descriptor['feature_count'], empty_source_evidence)
            geometry_validation = {'alignment_contract': 'verified-empty-core-no-native-cells-v1',
                                   'native_geometry_available': False,
                                   'empty_core_evidence': empty_source_evidence}
        assert source_cells is not None and source_graph is not None
        source_cells, source_graph, source_selection = select_native_source_outputs(
            source_cells, source_graph, outdir, input_preparation['width'], input_preparation['height'],
            shard['local_core'] if shard else None)
        metadata = {
            "model": args.model, "taxonomy": "pannuke", "requested_taxonomies": requested_taxonomies,
            "source_mpp": mpp, "segmentation_runs": 0 if args.cpu_empty_core else 1,
            "segmentation_origin": "grandqc-empty-skip" if args.cpu_empty_core else "official-process_wsi",
            "input_preparation": input_preparation,
            "geometry_validation": geometry_validation,
            "empty_core_evidence": empty_source_evidence,
            "source_extent_selection": source_selection,
            "shard": ({"id": shard['id'], "input_xyxy": shard['input'],
                       "core_xyxy": shard['core'], "plan_sha256": plan['identity_sha256']}
                      if shard else None),
            "extraction_contract": extraction_contract,
            "cli_runtime": cli_runtime,
            "sampling_preflight": sampling_preflight,
            "batch_size": batch_size, "ray_workers": ray_workers,
            "ray_worker_cpus": ray_worker_cpus,
            "model_provenance": {
                "source_repository": "https://github.com/TIO-IKIM/CellViT-plus-plus",
                "requested_revision": cellvit_version,
                "resolved_revision": (
                    f"package:cellvit=={cellvit_version}"
                    if cellvit_version else None
                ),
                "cache_path": str(cache_dir.resolve()),
                "checkpoints": [checkpoint_record(model_path, logical_name=args.model)],
            },
        }
        normalized_path = outdir / "cellvit_cells.json"
        normalize_output(source_cells, normalized_path, "pannuke", metadata)
        manifest_path = export_graph_features(source_cells, source_graph, outdir, image, shift,
                                             model_path, args.model, cellvit_version,
                                             extraction_contract=extraction_contract,
                                             source_evidence={"input_preparation": input_preparation,
                                                              "geometry_validation": geometry_validation,
                                                              "source_extent_selection": source_selection,
                                                              "empty_core_evidence": empty_source_evidence})
        if args.clean_tissue_mask:
            from grandqc_mask import CropCleanTissueMask, filter_cell_payload
            if shard:
                # The clean mask is registered to the full source crop, not
                # the temporary halo image. Test global centroids exactly.
                clean = CropCleanTissueMask(args.clean_tissue_mask, full_shift)
                payload = json.loads(normalized_path.read_text())
                before = len(payload['cells'])
                ox, oy = shard['input'][:2]
                payload['cells'] = [cell for cell in payload['cells'] if clean.contains_xy(
                    float(cell['centroid'][0]) + ox, float(cell['centroid'][1]) + oy)]
                filter_summary = {'input_cells': before, 'retained_cells': len(payload['cells']),
                                  'removed_outside_grandqc_clean_tissue': before - len(payload['cells'])}
            else:
                payload, filter_summary = filter_cell_payload(
                    json.loads(normalized_path.read_text()), args.clean_tissue_mask, shift,
                )
            normalized_path.write_text(json.dumps(payload))
            metadata["grandqc_filter"] = filter_summary
            # Keep the immutable raw cache and classify only retained clean-tissue cells.
            from cellvit_models import read_feature_manifest
            import csv
            import gzip
            import io
            manifest, feature_cells, feature_rows = read_feature_manifest(manifest_path)
            retained_ids = {str(cell["id"]) for cell in payload["cells"]}
            keep = [i for i, cell in enumerate(feature_cells) if cell["source_id"] in retained_ids]
            if len(keep) != len(retained_ids):
                raise RuntimeError("GrandQC retained IDs do not join uniquely to cached CellViT features")
            csv_path = outdir / manifest["feature_set"]["path"]
            with csv_path.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["cell_id", "source_id", "x", "y", *manifest["feature_set"]["feature_columns"]])
                        for i in keep:
                            c = feature_cells[i]
                            writer.writerow([c["cell_id"], c["source_id"], c["x"], c["y"], *feature_rows[i]])
            cells_path = outdir / manifest["cell_metadata"]["path"]
            write_json(cells_path, {"image": manifest["image"], "cells": [feature_cells[i] for i in keep]})
            manifest["cell_metadata"]["sha256"] = digest_file(cells_path)
            manifest["feature_set"].update(sha256=digest_file(csv_path), row_count=len(keep))
            manifest["source"]["grandqc_filter"] = filter_summary
            write_json(manifest_path, manifest)
        classification = classify_feature_manifest(manifest_path, classifier_cache, requested_taxonomies,
                                                   outdir, args.classification_device)
        frame = json.loads(manifest_path.read_text())["image"]
        payload = json.loads(normalized_path.read_text())
        identity = {c["source_id"]: c["cell_id"] for c in json.loads(classification.read_text())["cells"]}
        for cell in payload["cells"]:
            cell["source_id"] = cell["id"]
            cell["cell_id"] = identity[cell["id"]]
        geometry_frame = dict(frame)
        geometry_frame["source_image_sha256"] = frame["sha256"]
        metadata.update(geometry_frame)
        metadata["classification_path"] = classification.name
        metadata["feature_manifest_path"] = manifest_path.name
        payload["metadata"] = metadata
        from export_experiment import atomic_json
        atomic_json(normalized_path, payload)
        finalize_normalized_lineage(normalized_path, manifest_path, classification, source_cells)
        (outdir / "cellvit_metadata.json").write_text(json.dumps(metadata, indent=2))
        if shard:
            from cellvit_shards import complete_shard, release_verified_input_crops, validate_plan
            validate_plan(plan, plan['image_path'], plan['shift_path'])
            complete_shard(plan, shard, outdir)
            release_verified_input_crops(plan, shard, outdir, image.parent)
        else:
            prepared.unlink(missing_ok=True)
    finally:
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        if ray_temp_dir is not None:
            shutil.rmtree(ray_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
