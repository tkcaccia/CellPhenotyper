import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

pytest.importorskip("skimage")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import medsam_border_refine as medsam
import refine_grown_tissue_medsam as refine
from tissue_appearance_refine import refine_tissue_domains_by_appearance


def test_encoder_embedding_is_reused_across_multiple_box_prompts(monkeypatch):
    encoded, decoded = [], []
    def encode(image, config):
        encoded.append(image.shape)
        return np.full((1, 4, 4, 4), 3, np.float32)
    def decode(embedding, shape, box, config):
        decoded.append(np.asarray(embedding).copy())
        return np.ones(shape[:2], bool), np.ones(shape[:2], np.float32)
    monkeypatch.setattr(medsam, "_encode_image_tile", encode)
    monkeypatch.setattr(medsam, "_decode_box_prompt", decode)
    cache = medsam.TileEmbeddingCache()
    directory = Path(cache.directory.name)
    config = replace(medsam.MedSAMConfig(), _embedding_cache=cache)
    image = np.full((20, 20, 3), 100, np.uint8)
    for box in ([1, 2, 6, 7], [5, 7, 12, 13], [2, 3, 14, 15]):
        medsam._infer_box_prompt(image[:10, :10], np.array(box), config)
    medsam._infer_box_prompt(image[10:, 10:], np.array([1, 1, 9, 9]), config)
    assert len(encoded) == 2 and len(decoded) == 4
    assert cache.encoder_calls == 2 and cache.hits == 2
    np.testing.assert_array_equal(decoded[0], decoded[2])
    cache.close()
    assert not directory.exists()


def test_global_tile_lattice_and_real_refinement_loop_share_encoder(monkeypatch):
    count = []
    monkeypatch.setattr(medsam, "_encode_image_tile", lambda image, config: count.append(1) or np.ones((1, 4, 4, 4), np.float32))
    monkeypatch.setattr(medsam, "_decode_box_prompt", lambda embedding, shape, box, config: (np.ones(shape[:2], bool), np.ones(shape[:2], np.float32)))
    labels = np.ones((48, 48), np.uint16)
    labels[:, 24:] = 2
    seeds = np.zeros_like(labels)
    seeds[20:24, 10:14] = 1
    seeds[20:24, 34:38] = 2
    config = medsam.MedSAMConfig(device="cpu", component_min_area=1, component_merge_distance=0, seed_dilation_radius=1, core_erosion_radius=2, outer_dilation_radius=3, min_object_size=1, smooth_radius=0, cluster_tile_size=64, cluster_tile_overlap=0)
    _, _, _, meta, _ = medsam.run_medsam_border_refine(np.full((48, 48, 3), 100, np.uint8), seeds, labels > 0, config, baseline_label_map=labels)
    assert len(count) == 1
    assert meta["image_encoder_calls"] == 1
    assert meta["image_embedding_cache_hits"] == 1


def test_encoder_cache_is_cleaned_even_after_refinement_failure(monkeypatch):
    caches = []
    original_cache = medsam.TileEmbeddingCache
    def cache_factory():
        cache = original_cache()
        caches.append(cache)
        return cache
    monkeypatch.setattr(medsam, "TileEmbeddingCache", cache_factory)
    with pytest.raises(ValueError, match="shape mismatch"):
        medsam.run_medsam_border_refine(np.zeros((3, 3, 3), np.uint8), np.ones((3, 3), np.uint16), np.ones((2, 2), bool), medsam.MedSAMConfig())
    assert len(caches) == 1
    assert not Path(caches[0].directory.name).exists()


def test_original_uncertainty_survives_hole_fill_and_appearance_assignment():
    original = np.array([[1, 0, 1, 0, 2, 2, 0]], np.uint16)
    result = np.array([[1, 2, 2, 3, 0, 2, 0]], np.uint16)
    tissue = np.array([[1, 1, 1, 1, 1, 1, 0]], bool)
    uncertainty = np.array([[0, 3, 2, 0, 0, 0, 0]], np.uint8)
    protected = np.array([[1, 0, 0, 0, 0, 0, 0]], np.uint16)
    provenance, categories = refine.refinement_provenance(original, result, tissue, uncertainty, protected)
    assert provenance.tolist() == [[7, 3, 3, 5, 6, 1, 0]]
    assert categories.tolist() == [[0, 3, 2, 251, 252, 0, 0]]
    with pytest.raises(RuntimeError, match="Protected"):
        refine.refinement_provenance(original, result + 1, tissue, uncertainty, protected)


def test_growth_inference_is_not_promoted_to_original_observation_confidence():
    grown = np.ones((1, 5), np.uint16)
    original_cluster_support = np.array([[1, 0, 0, 1, 0]], np.uint16)
    incoming = np.array([[0, 0, 4, 0, 0]], np.uint8)
    final = np.array([[1, 1, 1, 2, 2]], np.uint16)
    provenance, uncertainty = refine.refinement_provenance(grown, final, np.ones(grown.shape, bool), incoming, None, original_cluster_support)
    assert provenance.tolist() == [[1, 8, 3, 2, 2]]
    assert uncertainty.tolist() == [[0, 254, 4, 250, 250]]


def test_uncertainty_constraints_preserve_confident_cores_and_background():
    original = np.ones((80, 80), np.uint16)
    original[:, 40:] = 2
    tissue = np.ones_like(original, bool)
    tissue[30:40, 30:40] = False
    original[~tissue] = 0
    uncertain = np.zeros_like(original, np.uint8)
    uncertain[10:20, 10:20] = 2
    editable, protected = refine.refinement_constraints(original, tissue, uncertain, 5, 8)
    assert protected[60, 20] == 1 and protected[60, 60] == 2
    assert protected[15, 15] == 0 and editable[15, 15]
    candidate = np.full_like(original, 3)
    final = refine.enforce_refinement_constraints(candidate, original, tissue, editable, protected)
    assert final[60, 20] == 1 and final[60, 60] == 2
    assert final[15, 15] == 3
    assert np.all(final[~tissue] == 0)


def test_grandqc_kodama_outlier_code_is_hard_excluded_from_refinement():
    original = np.ones((40, 40), np.uint16)
    tissue = np.ones_like(original, bool)
    uncertainty = np.zeros_like(original, np.uint8)
    uncertainty[10:20, 10:20] = refine.GRANDQC_KODAMA_OUTLIER_CODE
    editable, protected = refine.refinement_constraints(original, tissue, uncertainty, 3, 5)
    assert not editable[15, 15]
    assert protected[15, 15] == 0
    effective_tissue = tissue & (uncertainty != refine.GRANDQC_KODAMA_OUTLIER_CODE)
    final = refine.enforce_refinement_constraints(np.ones_like(original), original, effective_tissue, editable, protected)
    assert np.all(final[10:20, 10:20] == 0)


def test_multiclass_appearance_corrects_only_editable_supported_disagreements():
    image = np.zeros((120, 180, 3), np.uint8)
    colors = [(70, 30, 110), (220, 160, 180), (130, 90, 70)]
    labels = np.ones((120, 180), np.uint16)
    for i, color in enumerate(colors):
        image[:, i * 60:(i + 1) * 60] = color
        labels[:, i * 60:(i + 1) * 60] = i + 1
    labels[25:50, 20:45] = 2
    editable = np.zeros(labels.shape, bool)
    editable[20:55, 15:50] = True
    protected = np.zeros(labels.shape, bool)
    protected[30:35, 25:30] = True
    trusted = ~editable
    result, metadata = refine_tissue_domains_by_appearance(image, labels, np.ones(labels.shape, bool), clusters=9, core_erosion_px=2, smooth_sigma=.5, min_region_area_px=50, sample_pixels=10000, allow_multiclass=True, editable_mask=editable, protected_mask=protected, core_support=trusted)
    assert metadata["applied"] and metadata["multiclass"]
    assert set(metadata["supported_labels"]) == {1, 2, 3}
    assert result[25:50, 20:45][~protected[25:50, 20:45]].mean() < 1.1
    np.testing.assert_array_equal(result[protected], labels[protected])
    np.testing.assert_array_equal(result[~editable], labels[~editable])


def test_ambiguous_multiclass_appearance_votes_do_not_invent_assignment():
    image = np.full((90, 90, 3), 110, np.uint8)
    labels = np.ones((90, 90), np.uint16)
    labels[:, 30:60] = 2
    labels[:, 60:] = 3
    result, metadata = refine_tissue_domains_by_appearance(image, labels, np.ones(labels.shape, bool), clusters=2, allow_multiclass=True, core_erosion_px=2, min_region_area_px=1)
    np.testing.assert_array_equal(result, labels)
    assert metadata["changed_pixels"] == 0


def test_physical_units_require_verified_calibration_and_scale_correctly(tmp_path):
    path = tmp_path / "resolution.json"
    path.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    args = SimpleNamespace(resolution_json=str(path), source_mpp_x=.5, source_mpp_y=.5, medsam_core_erosion_um=10, appearance_min_region_area_um2=100)
    refine.apply_physical_parameters(args)
    assert args.medsam_core_erosion_radius == 20
    assert args.appearance_min_region_area_px == 400
    args.source_mpp_x = args.source_mpp_y = .25
    refine.apply_physical_parameters(args)
    assert args.medsam_core_erosion_radius == 40
    assert args.appearance_min_region_area_px == 1600
    path.write_text(json.dumps({"source_mpp": .5}))
    with pytest.raises(ValueError, match="verified resolution"):
        refine.apply_physical_parameters(args)


def test_provenance_writer_outputs_lossless_categories_and_declines_input_overwrite(tmp_path):
    original = np.ones((8, 8), np.uint16)
    result = original.copy()
    result[2:5, 2:5] = 2
    uncertainty = np.zeros((8, 8), np.uint8)
    uncertainty[2:5, 2:5] = 4
    paths = {}
    for name, values in [("grown", original), ("tissue", np.ones((8, 8), np.uint8)), ("uncertain", uncertainty)]:
        paths[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(paths[name], values)
    args = SimpleNamespace(out=str(tmp_path / "refined.tif"), provenance_out="", uncertainty_out="", clustering_uncertainty=str(paths["uncertain"]), grown_mask=str(paths["grown"]), tissue_mask=str(paths["tissue"]), source_mpp_x=.5, source_mpp_y=.5, stream_block_rows=3, overwrite=False)
    reader = refine.TiffWindowReader(str(paths["grown"]), "grown")
    tissue = refine.ScaledBinaryMaskReader(str(paths["tissue"]), (8, 8))
    incoming = refine.TiffWindowReader(str(paths["uncertain"]), "uncertainty")
    try:
        meta = refine.write_refinement_provenance_outputs(args, result, reader, tissue, incoming)
        np.testing.assert_array_equal(tifffile.imread(meta["uncertainty_path"]), uncertainty)
        provenance = tifffile.imread(meta["provenance_path"])
        assert np.all(provenance[2:5, 2:5] == 3)
        assert meta["output_binding_status"] == "pending_final_label_export"
        with pytest.raises(FileNotFoundError, match="Final refinement output"):
            refine.finalize_refinement_provenance_outputs(args, meta)
        # Binding must refer to the finalized/compressed destination, not an
        # in-memory label array or a temporary flat checkpoint's file bytes.
        tifffile.imwrite(args.out, result, compression="deflate")
        meta = refine.finalize_refinement_provenance_outputs(args, meta)
        assert meta["output_binding_status"] == "complete"
        assert meta["output_artifacts"]["labels"]["sha256"] == hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
        assert json.loads(Path(str(args.out) + ".provenance.json").read_text()) == meta
        args.uncertainty_out = str(paths["uncertain"])
        with pytest.raises((ValueError, FileExistsError)):
            refine.write_refinement_provenance_outputs(args, result, reader, tissue, incoming)
    finally:
        reader.close(); tissue.close(); incoming.close()
    np.testing.assert_array_equal(tifffile.imread(paths["uncertain"]), uncertainty)


def test_finalization_rejects_changed_flushed_sidecar_even_with_same_histogram(tmp_path):
    shape = (8, 8)
    original = np.ones(shape, np.uint16)
    incoming = np.zeros(shape, np.uint8)
    incoming[1:3, 1:3] = 3
    paths = {name: tmp_path / f"{name}.tif" for name in ("grown", "support", "incoming")}
    for name, values in (("grown", original), ("support", np.ones(shape, np.uint8)), ("incoming", incoming)):
        tifffile.imwrite(paths[name], values)
    args = SimpleNamespace(out=str(tmp_path / "final.tif"), provenance_out="", uncertainty_out="", clustering_uncertainty=str(paths["incoming"]),
        grown_mask=str(paths["grown"]), tissue_mask=str(paths["support"]), source_mpp_x=.5, source_mpp_y=.5, stream_block_rows=3, overwrite=False)
    reader = refine.TiffWindowReader(str(paths["grown"]), "grown")
    tissue = refine.ScaledBinaryMaskReader(str(paths["support"]), shape)
    uncertain = refine.TiffWindowReader(str(paths["incoming"]), "uncertainty")
    try:
        metadata = refine.write_refinement_provenance_outputs(args, original, reader, tissue, uncertain)
    finally:
        reader.close(); tissue.close(); uncertain.close()
    tifffile.imwrite(args.out, original, compression="deflate")
    output_uncertainty = Path(metadata["uncertainty_path"])
    values = tifffile.imread(output_uncertainty)
    tifffile.imwrite(output_uncertainty, np.roll(values, 1, axis=1))
    with pytest.raises(ValueError, match="sidecar changed"):
        refine.finalize_refinement_provenance_outputs(args, metadata)
    assert json.loads(Path(str(args.out) + ".provenance.json").read_text())["output_binding_status"] == "pending_final_label_export"


@pytest.mark.parametrize("mode", ["full", "stream"])
@pytest.mark.parametrize("appearance", [False, True])
@pytest.mark.parametrize("pyramid_failure", [False, True])
def test_full_and_stream_cli_preserve_unknown_after_refinement_and_fill(tmp_path, monkeypatch, mode, appearance, pyramid_failure):
    labels = np.ones((80, 80), np.uint16)
    labels[:, 40:] = 2
    support = np.ones(labels.shape, np.uint8)
    support[40:48, 25:33] = 0
    labels[support == 0] = 0
    seed = labels.copy()
    seed[55:65, 15:25] = 0  # already inferred during tissue growth
    incoming = np.zeros(labels.shape, np.uint8)
    incoming[20:30, 15:25] = 3
    incoming[10:15, 30:35] = refine.GRANDQC_KODAMA_OUTLIER_CODE
    paths = {}
    for name, values in [("image", np.full((80, 80, 3), 100, np.uint8)), ("seed", seed), ("grown", labels), ("tissue", support), ("uncertain", incoming)]:
        paths[name] = tmp_path / (name + ".tif")
        tifffile.imwrite(paths[name], values)
    original_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    out = tmp_path / "refined.ome.tif"
    calls = []
    def mock_medsam(**kwargs):
        calls.append(kwargs)
        assert not kwargs["seed_labels"][20:30, 15:25].any()
        # Deliberately overwrite cores, fill uncertain pixels and support holes;
        # the production orchestration must enforce every immutable constraint.
        candidate = np.full(kwargs["seed_labels"].shape, 2, np.uint16)
        return candidate > 0, None, .1, {"image_encoder_calls": 1, "image_embedding_cache_hits": 1}, {"label_map": candidate}
    monkeypatch.setattr(refine, "run_medsam_border_refine", mock_medsam)
    monkeypatch.setattr(refine, "medsam_model_provenance", lambda config: {"used_model": False, "test_mock": True})
    def pyramidize(source, target, args):
        if pyramid_failure:
            raise RuntimeError("synthetic pyramid failure")
        # Force genuinely different file bytes while preserving native labels.
        tifffile.imwrite(target, tifffile.imread(source), compression="deflate", tile=(16, 16))
    monkeypatch.setattr(refine, "pyramidize_or_fallback", pyramidize)
    monkeypatch.setattr(refine, "make_panel", lambda *args, **kwargs: None)
    monkeypatch.setattr(refine, "save_png", lambda *args, **kwargs: None)
    monkeypatch.setattr(refine, "save_preview_png", lambda *args, **kwargs: None)
    if appearance:
        # Simulate an aggressive later appearance correction. The real output
        # orchestrator must retain uncertainty and restore protected labels.
        monkeypatch.setattr(refine, "appearance_refine_working_scale", lambda image, labels, tissue, args, step, **kwargs: (np.ones_like(labels), {"applied": True, "changed_pixels": int(np.count_nonzero(labels != 1))}))
    argv = ["refine", "--sample-id", "specimen", "--image", str(paths["image"]), "--seed-mask", str(paths["seed"]), "--grown-mask", str(paths["grown"]), "--tissue-mask", str(paths["tissue"]), "--clustering-uncertainty", str(paths["uncertain"]), "--resolution-json", str(resolution), "--out", str(out), "--preview", str(tmp_path / "preview.png"), "--large-image-mode", mode, "--medsam-core-erosion-radius", "5", "--internal-boundary-radius", "8", "--medsam-cluster-tile-size", "512", "--medsam-cluster-tile-overlap", "0", "--medsam-qc-crop-size", "0", "--no-medsam-save-debug", "--no-image-guided-internal-refine", "--no-appearance-refine"]
    if appearance:
        argv[-1] = "--appearance-refine"
    monkeypatch.setattr(sys, "argv", argv)
    if pyramid_failure:
        with pytest.raises(RuntimeError, match="synthetic pyramid failure"):
            refine.main()
        metadata = json.loads(Path(str(out) + ".provenance.json").read_text())
        assert metadata["output_binding_status"] == "pending_final_label_export"
        assert "output_artifacts" not in metadata
        assert not (tmp_path / "specimen_medsam_summary.json").exists()
        assert not out.exists()
        for progress in tmp_path.glob("*progress*.json"):
            assert json.loads(progress.read_text()).get("status") != "complete"
        return
    refine.main()
    assert len(calls) == 1
    final = tifffile.imread(out)
    assert final[60, 20] == 1 and final[25, 20] == (1 if appearance else 2)
    assert not final[10:15, 30:35].any()
    assert not final[support == 0].any()
    uncertainty = tifffile.imread(str(out) + ".uncertainty.tif")
    provenance = tifffile.imread(str(out) + ".provenance.tif")
    np.testing.assert_array_equal(uncertainty[incoming > 0], incoming[incoming > 0])
    assert np.all(provenance[incoming == 3] == 3)
    assert np.all(provenance[incoming == refine.GRANDQC_KODAMA_OUTLIER_CODE] == 4)
    assert uncertainty[60, 20] == 254 and provenance[60, 20] == 8
    assert uncertainty[60, 60] == 0 and provenance[60, 60] == 7
    metadata = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert metadata["output_binding_status"] == "complete"
    for name, path in (("labels", out), ("uncertainty", Path(str(out) + ".uncertainty.tif")), ("provenance", Path(str(out) + ".provenance.tif"))):
        assert metadata["output_artifacts"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert metadata["output_artifacts"][name]["size_bytes"] == path.stat().st_size
    assert json.loads((tmp_path / "specimen_medsam_summary.json").read_text())["refinement_provenance"] == metadata
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()} == original_hashes


def test_nextflow_refinement_uncertainty_and_provenance_contract():
    module = (ROOT / "modules/refine_grown_tissue_medsam.nf").read_text()
    post = (ROOT / "subworkflows/post_grow_spatial_outputs.nf").read_text()
    auxiliary = (ROOT / "subworkflows/run_auxiliary_cell_spatial.nf").read_text()
    assert '--clustering-uncertainty "${clustering_uncertainty_tif}"' in module
    assert 'path(clustering_uncertainty_tif, stageAs:' in module
    for emitted in ("refined_uncertainty", "refined_provenance", "provenance_metadata"):
        assert "emit: " + emitted in module
        assert ".out." + emitted in post
        assert ".out." + emitted in auxiliary
    assert "cluster_uncertainty_ch" in post
    assert ".out.uncertainty_mask" in auxiliary
    assert "failOnMismatch: true" in post and "failOnMismatch: true" in auxiliary
