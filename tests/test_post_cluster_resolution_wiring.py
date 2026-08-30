from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_growth_and_medsam_receive_stardist_resolution_sidecar() -> None:
    workflow = (ROOT / "main.nf").read_text(encoding="utf-8")
    assert "def resolution_for_growth_variant_ch = Channel.empty()" in workflow
    assert "resolution_for_growth_variant_ch = shift_json_ch" in workflow
    assert workflow.count(".join(resolution_for_growth_variant_ch)") == 2

    grow = (ROOT / "modules/grow_to_tissue.nf").read_text(encoding="utf-8")
    medsam = (ROOT / "modules/refine_grown_tissue_medsam.nf").read_text(encoding="utf-8")
    assert "path(resolution_json)" in grow
    assert '--resolution-json "${resolution_json}"' in grow
    assert "path(resolution_json)" in medsam
    assert '--resolution-json "${resolution_json}"' in medsam


def test_neoplastic_section_uses_validated_bioformats_ome_output() -> None:
    source = (ROOT / "bin/select_neoplastic_section.py").read_text(encoding="utf-8")
    assert "pyramidize_with_raw2ometiff(" in source
    assert "validate_ome_tiff(" in source
    assert 'compression="LZW"' in source


def test_categorical_pyramids_use_nearest_neighbor_downsampling() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "grow_downsample                = 'SIMPLE'" in config


def test_medsam_keeps_native_compute_dtype_and_big_endian_storage() -> None:
    source = (ROOT / "bin/refine_grown_tissue_medsam.py").read_text(encoding="utf-8")
    assert 'label_dtype = np.dtype(label_dtype).newbyteorder("=")' in source
    assert 'storage_dtype = label_dtype.newbyteorder(">")' in source
    assert "dtype=storage_dtype" in source


def test_explicit_false_is_not_replaced_for_default_true_stages() -> None:
    workflow = (ROOT / "main.nf").read_text(encoding="utf-8")
    assert "params.tma_enable ?: true" not in workflow
    assert "params.cell_consensus_enable ?: true" not in workflow
    assert "params.tma_enable == null" in workflow
    assert "params.cell_consensus_enable == null" in workflow
