"""A chosen StarDist run must not import stale CellViT classification files."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import export_experiment as exporter


def test_stardist_export_ignores_unselected_cellvit_outputs(tmp_path, monkeypatch):
    contract = tmp_path / '00_execution' / 'analysis_contract.json'
    contract.parent.mkdir()
    contract.write_text(json.dumps({'scientific_route': {'cell_detection_mode': 'stardist'}}))
    monkeypatch.setattr(exporter, '_discover_virchow_tissues', lambda run: [{'name': 'tissue', 'provenance': {}}])
    def must_not_import(run):
        raise AssertionError('StarDist must not discover an older CellViT population')
    monkeypatch.setattr(exporter, '_cell_producer_outputs', must_not_import)
    monkeypatch.setattr(exporter, '_discover_continuous_rgb', lambda *args: None)
    monkeypatch.setattr(exporter, '_discover_bora_groups', lambda *args: None)
    tissue, = exporter.discover_run(tmp_path)
    assert tissue['provenance']['cell_detection_mode'] == 'stardist'
    assert tissue['provenance']['cell_detection_contract_sha256'] == exporter.sha256(contract)
    assert 'cell_classification' not in tissue


def test_cellvit_export_keeps_its_selected_population(tmp_path, monkeypatch):
    contract = tmp_path / '00_execution' / 'analysis_contract.json'
    contract.parent.mkdir()
    contract.write_text(json.dumps({'scientific_route': {'cell_detection_mode': 'cellvitpp'}}))
    requested = []
    monkeypatch.setattr(exporter, '_discover_virchow_tissues', lambda run: [{'name': 'tissue'}])
    monkeypatch.setattr(exporter, '_cell_producer_outputs', lambda run: requested.append(run) or [])
    monkeypatch.setattr(exporter, '_discover_continuous_rgb', lambda *args: None)
    monkeypatch.setattr(exporter, '_discover_bora_groups', lambda *args: None)
    tissue, = exporter.discover_run(tmp_path)
    assert requested == [tmp_path]
    assert tissue['provenance']['cell_detection_mode'] == 'cellvitpp'
