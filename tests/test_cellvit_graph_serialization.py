"""Real PyTorch graph serialization -> portable CellViT bundle, no model.

This optional test uses only tiny synthetic tensors and an inert dataclass;
no CellViT package, pretrained weights, CUDA context or learned inference.
"""
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not hasattr(torch.serialization, "safe_globals"),
    reason="CellViT restricted graph decoding requires torch.serialization.safe_globals; this runtime is unsupported")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cellvit_embeddings as embeddings
import cellvit_embedding_io as bundle_io
from test_cellvit_runtime_identity import make_bound_embedding_fixture


def save_synthetic_graph(path, values, positions):
    # torch.save checks the class's declared module identity. Register that
    # inert schema temporarily; never import executable upstream CellViT code.
    names = ("cellvit", "cellvit.data", "cellvit.data.dataclass", "cellvit.data.dataclass.cell_graph")
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules[names[-1]].CellGraphDataWSI = embeddings.CellGraphDataWSI
    graph = embeddings.CellGraphDataWSI(values, positions,
        {"synthetic_tensor_fixture": True, "used_model": False}, {"1": "synthetic"})
    with patch.dict(sys.modules, modules):
        torch.save(graph, path)


@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_actual_restricted_graph_decoder_export_and_bundle_roundtrip(tmp_path, precision):
    fixture = make_bound_embedding_fixture(tmp_path / "inputs")
    output = tmp_path / "actual_serialization"
    raw = output / "raw"
    raw.mkdir(parents=True)
    for source, target in ((fixture.outdir / "raw/cells.json", raw / "cells.json"),
                           (fixture.outdir / "cellvit_cells.json", output / "cellvit_cells.json")):
        shutil.copy2(source, target)
    values = torch.arange(12, dtype=getattr(torch, precision)).reshape(3, 4)
    positions = torch.tensor([[2., 3.], [7., 4.], [10., 6.]], dtype=torch.float64)
    graph_path = raw / "cells.pt"
    save_synthetic_graph(graph_path, values, positions)
    graph_hash = bundle_io.sha256(graph_path)
    metadata = embeddings.export_embeddings(raw / "cells.json", output / "cellvit_cells.json", output,
        {"synthetic_graph_tensors": True, "used_model": False}, execution=fixture.execution)
    receipt = bundle_io.complete_embedding_bundle(output, fixture.execution, source_paths=fixture.paths)
    data, ids, contract, sources = bundle_io.load_cellvit_embedding_bundle(output,
        expected_inputs={key: value["sha256"] for key, value in fixture.execution["inputs"].items()},
        expected_geometry=fixture.execution["geometry"])
    np.testing.assert_array_equal(data, np.array([[4., 5., 6., 7.], [0., 1., 2., 3.]], dtype=np.float32))
    assert data.dtype == np.float32
    assert ids.cellvitpp_id.tolist() == ["NA", "007"]
    assert ids.source_graph_row.tolist() == [1, 0]
    assert receipt["raw_cell_count"] == 3 and receipt["retained_cell_count"] == 2
    assert receipt["excluded_cell_count"] == 1
    assert metadata["upstream_artifacts"]["raw_graph"]["sha256"] == graph_hash
    assert contract["status"] == "verified_exact_inputs_and_population"
    assert contract["reference_compatible"] is False  # No actual learned runtime was claimed.
    assert bundle_io.sha256(graph_path) == graph_hash
    assert all(path.suffix in (".json", ".csv", ".npy") for path in sources.values())
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 100_000


def test_actual_torch_loader_rejects_unallowlisted_graph_type(tmp_path):
    graph_path = tmp_path / "unsupported.pt"
    torch.save(SimpleNamespace(x=torch.zeros((1, 4)), positions=torch.zeros((1, 2))), graph_path)
    with pytest.raises(RuntimeError, match="restricted 1.0.9 graph schema"):
        embeddings.load_graph_arrays(graph_path)
