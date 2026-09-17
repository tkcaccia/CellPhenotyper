import csv
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
import cellvit_embeddings as embeddings
import run_cellvitpp as wrapper
import cellvit_embedding_io as bundle_io


def payload(ids=("10", "02", "35")):
    return {"cells": [
        {"id": value, "centroid": [index * 10 + 1, index * 10 + 2], "type": 1}
        for index, value in enumerate(ids)
    ]}


class CellViTEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.raw = payload()
        self.vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.positions = np.array([[1, 2], [11, 12], [21, 22]], dtype=float)

    def test_filter_and_reorder_preserve_upstream_string_ids(self):
        retained = {"cells": [self.raw["cells"][2], self.raw["cells"][1]]}
        result, rows, metadata = embeddings.align_embeddings(
            self.raw, retained, self.vectors, self.positions,
        )
        np.testing.assert_array_equal(result, self.vectors[[2, 1]])
        self.assertEqual([row["cellvitpp_id"] for row in rows], ["35", "02"])
        self.assertEqual([row["source_graph_row"] for row in rows], [2, 1])
        self.assertEqual(metadata["excluded_cell_count"], 1)
        self.assertEqual(metadata["missing_embedding_count"], 0)
        self.assertEqual(metadata["embedding_dimension"], 4)

    def test_normalization_fallback_ids_match_original_graph_rows(self):
        raw = {"cells": [{"centroid": [1, 2], "type": 1}, {"centroid": [11, 12], "type": 3}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw.json").write_text(json.dumps(raw))
            wrapper.normalize_output(root / "raw.json", root / "normalized.json", "pannuke", {})
            normalized = json.loads((root / "normalized.json").read_text())
        normalized["cells"] = normalized["cells"][1:]
        result, rows, _ = embeddings.align_embeddings(raw, normalized, self.vectors[:2], self.positions[:2])
        self.assertEqual(rows[0]["cellvitpp_id"], "1")
        np.testing.assert_array_equal(result[0], self.vectors[1])

    def test_dict_ids_preserve_insertion_order(self):
        raw = {"cells": {"c9": {"centroid": [1, 2]}, "c1": {"centroid": [11, 12]}}}
        _, rows, _ = embeddings.align_embeddings(raw, raw, self.vectors[:2], self.positions[:2])
        self.assertEqual([row["cellvitpp_id"] for row in rows], ["c9", "c1"])

    def test_duplicate_ids_are_rejected_in_raw_and_retained(self):
        duplicate = {"cells": [self.raw["cells"][0], self.raw["cells"][0]]}
        for raw, retained in [(duplicate, self.raw), (self.raw, duplicate)]:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "duplicate"):
                embeddings.align_embeddings(raw, retained, self.vectors, self.positions)

    def test_missing_vectors_and_retained_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            embeddings.align_embeddings(self.raw, self.raw, self.vectors[:2], self.positions)
        unknown = {"cells": [{"id": "absent", "centroid": [1, 2]}]}
        with self.assertRaisesRegex(ValueError, "Missing CellViT embedding"):
            embeddings.align_embeddings(self.raw, unknown, self.vectors, self.positions)

    def test_permuted_graph_rows_are_rejected_by_centroids(self):
        with self.assertRaisesRegex(ValueError, "row order"):
            embeddings.align_embeddings(self.raw, self.raw, self.vectors, self.positions[::-1])

    def test_changed_retained_centroid_is_rejected(self):
        retained = payload()
        retained["cells"][1]["centroid"] = [400, 500]
        with self.assertRaisesRegex(ValueError, "centroid differs"):
            embeddings.align_embeddings(self.raw, retained, self.vectors, self.positions)

    def test_missing_and_nonnumeric_embeddings_are_rejected(self):
        vectors = self.vectors.copy()
        vectors[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            embeddings.align_embeddings(self.raw, self.raw, vectors, self.positions)
        with self.assertRaisesRegex(ValueError, "numeric"):
            embeddings.align_embeddings(self.raw, self.raw, vectors.astype(str), self.positions)

    def test_empty_retained_cells_have_correct_vector_width(self):
        features, rows, metadata = embeddings.align_embeddings(
            self.raw, {"cells": []}, self.vectors, self.positions,
        )
        self.assertEqual(features.shape, (0, 4))
        self.assertEqual(rows, [])
        self.assertEqual(metadata["excluded_cell_count"], 3)

    def test_exact_graph_option_required(self):
        embeddings.validate_graph_option("usage: cellvit-inference [--graph]\n --graph  Export graph")
        for text in ("--graphical", "--graph-format", "--geojson", ""):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "--graph"):
                embeddings.validate_graph_option(text)

    def test_actual_executable_help_is_checked(self):
        with patch.object(embeddings.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "[--graph]", "")
            summary = embeddings.check_graph_support("/cellvit/bin/cellvit-inference")
        self.assertEqual(run.call_args.args[0], ["/cellvit/bin/cellvit-inference", "--help"])
        self.assertTrue(summary["graph_cli_verified"])
        self.assertEqual(len(summary["help_sha256"]), 64)

    def test_failed_cli_check_has_clear_error(self):
        with patch.object(embeddings.subprocess, "run", side_effect=subprocess.TimeoutExpired("help", 60)):
            with self.assertRaisesRegex(RuntimeError, "verify CellViT"):
                embeddings.check_graph_support("cellvit-inference")

    def test_portable_export_has_no_pickle_and_complete_id_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw_json = raw_dir / "cells.json"
            raw_json.write_text(json.dumps(self.raw))
            (raw_dir / "cells.pt").touch()
            retained_json = root / "cellvit_cells.json"
            retained_json.write_text(json.dumps({"cells": [self.raw["cells"][1]]}))
            with patch.object(embeddings, "load_graph_arrays", return_value=(self.vectors, self.positions)):
                metadata = embeddings.export_embeddings(raw_json, retained_json, root, {"model": "HIPT"})
            actual = np.load(root / "cellvit_embeddings.npy", allow_pickle=False)
            with (root / "cellvit_embedding_ids.csv").open() as handle:
                ids = list(csv.DictReader(handle))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, self.vectors[1:2])
            self.assertEqual(ids[0]["cellvitpp_id"], "02")
            self.assertEqual(metadata["provenance"]["model"], "HIPT")
            self.assertEqual(json.loads((root / "cellvit_embeddings_metadata.json").read_text()), metadata)

    def test_missing_graph_fails_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "no matching graph"):
                embeddings.export_embeddings(root / "cells.json", root / "filtered.json", root, {})
            self.assertFalse((root / "cellvit_embeddings.npy").exists())

    def test_graph_loading_is_restricted_and_never_retries_unsafe_pickle(self):
        @contextmanager
        def safe_globals(allowed):
            self.assertEqual(allowed, [embeddings.CellGraphDataWSI])
            yield

        fake_torch = SimpleNamespace(serialization=SimpleNamespace(safe_globals=safe_globals))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch.object(fake_torch, "load", create=True, side_effect=ValueError("untrusted global")) as load:
                with self.assertRaisesRegex(RuntimeError, "restricted"):
                    embeddings.load_graph_arrays(Path("cells.pt"))
                self.assertEqual(load.call_count, 1)
                self.assertIs(load.call_args.kwargs["weights_only"], True)

    def test_old_torch_fails_before_inference(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(serialization=SimpleNamespace())}):
            with self.assertRaisesRegex(RuntimeError, "safe_globals"):
                embeddings.require_graph_loader()

    def test_wrapper_requests_graph_only_when_enabled(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                raw_dir = output / "raw" / "cellvit_input"
                raw_dir.mkdir(parents=True)
                (raw_dir / "cells.json").write_text(json.dumps(self.raw))
                shift = root / "shift.json"
                shift.write_text(json.dumps({"source_mpp": 0.25}))
                argv = [
                    "run_cellvitpp.py", "--image", str(root / "image.tif"),
                    "--shift", str(shift), "--outdir", str(output),
                    "--executable", sys.executable,
                ] + (["--export-embeddings"] if enabled else [])
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(wrapper, "require_readable_file"),
                    patch.object(wrapper, "make_pyramid", return_value={}),
                    patch.object(wrapper, "auto_batch_size", return_value=2),
                    patch.object(wrapper, "checkpoint_record", return_value={}),
                    patch.object(wrapper, "capture_inputs", return_value=({"inputs": {}, "geometry": {"source_mpp": .25}}, {})),
                    patch.object(wrapper, "check_inputs"),
                    patch.object(wrapper, "file_record", return_value={}),
                    patch.object(wrapper, "executable_runtime_identity", return_value={"status": "unverified"}),
                    patch.object(bundle_io, "complete_embedding_bundle") as complete,
                    patch.object(wrapper.subprocess, "run") as run,
                    patch.object(embeddings, "require_graph_loader") as loader_check,
                    patch.object(embeddings, "check_graph_support", return_value={}) as cli_check,
                    patch.object(embeddings, "export_embeddings", return_value={}) as export,
                ):
                    wrapper.main()
                command = run.call_args.args[0]
                self.assertEqual("--graph" in command, enabled)
                if enabled:
                    self.assertLess(command.index("--graph"), command.index("process_wsi"))
                self.assertEqual(export.call_count, int(enabled))
                self.assertEqual(cli_check.call_count, int(enabled))
                self.assertEqual(loader_check.call_count, int(enabled))
                self.assertEqual(complete.call_count, int(enabled))
                metadata = json.loads((output / "cellvit_metadata.json").read_text())
                self.assertIsNone(metadata["model_provenance"]["resolved_revision"])


if __name__ == "__main__":
    unittest.main()
