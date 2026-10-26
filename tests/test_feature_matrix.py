import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import feature_matrix as fm


def _gzip_csv(path, rows):
    import io
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                csv.writer(text).writerows(rows)


class FeatureMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv.gz"
        _gzip_csv(self.source, [
            ["cell_id", "source_id", "x", "y", "feat_1", "feat_2"],
            ["cell-1", "src-1", "1.0", "2.125", "0.5", "-0.0"],
            ["cell-2", "src-2", "3", "4.25", "1.5", "-2.0"],
        ])
        self.bundle = self.root / "bundle"
        fm.pack(self.source, self.bundle)

    def _rewrite_index(self, bundle, transform, rehash=False):
        with gzip.open(bundle / fm.INDEX, "rt", newline="") as stream:
            rows = list(csv.reader(stream))
        transform(rows)
        _gzip_csv(bundle / fm.INDEX, rows)
        if rehash:
            manifest = json.loads((bundle / fm.MANIFEST).read_text())
            manifest["index"]["sha256"] = fm.digest(bundle / fm.INDEX)
            manifest["index"]["size"] = (bundle / fm.INDEX).stat().st_size
            (bundle / fm.MANIFEST).write_text(json.dumps(manifest))

    def test_exact_csv_restoration_and_rows(self):
        restored = self.root / "restored.csv.gz"
        fm.restore_csv(self.bundle, restored)
        self.assertEqual(restored.read_bytes(), self.source.read_bytes())
        rows = list(fm.iter_rows(self.bundle))
        self.assertEqual([row[0][0] for row in rows], ["cell-1", "cell-2"])
        self.assertEqual(struct.pack("<f", rows[0][1][1]), struct.pack("<f", -0.0))

    def test_deterministic_bundle_hashes(self):
        second = self.root / "second"
        fm.pack(self.source, second)
        for name in (fm.MATRIX, fm.INDEX, fm.MANIFEST):
            self.assertEqual(fm.digest(self.bundle / name), fm.digest(second / name))

    def test_corrupt_matrix_rejected_without_publishing(self):
        matrix = self.bundle / fm.MATRIX
        data = bytearray(matrix.read_bytes())
        data[len(data) // 2] ^= 1
        matrix.write_bytes(data)
        target = self.root / "corrupt-output.csv.gz"
        with self.assertRaises(ValueError):
            fm.restore_csv(self.bundle, target)
        self.assertFalse(target.exists())

    def test_truncated_matrix_rejected_without_publishing(self):
        matrix = self.bundle / fm.MATRIX
        matrix.write_bytes(matrix.read_bytes()[:-5])
        target = self.root / "truncated-output.csv.gz"
        with self.assertRaises(ValueError):
            fm.restore_csv(self.bundle, target)
        self.assertFalse(target.exists())

    def test_changed_matrix_rejected_even_with_rehashed_file(self):
        matrix = self.bundle / fm.MATRIX
        with gzip.open(matrix, "rb") as stream:
            data = bytearray(stream.read())
        data[0] ^= 1
        with matrix.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(data)
        manifest = json.loads((self.bundle / fm.MANIFEST).read_text())
        manifest["matrix"]["sha256"] = fm.digest(matrix)
        manifest["matrix"]["size"] = matrix.stat().st_size
        (self.bundle / fm.MANIFEST).write_text(json.dumps(manifest))
        target = self.root / "changed-output.csv.gz"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            fm.restore_csv(self.bundle, target)
        self.assertFalse(target.exists())

    def test_swapped_ids_rejected_even_with_rehashed_index(self):
        def swap(rows):
            rows[1][0], rows[2][0] = rows[2][0], rows[1][0]
        self._rewrite_index(self.bundle, swap, rehash=True)
        target = self.root / "swapped-output.csv.gz"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            fm.restore_csv(self.bundle, target)
        self.assertFalse(target.exists())

    def test_truncated_index_rejected(self):
        index = self.bundle / fm.INDEX
        index.write_bytes(index.read_bytes()[:-8])
        target = self.root / "truncated-index-output.csv.gz"
        with self.assertRaises(ValueError):
            fm.restore_csv(self.bundle, target)
        self.assertFalse(target.exists())

    def test_non_float32_input_is_rejected(self):
        bad = self.root / "bad.csv.gz"
        _gzip_csv(bad, [["cell_id", "source_id", "x", "y", "feat_1"],
                        ["x", "s", "1", "1", "0.1"]])
        rejected = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "loses precision"):
            fm.pack(bad, rejected)
        self.assertFalse(rejected.exists())

    def test_mismatched_scientific_manifest_rejected(self):
        manifest = self.root / "scientific.json"
        manifest.write_text(json.dumps({"schema": "cellphenotyper-cell-features",
                                        "feature_set": {"sha256": "0" * 64}}))
        rejected = self.root / "wrong-manifest"
        with self.assertRaisesRegex(ValueError, "does not match"):
            fm.pack(self.source, rejected, manifest)
        self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
