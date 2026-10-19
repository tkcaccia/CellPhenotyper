import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
from run_stardist_roi_segmentation import _local_stardist_candidates, stardist_model_dir


class StarDistCacheDiscoveryTest(unittest.TestCase):
    def test_nested_extracted_model_is_loadable_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models" / "StarDist2D" / "2D_versatile_he"
            root.mkdir(parents=True)
            # A release archive alone is not a loadable model.
            (root / "2D_versatile_he.zip").touch()
            with patch.dict(os.environ, {"KERAS_HOME": directory}):
                self.assertEqual(_local_stardist_candidates("2D_versatile_he"), [])
                extracted = root / "2D_versatile_he_extracted"
                extracted.mkdir()
                (extracted / "config.json").touch()
                (extracted / "weights_best.h5").touch()
                self.assertEqual(
                    _local_stardist_candidates("2D_versatile_he"),
                    [(str(root), "2D_versatile_he_extracted")],
                )
                self.assertEqual(stardist_model_dir("2D_versatile_he"), extracted.resolve())


if __name__ == "__main__":
    unittest.main()
