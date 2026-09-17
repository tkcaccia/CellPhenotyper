import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location("gap_completion", Path(__file__).parents[1]/"bin/complete_tissue_cluster_gaps.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestGapCompletion(unittest.TestCase):
    def test_native_tiled_writer_roundtrip(self):
        import tifffile
        shape=(301,47)
        expected=np.arange(np.prod(shape),dtype=np.uint16).reshape(shape)
        def tiles():
            for y in range(0,shape[0],256):
                tile=np.zeros((256,48),np.uint16)
                part=expected[y:y+256]
                tile[:len(part),:47]=part
                yield tile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"test.ome.tif"
            tifffile.imwrite(path,tiles(),shape=shape,dtype=np.uint16,
                tile=(256,48),compression="zlib",metadata={"axes":"YX"})
            np.testing.assert_array_equal(tifffile.imread(path),expected)

    def test_stained_hole_filled_white_and_exterior_preserved(self):
        rgb = np.full((30,30,3), [190,130,180], np.uint8)
        original = np.zeros((30,30), np.uint16)
        original[5:25,5:25] = 1
        original[5:25,15:25] = 2
        original[10:15,10:15] = 0
        original[17:20,17:20] = 0
        rgb[17:20,17:20] = 255
        raw = np.zeros_like(original)
        raw[11,11] = 2
        new, prov, recovered, eligible = module.complete(rgb,original,original>0,raw,pixel_um=2)
        np.testing.assert_array_equal(new[original>0], original[original>0])
        self.assertTrue((new[10:15,10:15]>0).all())
        self.assertEqual(new[11,11],2)
        self.assertEqual(prov[11,11],2)
        self.assertTrue((new[17:20,17:20]==0).all())
        self.assertTrue((new[:5]==0).all())
        self.assertTrue(recovered[11,11])

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            module.complete(np.zeros((2,2,3)),np.ones((2,2),int),
                np.ones((2,2),bool),np.full((2,2),6),pixel_um=1)

    def test_artifacts_never_assigned_and_clean_pale_tissue_is_eligible(self):
        rgb = np.full((10,10,3),250,np.uint8)
        old = np.ones((10,10),np.uint16)
        old[3:7,3:7]=0
        art = np.zeros((10,10),bool)
        art[4:6,4:6]=True
        new,*_ = module.complete(rgb,old,np.ones_like(art),np.ones_like(old),artifacts=art,pixel_um=1)
        self.assertTrue((new[art]==0).all())
        self.assertTrue((new[~art]>0).all())

    def test_distance_limit(self):
        rgb = np.full((30,30,3),[190,130,180],np.uint8)
        old = np.ones((30,30),np.uint16)
        old[5:25,5:25]=0
        new,*_ = module.complete(rgb,old,old>0,np.zeros_like(old),pixel_um=10,max_distance_um=20)
        self.assertEqual(new[15,15],0)


if __name__ == "__main__":
    unittest.main()
