"""Synthetic source, geometry, leakage and same-slide evaluation checks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BORA_SRC = Path(os.environ.get("BORA_SEMANTIC_SOURCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "Bora" / "src")))
sys.path.insert(0, str(ROOT / "bin"))
import bora_semantic_roi_intake as intake


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _ref(path):
    return {"path": str(path), "sha256": intake.sha256(path)}


def _frame(path, mpp=1.):
    with Image.open(path) as im:
        w,h=im.size
    value=dict(sha256=intake.sha256(path),width=w,height=h,mpp_x=mpp,mpp_y=mpp,
        origin_x=0,origin_y=0,coordinate_space="crop-native-pixels",
        coordinate_convention="pixel-edge-origin_centres-in-native-pixels")
    value["frame_id"]="cellvit-frame-"+hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]
    return value


def _polygon(x0,y0,x1,y1):
    return {"type":"Polygon","coordinates":[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]}


def _feature(aid,cid,split,roi,box):
    return {"type":"Feature","geometry":_polygon(*box),"properties":{
        "annotation_id":aid,"class_id":cid,"split":split,"roi_id":roi,
        "annotator_id":"expert-1","annotation_source":"independent_expert"}}


@unittest.skipUnless((BORA_SRC/"bora/local_semantic.py").is_file(),
                     "Staged Bora semantic source is unavailable")
class SemanticIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix="semantic-roi-")
        root=Path(cls.tmp.name)
        cls.root=root
        h,w=700,1260
        he=np.empty((h,w,3),np.uint8)
        he[:,:]=[230,100,130]
        he[320:,:]=[75,95,230]
        support=np.ones((h,w),np.uint8)
        checkpoint=root/"virchow2-checkpoint.bin"
        checkpoint.write_bytes(b"synthetic-model-byte-identity")
        native=root/"native.tif"; working=root/"working.tif"
        Image.fromarray(he).save(native)
        Image.fromarray(he).save(working)
        hp=root/"he.npy"; sp=root/"support.npy"
        np.save(hp,he); np.save(sp,support)
        nf,wf=_frame(native),_frame(working)
        registration={"format":"cellphenotyper-dual-mpp-registration-v1",
            "verified":True,
            "support_sampling":"absolute pixel centre, nearest, full-frame GrandQC mask",
            "cell_full_image":nf,"image":wf,
            "tissue_to_cell_full_affine":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
        rp=_write_json(root/"registration.json",registration)
        ep=_write_json(root/"evidence.json",{"format":"coordinator-bora-global-evidence-v1",
            "source_sha256":{"registration":intake.sha256(rp)},
            "output_sha256":{"working_he":intake.sha256(hp)},
            "working_frame_id":wf["frame_id"],"working_image_sha256":wf["sha256"],
            "working_shape_yx":[h,w]})
        cp=_write_json(root/"coarse.json",{"format":"coordinator-bora-global-coarse-v1",
            "input_sha256":{"registration":intake.sha256(rp)},
            "output_sha256":{"support":intake.sha256(sp)},
            "working_frame_id":wf["frame_id"],"working_image_sha256":wf["sha256"],
            "working_shape_yx":[h,w]})
        sys.path.insert(0,str(BORA_SRC))
        from bora.chunk_plan import make_chunk_plan
        from bora.chunk_store import relative_name
        plan=make_chunk_plan(h,w)
        gridroot=root/"tokens"
        gridroot.mkdir()
        hashes={}
        for case in plan["cases"]:
            cy,cx=case["chunk_shape_yxc"][:2]
            array=np.zeros((cy,cx,1280),np.float32)
            if case["scale_factor"]==1:
                for iy in range(cy):
                    global_y=case["token_y0"]+iy
                    array[iy,:,0]=1. if global_y*14<320 else -1.
                    array[iy,:,1]=.25 if global_y*14<320 else -.25
            path=gridroot/relative_name(case)
            path.parent.mkdir(exist_ok=True)
            np.save(path,array)
            hashes[str(case["case_id"])]=intake.sha256(path)
        inventory={"format":"coordinator-bora-token-chunk-inventory-v1",
            "plan":plan,"sha256_by_case":hashes,
            "working_frame_id":wf["frame_id"],"working_image_sha256":wf["sha256"],
            "checkpoint_sha256":intake.sha256(checkpoint),"completed_case_count":plan["case_count"],
            "all_cleanup_verified":True,"one_global_absolute_field":True}
        ip=_write_json(gridroot/"CHUNK_INVENTORY.json",inventory)
        cls.features=[
            _feature("t1",1,"train","train-upper",(40,40,300,300)),
            _feature("t2",2,"train","train-lower",(40,350,300,610)),
            _feature("h1a",1,"holdout","upper-a",(650,40,900,300)),
            _feature("h1b",1,"holdout","upper-b",(920,40,1170,300)),
            _feature("h2a",2,"holdout","lower-a",(650,350,900,610)),
            _feature("h2b",2,"holdout","lower-b",(920,350,1170,610))]
        doc={"type":"FeatureCollection","schema":"cellphenotyper-expert-tissue-v1",
            "source":{"slide_id":"synthetic","source_frame_id":nf["frame_id"],
                      "image_sha256":nf["sha256"],"native_shape_yx":[h,w],
                      "native_mpp_xy":[1.,1.],"token_inventory_sha256":intake.sha256(ip),
                      "coordinate_space":"native_image_pixel_edges"},
            "class_names":{"1":"upper","2":"lower"},"features":cls.features}
        ap=_write_json(root/"expert.geojson",doc)
        source={name:intake.sha256(BORA_SRC/"bora"/name) for name in intake.SOURCE_MODULES}
        spec={"format":intake.FORMAT,"bora_source_root":str(BORA_SRC),
            "bora_source_sha256":source,"seed":11,"max_samples_per_class":256,
            "inputs":{name:_ref(path) for name,path in {
                "registration":rp,"native_image":native,"working_image":working,
                "working_he":hp,"he_evidence":ep,"working_support":sp,
                "support_receipt":cp,"token_inventory":ip,
                "virchow2_checkpoint":checkpoint,"expert_geojson":ap}.items()}}
        cls.spec=spec
        cls.registration=registration
        cls.inventory_sha=intake.sha256(ip)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _case(self, name, changes=None, annotation_changes=None):
        root=self.root/name
        root.mkdir()
        spec=json.loads(json.dumps(self.spec))
        if annotation_changes:
            document=json.loads(Path(spec["inputs"]["expert_geojson"]["path"]).read_text())
            annotation_changes(document)
            ap=_write_json(root/"expert.geojson",document)
            spec["inputs"]["expert_geojson"]=_ref(ap)
        if changes:
            changes(spec)
        sp=_write_json(root/"spec.json",spec)
        return sp,root/"output"

    def test_same_slide_pass_has_no_fullslide_projection(self):
        spec,output=self._case("good")
        report=intake.run(spec,output)
        import export_bora_semantic_project as semantic_export
        verified, _, model = semantic_export._evaluation(
            spec, output, BORA_SRC, Path(self.spec["inputs"]["registration"]["path"]))
        self.assertEqual(verified["confusion"], report["confusion"])
        self.assertTrue(model.is_file())
        self.assertIn("passed",report["status"])
        self.assertTrue(all(report["gates"].values()))
        self.assertGreaterEqual(report["macro_iou"],.99)
        self.assertFalse(report["independent_patient_validation"])
        self.assertFalse(report["semantic_annotation_published"])
        self.assertFalse(report["fullslide_projection_started"])
        self.assertEqual(report["source_sha256"]["token_inventory"],self.inventory_sha)
        self.assertTrue((output/"local_semantic_model.npz").is_file())
        self.assertTrue((output/"EVALUATION.json").is_file())
        with self.assertRaisesRegex(ValueError,"fresh absolute output"):
            intake.run(spec,output)

    def test_missing_expert_features_fails_before_output(self):
        spec,output=self._case("empty",annotation_changes=lambda doc: doc.update(features=[]))
        with self.assertRaisesRegex(ValueError,"expert polygons"):
            intake.run(spec,output)
        self.assertFalse(output.exists())

    def test_annotation_frame_and_inventory_binding(self):
        def change(document):
            document["source"]["token_inventory_sha256"]="0"*64
        spec,output=self._case("wrong-inventory",annotation_changes=change)
        with self.assertRaisesRegex(ValueError,"not bound"):
            intake.run(spec,output)
        self.assertFalse(output.exists())

    def test_holdout_roi_overlap_and_split_leakage(self):
        def change(document):
            document["features"][2]["properties"]["roi_id"]="train-upper"
        spec,output=self._case("roi-leak",annotation_changes=change)
        with self.assertRaisesRegex(ValueError,"separate train/holdout"):
            intake.run(spec,output)
        self.assertFalse(output.exists())

    def test_source_module_mutation_is_rejected(self):
        def change(spec):
            spec["bora_source_sha256"]["local_semantic.py"]="0"*64
        spec,output=self._case("wrong-source",changes=change)
        with self.assertRaisesRegex(ValueError,"source differs"):
            intake.run(spec,output)
        self.assertFalse(output.exists())

    def test_wrong_holdout_labels_are_reported_as_failed_not_published(self):
        def change(document):
            for feature in document["features"]:
                props=feature["properties"]
                if props["split"]=="holdout":
                    props["class_id"]=3-props["class_id"]
        spec,output=self._case("wrong-holdout",annotation_changes=change)
        report=intake.run(spec,output)
        self.assertEqual(report["status"],"same_slide_holdout_failed")
        self.assertFalse(report["gates"]["per_class_iou"])
        self.assertFalse(report["semantic_annotation_published"])
        self.assertTrue((output/"EVALUATION.json").is_file())

    def test_unverified_checkpoint_is_rejected(self):
        def change(spec):
            spec["inputs"]["virchow2_checkpoint"]["sha256"]="0"*64
        spec,output=self._case("wrong-checkpoint",changes=change)
        with self.assertRaisesRegex(ValueError,"checkpoint source"):
            intake.run(spec,output)
        self.assertFalse(output.exists())


if __name__=="__main__":
    unittest.main()
