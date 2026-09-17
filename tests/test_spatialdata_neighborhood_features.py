"""Actual SpatialData I/O of source-bound, separately array-backed groups."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sd = pytest.importorskip("spatialdata")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import export_spatialdata as exporter
import neighborhood_feature_io as nio
from test_spatialdata_export import specimen, affine


def add_store(root, *, dimensions=4):
    header=root/"cell_profiles_manifest.json"
    manifest=json.loads(header.read_text())
    if dimensions!=4:
        values=np.arange(2*dimensions,dtype=np.float32).reshape(2,dimensions)
        values[1,-1]=np.nan
        np.save(root/"context.npy",values)
        manifest["feature_blocks"]["context"].update(shape=list(values.shape),sha256=nio._sha(root/"context.npy"))
    own=np.load(root/"context.npy")
    columns=[f"own_context:feature_{i}" for i in range(dimensions)]
    groups={"own:context":{"columns":columns,"segments":[{
        "path":"context.npy","sha256":nio._sha(root/"context.npy"),"shape":list(own.shape),
        "dtype":str(own.dtype),"columns":columns}]},"context":{"columns":[],"segments":[]}}
    expected={"own:context":own.astype(np.float64)}
    arrays=[]
    for radius in (25,50):
        values=np.nextafter(np.arange(2*dimensions,dtype=np.float64).reshape(2,dimensions)/radius,np.inf)
        values[1,1]=np.nan
        relative=f"neighborhood_features/r{radius}_context.npy"
        path=root/relative; path.parent.mkdir(exist_ok=True)
        np.save(path,values)
        columns=[f"r{radius}um_context:feature_{i}" for i in range(dimensions)]
        groups["context"]["columns"].extend(columns)
        groups["context"]["segments"].append({"path":relative,"sha256":nio._sha(path),
            "shape":list(values.shape),"dtype":str(values.dtype),"columns":columns})
        arrays.append(values)
    expected["context"]=np.concatenate(arrays,axis=1)
    summary={"feature_groups":{name:value["columns"] for name,value in groups.items()},
        "feature_group_definitions":{"own:context":{"aggregation":"identity_no_imputation","focal_cell_included":True},
            "context":{"aggregation":"mean_of_finite_neighbor_values","focal_cell_included":False,"radii_um":[25.,50.]}}}
    (root/"neighborhood_summary.json").write_text(json.dumps(summary))
    manifest["neighborhoods"]={"summary":"neighborhood_summary.json"}
    manifest["files"]["neighborhood_summary.json"]=nio._sha(root/"neighborhood_summary.json")
    manifest["neighborhood_feature_store"]=nio.finalize_store(root,groups)
    header.write_text(json.dumps(manifest))
    return expected,groups,manifest


def test_groups_preserve_scalar_obs_original_arrays_precision_axes_and_portability(specimen,tmp_path):
    args,image,labels,original_context=specimen
    root=args["profile_dir"]
    expected,groups,manifest=add_store(root)
    before={p:nio._sha(p) for p in root.rglob("*") if p.is_file()}
    original=pd.read_parquet(root/"cell_profiles.parquet")
    output=tmp_path/"array_groups.zarr"
    summary=exporter.export_spatialdata(**args,outdir=output)
    loaded=sd.SpatialData.read(output); table=loaded.tables["cells"]
    record=summary["neighborhood_features"]
    assert record==loaded.attrs["cellphenotyper"]["neighborhood_features"]
    assert json.loads(table.uns["cellphenotyper"]["neighborhood_features_json"])==record
    mapping=json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    assert set(table.obs)==set(original)|{"instance_id","spatial_region"}
    assert table.obs_names.tolist()==original.cell_uid.tolist()
    assert set(table.obsm)=={"spatial","context","local","neighborhood_group_0000","neighborhood_group_0001"}
    safe=exporter._safe_obs(original.set_axis(table.obs.index))
    for name in original:
        pd.testing.assert_series_equal(table.obs[name],safe[name],check_exact=True)
    assert table.obsm["context"].dtype==np.dtype("float32")
    np.testing.assert_array_equal(table.obsm["context"],original_context)
    for group,matrix in expected.items():
        entry=record["groups"][group]; target=entry["obsm"]
        assert mapping["obsm"]["neighborhood_features/"+group]==target
        assert table.obsm[target].dtype==np.dtype("float64")
        np.testing.assert_array_equal(table.obsm[target],matrix)
        assert entry["logical_columns"]==groups[group]["columns"]
        assert entry["source_segments"]==groups[group]["segments"]
        assert entry["aggregation_definition_status"]=="source_declared"
    assert record["groups"]["context"]["aggregation_definition"]["radii_um"]==[25.,50.]
    np.testing.assert_array_equal(loaded.images["he_image"].data.compute(),np.moveaxis(image,-1,0))
    np.testing.assert_array_equal(loaded.labels["canonical_cells"].data.compute(),labels)
    np.testing.assert_array_equal(affine(loaded.labels["canonical_cells"]),[[.5,0,50],[0,.5,100],[0,0,1]])
    assert all(nio._sha(path)==digest for path,digest in before.items())
    assert manifest["feature_blocks"]==json.loads(table.uns["cellphenotyper"]["profile_manifest_json"])["feature_blocks"]
    root.rename(tmp_path/"unmounted_profiles")
    moved=sd.SpatialData.read(output)
    for name,entry in record["groups"].items():
        np.testing.assert_array_equal(moved.tables["cells"].obsm[entry["obsm"]],expected[name])


def test_wide_group_export_reads_both_axes_in_bounded_blocks(specimen,tmp_path,monkeypatch):
    args=specimen[0]; expected,_,_=add_store(args["profile_dir"],dimensions=300)
    original=nio.FeatureColumns.read; calls=[]
    def bounded(self,rows,columns):
        assert isinstance(rows,slice) and rows.stop-rows.start<=args["tile_size"]
        assert len(columns)<=256
        calls.append((rows.start,rows.stop,len(columns)))
        return original(self,rows,columns)
    monkeypatch.setattr(nio.FeatureColumns,"read",bounded)
    output=tmp_path/"bounded.zarr"
    summary=exporter.export_spatialdata(**args,outdir=output)
    assert len(calls)>=10 and max(size for _,_,size in calls)==256
    table=sd.SpatialData.read(output).tables["cells"]
    for name,values in expected.items():
        np.testing.assert_array_equal(table.obsm[summary["neighborhood_features"]["groups"][name]["obsm"]],values)


def test_no_store_preserves_legacy_export(specimen,tmp_path):
    output=tmp_path/"legacy.zarr"
    summary=exporter.export_spatialdata(**specimen[0],outdir=output)
    table=sd.SpatialData.read(output).tables["cells"]
    assert summary["neighborhood_features"]=={"status":"not_provided"}
    assert "neighborhood_features_json" not in table.uns["cellphenotyper"]
    assert set(table.obsm)=={"spatial","context","local"}


@pytest.mark.parametrize("target",["array","store","rows","summary"])
def test_corrupt_sources_fail_before_output(specimen,tmp_path,target):
    args=specimen[0]; root=args["profile_dir"]; add_store(root)
    name={"array":"neighborhood_features/r25_context.npy","store":nio.STORE_PATH,
        "rows":"feature_rows.csv","summary":"neighborhood_summary.json"}[target]
    with (root/name).open("ab") as stream: stream.write(b"corrupted")
    output=tmp_path/"corrupt.zarr"
    with pytest.raises(ValueError): exporter.export_spatialdata(**args,outdir=output)
    assert not output.exists()


def test_existing_obsm_is_never_overwritten(specimen,tmp_path):
    args=specimen[0]; root=args["profile_dir"]; _,_,manifest=add_store(root)
    manifest["feature_blocks"]["neighborhood_group_0000"]=manifest["feature_blocks"]["context"]
    (root/"cell_profiles_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="collides"):
        exporter.export_spatialdata(**args,outdir=tmp_path/"collision.zarr")


def test_source_change_during_write_is_rejected(specimen,tmp_path,monkeypatch):
    args=specimen[0]; root=args["profile_dir"]; add_store(root)
    original=sd.SpatialData.write
    def changed(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        path=root/"neighborhood_features/r25_context.npy"
        values=np.load(path).copy(); values[0,0]=.2; np.save(path,values)
        return result
    monkeypatch.setattr(sd.SpatialData,"write",changed)
    with pytest.raises(ValueError,match="source changed"):
        exporter.export_spatialdata(**args,outdir=tmp_path/"source_changed.zarr")


@pytest.mark.parametrize("target",["one_ulp","nan","axis_metadata","package_metadata"])
def test_exact_readback_rejects_altered_arrays_and_metadata(specimen,tmp_path,monkeypatch,target):
    import zarr
    args=specimen[0]; add_store(args["profile_dir"])
    original=sd.SpatialData.write
    def changed(self,path,*args,**kwargs):
        result=original(self,path,*args,**kwargs); store=zarr.open_group(path,mode="r+")
        if target in ("one_ulp","nan"):
            values=store["tables/cells/obsm/neighborhood_group_0001"]
            if target=="nan": values[1,1]=0.
            else: values[0,0]=np.nextafter(values[0,0],np.inf)
        elif target=="axis_metadata":
            store["tables/cells/uns/cellphenotyper/neighborhood_features_json"][()]="{}"
        else:
            attrs=dict(store.attrs["cellphenotyper"]); attrs["neighborhood_features"]={"changed":True}
            store.attrs["cellphenotyper"]=attrs
        return result
    monkeypatch.setattr(sd.SpatialData,"write",changed)
    with pytest.raises(RuntimeError,match="neighbourhood feature"):
        exporter.export_spatialdata(**args,outdir=tmp_path/"bad_readback.zarr")
