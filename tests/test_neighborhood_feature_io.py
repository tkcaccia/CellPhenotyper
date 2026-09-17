"""Portable neighbourhood array axes and bounded scalar/NPY feature reads."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import neighborhood_feature_io as nio


def make_store(root, *, finish=True):
    root.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame({"sample_id":["S"]*5, "cell_id":["001","NA","3","4","5"],
        "cell_uid":["S:"+value for value in ("001","NA","3","4","5")],
        "density":np.array([.1,np.nan,.3,.4,.5],dtype=np.float64),
        "neighbor_count":pd.array([1,2,None,4,5],dtype="Int64")})
    cells.to_parquet(root/"cell_profiles.parquet",index=False)
    cells[nio.KEYS].to_csv(root/"feature_rows.csv",index=False)
    manifest = {"cell_count":5,"feature_blocks":{},"files":{name:nio._sha(root/name)
        for name in ("cell_profiles.parquet","feature_rows.csv")}}
    own = np.arange(15,dtype=np.float32).reshape(5,3); own[3,1]=np.nan
    arrays = {"feature_blocks/context.npy":own,
        "neighborhood_features/r25_context.npy":np.nextafter(own.astype(np.float64),np.inf),
        "neighborhood_features/r50_context.npy":own.astype(np.float64)/3.}
    groups = {"own:context":{"columns":[],"segments":[]},"context":{"columns":[],"segments":[]}}
    for index,(name,values) in enumerate(arrays.items()):
        path=root/name; path.parent.mkdir(exist_ok=True)
        np.save(path,values,allow_pickle=False)
        columns=[f"{('own' if index==0 else 'r'+str(25*index)+'um')}_context:{i}" for i in range(3)]
        group=groups["own:context" if index==0 else "context"]
        group["columns"].extend(columns)
        group["segments"].append({"path":name,"sha256":nio._sha(path),"shape":list(values.shape),
                                  "dtype":str(values.dtype),"columns":columns})
    manifest["feature_blocks"]["context"]={key:value for key,value in groups["own:context"]["segments"][0].items() if key!="columns"}
    summary={"feature_groups":{**{name:record["columns"] for name,record in groups.items()},"density":["density"]},
        "feature_group_definitions":{"own:context":{"aggregation":"identity_no_imputation"},
            "context":{"aggregation":"mean_of_finite_neighbor_values","radii_um":[25.,50.]}}}
    (root/"neighborhood_summary.json").write_text(json.dumps(summary))
    manifest["neighborhoods"]={"summary":"neighborhood_summary.json"}
    manifest["files"]["neighborhood_summary.json"]=nio._sha(root/"neighborhood_summary.json")
    if finish: manifest["neighborhood_feature_store"]=nio.finalize_store(root,groups)
    (root/"cell_profiles_manifest.json").write_text(json.dumps(manifest))
    return cells,arrays,groups,manifest


def alter_store(root, change):
    path=root/nio.STORE_PATH
    record=json.loads(path.read_text()); change(record); path.write_text(json.dumps(record))
    header=root/"cell_profiles_manifest.json"
    manifest=json.loads(header.read_text())
    manifest["neighborhood_feature_store"]["sha256"]=nio._sha(path)
    header.write_text(json.dumps(manifest))


def test_exact_ordered_mixed_reads_precision_missingness_and_original_array_reuse(tmp_path):
    cells,arrays,groups,manifest=make_store(tmp_path)
    before={name:nio._sha(tmp_path/name) for name in arrays}
    reader=nio.FeatureColumns(tmp_path,manifest=manifest)
    assert reader.available_columns==list(cells)+groups["own:context"]["columns"]+groups["context"]["columns"]
    columns=["r50um_context:2","density","own_context:1","r25um_context:0","neighbor_count"]
    expected=np.column_stack([arrays["neighborhood_features/r50_context.npy"][:,2],cells.density,
        arrays["feature_blocks/context.npy"][:,1],arrays["neighborhood_features/r25_context.npy"][:,0],
        cells.neighbor_count.to_numpy(dtype=np.float64,na_value=np.nan)])
    for rows,indices in ((slice(None),np.arange(5)),([4,1,4,0],[4,1,4,0]),(slice(None,None,-2),[4,2,0])):
        result=reader.read(rows,columns)
        assert result.dtype==np.float64
        np.testing.assert_array_equal(result,expected[indices])
    assert reader.read([],columns).shape==(0,5)
    assert reader.read([0,4],[]).shape==(2,0)
    assert reader.groups==groups
    assert reader.group_definitions["context"]["radii_um"]==[25.,50.]
    assert set(reader.source_files)=={*arrays,"cell_profiles_manifest.json","cell_profiles.parquet",
        "feature_rows.csv","neighborhood_summary.json",nio.STORE_PATH}
    assert {name:nio._sha(tmp_path/name) for name in arrays}==before
    assert len(list(tmp_path.rglob("*.npy")))==3  # No copied own-cell feature block.
    reader.recheck()


def test_wide_legacy_profile_remains_available_without_array_store(tmp_path):
    cells,_,_,_=make_store(tmp_path,finish=False)
    reader=nio.FeatureColumns(tmp_path)
    assert reader.store_record is None and reader.groups=={}
    assert reader.available_columns==list(cells)
    np.testing.assert_array_equal(reader.read([4,0],["density"]),[[.5],[.1]])
    with pytest.raises(ValueError,match="not numeric"):
        reader.read([0],["cell_id"])


@pytest.mark.parametrize("change",["row_hash","row_count","float_count","group_order","overlap","duplicate",
    "segment_order","shape","dtype","hash","unsafe_path","unknown_field"])
def test_rehashed_invalid_store_metadata_fails(tmp_path,change):
    make_store(tmp_path)
    def mutate(record):
        group=record["groups"]["context"]; segment=group["segments"][0]
        if change=="row_hash": record["feature_rows_sha256"]="0"*64
        elif change=="row_count": record["cell_count"]=4
        elif change=="float_count": record["cell_count"]=5.
        elif change=="group_order": group["columns"]=group["columns"][::-1]
        elif change=="overlap": group["columns"][0]="density"
        elif change=="duplicate": segment["columns"][1]=segment["columns"][0]
        elif change=="segment_order": group["segments"].reverse()
        elif change=="shape": segment["shape"]=[5,4]
        elif change=="dtype": segment["dtype"]="float32"
        elif change=="hash": segment["sha256"]="0"*64
        elif change=="unsafe_path": segment["path"]="../escape.npy"
        else: segment["unverified_axis"]="extra"
    alter_store(tmp_path,mutate)
    with pytest.raises(ValueError): nio.FeatureColumns(tmp_path)


@pytest.mark.parametrize("kind",["infinity","object","complex","lossy_uint64","lossy_int64","trailing_bytes"])
def test_invalid_npy_payload_fails_even_when_rehashed(tmp_path,kind):
    _,arrays,groups,_=make_store(tmp_path,finish=False)
    segment=groups["context"]["segments"][0]
    path=tmp_path/segment["path"]
    values=arrays[segment["path"]].copy()
    if kind=="infinity": values[-1,-1]=np.inf
    elif kind=="object": values=values.astype(object)
    elif kind=="complex": values=values.astype(complex)
    elif kind in ("lossy_uint64","lossy_int64"):
        values=np.ones(values.shape,dtype="uint64" if kind=="lossy_uint64" else "int64")
        values[-1,-1]=2**53+1
    np.save(path,values,allow_pickle=kind=="object")
    if kind=="trailing_bytes":
        with path.open("ab") as stream: stream.write(b"extra")
    segment.update(sha256=nio._sha(path),dtype=str(values.dtype))
    with pytest.raises(ValueError): nio.finalize_store(tmp_path,groups)
    assert not (tmp_path/nio.STORE_PATH).exists()


def test_extended_float_precision_is_rejected_when_platform_supports_it(tmp_path):
    if np.dtype(np.longdouble).itemsize<=8: pytest.skip("Platform long double is already float64")
    _,arrays,groups,_=make_store(tmp_path,finish=False)
    segment=groups["context"]["segments"][0]; path=tmp_path/segment["path"]
    values=arrays[segment["path"]].astype(np.longdouble); np.save(path,values)
    segment.update(dtype=str(values.dtype),sha256=nio._sha(path))
    with pytest.raises(ValueError,match="numeric matrices"):
        nio.finalize_store(tmp_path,groups)


def test_nullable_integer_precision_check_does_not_skip_observed_values(tmp_path):
    _,_,_,manifest=make_store(tmp_path,finish=False)
    path=tmp_path/"cell_profiles.parquet"
    frame=pd.read_parquet(path)
    frame["neighbor_count"]=pd.array([2**53+1,None,3,4,5],dtype="Int64")
    frame.to_parquet(path,index=False)
    manifest["files"][path.name]=nio._sha(path)
    (tmp_path/"cell_profiles_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="exactly as float64"):
        nio.FeatureColumns(tmp_path).read([0,1],["neighbor_count"])


@pytest.mark.parametrize("change",["reverse_rows","duplicate_uid","literal_alias","extra_row"])
def test_canonical_identity_changes_are_rejected(tmp_path,change):
    make_store(tmp_path)
    path=tmp_path/"feature_rows.csv"
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    if change=="reverse_rows": frame=frame.iloc[::-1]
    elif change=="duplicate_uid": frame.loc[1,"cell_uid"]=frame.loc[0,"cell_uid"]
    elif change=="literal_alias": frame.loc[0,"cell_id"]="1"
    else: frame=pd.concat([frame,frame.iloc[:1]])
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="identit|row count"):
        nio.FeatureColumns(tmp_path)


@pytest.mark.parametrize("rows,columns",[([-1],["density"]),([5],["density"]),([True],["density"]),
    ([.5],["density"]),([[0]],["density"]),([0],["missing"]),([0],["density","density"]),([0],"density")])
def test_invalid_requests_never_wrap_reorder_or_duplicate_features(tmp_path,rows,columns):
    make_store(tmp_path)
    with pytest.raises(ValueError): nio.FeatureColumns(tmp_path).read(rows,columns)


def test_constructor_hashes_each_segment_once_and_reads_requested_blocks(tmp_path,monkeypatch):
    _,arrays,_,_=make_store(tmp_path)
    original=nio._sha; calls=[]
    def counted(path): calls.append(path); return original(path)
    monkeypatch.setattr(nio,"_sha",counted)
    monkeypatch.setattr(nio,"BATCH_ROWS",2)
    monkeypatch.setattr(nio,"ARRAY_BLOCK_BYTES",32)
    reader=nio.FeatureColumns(tmp_path)
    for name in arrays: assert calls.count(tmp_path/name)==1
    # A selected scalar subset uses bounded Arrow batches, never pd.read_parquet.
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Unbounded scalar read"))
    calls.clear()
    result=reader.read([4,0],["density","r25um_context:2"])
    assert result.shape==(2,2) and calls==[]
    path=tmp_path/"neighborhood_features/r25_context.npy"
    values=np.load(path).copy(); values[0,0]=42.; np.save(path,values)
    with pytest.raises(ValueError,match="source changed"): reader.recheck()


def test_escaping_symlinks_and_summary_axis_disagreement_fail(tmp_path):
    root=tmp_path/"profile"; make_store(root)
    external=tmp_path/"external.npy"; np.save(external,np.ones((5,3)))
    link=root/"alias.npy"; link.symlink_to(external)
    alter_store(root,lambda store:store["groups"]["context"]["segments"][0].update(path="alias.npy"))
    with pytest.raises(ValueError,match="escaping"): nio.FeatureColumns(root)
    other=tmp_path/"other"; make_store(other)
    summary_path=other/"neighborhood_summary.json"; summary=json.loads(summary_path.read_text())
    summary["feature_groups"]["context"].reverse(); summary_path.write_text(json.dumps(summary))
    header=other/"cell_profiles_manifest.json"; manifest=json.loads(header.read_text())
    manifest["files"][summary_path.name]=nio._sha(summary_path); header.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="summary feature groups"): nio.FeatureColumns(other)


def test_finalize_never_overwrites_and_duplicate_json_is_rejected(tmp_path):
    _,_,groups,_=make_store(tmp_path)
    with pytest.raises(FileExistsError): nio.finalize_store(tmp_path,groups)
    path=tmp_path/nio.STORE_PATH
    text=path.read_text(); path.write_text(text.replace('"format":', '"format":"duplicate", "format":',1))
    header=tmp_path/"cell_profiles_manifest.json"; manifest=json.loads(header.read_text())
    manifest["neighborhood_feature_store"]["sha256"]=nio._sha(path); header.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Duplicate"): nio.FeatureColumns(tmp_path)


def test_own_group_must_reuse_original_block_and_canonical_hashes_are_required(tmp_path):
    make_store(tmp_path)
    original=tmp_path/"feature_blocks/context.npy"; copied=tmp_path/"copied.npy"
    copied.write_bytes(original.read_bytes())
    alter_store(tmp_path,lambda store:store["groups"]["own:context"]["segments"][0].update(path="copied.npy"))
    with pytest.raises(ValueError,match="unchanged original"): nio.FeatureColumns(tmp_path)
    other=tmp_path/"other"; make_store(other)
    header=other/"cell_profiles_manifest.json"; manifest=json.loads(header.read_text())
    manifest["files"].pop("cell_profiles.parquet"); header.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="require canonical source hashes"): nio.FeatureColumns(other)


def test_reader_never_imports_fitting_or_image_model_runtimes(tmp_path):
    make_store(tmp_path)
    script="""import sys
sys.path.insert(0, sys.argv[1])
from neighborhood_feature_io import FeatureColumns
reader=FeatureColumns(sys.argv[2])
assert reader.read([0], ['density', 'own_context:0']).shape == (1,2)
assert not any(name.split('.')[0] in {'sklearn','torch','fit_cohort_niches','analyze_cell_neighborhoods'} for name in sys.modules)
"""
    result=subprocess.run([sys.executable,"-c",script,str(Path(nio.__file__).parent),str(tmp_path)],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
