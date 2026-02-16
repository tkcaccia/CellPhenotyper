# How To Use The Pipeline

## Example run (repository data)

```bash
git clone https://github.com/tkcaccia/CellPhenotyper.git
cd CellPhenotyper

nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --image_input Data/ROI.ome.tif \
  --roi_geojson Data/ROI.geojson \
  --outdir_base results_example
```

Final output:

- `results_example/12_cluster_geojson/ROI_grown_mask.geojson`

## Runtime behavior

Automatic by default:

- architecture detection: `host_arch: auto`
- device detection: `compute_device: auto`
- image resolution from tags in `pipeline_paramers.yml`

Current tag mapping:

- arm64 CPU -> `ghcr.io/tkcaccia/cellphenotyper:0.2.0`
- amd64 CPU -> `ghcr.io/tkcaccia/cellphenotyper:0.2.0-amd64`
- amd64 GPU -> `ghcr.io/tkcaccia/cellphenotyper:0.2.0-gpu`

Force architecture when needed:

```bash
nextflow run main.nf \
  -profile singularity \
  -params-file pipeline_paramers.yml \
  --host_arch amd64
```

## Profile rule

Do not use both profiles in one run:

- use `-profile singularity` OR `-profile docker`

## UNI-2 token

```bash
printf 'HF_UNI2="%s"\n' "<your_hf_token>" > tokens.env
```

## Stage window

Set in `pipeline_paramers.yml`:

- `start_point`
- `end_point`

Allowed stages:

`convert`, `stardist`, `tissue_mask`, `cell_assignment`, `cytoplasm`, `uni2`, `kodama`, `clustering`, `cluster_mask`, `grow_tissue`, `cluster_geojson`.
