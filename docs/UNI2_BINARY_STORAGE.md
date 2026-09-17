# Portable binary UNI-2 intermediates

This option changes storage and loading, not the encoder, observation unit,
physical context, KODAMA ncomp, or requested tissue K. CSV remains the compatibility
default during end-to-end validation. Historical results are never migrated
implicitly.

```yaml
uni2_embedding_storage: binary  # csv | binary
uni2_rows_per_csv: 10000        # historical name: row limit per shard in either format
```

Single and shared/paired extraction pass storage explicitly. Each primary output
declares its actual family (tile, nuclei, cyto, inner_square); the paired secondary
is inner_square. R/KODAMA, canonical cell profiles and raw-grid hierarchy inputs
accept binary shards. The hierarchy producer's separate NPY bundle remains supported.

## Format and integrity

| File suffix | Content |
|---|---|
| `.features.bin` | Uncompressed little-endian C-order numerical matrix |
| `.rows.csv` | Small metadata table, including exact string cell/tile IDs |
| `.embedding.json` | Versioned shape, dtype, ordered schemas, family, payload names, byte sizes and SHA256 hashes |

Format `cellphenotyper_uni2_binary`, version `1.0.0`, supports float32 and float64
without implicit float16 quantization. Normal extraction preserves encoder float32
output. Float64 can preserve parsed historical CSV values during explicit conversion;
that does not establish equality with the original encoder tensor.

Feature writes/scans are block-bounded. Python can memory-map features and stream
metadata. R reads bounded binary blocks and retains variance-selected columns as
R doubles. The rest of PCA/KODAMA is not thereby out-of-core. No new Python storage
dependency or HDF5/Zarr R bridge is required; R uses data.table, jsonlite and digest.

The manifest is written last; the shard writer does not replace existing payloads.
Readers check shape, byte size, hashes, finiteness, IDs, schema and payload ownership.
Mixed CSV/binary sources, orphan payloads and incomplete bundles fail. Extraction
completion receipts bind grid receipts and every payload. Changing storage invalidates
cache compatibility: use a fresh output directory, not a mixture in existing results.

Storage integrity is not encoder/source provenance. A historical conversion cannot
claim newly verified weights, image calibration or cell identity. Python profiles
retain independent exact-source checks. When root/grid extraction receipts exist,
R requires their complete chain and verifies exact inventories, SHA256/size agreement,
representation family, observation type and full row coverage. Receipt-free historical
conversions remain explicitly legacy-unverified. R compares the declared cache-contract
SHA across receipts; it does not reconstruct Python canonical JSON or independently
verify the source image/encoder. Missing modalities and cell/grid distinctions remain
enforced.

CSV and binary readers now use matching round-trip parsing for physical metadata;
missing required physical settings cannot qualify a profile for reference use.
Older already-exported references may contain strings from the former CSV float
rounding. If that causes a definition mismatch, re-export both profiles/reference
definitions from cached features; do not silently relabel them as compatible or
rerun the image encoder solely to change metadata. No existing reference is changed
automatically.

## Default CSV identity checks

The R loader now applies the same no-loss population contract to the default
CSV route. IDs are parsed as literal strings (including leading zeros); empty,
whitespace/control, duplicate, foreign or missing IDs fail instead of being
trimmed, deduplicated or intersected. Every explicitly selected family must
cover the complete annotation population. Unselected modalities remain optional;
canonical profile exports still retain cells with explicitly missing modalities.

Every CSV shard must have the same ordered schema. All feature values must be
numeric and finite **before** variance selection, including discarded columns.
Parser warnings and damaged gzip streams fail. Explicit `x/y` metadata must
match annotation coordinates; rounded/clipped extraction `cx/cy` is not silently
equated to a floating-point centroid. A future extraction-geometry contract is
needed to authenticate those centers independently.

`embedding_input_provenance.json` records schema `cellphenotyper_uni2_csv_input`
version `1.0.0`, per-shard read hashes/byte counts, annotation identity, selected
features and exact source-to-analysis row mappings. PCA/KODAMA propagate this
receipt. These read hashes do **not** authenticate the original encoder or
upgrade historical calibration. Receipt-free historical RData can still run
when its IDs, selected populations and finite features are valid; malformed
inputs now fail, and no provenance is invented for valid legacy files.

The stricter reader preserved all 16,898 real grid observations, coordinates,
and the exact 100 selected features per family against the preserved CSV
baseline. Details and a reproducible verifier are in the spatial-atlas audit.
This changes input validation, not KODAMA ncomp, PCA dimensions or tissue K.

## Real saved-feature comparison

Both historical crop families contain 16,898 grid observations, 1,536 dimensions
and 93 shards. No expert annotation, image or learned model was used. All source
hashes were unchanged afterward.

| Family | CSV repeated read | Verified binary32 repeated read | CSV bytes | Binary32 bytes |
|---|---:|---:|---:|---:|
| Tile | 13.55–14.47 s | 0.87–1.29 s | 123,295,807 | 112,203,934 |
| Inner square | 12.65–14.35 s | 0.36–0.64 s | 123,709,704 | 112,204,678 |

These are three rotated-order local warm-cache reads. Binary checksums and finite
scans are included. This is not a controlled cold-cache, GPU-inference or whole-pipeline
speed measurement; competing CPU work can affect timings. Binary sizes include row
metadata/manifests. Binary is not guaranteed smaller than compressed CSV on every input.

Float64 read-back exactly preserved Python round-trip parsed CSV values. Float32
read-back exactly preserved their float32 casts; maximum absolute differences from
historical CSV doubles were 1.04e-7 (tile) and 1.96e-7 (inner square). Small differences
alone do not prove clustering equivalence or biological accuracy.

Evidence: `/tmp/cellphenotyper_uni2_binary_20260905.5pWDEo/real_comparison/verification.json`.
The helper is frozen and source hashes/software versions recorded. Reproduce with
`audits/spatial_atlas_20260904/benchmark_uni2_storage.py`, explicit `--tile`, `--inner`
and a fresh `--outdir`. R-loader and model-free producer/Nextflow evidence are recorded
separately in the implementation ledger. Independent histological/marker validation,
updated GPU inference and container-runtime verification remain required work.
