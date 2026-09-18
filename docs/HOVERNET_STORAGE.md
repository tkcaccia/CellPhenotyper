# HoVer-Net transient storage

Official HoVer-Net whole-slide inference creates a four-channel float32
prediction map and an int32 instance map over the complete processed slide
rectangle. At 0.25 micrometres per pixel these arrays can require more than
100 GB even when the input WSI is compressed.

CellPhenotyper therefore defaults to
`hovernet_execution_mode=streaming_tiles`. The crop is sampled at the same
target MPP as overlapping JPEG tiles. Only tiles whose ownership core
intersects the GrandQC clean-tissue mask are submitted. A 256-pixel halo
provides model context, while the half-open 4,096-pixel core owns detections by
centroid so overlap detections are retained exactly once. At most
`hovernet_stream_batch_tiles=64` normalized tiles exist at a time. The pinned
upstream model and tile post-processor run unchanged, but its unused MAT files,
RGB overlays and raw prediction exports are disabled. Per-tile JSON is gzip
compressed, merged into the final stream, and deleted before the next batch.
No cohort-sized collection of intermediate tile records is retained. The
result is
`hovernet_cells.json.gz`; redundant HoVer-Net contours are omitted by default
because the default broad-pair consensus obtains canonical geometry from
CellViT++. Set `hovernet_export_contours=true` only for a workflow that uses
HoVer-Net geometry directly.

`hovernet_execution_mode=wsi` retains the official slide-wide route for
compatibility. In that mode `hovernet_cache_backend=zarr` keeps the exact
float32 and int32 values in 1,024-pixel, losslessly Blosc/Zstd-compressed
chunks. `numpy` selects the uncompressed upstream representation and is
required for explicit `hovernet_prediction_cache` recovery.

The task removes tile batches, per-tile JSON, WSI prediction/instance maps,
normalized input, mask copies, raw upstream output, runtime caches and the
isolated upstream source copy whenever it exits, including after an error or
cancellation. These files are never published results.

## Lossless WSI-cache validation (2026-09-18)

The two backends were run independently on the same central 4,096 x 4,096
native-pixel patch from prostate sample `SAPC0052_4255_14` on Chiamaka. Source
resolution was 0.17203940921 micrometres per pixel; HoVer-Net target resolution
was 0.25 micrometres per pixel. Both runs used the same MoNuSAC checkpoint,
batch size 8, one post-processing worker, 8,192-pixel chunk setting and
2,048-pixel post-processing tiles.

| Measure | NumPy | Lossless Zarr |
|---|---:|---:|
| Cells | 1,369 | 1,369 |
| Elapsed wall time | 30 s | 32 s |
| Cache logical bytes | 181,065,572 | 94,630,195 |
| Cache allocated bytes | 172,281,856 | 94,666,752 |
| Cache allocated-space reduction | - | 45.05% |

After removing run-specific metadata, the complete ordered cell records
(centroids, contours, class IDs/names and type probabilities) were byte-for-byte
identical. Their canonical JSON SHA-256 was
`1581f7e60a04e42d15f96b93dcb8677ed38253877f464c83a823f092a4ddabf0` for both
backends. This validates numerical equivalence on the tested patch. Compression
ratios on other slides will depend on tissue coverage and prediction-map
entropy; this bounded benchmark does not validate the new overlap-tiled route.
The tiled route requires a matched WSI-versus-tile boundary-equivalence audit
before release-quality biological claims.
