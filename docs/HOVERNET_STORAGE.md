# HoVer-Net transient storage

HoVer-Net's upstream whole-slide route creates a four-channel float32
prediction map and an int32 instance map over the processed slide rectangle.
These maps are temporary and can be much larger than the compressed input
OME-TIFF. CellPhenotyper defaults to `hovernet_cache_backend=zarr`, which keeps
the same float32 and int32 values in bounded 1,024-pixel, losslessly
Blosc/Zstd-compressed Zarr chunks. Each prediction block is assembled and
written once, capping its extra buffer at 16 MiB rather than materializing a
complete 8,192-pixel HoVer-Net processing chunk.
`hovernet_cache_backend=numpy` retains the uncompressed
upstream representation for compatibility comparisons and explicit
`hovernet_prediction_cache` recovery.

The task removes the prediction map, instance map, normalized input, mask copy,
runtime cache and isolated upstream source copy whenever it exits, including
after an error or cancellation. These files are never published results.

## Bounded real-tissue validation (2026-09-18)

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
entropy; this bounded benchmark does not by itself establish a whole-cohort
runtime distribution.
