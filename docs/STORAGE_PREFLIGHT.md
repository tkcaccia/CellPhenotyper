# Storage Preflight

CellPhenotyper estimates disk demand before scheduling image-processing stages. The purpose is to reject a run that is likely to exhaust a filesystem and to make restart risk visible before expensive GPU work begins.

## What Is Estimated

`00_execution/storage_preflight.json` records:

- source file size, probed level-0 dimensions, metadata probe method and estimated active RGB bytes for every input;
- an ROI bounding-box adjustment when a matching GeoJSON is available before execution;
- separate stage-window coefficients for durable published output and retained/transient Nextflow work;
- an explicit HoVer-Net peak-disk model: the default streaming route budgets one bounded tile batch plus the incrementally written final result, while `hovernet_execution_mode=wsi` budgets the complete float32 prediction and int32 instance maps;
- the effect of `publish_dir_mode=copy` versus `rellink`;
- conservative remaining allowances for StarDist, GrandQC, Hugging Face, TITAN, PathoFMPred and Singularity caches when relevant;
- expected incremental demand and a restart worst case;
- paths aggregated by filesystem device, so output, work and caches on one volume are not evaluated independently against the same free bytes;
- current available capacity, required post-run reserve and expected/worst-case headroom.

The stage coefficients are explicit in the JSON report. They are conservative engineering allowances derived from the type and number of retained rasters, tables, embeddings and disk-backed accumulators. They are not measurements of the future run.

## Decisions

`storage_preflight_mode=fail` is the default.

- `pass`: expected and restart-worst-case demand fit while preserving `storage_min_free_gib`.
- `warning`: expected demand fits, but the restart-worst-case demand does not. The run continues and the warning appears in the review-first report.
- `fail`: expected demand plus the reserve does not fit. In `fail` mode the pipeline stops before image tasks are submitted. In `warn` mode it records the failure-level risk and continues.

Use `storage_preflight_mode=warn` only after an operator has accepted the capacity risk. `off` removes the guard and should be limited to controlled testing.

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `storage_preflight_mode` | `fail` | `off`, `warn`, or `fail`. |
| `storage_min_free_gib` | `20.0` | Free space that must remain after estimated demand. |
| `storage_safety_factor` | `1.25` | Multiplier applied to expected output and work demand. |
| `storage_restart_duplication_factor` | `1.0` | Additional retained-work allowance for the restart scenario. |
| `storage_source_expansion_factor` | `8.0` | Source-size fallback used only when image dimensions cannot be probed. |

## Responding To A Failure

1. Read every filesystem row in `00_execution/storage_preflight.json`; freeing space on the wrong volume will not help.
2. Put `-w /large_scratch/work` on a scratch filesystem with enough capacity.
3. Put `--outdir_base /durable/results` on durable storage. Keep `publish_dir_mode=copy` if the work cache will be removed.
4. Reuse shared model caches rather than duplicating them per checkout.
5. Keep `uni2_save_tiles=false` unless tile images are required for a bounded QC sample.
6. Supply the intended ROI before execution when analysis is genuinely restricted to that ROI.
7. Remove old work only after published outputs and checksums have been verified.

## Limitations

The check does not reserve space. Concurrent jobs, user quotas, sparse-file behavior, compression ratio and filesystem metadata/inode exhaustion can change real capacity. CZI files without a readable host-side dimension probe use the conservative source-expansion fallback. Docker engine storage is not portably inspectable from Nextflow and is not included unless represented by a supplied cache path. The report is therefore a fail-fast engineering control, not a guarantee that a run cannot exhaust storage.
