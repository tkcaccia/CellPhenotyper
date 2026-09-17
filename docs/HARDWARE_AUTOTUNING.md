# Hardware auto-tuning

CellPhenotyper builds a resource plan at workflow startup when `hardware_auto=true`.
The plan uses the CPU allowance exposed by a supported scheduler or the local
runtime, the effective host/container RAM limit, GPU inventory, and NVIDIA VRAM.
It is written to `00_execution/hardware_plan.json` so every run records the
allocation used for each stage.

## Safety boundary

Auto-tuning changes execution settings only. It does not change the scientific
route, detector set, model, MPP, ROI, feature definition, KODAMA dimensions,
landmark count, clustering method, or requested cluster count. In particular, a
consensus cell-detection run fails when no compatible GPU is available instead
of silently switching to StarDist-only cells.

## Global policy

- `max_cpus=auto` uses the scheduler CPU allocation when present. On an
  unscheduled host it preserves up to two CPUs for the operating system.
- `max_memory_gb=auto` uses the smallest visible physical/cgroup RAM limit and
  preserves up to 15% for the operating system.
- `hardware_profile=auto` resolves to `conservative`, `balanced`, or
  `aggressive` from usable CPU, RAM, and GPU capacity.
- Per-stage `*_cpus` and `*_memory_gb` values are ceilings. The policy never
  exceeds them or the global envelope.
- Serial/light processes receive small allocations so independent ready tasks
  can run concurrently. Parallel WSI and model stages receive a larger share.
- `max_parallel_tasks=auto` sizes the executor queue from usable CPU count;
  Nextflow still gates launches by the aggregate CPU and memory envelopes.

## Stage decisions

| Stage group | Automatically resolved settings |
|---|---|
| Input conversion and crops | task CPUs, writer workers, RAM |
| GrandQC | CPU threads, RAM, CPU/CUDA device; model geometry stays fixed |
| StarDist | CPU/RAM allocation and live-free-VRAM-aware large-image block ladder with OOM fallback |
| HoVer-Net MoNuSAC | inference workers, RAM-bounded post-processing workers, batch size from live free VRAM |
| CellViT++ | task/Ray CPUs, RAM, batch size from live free VRAM |
| Consensus and TMA | right-sized CPU/RAM allocation |
| GigaTIME and marker export | CPU/RAM allocation, live RAM/VRAM batch plan, OOM backoff, writer workers |
| ROI assignment and cytoplasm | CPU worker count and RAM |
| Grid generation and UNI-2 | CPU/RAM allocation, Torch threads, batch size from the assigned GPU's live free VRAM, OOM batch halving |
| KODAMA and clustering | allocated R cores and RAM; raw matrices are released after PCA |
| Cluster masks and growth | CPU/RAM allocation and bounded writer/processing workers |
| MedSAM refinement | CUDA/CPU selection, CPU writer workers, RAM, and GPU admission; refinement geometry stays fixed |
| TITAN | CPU/RAM allocation and batch size from live free VRAM |
| GeoJSON and PathoFMPred | right-sized CPU/RAM allocation |

All GPU-capable stages first enter the host-wide GPU admission layer. It chooses
a device with enough live free VRAM, locks task and VRAM tokens, and exports
`CUDA_VISIBLE_DEVICES`. StarDist, HoVer-Net, CellViT++, UNI-2, GigaTIME, and
TITAN then inspect that assigned device rather than assuming GPU 0 or using only
nominal VRAM. This permits larger batches on idle high-memory GPUs while
reducing them on smaller or partially occupied GPUs.

## Overrides

Set `hardware_auto=false` to retain explicit stage allocations, or set
`hardware_profile` to `conservative`, `balanced`, or `aggressive` to override
profile selection while retaining hard resource ceilings. A positive explicit
model batch size disables that tool's automatic batch selection. OOM-aware
tools may still retry with a smaller batch to complete safely.
