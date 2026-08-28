process RUN_HOVERNET_MONUSAC {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/03b_hovernet_monusac/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.hovernet_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.hovernet_memory_gb as int))} GB" }
    time { params.hovernet_time as String }
    maxForks 1

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json)

    output:
    tuple val(sample_id), path("hovernet_${sample_id}/hovernet_cells.json"), emit: cells_json
    tuple val(sample_id), path("hovernet_${sample_id}"), emit: hovernet_dir

    script:
    def scriptPath = "${projectDir}/${params.hovernet_script}"
    def predictionCacheArg = params.hovernet_prediction_cache?.toString()?.trim() ? "--prediction-cache \"${params.hovernet_prediction_cache}\"" : ""
    def memoryBoundWorkers = Math.max(1, Math.floor(Math.min(params.max_memory_gb as double, params.hovernet_memory_gb as double) / 6.0) as int)
    def requestedPostprocWorkers = params.hovernet_postproc_workers as int
    def postprocWorkers = requestedPostprocWorkers > 0 ? Math.max(1, Math.min(requestedPostprocWorkers, memoryBoundWorkers)) : Math.min(task.cpus as int, memoryBoundWorkers)
    """
    set -euo pipefail
    test "${params._resolved_compute_device ?: params.compute_device}" = "gpu" || { echo "HoVer-Net MoNuSAC requires a resolved GPU runtime" >&2; exit 2; }
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    python "${scriptPath}" \
      --image "${crop_tif}" --shift "${shift_json}" --outdir "hovernet_${sample_id}" \
      --repo "${params.hovernet_repo_dir}" --checkpoint "${params.hovernet_monusac_checkpoint}" \
      --target-mpp ${params.hovernet_target_mpp} --default-mpp ${params.hovernet_default_mpp} \
      --gpu ${params.hovernet_gpu} --batch-size ${params.hovernet_batch_size} \
      --inference-workers ${task.cpus} --postproc-workers ${postprocWorkers} \
      --chunk-shape ${params.hovernet_chunk_shape} --tile-shape ${params.hovernet_tile_shape} ${predictionCacheArg}
    """

    stub:
    """
    mkdir -p "hovernet_${sample_id}"
    printf '{"model":"HoVer-Net","checkpoint":"MoNuSAC","cells":[]}' > "hovernet_${sample_id}/hovernet_cells.json"
    """
}
