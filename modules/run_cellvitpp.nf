process RUN_CELLVITPP {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/03c_cellvitpp/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cellvit_cpus as int)) }
    memory { "${Math.max(8, Math.min(params.max_memory_gb as int, params.cellvit_memory_gb as int))} GB" }
    time { params.cellvit_time as String }
    maxForks 1

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json)

    output:
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cells.json"), emit: cells_json
    tuple val(sample_id), path("cellvit_${sample_id}"), emit: cellvit_dir

    script:
    def scriptPath = "${projectDir}/${params.cellvit_script}"
    def ampFlag = params.cellvit_amp ? '--amp' : ''
    def memoryMb = Math.max(8192, Math.min(params.max_memory_gb as int, params.cellvit_memory_gb as int) * 1024)
    """
    set -euo pipefail
    test "${params._resolved_compute_device ?: params.compute_device}" = "gpu" || { echo "CellViT++ requires a resolved GPU runtime" >&2; exit 2; }
    export CELLVIT_CACHE="${params.cellvit_cache_dir}"
    mkdir -p "\$CELLVIT_CACHE"
    python "${scriptPath}" \
      --image "${crop_tif}" --shift "${shift_json}" --outdir "cellvit_${sample_id}" \
      --executable "${params.cellvit_executable}" --model "${params.cellvit_model}" \
      --taxonomy "${params.cellvit_taxonomy}" --gpu ${params.cellvit_gpu} \
      --batch-size ${params.cellvit_batch_size} --cpus ${task.cpus} --memory-mb ${memoryMb} \
      --ray-workers ${params.cellvit_ray_workers} --ray-worker-cpus ${params.cellvit_ray_worker_cpus} \
      --default-mpp ${params.cellvit_default_mpp} ${ampFlag}
    """

    stub:
    """
    mkdir -p "cellvit_${sample_id}"
    printf '{"pipeline_metadata":{"model":"HIPT"},"cells":[]}' > "cellvit_${sample_id}/cellvit_cells.json"
    """
}
