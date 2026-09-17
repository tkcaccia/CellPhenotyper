process RUN_CELLVITPP {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_cellvitpp', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([crop_tif, shift_json, clean_tissue_mask, resolution_json]) }
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/03c_cellvitpp/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'cellvit') }
    memory { TaskRuntime.memory(runtime_plan, 'cellvit') }
    time { params.cellvit_time as String }

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json), path(clean_tissue_mask), path(resolution_json)
    val(runtime_plan)

    output:
    tuple val(sample_id), path("cellvit_${sample_id}/cellvit_cells.json"), emit: cells_json
    tuple val(sample_id), path("cellvit_${sample_id}"), emit: cellvit_dir

    script:
    def scriptPath = "${projectDir}/${params.cellvit_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/grandqc_mask.py", "${projectDir}/bin/cellvit_embeddings.py"])
    def ampFlag = params.cellvit_amp ? '--amp' : ''
    def embeddingFlag = params.cellvit_export_embeddings ? '--export-embeddings' : ''
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def resolvedHardwareProfile = TaskRuntime.profile(runtime_plan)
    def memoryMb = task.memory ? Math.max(1, task.memory.toMega() as int) : Math.max(1, (runtime_plan.memory_budget_gb as double) * 1024) as int
    def configuredRayWorkers = params.containsKey('cellvit_ray_workers') ? params.cellvit_ray_workers : 1
    def configuredRayWorkerCpus = params.containsKey('cellvit_ray_worker_cpus') ? params.cellvit_ray_worker_cpus : 0
    def rayWorkers = Math.max(1, Math.min(task.cpus as int, TaskRuntime.setting(runtime_plan, 'cellvit_ray_workers', configuredRayWorkers) as int))
    def workerCpuCap = Math.max(1, Math.floor((task.cpus as int) / (rayWorkers as double)) as int)
    def requestedWorkerCpus = TaskRuntime.setting(runtime_plan, 'cellvit_ray_worker_cpus', configuredRayWorkerCpus) as int
    def rayWorkerCpus = requestedWorkerCpus > 0 ? Math.min(workerCpuCap, requestedWorkerCpus) : workerCpuCap
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] CellViT++ wrapper code fingerprint: ${codeFingerprint}"
    test "${resolvedComputeDevice}" = "gpu" || { echo "CellViT++ requires a resolved GPU runtime" >&2; exit 2; }
    echo "[INFO] CellViT++ runtime plan: device=${resolvedComputeDevice}, profile=${resolvedHardwareProfile}, cpus=${task.cpus}, memory_mb=${memoryMb}, ray_workers=${rayWorkers}, ray_worker_cpus=${rayWorkerCpus}"
    export CELLVIT_CACHE="${params.cellvit_cache_dir}"
    mkdir -p "\$CELLVIT_CACHE"
    python "${scriptPath}" \
      --image "${crop_tif}" --shift "${shift_json}" --outdir "cellvit_${sample_id}" \
      --clean-tissue-mask "${clean_tissue_mask}" --resolution-json "${resolution_json}" \
      --executable "${params.cellvit_executable}" --model "${params.cellvit_model}" \
      --taxonomy "${params.cellvit_taxonomy}" --gpu ${params.cellvit_gpu} \
      --batch-size ${params.cellvit_batch_size} --cpus ${task.cpus} --memory-mb ${memoryMb} \
      --ray-workers ${rayWorkers} --ray-worker-cpus ${rayWorkerCpus} \
      --default-mpp ${params.cellvit_default_mpp} ${ampFlag} ${embeddingFlag}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "cellvit_${sample_id}"
    printf '{"pipeline_metadata":{"model":"HIPT"},"cells":[]}' > "cellvit_${sample_id}/cellvit_cells.json"
    printf '{"model_provenance":{"used_model":false}}\n' > "cellvit_${sample_id}/cellvit_metadata.json"
    """
}
