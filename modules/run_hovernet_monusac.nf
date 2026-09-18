process RUN_HOVERNET_MONUSAC {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_hovernet_monusac', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([crop_tif, shift_json, clean_tissue_mask]) }
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/03b_hovernet_monusac/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'hovernet') }
    memory { TaskRuntime.memory(runtime_plan, 'hovernet') }
    time { params.hovernet_time as String }
    containerOptions { (System.getenv('CELLPHENOTYPER_HOVERNET_CONTAINER_OPTIONS') ?: '').trim() }

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json), path(clean_tissue_mask)
    val(runtime_plan)

    output:
    tuple val(sample_id), path("hovernet_${sample_id}/hovernet_cells.json.gz"), emit: cells_json
    tuple val(sample_id), path("hovernet_${sample_id}"), emit: hovernet_dir

    script:
    def scriptPath = "${projectDir}/${params.hovernet_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/grandqc_mask.py"])
    def predictionCacheArg = params.hovernet_prediction_cache?.toString()?.trim() ? "--prediction-cache \"${params.hovernet_prediction_cache}\"" : ""
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def resolvedHardwareProfile = TaskRuntime.profile(runtime_plan)
    def taskMemoryGb = task.memory.toBytes() / (1024.0d * 1024.0d * 1024.0d)
    def memoryBoundWorkers = Math.max(1, Math.floor(taskMemoryGb / 6.0d) as int)
    def requestedPostprocWorkers = TaskRuntime.setting(runtime_plan, 'hovernet_postproc_workers', params.hovernet_postproc_workers) as int
    def postprocCap = Math.max(1, Math.min(task.cpus as int, memoryBoundWorkers))
    def postprocWorkers = requestedPostprocWorkers > 0 ? Math.min(requestedPostprocWorkers, postprocCap) : postprocCap
    """
    set -euo pipefail
    cleanup_hovernet_transients() {
      # These slide-sized maps and runtime copies are implementation details,
      # never published outputs. Do not leave them in the Nextflow cache after
      # success, failure, or scheduler cancellation.
      rm -rf -- \
        "hovernet_${sample_id}/cache" \
        "hovernet_${sample_id}/input" \
        "hovernet_${sample_id}/input_mask" \
        "hovernet_${sample_id}/raw" \
        "hovernet_${sample_id}/raw_cells" \
        "hovernet_${sample_id}/runtime_cache" \
        "hovernet_${sample_id}/hovernet_runtime"
    }
    trap cleanup_hovernet_transients EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] HoVer-Net wrapper code fingerprint: ${codeFingerprint}"
    test "${resolvedComputeDevice}" = "gpu" || { echo "HoVer-Net MoNuSAC requires a resolved GPU runtime" >&2; exit 2; }
    echo "[INFO] HoVer-Net runtime plan: device=${resolvedComputeDevice}, profile=${resolvedHardwareProfile}, cpus=${task.cpus}, memory_gb=${taskMemoryGb}, postproc_workers=${postprocWorkers}"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    python "${scriptPath}" \
      --image "${crop_tif}" --shift "${shift_json}" --outdir "hovernet_${sample_id}" \
      --clean-tissue-mask "${clean_tissue_mask}" \
      --repo "${params.hovernet_repo_dir}" --checkpoint "${params.hovernet_monusac_checkpoint}" \
      --target-mpp ${params.hovernet_target_mpp} --default-mpp ${params.hovernet_default_mpp} \
      --gpu ${params.hovernet_gpu} --batch-size ${params.hovernet_batch_size} \
      --inference-workers ${task.cpus} --postproc-workers ${postprocWorkers} \
      --chunk-shape ${params.hovernet_chunk_shape} --tile-shape ${params.hovernet_tile_shape} \
      --cache-backend ${params.hovernet_cache_backend} \
      --execution-mode ${params.hovernet_execution_mode} \
      --stream-core-size ${params.hovernet_stream_core_size} \
      --stream-halo ${params.hovernet_stream_halo} \
      --stream-batch-tiles ${params.hovernet_stream_batch_tiles} \
      --tile-jpeg-quality ${params.hovernet_tile_jpeg_quality} \
      ${params.hovernet_export_contours as boolean ? '--export-contours' : ''} ${predictionCacheArg}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "hovernet_${sample_id}"
    printf '{"model":"HoVer-Net","checkpoint":"MoNuSAC","cells":[]}' | gzip -c > "hovernet_${sample_id}/hovernet_cells.json.gz"
    printf '{"model_provenance":{"used_model":false}}\n' > "hovernet_${sample_id}/hovernet_metadata.json"
    """
}
