process RUN_GRANDQC_ARTIFACT_ANALYSIS {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_grandqc_artifact_analysis', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([ome_tif]) }
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/02_grandqc/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { TaskRuntime.cpus(runtime_plan, 'grandqc') }
    memory { TaskRuntime.memory(runtime_plan, 'grandqc') }
    time { params.grandqc_time as String }

    input:
    tuple val(sample_id), path(ome_tif)
    val(runtime_plan)

    output:
    tuple val(sample_id), path("grandqc_${sample_id}"), emit: grandqc_dir
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc_clean_tissue_mask.tif"), emit: clean_tissue_mask
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc_artifact_mask.tif"), emit: artifact_mask
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc.geojson"), optional: true, emit: artifact_geojson

    script:
    def script_path = "${projectDir}/${params.grandqc_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([script_path])
    def cache_dir = (params.grandqc_cache_dir ?: "${baseDir}/.cache/grandqc").toString()
    def bootstrap_flag = params.grandqc_bootstrap_deps ? '--bootstrap-deps' : ''
    def download_flag = params.grandqc_download_models ? '--download-models' : ''
    def geojson_flag = params.grandqc_create_geojson ? '--create-geojson' : ''
    def requestedGrandqcDevice = (params.grandqc_device ?: 'auto').toString().trim().toLowerCase()
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def resolvedHardwareProfile = TaskRuntime.profile(runtime_plan)
    // Resolving a device is independent of optional performance auto-tuning.
    // Passing Python 'auto' could rediscover CUDA/MPS despite a CPU task plan.
    def resolvedGrandqcDevice = requestedGrandqcDevice == 'auto'
      ? (resolvedComputeDevice == 'gpu' ? 'cuda' : 'cpu') : requestedGrandqcDevice
    if (!(resolvedGrandqcDevice in ['cpu', 'cuda', 'mps']))
      throw new IllegalArgumentException('grandqc_device must be auto, cpu, cuda, or mps')
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] GrandQC code fingerprint: ${codeFingerprint}"

    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    echo "[INFO] GrandQC runtime tune: profile=${resolvedHardwareProfile}, requested_device=${requestedGrandqcDevice}, resolved_device=${resolvedGrandqcDevice}, tissue_patch_size=${params.grandqc_patch_size}, tissue_probability_threshold=${params.grandqc_tissue_probability_threshold}, clean_tissue_policy=${params.grandqc_clean_tissue_policy}, artifact_tile_size=${params.grandqc_artifact_tile_size}, overlap=${params.grandqc_artifact_overlap_fraction}"

    python "${script_path}" \
      --image "${ome_tif}" \
      --outdir "grandqc_${sample_id}" \
      --sample-id "${sample_id}" \
      --device "${resolvedGrandqcDevice}" \
      --cache-dir "${cache_dir}" \
      --default-source-mpp ${params.grandqc_default_source_mpp} \
      --artifact-mpp-model ${params.grandqc_artifact_mpp_model} \
      --tissue-mpp-model ${params.grandqc_tissue_mpp_model} \
      --tissue-probability-threshold ${params.grandqc_tissue_probability_threshold} \
      --clean-tissue-policy ${params.grandqc_clean_tissue_policy} \
      --patch-size ${params.grandqc_patch_size} \
      --artifact-tile-size ${params.grandqc_artifact_tile_size} \
      --artifact-overlap-fraction ${params.grandqc_artifact_overlap_fraction} \
      --overlay-factor ${params.grandqc_overlay_factor} \
      --preview-max-side ${params.grandqc_preview_max_side} \
      ${bootstrap_flag} \
      ${download_flag} \
      ${geojson_flag}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "grandqc_${sample_id}"
    touch "grandqc_${sample_id}/${sample_id}_grandqc_clean_tissue_mask.tif"
    touch "grandqc_${sample_id}/${sample_id}_grandqc_artifact_mask.tif"
    printf '{"model_provenance":{"tissue":{"used_model":false},"artifact":{"used_model":false}}}\n' > "grandqc_${sample_id}/${sample_id}_grandqc_summary.json"
    """
}
