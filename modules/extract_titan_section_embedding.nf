process EXTRACT_TITAN_SECTION_EMBEDDING {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'extract_titan_section_embedding', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([selected_image, selected_mask, selected_shift, selected_summary]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/17_titan/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'titan') }
    memory { TaskRuntime.memory(runtime_plan, 'titan') }
    time { params.titan_time as String }
    containerOptions {
        def modelPath = (params.titan_model ?: '').toString().trim()
        if (!modelPath || !new File(modelPath).isAbsolute()) return ''
        if (workflow.containerEngine == 'singularity') return "-B ${modelPath}:${modelPath}:ro"
        if (workflow.containerEngine == 'docker') return "-v ${modelPath}:${modelPath}:ro"
        ''
    }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(selected_image), path(selected_mask), path(selected_shift), path(selected_summary)
    val(runtime_plan)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}/titan_embedding.csv"), emit: embedding_csv
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}/titan_patch_features.h5"), emit: patch_features
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}"), emit: titan_dir

    script:
    def scriptPath = "${projectDir}/${params.titan_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    codeDigest.update(new File(scriptPath).bytes)
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    def offlineFlag = (params.titan_offline as boolean) ? '--offline' : ''
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def hardwareProfile = TaskRuntime.profile(runtime_plan)
    def memoryBudgetGb = task.memory.toBytes() / (1024.0d * 1024.0d * 1024.0d)
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] TITAN extraction code fingerprint: ${codeFingerprint}"
    echo "[INFO] TITAN runtime: profile=${hardwareProfile}, memory_budget_gb=${memoryBudgetGb}, threads=${task.cpus}"
    test "${resolvedComputeDevice}" = "gpu" || { echo "TITAN requires a resolved GPU runtime" >&2; exit 2; }
    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}
    export NUMEXPR_NUM_THREADS=${task.cpus}
    case "${params.titan_model}" in
      /*) test -d "${params.titan_model}" || { echo "Absolute TITAN model directory is not visible inside the container: ${params.titan_model}" >&2; exit 2; } ;;
    esac
    export HF_HOME="${params.titan_cache_dir}"
    export HUGGINGFACE_HUB_CACHE="${params.titan_cache_dir}/hub"
    mkdir -p "\$HF_HOME" "\$HUGGINGFACE_HUB_CACHE"
    python "${scriptPath}" \
      --image "${selected_image}" --mask "${selected_mask}" \
      --section-summary "${selected_summary}" --shift "${selected_shift}" \
      --sample-id "${sample_id}" --outdir "titan_${sample_id}_${cluster_variant}" \
      --model "${params.titan_model}" --revision "${params.titan_revision}" \
      --target-mpp ${params.titan_target_mpp} --default-mpp ${params.titan_default_mpp} \
      --patch-size ${params.titan_patch_size} --min-tissue-coverage ${params.titan_min_tissue_coverage} \
      --batch-size ${params.titan_batch_size} --gpu ${params.titan_gpu} ${offlineFlag}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "titan_${sample_id}_${cluster_variant}"
    touch "titan_${sample_id}_${cluster_variant}/titan_embedding.csv"
    touch "titan_${sample_id}_${cluster_variant}/titan_patch_features.h5"
    printf '{"model_provenance":{"used_model":false}}\n' > "titan_${sample_id}_${cluster_variant}/titan_metadata.json"
    """
}
