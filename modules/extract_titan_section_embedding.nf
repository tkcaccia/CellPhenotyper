process EXTRACT_TITAN_SECTION_EMBEDDING {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_heavy'
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/17_titan/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.titan_cpus as int)) }
    memory { "${Math.max(8, Math.min(params.max_memory_gb as int, params.titan_memory_gb as int))} GB" }
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

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}/titan_embedding.csv"), emit: embedding_csv
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}/titan_patch_features.h5"), emit: patch_features
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("titan_${sample_id}_${cluster_variant}"), emit: titan_dir

    script:
    def scriptPath = "${projectDir}/${params.titan_script}"
    def offlineFlag = (params.titan_offline as boolean) ? '--offline' : ''
    """
    set -euo pipefail
    test "${params._resolved_compute_device ?: params.compute_device}" = "gpu" || { echo "TITAN requires a resolved GPU runtime" >&2; exit 2; }
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
    mkdir -p "titan_${sample_id}_${cluster_variant}"
    touch "titan_${sample_id}_${cluster_variant}/titan_embedding.csv"
    touch "titan_${sample_id}_${cluster_variant}/titan_patch_features.h5"
    """
}
