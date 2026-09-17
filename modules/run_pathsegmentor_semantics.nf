process RUN_PATHSEGMENTOR_SEMANTICS {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'run_pathsegmentor_semantics', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, tissue_mask_tif, resolution_json, pathsegmentor_repo, pathsegmentor_config, pathsegmentor_checkpoint, prompt_panel]) }
    tag "${sample_id}:pathsegmentor"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/09b_pathsegmentor/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.pathsegmentor_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.pathsegmentor_memory_gb as int))} GB" }
    time { params.pathsegmentor_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(tissue_mask_tif), path(resolution_json)
    path(pathsegmentor_repo, stageAs: 'PathSegmentor')
    path(pathsegmentor_config, stageAs: 'pathsegmentor_config.yaml')
    path(pathsegmentor_checkpoint, stageAs: 'pathsegmentor_checkpoint.pt')
    path(prompt_panel, stageAs: 'pathsegmentor_prompts.json')
    val(runtime_plan)

    output:
    tuple val(sample_id), path("pathsegmentor_${sample_id}"), emit: semantic_dir
    tuple val(sample_id), path("pathsegmentor_${sample_id}/pathsegmentor_probabilities.ome.tif"), path("pathsegmentor_${sample_id}/pathsegmentor_manifest.json"), emit: semantic_bundle

    script:
    def scriptPath = "${projectDir}/${params.pathsegmentor_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    [scriptPath, "${projectDir}/bin/extract_uni2_embeddings.py", "${projectDir}/resources/pathsegmentor_breast_prompts.json"].each {
        codeDigest.update(new File(it).bytes)
    }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] PathSegmentor adapter code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --image "${image_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --resolution-json "${resolution_json}" \
      --repo "${pathsegmentor_repo}" \
      --config "${pathsegmentor_config}" \
      --checkpoint "${pathsegmentor_checkpoint}" \
      --prompt-panel "${prompt_panel}" \
      --outdir "pathsegmentor_${sample_id}" \
      --target-mpp ${params.pathsegmentor_target_mpp} \
      --output-mpp ${params.pathsegmentor_output_mpp} \
      --tile-size ${params.pathsegmentor_tile_size} \
      --overlap-fraction ${params.pathsegmentor_overlap_fraction} \
      --min-tissue-fraction ${params.pathsegmentor_min_tissue_fraction} \
      --device cuda
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "pathsegmentor_${sample_id}"
    touch "pathsegmentor_${sample_id}/pathsegmentor_probabilities.ome.tif"
    touch "pathsegmentor_${sample_id}/pathsegmentor_tile_scores.csv.gz"
    printf '{"schema_version":"1.0.0","stub":true,"scientific_role":"supervised_semantic_evidence_not_unsupervised_cluster_identity"}\n' > "pathsegmentor_${sample_id}/pathsegmentor_manifest.json"
    """
}
