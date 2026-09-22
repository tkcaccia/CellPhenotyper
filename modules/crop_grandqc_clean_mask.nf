process CROP_GRANDQC_CLEAN_MASK {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'crop_grandqc_clean_mask', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([clean_tissue_mask, artifact_mask, shift_json, roi_crop_geojson]) }
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/04_tissue_mask/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus 1
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.grandqc_crop_mask_memory_gb as int))} GB" }
    time { params.grandqc_crop_mask_time as String }

    input:
    tuple val(sample_id), path(clean_tissue_mask), path(artifact_mask), path(shift_json), path(roi_crop_geojson)

    output:
    tuple val(sample_id), path("${sample_id}_tissue_mask.tif"), emit: tissue_mask
    tuple val(sample_id), path("${sample_id}_grandqc_artifact_candidates.tif"), emit: artifact_candidates
    tuple val(sample_id), path("${sample_id}_grandqc_crop_mask_summary.json"), emit: summary_json

    script:
    def scriptPath = "${projectDir}/${params.grandqc_crop_mask_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/prepare_analysis_crop.py"])
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] GrandQC crop-mask code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --mask "${clean_tissue_mask}" --shift "${shift_json}" \
      --artifact-mask "${artifact_mask}" \
      --roi "${roi_crop_geojson}" \
      --output "${sample_id}_tissue_mask.tif" \
      --artifact-output "${sample_id}_grandqc_artifact_candidates.tif" \
      --summary "${sample_id}_grandqc_crop_mask_summary.json"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_tissue_mask.tif"
    touch "${sample_id}_grandqc_artifact_candidates.tif"
    printf '{"clean_tissue_fraction":1.0}' > "${sample_id}_grandqc_crop_mask_summary.json"
    """
}
