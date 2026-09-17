process ROI_GEOJSON_TO_MASK {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'roi_geojson_to_mask', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([roi_geojson, reference_tif]) }
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/06_roi/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.input_roi_mask_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.input_roi_mask_memory_gb as int))} GB" }
    time { params.input_roi_mask_time as String }

    input:
    tuple val(sample_id), path(roi_geojson), path(reference_tif)

    output:
    tuple val(sample_id), path("${sample_id}_input_roi_mask.tif"), emit: roi_mask
    tuple val(sample_id), path("${sample_id}_input_roi_mask_preview.png"), emit: roi_mask_preview
    tuple val(sample_id), path("${sample_id}_input_roi_mask_labels.json"), emit: roi_mask_labels

    script:
    def rasterize_script = "${projectDir}/${params.geojson_to_mask_script}"
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"

    python "${rasterize_script}" \
      --geojson "${roi_geojson}" \
      --reference "${reference_tif}" \
      --reference-page 0 \
      --out "${sample_id}_input_roi_mask.tif" \
      --label-mode "${params.input_roi_mask_label_mode}" \
      --value-prop "${params.input_roi_mask_value_prop}" \
      --annotation-props "${params.input_roi_mask_annotation_props}" \
      --default-value ${params.input_roi_mask_default_value} \
      --fill-value 0 \
      --compression "${params.input_roi_mask_compression}" \
      --preview "${sample_id}_input_roi_mask_preview.png" \
      --preview-factor ${params.input_roi_mask_preview_factor} \
      --preview-threshold-mb ${params.input_roi_mask_preview_threshold_mb} \
      --preview-alpha ${params.input_roi_mask_preview_alpha} \
      --label-map-out "${sample_id}_input_roi_mask_labels.json"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_input_roi_mask.tif"
    touch "${sample_id}_input_roi_mask_preview.png"
    touch "${sample_id}_input_roi_mask_labels.json"
    """
}
