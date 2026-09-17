process PREPARE_ROI_GEOJSON {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'prepare_roi_geojson', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([ome_tif, image_qc_report]) }
    tag "${sample_id}"
    label 'io_heavy'

    publishDir "${params.outdir_base}/06_roi/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus 1
    memory '2 GB'
    time '1h'

    input:
    tuple val(sample_id), path(ome_tif), path(image_qc_report), val(roi_hint_name), val(roi_hint_b64)

    output:
    tuple val(sample_id), path("${sample_id}.roi.geojson"), emit: roi_geojson
    tuple val(sample_id), path("${sample_id}.roi_qc.json"), emit: roi_qc

    script:
    def roi_script = "${projectDir}/bin/create_full_image_roi_geojson.py"
    def roi_validator_script = "${projectDir}/bin/validate_roi_geojson.py"
    def codeFingerprint = PipelineHelpers.codeFingerprint([roi_script, roi_validator_script])
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] ROI preparation code fingerprint: ${codeFingerprint}"

    ROI_SOURCE_KIND=generated
    if [[ -n "${roi_hint_b64}" ]]; then
      printf '%s' '${roi_hint_b64}' | base64 --decode > "${sample_id}.roi.geojson"
      ROI_SOURCE_KIND=provided
      echo "[INFO] Using provided ROI GeoJSON for ${sample_id}: ${roi_hint_name}"
    else
      python "${roi_script}" \\
        --image "${ome_tif}" \\
        --out "${sample_id}.roi.geojson"
      echo "[INFO] No ROI GeoJSON found for ${sample_id}; generated full-image ROI."
    fi

    python "${roi_validator_script}" \
      --geojson "${sample_id}.roi.geojson" \
      --image "${ome_tif}" \
      --image-report "${image_qc_report}" \
      --report "${sample_id}.roi_qc.json" \
      --sample-id "${sample_id}" \
      --source-kind "\$ROI_SOURCE_KIND" \
      --source-name "${roi_hint_name}" \
      --mode "${params.roi_validation_mode}" \
      --bounds-tolerance-px2 ${params.roi_bounds_tolerance_px2}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    cat > "${sample_id}.roi.geojson" <<'JSON'
    {"type":"FeatureCollection","features":[]}
    JSON
    echo '{"status":"stub"}' > "${sample_id}.roi_qc.json"
    """
}
