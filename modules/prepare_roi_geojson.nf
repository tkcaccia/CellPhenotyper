process PREPARE_ROI_GEOJSON {
    tag "${sample_id}"
    label 'io_heavy'

    publishDir "${params.outdir_base}/04_roi/${sample_id}", mode: 'copy', overwrite: true

    cpus 1
    memory '2 GB'
    time '1h'

    input:
    tuple val(sample_id), path(ome_tif), val(roi_hint_name), val(roi_hint_b64)

    output:
    tuple val(sample_id), path("${sample_id}.roi.geojson"), emit: roi_geojson

    script:
    def roi_script = "${projectDir}/bin/create_full_image_roi_geojson.py"
    """
    set -euo pipefail

    if [[ -n "${roi_hint_b64}" ]]; then
      printf '%s' '${roi_hint_b64}' | base64 --decode > "${sample_id}.roi.geojson"
      echo "[INFO] Using provided ROI GeoJSON for ${sample_id}: ${roi_hint_name}"
    else
      python "${roi_script}" \\
        --image "${ome_tif}" \\
        --out "${sample_id}.roi.geojson"
      echo "[INFO] No ROI GeoJSON found for ${sample_id}; generated full-image ROI."
    fi
    """

    stub:
    """
    cat > "${sample_id}.roi.geojson" <<'JSON'
    {"type":"FeatureCollection","features":[]}
    JSON
    """
}
