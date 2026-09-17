process PREPARE_ANALYSIS_CROP {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'prepare_analysis_crop', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([ome_tif, roi_geojson, clean_tissue_mask, resolution_json]) }
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/03_stardist/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "prepared_crop"
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.prepare_crop_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.prepare_crop_memory_gb as int))} GB" }
    time { params.prepare_crop_time as String }

    input:
    tuple val(sample_id), path(ome_tif), path(roi_geojson), path(clean_tissue_mask), path(resolution_json)

    output:
    tuple val(sample_id), path("prepared_crop/crop_roi.tif"), emit: crop_roi
    tuple val(sample_id), path("prepared_crop/roi_all_crop.geojson"), emit: roi_crop_geojson
    tuple val(sample_id), path("prepared_crop/shift.json"), emit: shift_json
    tuple val(sample_id), path("prepared_crop"), emit: crop_dir

    script:
    def scriptPath = "${projectDir}/${params.prepare_crop_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/profile_cell_morphology.py"])
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Shared analysis crop code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --image "${ome_tif}" --roi "${roi_geojson}" \
      --clean-tissue-mask "${clean_tissue_mask}" --outdir prepared_crop \
      --resolution-json "${resolution_json}" \
      --pad ${params.stardist_crop_pad}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p prepared_crop
    touch prepared_crop/crop_roi.tif
    printf '{"type":"FeatureCollection","features":[]}' > prepared_crop/roi_all_crop.geojson
    printf '{"crop_size":{"width":1,"height":1},"full_size":{"width":1,"height":1},"crop_bbox_xyxy":{"x0":0,"y0":0,"x1":1,"y1":1}}' > prepared_crop/shift.json
    """
}
