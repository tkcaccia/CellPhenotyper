process REVIEW_CELL_SEGMENTATION {
    tag "${sample_id}:agreement-review"
    label 'compute_medium'
    publishDir "${params.outdir_base}/03e_cell_segmentation_review/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cell_segmentation_review_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.cell_segmentation_review_memory_gb as int))} GB" }
    time { params.cell_segmentation_review_time as String }
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(sample_id), path(crop_tif), path(cells_json), path(cellvit_metadata), path(stardist_labels), path(stardist_frame)

    output:
    tuple val(sample_id), path("segmentation_review_${sample_id}/segmentation_qc.json"), emit: qc_json
    tuple val(sample_id), path("segmentation_review_${sample_id}/segmentation_qc.csv"), emit: qc_csv
    tuple val(sample_id), path("segmentation_review_${sample_id}/segmentation_qc.geojson"), emit: qc_geojson
    tuple val(sample_id), path("segmentation_review_${sample_id}/original_labels.tif"), emit: original_labels
    tuple val(sample_id), path("segmentation_review_${sample_id}/tuned_candidate_labels.tif"), emit: candidate_labels
    tuple val(sample_id), path("segmentation_review_${sample_id}"), emit: review_dir

    script:
    def scriptPath = "${projectDir}/bin/tune_cell_segmentation.py"
    def codeFingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/cell_geometry_refinement.py"])
    def partialImageFlag = params.cell_segmentation_review_allow_partial_image_cells ? '--allow-partial-image-cells' : ''
    def autoAdjustFlag = params.cell_segmentation_review_auto_adjust ? '--auto-adjust' : ''
    def quarantineFlag = params.cell_segmentation_review_quarantine_invalid_geometry ? '--quarantine-invalid-geometry' : ''
    def overlapQuarantineFlag = params.cell_segmentation_review_quarantine_overlapping_geometry ? '--quarantine-overlapping-geometry' : ''
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    echo "[INFO] Review-only geometry comparison: ${codeFingerprint}"
    python "${scriptPath}" \
      --source-image "${crop_tif}" \
      --cellvit-cells "${cells_json}" --cellvit-metadata "${cellvit_metadata}" \
      --stardist-labels "${stardist_labels}" --stardist-metadata "${stardist_frame}" \
      --max-pixels ${params.cell_segmentation_review_max_pixels} ${partialImageFlag} ${autoAdjustFlag} ${quarantineFlag} ${overlapQuarantineFlag} \
      --outdir "segmentation_review_${sample_id}"
    cp "${stardist_frame}" "segmentation_review_${sample_id}/stardist_frame.json"
    cp "${cellvit_metadata}" "segmentation_review_${sample_id}/cellvit_frame.json"
    """

    stub:
    """
    mkdir -p "segmentation_review_${sample_id}"
    touch "segmentation_review_${sample_id}/original_labels.tif" "segmentation_review_${sample_id}/tuned_candidate_labels.tif"
    printf '{"stub_only":true,"reference_quality":"model_agreement_only","cells":[]}\n' > "segmentation_review_${sample_id}/segmentation_qc.json"
    printf 'cell_id,segmentation_review_required\n' > "segmentation_review_${sample_id}/segmentation_qc.csv"
    printf '{"type":"FeatureCollection","stub_only":true,"features":[]}\n' > "segmentation_review_${sample_id}/segmentation_qc.geojson"
    """
}
