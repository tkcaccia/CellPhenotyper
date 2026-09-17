process EXPAND_LABELS_TO_CYTOPLASM {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'expand_labels_to_cytoplasm', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([labels_tif, shift_json, resolution_json, tissue_mask]) }
    tag "${sample_id}:${label_kind}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/08_cytoplasm/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.expand_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.expand_memory_gb as int))} GB" }
    time { params.expand_time as String }

    input:
    tuple val(sample_id), path(labels_tif), val(label_kind), val(preview_background_path), path(shift_json), path(resolution_json), path(tissue_mask), val(frame)

    output:
    tuple val(sample_id), path("${sample_id}_${label_kind}.tif"), val(label_kind), emit: expanded_labels
    tuple val(sample_id), path("${sample_id}_${label_kind}_preview.png"), val(label_kind), emit: expanded_preview
    tuple val(sample_id), path("${sample_id}_${label_kind}_compartments/labels_perinuclear_ring.tif"), val(label_kind), optional: true, emit: ring_labels
    tuple val(sample_id), path("${sample_id}_${label_kind}_compartments"), val(label_kind), optional: true, emit: compartments

    script:
    def compression_flag = params.expand_compression ? "--compression ${params.expand_compression}" : ''
    def expand_script = "${projectDir}/${params.expand_script}"
    def codeFingerprint = PipelineHelpers.codeFingerprint([expand_script, "${projectDir}/bin/profile_cell_morphology.py"])
    def physical = (params.expand_um as double) >= 0
    def radiusFlag = physical ? "--expand-um ${params.expand_um} --resolution-json '${resolution_json}' --shift '${shift_json}' --tissue-mask '${tissue_mask}' --tissue-frame ${frame} --label-frame ${frame} --compartments-dir '${sample_id}_${label_kind}_compartments'" : "--expand-px ${params.expand_px}"
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Compartment code fingerprint: ${codeFingerprint}; physical=${physical}"

    python "${expand_script}" \\
      --labels "${labels_tif}" \\
      --out "${sample_id}_${label_kind}.tif" \\
      ${radiusFlag} \\
      --mode "${params.expand_mode}" \\
      --tile-size ${params.expand_tile_size} \\
      --auto-threshold-mpix ${params.expand_auto_threshold_mpix} \\
      --preview "${sample_id}_${label_kind}_preview.png" \\
      --preview-background "${preview_background_path}" \\
      --preview-factor ${params.expand_preview_factor} \\
      --preview-threshold-mb ${params.expand_preview_threshold_mb} \\
      --preview-alpha ${params.expand_preview_alpha} \\
      ${compression_flag}
    """

    stub:
    def stubPhysical = (params.expand_um as double) >= 0
    def compartmentStub = stubPhysical ? "mkdir -p '${sample_id}_${label_kind}_compartments'; touch '${sample_id}_${label_kind}_compartments/labels_perinuclear_ring.tif'" : ''
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${label_kind}.tif"
    touch "${sample_id}_${label_kind}_preview.png"
    ${compartmentStub}
    """
}
