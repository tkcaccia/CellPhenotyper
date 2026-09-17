process BUILD_CELL_CONSENSUS {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'build_cell_consensus', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([stardist_objects, hovernet_cells, cellvit_cells, crop_tif, shift_json]) }
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/03d_cell_consensus/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cell_consensus_cpus as int)) }
    memory { "${Math.max(1, Math.min(params._executor_max_memory_gb as int, Math.max(4, params.cell_consensus_memory_gb as int)))} GB" }
    time { params.cell_consensus_time as String }

    input:
    tuple val(sample_id), path(stardist_objects), path(hovernet_cells), path(cellvit_cells), path(crop_tif), path(shift_json)

    output:
    tuple val(sample_id), path("consensus_${sample_id}/labels.tif"), emit: labels_tif
    tuple val(sample_id), path("consensus_${sample_id}/objects.csv"), emit: objects_csv
    tuple val(sample_id), path("consensus_${sample_id}/alignment.csv"), emit: alignment_csv
    tuple val(sample_id), path("consensus_${sample_id}/consensus_cells.geojson"), emit: cells_geojson
    tuple val(sample_id), path("consensus_${sample_id}/rejected_detector_candidates.geojson"), emit: rejected_candidates_geojson
    tuple val(sample_id), path("consensus_${sample_id}/consensus_summary.json"), emit: summary_json
    tuple val(sample_id), path("consensus_${sample_id}/detector_agreement_benchmark.csv"), emit: benchmark_csv
    tuple val(sample_id), path("consensus_${sample_id}/detector_agreement_benchmark.json"), emit: benchmark_json
    tuple val(sample_id), path("consensus_${sample_id}"), emit: consensus_dir

    script:
    def scriptPath = "${projectDir}/${params.cell_consensus_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    codeDigest.update(new File(scriptPath).bytes)
    ['detector_candidate_review.py', 'cell_profile_io.py', 'profile_cell_morphology.py'].each {
        codeDigest.update(new File("${projectDir}/bin/${it}").bytes)
    }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Cell consensus code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --stardist-objects "${stardist_objects}" --hovernet-cells "${hovernet_cells}" \
      --cellvit-cells "${cellvit_cells}" --image "${crop_tif}" --shift "${shift_json}" \
      --outdir "consensus_${sample_id}" --min-support ${params.cell_consensus_min_support} \
      --match-radius-um ${params.cell_consensus_match_radius_um} \
      --default-mpp ${params.cell_consensus_default_mpp} \
      --geometry-priority "${params.cell_consensus_geometry_priority}" \
      --count-ratio-warning ${params.cell_consensus_count_ratio_warning} \
      --min-agreement-score ${params.cell_consensus_min_agreement_score} \
      --fusion-acceptance-policy "${params.cell_consensus_fusion_acceptance_policy}" \
      --tile-size ${params.cell_consensus_tile_size} --compression "${params.cell_consensus_compression}" \
      --preview-max-side ${params.cell_consensus_preview_max_side}
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "consensus_${sample_id}"
    touch "consensus_${sample_id}/labels.tif" "consensus_${sample_id}/objects.csv" "consensus_${sample_id}/alignment.csv" "consensus_${sample_id}/consensus_preview.png"
    printf 'source_a,source_b,matched_pairs\n' > "consensus_${sample_id}/detector_agreement_benchmark.csv"
    printf '{"interpretation":"Inter-detector agreement; not accuracy against ground truth.","pairs":[]}' > "consensus_${sample_id}/detector_agreement_benchmark.json"
    printf '{"type":"FeatureCollection","features":[]}' > "consensus_${sample_id}/consensus_cells.geojson"
    printf '{"type":"FeatureCollection","features":[],"metadata":{"stub":true}}' > "consensus_${sample_id}/rejected_detector_candidates.geojson"
    printf '{"consensus_count":0}' > "consensus_${sample_id}/consensus_summary.json"
    """
}
