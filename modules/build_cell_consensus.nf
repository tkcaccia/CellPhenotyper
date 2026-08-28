process BUILD_CELL_CONSENSUS {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/03d_cell_consensus/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.cell_consensus_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.cell_consensus_memory_gb as int))} GB" }
    time { params.cell_consensus_time as String }

    input:
    tuple val(sample_id), path(stardist_objects), path(hovernet_cells), path(cellvit_cells), path(crop_tif), path(shift_json)

    output:
    tuple val(sample_id), path("consensus_${sample_id}/labels.tif"), emit: labels_tif
    tuple val(sample_id), path("consensus_${sample_id}/objects.csv"), emit: objects_csv
    tuple val(sample_id), path("consensus_${sample_id}/alignment.csv"), emit: alignment_csv
    tuple val(sample_id), path("consensus_${sample_id}/consensus_cells.geojson"), emit: cells_geojson
    tuple val(sample_id), path("consensus_${sample_id}/consensus_summary.json"), emit: summary_json
    tuple val(sample_id), path("consensus_${sample_id}"), emit: consensus_dir

    script:
    def scriptPath = "${projectDir}/${params.cell_consensus_script}"
    """
    set -euo pipefail
    python "${scriptPath}" \
      --stardist-objects "${stardist_objects}" --hovernet-cells "${hovernet_cells}" \
      --cellvit-cells "${cellvit_cells}" --image "${crop_tif}" --shift "${shift_json}" \
      --outdir "consensus_${sample_id}" --min-support ${params.cell_consensus_min_support} \
      --match-radius-um ${params.cell_consensus_match_radius_um} \
      --default-mpp ${params.cell_consensus_default_mpp} \
      --geometry-priority "${params.cell_consensus_geometry_priority}" \
      --tile-size ${params.cell_consensus_tile_size} --compression "${params.cell_consensus_compression}" \
      --preview-max-side ${params.cell_consensus_preview_max_side}
    """

    stub:
    """
    mkdir -p "consensus_${sample_id}"
    touch "consensus_${sample_id}/labels.tif" "consensus_${sample_id}/objects.csv" "consensus_${sample_id}/alignment.csv" "consensus_${sample_id}/consensus_preview.png"
    printf '{"type":"FeatureCollection","features":[]}' > "consensus_${sample_id}/consensus_cells.geojson"
    printf '{"consensus_count":0}' > "consensus_${sample_id}/consensus_summary.json"
    """
}
