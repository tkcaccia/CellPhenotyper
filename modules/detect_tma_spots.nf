process DETECT_TMA_SPOTS {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/04_TMA/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.tma_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.tma_memory_gb as int))} GB" }
    time { params.tma_time as String }

    input:
    tuple val(sample_id), path(crop_tif), path(objects_csv), path(shift_json)

    output:
    tuple val(sample_id), path("tma_${sample_id}"), emit: tma_dir
    tuple val(sample_id), path("tma_${sample_id}/${sample_id}_objects_tma_assigned.csv"), emit: objects_tma_assigned
    tuple val(sample_id), path("tma_${sample_id}/${sample_id}_tma_spots.geojson"), emit: tma_spots_geojson
    tuple val(sample_id), path("tma_${sample_id}/${sample_id}_tma_summary.json"), emit: tma_summary

    script:
    def tma_script = "${projectDir}/${params.tma_script}"
    """
    set -euo pipefail

    mkdir -p "tma_${sample_id}"
    python "${tma_script}" \
      --image "${crop_tif}" \
      --objects "${objects_csv}" \
      --shift "${shift_json}" \
      --outdir "tma_${sample_id}" \
      --sample-id "${sample_id}" \
      --thumbnail-max-side ${params.tma_thumbnail_max_side} \
      --min-spots ${params.tma_min_spots} \
      --min-spot-area-fraction ${params.tma_min_spot_area_fraction} \
      --max-spot-area-fraction ${params.tma_max_spot_area_fraction} \
      --max-area-cv ${params.tma_max_area_cv} \
      --min-circularity ${params.tma_min_circularity} \
      --min-solidity ${params.tma_min_solidity} \
      --max-eccentricity ${params.tma_max_eccentricity} \
      --grid-tolerance-fraction ${params.tma_grid_tolerance_fraction} \
      --close-radius ${params.tma_close_radius}
    """

    stub:
    """
    mkdir -p "tma_${sample_id}"
    printf '{"sample_id":"%s","is_tma":false,"spot_count":0}\n' "${sample_id}" > "tma_${sample_id}/${sample_id}_tma_summary.json"
    printf '{"type":"FeatureCollection","features":[]}\n' > "tma_${sample_id}/${sample_id}_tma_spots.geojson"
    printf 'spot_id,spot_label,row,column\n' > "tma_${sample_id}/${sample_id}_tma_spots.csv"
    cp "${objects_csv}" "tma_${sample_id}/${sample_id}_objects_tma_assigned.csv"
    touch "tma_${sample_id}/${sample_id}_tma_spots_preview.png"
    """
}
