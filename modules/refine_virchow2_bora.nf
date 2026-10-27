process REFINE_VIRCHOW2_BORA {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_heavy'
    label 'gpu_capable'
    publishDir "${params.outdir_base}/15_cluster_geojson/${sample_id}",
        mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.bora_cpus as int)) }
    memory { "${Math.max(4, Math.min(params._executor_max_memory_gb as int, params.bora_memory_gb as int))} GB" }
    time { params.bora_time as String }
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant),
          path(image_tif), path(coarse_mask_tif), path(tissue_mask_tif),
          path(grid_metadata_json), path(resolution_json), path(classifier_model), path(checkpoint_dir), path(bora_source_dir)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant),
          path("bora_${sample_id}_${cluster_variant}/mask.ome.tif"), emit: refined_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant),
          path("bora_${sample_id}_${cluster_variant}/mask.geojson"), emit: cluster_geojson
    tuple val(sample_key), val(sample_id), val(cluster_variant),
          path("bora_${sample_id}_${cluster_variant}/PIPELINE_COMPLETE.json"), emit: report

    script:
    def scriptPath = "${projectDir}/bin/run_virchow2_bora_wsi.py"
    def deadline = Math.max(180, params.bora_deadline_seconds as int)
    """
    set -euo pipefail
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    python "${scriptPath}" \
      --image "${image_tif}" \
      --coarse "${coarse_mask_tif}" \
      --tissue "${tissue_mask_tif}" \
      --grid-metadata "${grid_metadata_json}" \
      --resolution-json "${resolution_json}" \
      --classifier "${classifier_model}" \
      --checkpoint "${checkpoint_dir}" \
      --bora-source "${bora_source_dir}" \
      --output "bora_${sample_id}_${cluster_variant}" \
      --core-pixels ${params.bora_core_pixels} \
      --halo-pixels ${params.bora_halo_pixels} \
      --batch ${params.bora_batch} \
      --vector-workers ${params.bora_vector_workers} \
      --deadline-seconds ${deadline}
    """

    stub:
    """
    mkdir -p "bora_${sample_id}_${cluster_variant}"
    touch "bora_${sample_id}_${cluster_variant}/mask.ome.tif"
    printf '{"type":"FeatureCollection","features":[]}\n' > "bora_${sample_id}_${cluster_variant}/mask.geojson"
    printf '{"pilot_only":false,"encoder_tile_overlap_px":0}\n' > "bora_${sample_id}_${cluster_variant}/PIPELINE_COMPLETE.json"
    """
}
