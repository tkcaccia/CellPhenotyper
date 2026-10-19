process BUILD_UNI2_SPATIAL_GRID {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/09_grid_tiles/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.uni2_grid_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, params.uni2_grid_memory_gb as int))} GB" }
    time { params.uni2_grid_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(tissue_mask_tif), path(resolution_json)

    output:
    tuple val(sample_id), path("${sample_id}_uni2_grid_objects.csv"), emit: grid_objects
    tuple val(sample_id), path("${sample_id}_uni2_grid_metadata.json"), emit: grid_metadata
    tuple val(sample_id), path("${sample_id}_uni2_grid_preview.png"), emit: grid_preview

    script:
    def scriptPath = "${projectDir}/${params.uni2_grid_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    [scriptPath, "${projectDir}/bin/uni2_grid.py", "${projectDir}/bin/ome_tiff_metadata.py"].each {
      codeDigest.update(new File(it).bytes)
    }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] UNI2 spatial-grid code fingerprint: ${codeFingerprint}"

    python "${scriptPath}" \
      --image "${image_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --resolution-json "${resolution_json}" \
      --objects-out "${sample_id}_uni2_grid_objects.csv" \
      --metadata-out "${sample_id}_uni2_grid_metadata.json" \
      --preview-out "${sample_id}_uni2_grid_preview.png" \
      --model-tile-size ${params.uni2_tile_size} \
      --target-mpp ${params.uni2_target_mpp} \
      --default-source-mpp ${params.uni2_default_source_mpp} \
      --min-tissue-fraction ${params.uni2_grid_min_tissue_fraction} \
      --preview-max-side ${params.uni2_grid_preview_max_side}
    """

    stub:
    """
    printf 'label,x,y,polygon_label,grid_row,grid_col,core_x0,core_y0,core_x1,core_y1,tile_x0,tile_y0,tile_x1,tile_y1,tissue_px,tissue_fraction,area_px\n' > "${sample_id}_uni2_grid_objects.csv"
    printf '{"observation_type":"spatial_grid"}\n' > "${sample_id}_uni2_grid_metadata.json"
    touch "${sample_id}_uni2_grid_preview.png"
    """
}
