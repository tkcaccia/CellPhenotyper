process GROW_TO_TISSUE {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/13_grown_tissue/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.grow_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.grow_memory_gb as int))} GB" }
    time { params.grow_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(image_tif), path(cluster_mask_tif), path(tissue_mask_tif)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask.ome.tif"), emit: grown_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_qc_preview.png"), emit: grown_preview

    script:
    def grow_script = "${projectDir}/${params.grow_script}"
    def grow_method = 'classic_existing'
    def restrict_flag = (params.grow_restrict_to_seeded_components as boolean) ? '--restrict-to-seeded-components' : ''
    def add_nuclei_flag = (params.grow_add_nuclei as boolean) ? '' : '--no-add-nuclei'
    def legacy_flag = (params.grow_legacy as boolean) ? '--legacy' : ''
    def overwrite_flag = (params.grow_overwrite as boolean) ? '--overwrite' : ''
    def keep_tmp_flag = (params.grow_keep_tmp as boolean) ? '--keep-tmp' : ''
    def tail_flags = [legacy_flag, overwrite_flag, keep_tmp_flag].findAll { it?.trim() }.join(' ')
    """
    set -euo pipefail

    python "${grow_script}" \
      --image "${image_tif}" \
      --mask "${cluster_mask_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --out "${sample_id}_${cluster_variant}_grown_mask.ome.tif" \
      --preview "${sample_id}_${cluster_variant}_grown_qc_preview.png" \
      --preview-factor ${params.grow_preview_factor} \
      --preview-threshold-mb ${params.grow_preview_threshold_mb} \
      --preview-alpha ${params.grow_preview_alpha} \
      --sigma ${params.grow_sigma} \
      --method ${grow_method} \
      ${restrict_flag} \
      --min-seed-area ${params.grow_min_seed_area} \
      --fill-holes-area ${params.grow_fill_holes_area} \
      --close-radius ${params.grow_close_radius} \
      ${add_nuclei_flag} \
      --nuclei-thresh ${params.grow_nuclei_thresh} \
      --nuclei-dilate ${params.grow_nuclei_dilate} \
      --pyr-compression ${params.grow_pyr_compression} \
      --max-workers ${Math.max(1, Math.min(task.cpus as int, params.grow_max_workers as int))} \
      --downsample ${params.grow_downsample} \
      ${tail_flags}
    """

    stub:
    """
    touch "${sample_id}_${cluster_variant}_grown_mask.ome.tif"
    touch "${sample_id}_${cluster_variant}_grown_qc_preview.png"
    """
}
