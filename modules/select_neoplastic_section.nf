process SELECT_NEOPLASTIC_SECTION {
    tag "${sample_id}:${cluster_variant}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/16_neoplastic_section/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params.max_cpus as int, params.neoplastic_section_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.neoplastic_section_memory_gb as int))} GB" }
    time { params.neoplastic_section_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(cluster_geojson), path(objects_csv), path(image_tif), path(shift_json)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("neoplastic_${sample_id}_${cluster_variant}/selected_section.ome.tif"), emit: selected_image
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("neoplastic_${sample_id}_${cluster_variant}/selected_section_mask.tif"), emit: selected_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("neoplastic_${sample_id}_${cluster_variant}/selected_section_shift.json"), emit: selected_shift
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("neoplastic_${sample_id}_${cluster_variant}/selected_section_summary.json"), emit: selected_summary
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("neoplastic_${sample_id}_${cluster_variant}"), emit: selected_dir

    script:
    def scriptPath = "${projectDir}/${params.neoplastic_section_script}"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    [
      scriptPath,
      "${projectDir}/bin/grow_to_tissue.py",
      "${projectDir}/bin/ome_tiff_metadata.py",
    ].each { codeDigest.update(new File(it).bytes) }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    def requireFlag = (params.neoplastic_section_require_cells as boolean) ? '--require-neoplastic' : ''
    """
    set -euo pipefail
    echo "[INFO] Neoplastic section code fingerprint: ${codeFingerprint}"
    python "${scriptPath}" \
      --cluster-geojson "${cluster_geojson}" --objects "${objects_csv}" \
      --image "${image_tif}" --shift "${shift_json}" --cluster-variant "${cluster_variant}" \
      --outdir "neoplastic_${sample_id}_${cluster_variant}" \
      --neoplastic-names "${params.neoplastic_section_names}" \
      --default-mpp ${params.neoplastic_section_default_mpp} \
      --padding-um ${params.neoplastic_section_padding_um} \
      --spatial-bin-size ${params.neoplastic_section_spatial_bin_size} \
      --tile-size ${params.neoplastic_section_tile_size} \
      --max-workers ${Math.max(1, task.cpus as int)} ${requireFlag}
    """

    stub:
    """
    mkdir -p "neoplastic_${sample_id}_${cluster_variant}"
    touch "neoplastic_${sample_id}_${cluster_variant}/selected_section.ome.tif"
    touch "neoplastic_${sample_id}_${cluster_variant}/selected_section_mask.tif"
    printf '{"source_mpp":0.5}' > "neoplastic_${sample_id}_${cluster_variant}/selected_section_shift.json"
    printf '{"selected_section_id":"stub"}' > "neoplastic_${sample_id}_${cluster_variant}/selected_section_summary.json"
    """
}
