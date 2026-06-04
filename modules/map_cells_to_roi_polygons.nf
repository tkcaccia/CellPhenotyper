process MAP_CELLS_TO_ROI_POLYGONS {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/07_cell_assignments/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.assign_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.assign_memory_gb as int))} GB" }
    time { params.assign_time as String }

    input:
    tuple val(sample_id), path(objects_csv), path(roi_geojson), path(shift_json)

    output:
    tuple val(sample_id), path("${sample_id}_objects_assigned.csv"), emit: objects_assigned

    script:
    def choose_flag = params.assign_choose ? "--choose ${params.assign_choose}" : ''
    def chunk_rows_flag = params.assign_chunk_rows ? "--chunk-rows ${params.assign_chunk_rows}" : ''
    def xcol_flag = params.assign_xcol ? "--xcol ${params.assign_xcol}" : ''
    def ycol_flag = params.assign_ycol ? "--ycol ${params.assign_ycol}" : ''
    def assign_script = "${projectDir}/${params.assign_script}"
    """
    set -euo pipefail

    python "${assign_script}" \\
      --objects "${objects_csv}" \\
      --roi "${roi_geojson}" \\
      --shift "${shift_json}" \\
      --label-prop "${params.assign_label_prop}" \\
      --out "${sample_id}_objects_assigned.csv" \\
      --out-col "${params.assign_out_col}" \\
      --workers ${task.cpus} \\
      ${choose_flag} \\
      ${chunk_rows_flag} \\
      ${xcol_flag} \\
      ${ycol_flag}
    """

    stub:
    """
    touch "${sample_id}_objects_assigned.csv"
    """
}
