process RUN_STARDIST_ROI_SEGMENTATION {
    tag "${sample_id}"
    label 'compute_heavy'

    publishDir "${params.outdir_base}/02_stardist", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.stardist_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.stardist_memory_gb as int))} GB" }
    time { params.stardist_time as String }

    input:
    tuple val(sample_id), path(ome_tif), path(roi_geojson)

    output:
    tuple val(sample_id), path("stardist_out/crop_roi.tif"), emit: crop_roi
    tuple val(sample_id), path("stardist_out/labels.tif"), emit: labels_tif
    tuple val(sample_id), path("stardist_out/labels_full.tif"), optional: true, emit: labels_full_tif
    tuple val(sample_id), path("stardist_out/objects.csv"), emit: objects_csv
    tuple val(sample_id), path("stardist_out/shift.json"), emit: shift_json
    tuple val(sample_id), path("stardist_out"), emit: stardist_dir

    script:
    def write_full_flag = params.write_full_labels ? '--write-full-labels' : ''
    def allow_huge_flag = params.allow_huge_tif ? '--allow-huge-tif' : ''
    def stardist_script = "${projectDir}/${params.stardist_script}"
    """
    set -euo pipefail

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    export LOKY_MAX_CPU_COUNT=${task.cpus}
    if [[ -n "${params.stardist_pythonpath}" ]]; then
      export PYTHONPATH="${params.stardist_pythonpath}:\${PYTHONPATH:-}"
    fi

    mkdir -p stardist_out

    python "${stardist_script}" \\
      --in "${ome_tif}" \\
      --roi "${roi_geojson}" \\
      --outdir stardist_out \\
      --model "${params.stardist_model}" \\
      --prob ${params.stardist_prob} \\
      --nms ${params.stardist_nms} \\
      --tiles ${params.stardist_tiles_y} ${params.stardist_tiles_x} \\
      --full-format "${params.full_format}" \\
      --full-out "stardist_out/labels_full.tif" \\
      ${write_full_flag} \\
      ${allow_huge_flag}
    """

    stub:
    """
    mkdir -p stardist_out
    touch stardist_out/crop_roi.tif
    touch stardist_out/labels.tif
    touch stardist_out/objects.csv
    touch stardist_out/shift.json
    if [[ "${params.write_full_labels}" == "true" ]]; then
      touch stardist_out/labels_full.tif
    fi
    """
}
