process MAP_CELL_REFERENCE_ATLAS {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'map_cell_reference_atlas', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([profiles, reference_atlas]) }
    tag "${sample_id}"
    label 'compute_medium'
    cache 'deep'
    publishDir "${params.outdir_base}/21_reference_mapping/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'reference_mapping') }
    memory { TaskRuntime.memory(runtime_plan, 'reference_mapping') }
    time '24h'

    input:
    tuple val(sample_id), path(profiles, stageAs: 'query_profiles'), path(reference_atlas, stageAs: 'frozen_reference'), val(source_fingerprint)

    val(runtime_plan)

    output:
    tuple val(sample_id), path('reference_assignments.csv'), emit: assignments
    tuple val(sample_id), path('reference_assignments.{csv,atlas.json,mapping.json}'), emit: mapping_bundle

    script:
    def codeFingerprint = PipelineHelpers.codeFingerprint(['cell_reference_atlas.py', 'cell_profile_io.py', 'reference_mapping_io.py'].collect { "${projectDir}/bin/${it}" })
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Reference mapping code fingerprint: ${codeFingerprint}"
    echo "[INFO] Reference mapping code-and-data fingerprint: ${source_fingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    "${params.cell_atlas_python}" "${projectDir}/bin/cell_reference_atlas.py" map \
      --query '${profiles}' --atlas '${reference_atlas}' --output reference_assignments.csv
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    printf 'observation_uid,reference_assignment,reference_status\n' > reference_assignments.csv
    printf '{"stub":true}' > reference_assignments.atlas.json
    printf '{"stub":true}' > reference_assignments.mapping.json
    """
}
