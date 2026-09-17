process MAP_REGION_REFERENCE_ATLAS {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'map_region_reference_atlas', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([region_profiles, reference_atlas]) }
    tag "${sample_id}:${variant}"
    label 'compute_medium'
    cache 'deep'
    publishDir "${params.outdir_base}/23_region_reference_mapping/${sample_id}/${variant}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'region_reference_mapping') }
    memory { TaskRuntime.memory(runtime_plan, 'region_reference_mapping') }
    time '12h'
    input:
    tuple val(sample_key), val(sample_id), val(variant), path(region_profiles, stageAs: 'query_profiles'), path(reference_atlas, stageAs: 'frozen_reference'), val(source_fingerprint)
    val(runtime_plan)

    output:
    tuple val(sample_key), val(sample_id), val(variant), path('reference_assignments.csv'), emit: assignments
    tuple val(sample_key), val(sample_id), val(variant), path('reference_assignments.{csv,atlas.json,mapping.json}'), emit: mapping_bundle
    script:
    def fingerprint = PipelineHelpers.codeFingerprint(['cell_reference_atlas.py', 'cell_profile_io.py', 'reference_mapping_io.py'].collect { "${projectDir}/bin/${it}" })
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Region reference code fingerprint: ${fingerprint}"
    echo "[INFO] Region reference code-and-data fingerprint: ${source_fingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    "${params.cell_atlas_python}" "${projectDir}/bin/cell_reference_atlas.py" map \
      --query '${region_profiles}' --atlas '${reference_atlas}' --output reference_assignments.csv
    """
    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    printf 'sample_id,region_uid,reference_assignment,reference_status\n' > reference_assignments.csv
    printf '{"stub":true}' > reference_assignments.atlas.json
    printf '{"stub":true}' > reference_assignments.mapping.json
    """
}
