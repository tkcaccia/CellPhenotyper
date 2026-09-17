process LINK_CELL_TISSUE_HIERARCHY {
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'link_cell_tissue_hierarchy', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([profiles, labels_tif, hierarchy, compartments]) }
    tag "${sample_id}"
    label 'compute_medium'
    cache 'deep'
    // Keep the original profiles immutable and publish the linked version separately.
    publishDir "${params.outdir_base}/24_cell_tissue_links/${sample_id}", mode: (params.publish_dir_mode ?: 'copy'), overwrite: true
    cpus { TaskRuntime.cpus(runtime_plan, 'cell_tissue_links') }
    memory { TaskRuntime.memory(runtime_plan, 'cell_tissue_links') }
    time '24h'

    input:
    tuple val(sample_id), path(profiles, stageAs: 'source_profiles'), path(labels_tif), path(hierarchy, stageAs: 'hierarchy'), path(compartments, stageAs: 'compartments'), val(have_ring)

    val(runtime_plan)

    output:
    tuple val(sample_id), path('cell_profiles'), emit: profiles

    script:
    def codeFingerprint = PipelineHelpers.codeFingerprint(['link_cell_tissue_hierarchy.py', 'cell_profile_io.py'].collect { "${projectDir}/bin/${it}" })
    def ringArg = have_ring ? "--ring-labels '${compartments}/labels_perinuclear_ring.tif'" : ''
    """
    set -euo pipefail
    source "${projectDir}/bin/activate_source_python.sh"
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] Cell-hierarchy linkage fingerprint: ${codeFingerprint}"
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    "${params.cell_atlas_python}" "${projectDir}/bin/link_cell_tissue_hierarchy.py" \
      --profile-dir '${profiles}' --labels '${labels_tif}' --hierarchy-dir '${hierarchy}' \
      ${ringArg} --outdir cell_profiles
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p cell_profiles
    printf '{"stub":true,"cell_count":0}' > cell_profiles/cell_profiles_manifest.json
    printf 'sample_id,cell_id,cell_uid\n' > cell_profiles/cell_profiles.csv
    printf 'cell_uid,compartment,parent_domain_id,subdomain_id,region_id,pixel_count\n' > cell_profiles/cell_hierarchy_overlaps.csv
    """
}
