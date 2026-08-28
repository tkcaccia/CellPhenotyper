process RUN_GRANDQC_ARTIFACT_ANALYSIS {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/02_grandqc/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.grandqc_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.grandqc_memory_gb as int))} GB" }
    time { params.grandqc_time as String }

    input:
    tuple val(sample_id), path(ome_tif)

    output:
    tuple val(sample_id), path("grandqc_${sample_id}"), emit: grandqc_dir
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc_clean_tissue_mask.tif"), emit: clean_tissue_mask
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc_artifact_mask.tif"), emit: artifact_mask
    tuple val(sample_id), path("grandqc_${sample_id}/${sample_id}_grandqc.geojson"), optional: true, emit: artifact_geojson

    script:
    def script_path = "${projectDir}/${params.grandqc_script}"
    def cache_dir = (params.grandqc_cache_dir ?: "${baseDir}/.cache/grandqc").toString()
    def bootstrap_flag = params.grandqc_bootstrap_deps ? '--bootstrap-deps' : ''
    def download_flag = params.grandqc_download_models ? '--download-models' : ''
    def geojson_flag = params.grandqc_create_geojson ? '--create-geojson' : ''
    def requestedGrandqcDevice = (params.grandqc_device ?: 'auto').toString()
    def resolvedGrandqcDevice = requestedGrandqcDevice
    if (requestedGrandqcDevice == 'auto' && (params.hardware_auto as boolean) && (params.compute_device ?: 'cpu').toString().toLowerCase() == 'gpu') {
      resolvedGrandqcDevice = 'cuda'
    }
    """
    set -euo pipefail

    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    echo "[INFO] GrandQC runtime tune: profile=${params.hardware_profile}, requested_device=${requestedGrandqcDevice}, resolved_device=${resolvedGrandqcDevice}, tissue_patch_size=${params.grandqc_patch_size}, artifact_tile_size=${params.grandqc_artifact_tile_size}, overlap=${params.grandqc_artifact_overlap_fraction}"

    python "${script_path}" \
      --image "${ome_tif}" \
      --outdir "grandqc_${sample_id}" \
      --sample-id "${sample_id}" \
      --device "${resolvedGrandqcDevice}" \
      --cache-dir "${cache_dir}" \
      --default-source-mpp ${params.grandqc_default_source_mpp} \
      --artifact-mpp-model ${params.grandqc_artifact_mpp_model} \
      --tissue-mpp-model ${params.grandqc_tissue_mpp_model} \
      --patch-size ${params.grandqc_patch_size} \
      --artifact-tile-size ${params.grandqc_artifact_tile_size} \
      --artifact-overlap-fraction ${params.grandqc_artifact_overlap_fraction} \
      --overlay-factor ${params.grandqc_overlay_factor} \
      --preview-max-side ${params.grandqc_preview_max_side} \
      ${bootstrap_flag} \
      ${download_flag} \
      ${geojson_flag}
    """

    stub:
    """
    mkdir -p "grandqc_${sample_id}"
    touch "grandqc_${sample_id}/${sample_id}_grandqc_clean_tissue_mask.tif"
    touch "grandqc_${sample_id}/${sample_id}_grandqc_artifact_mask.tif"
    touch "grandqc_${sample_id}/${sample_id}_grandqc_summary.json"
    """
}
