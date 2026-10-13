process REFINE_WITH_BORA {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'refine_with_bora', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, cluster_mask_tif, baseline_mask_tif, tissue_mask_tif, kodama_membership_png, resolution_json, clustering_uncertainty_tif]) }
    tag "${sample_id}:${cluster_variant}"
    label 'compute_heavy'

    publishDir "${params.outdir_base}/14_medsam_refine_tissue/${sample_id}",
        mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true,
        pattern: "${sample_id}_${cluster_variant}_{grown_mask_refined*,bora_*}"
    publishDir "${params.outdir_base}/15_cluster_geojson/${sample_id}",
        mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true,
        pattern: "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson*"

    cpus { TaskRuntime.cpus(runtime_plan, 'medsam_refine') }
    memory { TaskRuntime.memory(runtime_plan, 'medsam_refine') }
    time { params.bora_time as String }

    input:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path(image_tif), path(cluster_mask_tif, stageAs: 'source_cluster_mask.tif'), path(baseline_mask_tif, stageAs: 'baseline_labels.tif'), path(tissue_mask_tif), path(kodama_membership_png), path(resolution_json), path(clustering_uncertainty_tif, stageAs: 'input_clustering_uncertainty.tif')
    val(runtime_plan)

    output:
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"), emit: refined_mask
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif"), emit: refined_uncertainty
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif"), emit: refined_provenance
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif.provenance.json"), emit: provenance_metadata
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson"), emit: cluster_geojson
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson.provenance.json"), emit: provenance_summary
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_bora_refined_qc_preview.png"), emit: refined_preview
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_bora_kodama_membership.png"), emit: kodama_plot_png
    tuple val(sample_key), val(sample_id), val(cluster_variant), path("${sample_id}_${cluster_variant}_bora_summary.json"), emit: bora_summary

    script:
    def finalizeScript = "${projectDir}/${params.bora_finalize_script}"
    def stageMaskScript = "${projectDir}/bin/stage_bora_label_mask.py"
    def streamMode = (params.bora_stream_mode ?: 'auto').toString().trim().toLowerCase()
    def streamFlag = streamMode == 'always' ? '--stream' : (streamMode == 'never' ? '--no-stream' : '')
    def pyramidFlag = (params.bora_pyramid as boolean) ? '--pyramid' : '--no-pyramid'
    def wandFlag = (params.bora_annealed_wand as boolean) ? '--annealed-wand' : '--no-annealed-wand'
    def cudaFallbackFlag = (params.bora_geojson_cuda_fallback as boolean) ? '--geojson-cuda-fallback' : '--no-geojson-cuda-fallback'
    def sourceDir = (params.bora_source_dir ?: '').toString().trim()
    def sourceSetup = sourceDir ? "export BORA_HOME=\"${sourceDir}\"; export PYTHONPATH=\"${sourceDir}/src\${PYTHONPATH:+:\$PYTHONPATH}\"" : ''
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def requestedDevice = (params.bora_device ?: 'auto').toString().trim().toLowerCase()
    def boraDevice = requestedDevice == 'auto' ? (resolvedComputeDevice == 'gpu' ? 'cuda' : 'cpu') : requestedDevice
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    ${sourceSetup}

    resolved_revision=""
    if [ -n "\${BORA_HOME:-}" ] && [ -d "\${BORA_HOME}/.git" ]; then
      resolved_revision="\$(git -C "\${BORA_HOME}" rev-parse HEAD)"
    elif [ -n "\${BORA_REVISION:-}" ]; then
      resolved_revision="\${BORA_REVISION}"
    fi
    if [ -n "\$resolved_revision" ] && [ "\$resolved_revision" != "${params.bora_revision}" ]; then
      echo "FATAL: Bora revision mismatch: expected ${params.bora_revision}, resolved \$resolved_revision" >&2
      exit 2
    fi
    python - <<'PY'
import bora
from bora.cli import parser
print(f"[INFO] Bora runtime: {bora.__file__}")
parser()
PY

    bora_input_mask=".${sample_id}_${cluster_variant}_bora_input_labels.tif"
    trap 'rm -f -- "\$bora_input_mask"' EXIT
    python "${stageMaskScript}" \
      --input "${baseline_mask_tif}" \
      --output "\$bora_input_mask" \
      --block-rows ${params.bora_block_size} \
      --workers ${params.bora_pyramid_workers}

    python -m bora.cli refine \
      "${image_tif}" "\$bora_input_mask" \
      --output "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif" \
      --geojson "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson" \
      --report "${sample_id}_${cluster_variant}_bora_summary.json" \
      --backend "${params.bora_backend}" \
      --device "${boraDevice}" \
      --core-erosion ${params.bora_core_erosion} \
      --outer-dilation ${params.bora_outer_dilation} \
      --tile-size ${params.bora_tile_size} \
      --tile-overlap ${params.bora_tile_overlap} \
      --min-area ${params.bora_min_area} \
      --smooth-radius ${params.bora_smooth_radius} \
      --geojson-backend "${params.bora_geojson_backend}" \
      --geojson-profile "${params.bora_geojson_profile}" \
      --geojson-workers ${params.bora_geojson_workers} \
      --geojson-max-page-side ${params.bora_geojson_max_page_side} \
      --geojson-group-prefix "${params.cluster_geojson_group_prefix}" \
      --geojson-cuda-device ${params.bora_geojson_cuda_device} \
      ${cudaFallbackFlag} \
      ${streamFlag} --block-size ${params.bora_block_size} \
      ${pyramidFlag} --pyramid-compression "${params.bora_pyramid_compression}" \
      --pyramid-workers ${params.bora_pyramid_workers} \
      ${wandFlag} \
      --wand-boundary-radius ${params.bora_wand_boundary_radius} \
      --wand-downsample ${params.bora_wand_downsample} \
      --wand-iterations ${params.bora_wand_iterations} \
      --wand-initial-temperature ${params.bora_wand_initial_temperature} \
      --wand-final-temperature ${params.bora_wand_final_temperature} \
      --wand-data-weight ${params.bora_wand_data_weight} \
      --wand-smoothness-weight ${params.bora_wand_smoothness_weight} \
      --wand-edge-beta ${params.bora_wand_edge_beta} \
      --wand-connectivity ${params.bora_wand_connectivity} \
      --watershed-resolution "${params.bora_watershed_resolution}" \
      --watershed-max-side ${params.bora_watershed_max_side}

    rm -f -- "\$bora_input_mask"
    trap - EXIT

    python "${finalizeScript}" \
      --sample-id "${sample_id}_${cluster_variant}" \
      --image "${image_tif}" \
      --source-mask "${cluster_mask_tif}" \
      --baseline-mask "${baseline_mask_tif}" \
      --tissue-mask "${tissue_mask_tif}" \
      --uncertainty-mask "${clustering_uncertainty_tif}" \
      --resolution-json "${resolution_json}" \
      --shift-json "${resolution_json}" \
      --refined-mask "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif" \
      --geojson "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson" \
      --bora-report "${sample_id}_${cluster_variant}_bora_summary.json" \
      --uncertainty-out "${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif" \
      --provenance-out "${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif" \
      --geojson-provenance-out "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson.provenance.json" \
      --preview "${sample_id}_${cluster_variant}_bora_refined_qc_preview.png" \
      --bora-repository "${params.bora_repository}" \
      --bora-revision "${params.bora_revision}" \
      --block-rows ${params.bora_contract_block_rows} \
      --preview-factor ${params.grow_preview_factor} \
      --preview-threshold-mb ${params.grow_preview_threshold_mb} \
      --preview-alpha ${params.grow_preview_alpha} \
      --default-mpp ${params.input_resolution_override_mpp} \
      --overwrite

    cp "${kodama_membership_png}" "${sample_id}_${cluster_variant}_bora_kodama_membership.png"
    """

    stub:
    """
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined_uncertainty.tif"
    touch "${sample_id}_${cluster_variant}_grown_mask_refined_provenance.tif"
    printf '{"schema_version":"1.1.0","stub":true,"refinement_engine":"Bora"}\n' > "${sample_id}_${cluster_variant}_grown_mask_refined.ome.tif.provenance.json"
    printf '{"type":"FeatureCollection","features":[],"stub":true}\n' > "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson"
    printf '{"schema_version":"cellphenotyper.bora_geojson_provenance.v1","stub":true}\n' > "${sample_id}_${cluster_variant}_grown_mask_smooth_class.geojson.provenance.json"
    touch "${sample_id}_${cluster_variant}_bora_refined_qc_preview.png"
    touch "${sample_id}_${cluster_variant}_bora_kodama_membership.png"
    printf '{"backend":"watershed","stub":true}\n' > "${sample_id}_${cluster_variant}_bora_summary.json"
    """
}
