// Convert CellViT geometry into the same mask/table contract used downstream.
// This process performs no model inference or detector fusion.
process EXPORT_CELLVIT_PRIMARY_SEGMENTATION {
    tag "${sample_id}"
    label 'compute_medium'
    publishDir "${params.outdir_base}/03b_cellvitpp/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true
    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.cell_consensus_cpus as int)) }
    memory { "${Math.max(2, Math.min(params._executor_max_memory_gb as int, Math.max(4, params.cell_consensus_memory_gb as int)))} GB" }
    time { params.cell_consensus_time as String }
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json), path(clean_tissue_mask),
          val(source_kind), path(primary_source), path(source_metadata)

    output:
    tuple val(sample_id), path("cellvit_primary_${sample_id}/labels.tif"), emit: labels_tif
    tuple val(sample_id), path("cellvit_primary_${sample_id}/objects.csv"), emit: objects_csv
    tuple val(sample_id), path("cellvit_primary_${sample_id}/segmentation_metadata.json"), emit: metadata_json
    tuple val(sample_id), path("cellvit_primary_${sample_id}/cell_instances.geojson"), emit: annotations_geojson
    tuple val(sample_id), path("cellvit_primary_${sample_id}"), emit: segmentation_dir

    script:
    def q = { value -> "'"+value.toString().replace("'", "'\"'\"'")+"'" }
    if (!(source_kind in ['full_slide', 'roi'])) error 'CellViT primary source_kind must be full_slide or roi.'
    def scriptPath = "${projectDir}/bin/export_cellvit_primary_segmentation.py"
    def fingerprint = PipelineHelpers.codeFingerprint([scriptPath, "${projectDir}/bin/build_cell_consensus.py", "${projectDir}/bin/cellvit_models.py"])
    def sourceArgs = source_kind == 'full_slide' ? "--project-pointer ${q(primary_source)} --source-metadata ${q(source_metadata)}" : "--cellvit-cells ${q(primary_source)} --cellvit-metadata ${q(source_metadata)}"
    """
    set -euo pipefail
    echo "[INFO] CellViT primary conversion code fingerprint: ${fingerprint}"
    python ${q(scriptPath)} --image ${q(crop_tif)} --shift ${q(shift_json)} \
      --clean-tissue-mask ${q(clean_tissue_mask)} --outdir ${q("cellvit_primary_${sample_id}")} \
      ${sourceArgs} --tile-size ${params.cell_consensus_tile_size} --compression ${q(params.cell_consensus_compression)}
    """

    stub:
    """
    mkdir -p "cellvit_primary_${sample_id}"
    touch "cellvit_primary_${sample_id}/labels.tif"
    printf 'label,area,x,y,xmin,ymin,xmax,ymax,cell_id,source_id,geometry_source,cellvitpp_type,cellvitpp_type_id,cellvitpp_probability\n' > "cellvit_primary_${sample_id}/objects.csv"
    printf '{"stub_only":true,"cell_detection_mode":"cellvitpp","cell_count":0}\n' > "cellvit_primary_${sample_id}/segmentation_metadata.json"
    printf '{"type":"FeatureCollection","features":[]}\n' > "cellvit_primary_${sample_id}/cell_instances.geojson"
    """
}
