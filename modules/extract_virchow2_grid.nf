process EXTRACT_VIRCHOW2_GRID {
    tag "${sample_id}:nonoverlap"
    label 'compute_heavy'
    label 'gpu_capable'
    publishDir "${params.outdir_base}/09_embeddings/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus { Math.max(1, Math.min(params._executor_max_cpus as int, params.uni2_cpus as int)) }
    memory { "${Math.max(4, Math.min(params._executor_max_memory_gb as int, params.uni2_memory_gb as int))} GB" }
    time { params.uni2_time as String }
    errorStrategy 'terminate'
    maxRetries 0

    input:
    tuple val(sample_id), path(image_tif), path(grid_objects_csv), path(grid_metadata_json), path(checkpoint_dir)

    output:
    tuple val(sample_id), path("embeddings_${sample_id}_tile"), emit: tile_embeddings_dir

    script:
    def scriptPath = "${projectDir}/bin/extract_virchow2_grid.py"
    def requestedBatch = Math.max(1, Math.min(params.uni2_batch as int, 8))
    """
    set -euo pipefail
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    python "${scriptPath}" \
      --image "${image_tif}" \
      --grid-objects "${grid_objects_csv}" \
      --grid-metadata "${grid_metadata_json}" \
      --checkpoint "${checkpoint_dir}" \
      --outdir "embeddings_${sample_id}_tile" \
      --batch ${requestedBatch} --device cuda
    """

    stub:
    """
    mkdir -p "embeddings_${sample_id}_tile"
    printf 'cell_id,feat_1,feat_2\n' > "embeddings_${sample_id}_tile/virchow2_embeddings_shard0000.csv"
    printf '{"encoder":"Virchow2","tile_overlap_source_px":0,"inner_square_used":false}\n' > "embeddings_${sample_id}_tile/VIRCHOW2_COMPLETE.json"
    """
}
