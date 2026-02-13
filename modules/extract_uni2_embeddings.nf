process EXTRACT_UNI2_EMBEDDINGS {
    tag "${sample_id}:${embedding_mode}"
    label 'compute_heavy'

    publishDir "${params.outdir_base}/07_embeddings", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.uni2_cpus as int)) }
    memory { "${Math.max(4, Math.min(params.max_memory_gb as int, params.uni2_memory_gb as int))} GB" }
    time { params.uni2_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(mask_tif), val(embedding_mode), val(zero_outside_mask)

    output:
    tuple val(sample_id), val(embedding_mode), path("embeddings_${sample_id}_${embedding_mode}"), emit: embeddings_dir

    script:
    def zero_outside_flag = zero_outside_mask ? '--zero-outside-mask --outside-fill 255' : ''
    def force_full_flag = params.uni2_force_full_image ? '--force-full-image' : ''
    def save_tiles_flag = params.uni2_save_tiles ? '--save-tiles' : ''
    def device_value = params.compute_device == 'gpu' ? 'cuda' : (params.compute_device == 'auto' ? params.uni2_device_auto : 'cpu')
    def uni2_script = "${projectDir}/${params.uni2_script}"
    """
    set -euo pipefail

    TOKEN_VAR_NAME="${params.hf_token_env_var_name ?: 'HF_TOKEN'}"
    export HF_HOME="${params.hf_home}"
    export HF_HUB_CACHE="${params.hf_hub_cache}"
    export HF_TOKEN="\$(printenv "\$TOKEN_VAR_NAME" || true)"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1

    HF_TOKEN_ARGS=()
    if [[ -n "\$HF_TOKEN" ]]; then
      HF_TOKEN_ARGS=(--hf-token "\$HF_TOKEN")
    fi

    python "${uni2_script}" \\
      --image "${image_tif}" \\
      --mask "${mask_tif}" \\
      --outdir "embeddings_${sample_id}_${embedding_mode}" \\
      --image-level ${params.uni2_image_level} \\
      ${force_full_flag} \\
      --grid "${params.uni2_grid}" \\
      --tile-size ${params.uni2_tile_size} \\
      ${zero_outside_flag} \\
      ${save_tiles_flag} \\
      --tiles-root "${params.uni2_tiles_root}" \\
      --bucket-size ${params.uni2_bucket_size} \\
      --min-area ${params.uni2_min_area} \\
      --encoder "${params.uni2_encoder}" \\
      --backend "${params.uni2_backend}" \\
      --pooling "${params.uni2_pooling}" \\
      --img-size ${params.uni2_img_size} \\
      --device "${device_value}" \\
      --batch ${params.uni2_batch} \\
      --torch-threads ${Math.max(1, Math.min(task.cpus as int, params.uni2_torch_threads as int))} \\
      --rows-per-csv ${params.uni2_rows_per_csv} \\
      --mask-block ${params.uni2_mask_block} \\
      "\${HF_TOKEN_ARGS[@]}"
    """

    stub:
    """
    mkdir -p "embeddings_${sample_id}_${embedding_mode}"
    touch "embeddings_${sample_id}_${embedding_mode}/embeddings_000000.csv"
    """
}
