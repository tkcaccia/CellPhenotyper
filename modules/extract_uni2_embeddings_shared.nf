process EXTRACT_UNI2_EMBEDDINGS_SHARED {
    tag "${sample_id}:tile_inner_square_fixed90"
    label 'compute_heavy'
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/09_embeddings/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus {
      def requested = Math.max(1, Math.min(params.max_cpus as int, params.uni2_cpus as int))
      def compute = (params.compute_device ?: 'cpu').toString().toLowerCase()
      def maxMemGb = params.max_memory_gb as int
      if (compute != 'gpu' && maxMemGb <= 3) return 1
      return requested
    }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.uni2_memory_gb as int))} GB" }
    time { params.uni2_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(mask_tif)

    output:
    tuple val(sample_id), val('tile'), path("embeddings_${sample_id}_tile"), emit: tile_embeddings_dir
    tuple val(sample_id), val('inner_square'), path("embeddings_${sample_id}_inner_square"), emit: inner_square_embeddings_dir

    script:
    def force_full_flag = params.uni2_force_full_image ? '--force-full-image' : ''
    def save_tiles_flag = params.uni2_save_tiles ? '--save-tiles' : ''
    def tiles_root_path = "embeddings_${sample_id}_tile/${params.uni2_tiles_root}"
    def device_value = params.compute_device == 'gpu' ? 'cuda' : (params.compute_device == 'auto' ? params.uni2_device_auto : 'cpu')
    def task_mem_gb = task.memory ? Math.max(1, task.memory.toGiga() as int) : Math.max(1, params.max_memory_gb as int)
    def requested_batch = Math.max(1, params.uni2_batch as int)
    def initial_batch = requested_batch
    if (device_value == 'cpu') {
      if (task_mem_gb <= 4) initial_batch = Math.min(initial_batch, 2)
      else if (task_mem_gb <= 8) initial_batch = Math.min(initial_batch, 4)
      else if (task_mem_gb <= 12) initial_batch = Math.min(initial_batch, 8)
      else if (task_mem_gb <= 16) initial_batch = Math.min(initial_batch, 16)
    }
    def resolved_torch_threads = Math.max(1, Math.min(task.cpus as int, params.uni2_torch_threads as int))
    if (device_value == 'cpu' && task_mem_gb <= 4) {
      resolved_torch_threads = 1
    }
    def resolved_rows_per_csv = Math.max(1000, params.uni2_rows_per_csv as int)
    if (device_value == 'cpu' && task_mem_gb <= 4) {
      resolved_rows_per_csv = Math.min(resolved_rows_per_csv, 2000)
    }
    def paired_inner_square_mode = (params.uni2_paired_inner_square_mode ?: 'token_subset').toString()
    def uni2_script = "${projectDir}/${params.uni2_script}"
    def token_env_file = params.hf_token_env_file ? (params.hf_token_env_file.toString().startsWith('/') ? params.hf_token_env_file : "${projectDir}/${params.hf_token_env_file}") : ''
    """
    set -euo pipefail

    TOKEN_ENV_FILE="${token_env_file}"
    TOKEN_VAR_NAME="${params.hf_token_env_var_name ?: 'HF_TOKEN'}"
    if [[ -n "\$TOKEN_ENV_FILE" && -f "\$TOKEN_ENV_FILE" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "\$TOKEN_ENV_FILE"
      set +a
    fi

    if [[ -z "\${HF_HOME:-}" ]]; then
      export HF_HOME="${params.hf_home}"
    fi
    if [[ -z "\${HF_HUB_CACHE:-}" ]]; then
      export HF_HUB_CACHE="${params.hf_hub_cache}"
    fi
    if [[ -z "\${HF_HUB_OFFLINE:-}" ]]; then
      export HF_HUB_OFFLINE="${params.hf_hub_offline}"
    fi
    if [[ -z "\${TRANSFORMERS_OFFLINE:-}" ]]; then
      export TRANSFORMERS_OFFLINE="${params.hf_hub_offline}"
    fi
    export HF_TOKEN="\$(printenv "\$TOKEN_VAR_NAME" || true)"
    if [[ -z "\$HF_TOKEN" ]]; then
      export HF_TOKEN="\$(printenv HF_TOKEN || true)"
    fi
    if [[ -z "\$HF_TOKEN" ]]; then
      export HF_TOKEN="\$(printenv HF_UNI2 || true)"
    fi
    echo "[INFO] HF cache env: HF_HOME=\$HF_HOME HF_HUB_CACHE=\$HF_HUB_CACHE HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-unset} TRANSFORMERS_OFFLINE=\${TRANSFORMERS_OFFLINE:-unset}"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1

    OUTDIR_TILE="embeddings_${sample_id}_tile"
    OUTDIR_INNER="embeddings_${sample_id}_inner_square"
    ATTEMPT_BATCH=${initial_batch}
    echo "[INFO] UNI2 shared runtime tune: mem_gb=${task_mem_gb}, requested_batch=${requested_batch}, start_batch=${initial_batch}, torch_threads=${resolved_torch_threads}, rows_per_csv=${resolved_rows_per_csv}"

    while true; do
      mkdir -p "\$OUTDIR_TILE" "\$OUTDIR_INNER"

      ATTEMPT_ERR=".uni2_shared_attempt_batch_\${ATTEMPT_BATCH}.err"
      set +e
      python "${uni2_script}" \\
        --image "${image_tif}" \\
        --mask "${mask_tif}" \\
        --outdir "\$OUTDIR_TILE" \\
        --paired-inner-square-outdir "\$OUTDIR_INNER" \\
        --paired-inner-square-mode ${paired_inner_square_mode} \\
        --image-level ${params.uni2_image_level} \\
        ${force_full_flag} \\
        --grid "${params.uni2_grid}" \\
        --tile-size ${params.uni2_tile_size} \\
        --target-mpp ${params.uni2_target_mpp} \\
        --default-source-mpp ${params.uni2_default_source_mpp} \\
        ${save_tiles_flag} \\
        --tiles-root "${tiles_root_path}" \\
        --bucket-size ${params.uni2_bucket_size} \\
        --min-area ${params.uni2_min_area} \\
        --mask-context-mode none \\
        --inner-square-factor ${params.uni2_inner_square_factor} \\
        --inner-square-min-px ${params.uni2_inner_square_min_px} \\
        --inner-square-max-px ${params.uni2_inner_square_max_px} \\
        --inner-square-fixed-px ${params.uni2_inner_square_fixed_px} \\
        --encoder "${params.uni2_encoder}" \\
        --backend "${params.uni2_backend}" \\
        --pooling "${params.uni2_pooling}" \\
        --img-size ${params.uni2_img_size} \\
        --device "${device_value}" \\
        --batch "\$ATTEMPT_BATCH" \\
        --torch-threads ${resolved_torch_threads} \\
        --rows-per-csv ${resolved_rows_per_csv} \\
        --mask-block ${params.uni2_mask_block} \\
        2> >(tee "\$ATTEMPT_ERR" >&2)
      RC=\$?
      set -e

      if [[ "\$RC" -eq 0 ]]; then
        echo "[INFO] UNI2 shared tile+inner_square succeeded with batch=\$ATTEMPT_BATCH"
        break
      fi

      if grep -Eiq 'No space left on device|not enough free disk space' "\$ATTEMPT_ERR"; then
        echo "[ERROR] UNI2 shared cache path has insufficient free disk. Free disk space and rerun." >&2
        exit "\$RC"
      fi

      if [[ "\$RC" -ne 137 && "\$RC" -ne 134 && "\$RC" -ne 9 ]] && ! grep -Eiq 'Killed|out of memory|cannot allocate memory' "\$ATTEMPT_ERR"; then
        echo "[ERROR] UNI2 shared failed with non-OOM error (exit=\$RC)." >&2
        exit "\$RC"
      fi

      if [[ "\$ATTEMPT_BATCH" -le 1 ]]; then
        echo "[ERROR] UNI2 shared failed even at batch=1 (exit=\$RC)." >&2
        exit "\$RC"
      fi

      NEXT_BATCH=\$(( ATTEMPT_BATCH / 2 ))
      if [[ "\$NEXT_BATCH" -lt 1 ]]; then
        NEXT_BATCH=1
      fi
      if [[ "\$NEXT_BATCH" -ge "\$ATTEMPT_BATCH" ]]; then
        NEXT_BATCH=\$(( ATTEMPT_BATCH - 1 ))
      fi
      echo "[WARN] UNI2 shared failed with batch=\$ATTEMPT_BATCH (exit=\$RC). Retrying with batch=\$NEXT_BATCH." >&2
      ATTEMPT_BATCH="\$NEXT_BATCH"
    done
    """

    stub:
    """
    mkdir -p "embeddings_${sample_id}_tile" "embeddings_${sample_id}_inner_square"
    touch "embeddings_${sample_id}_tile/embeddings_000000.csv"
    touch "embeddings_${sample_id}_inner_square/embeddings_000000.csv"
    """
}
