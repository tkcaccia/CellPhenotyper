process EXTRACT_UNI2_EMBEDDINGS_SHARED {
    cache 'deep'
    ext code_fingerprint: { ProcessCode.fingerprint(projectDir, 'extract_uni2_embeddings_shared', params) },
        source_fingerprint: { ProcessCode.directoryFingerprint([image_tif, mask_tif, observations_file, resolution_file]) }
    tag "${sample_id}:${observation_mode}:tile_inner_square_fixed${params.uni2_inner_square_fixed_px}"
    label 'compute_heavy'
    label 'gpu_capable'

    publishDir "${params.outdir_base}/09_embeddings/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus {
      def requested = TaskRuntime.cpus(runtime_plan, 'uni2')
      def compute = TaskRuntime.device(runtime_plan)
      def maxMemGb = runtime_plan.memory_budget_gb as double
      if (compute != 'gpu' && maxMemGb <= 3) return 1
      return requested
    }
    memory { TaskRuntime.memory(runtime_plan, 'uni2') }
    time { params.uni2_time as String }

    input:
    tuple val(sample_id), path(image_tif), path(mask_tif), val(observation_mode), path(observations_file), path(resolution_file)
    val(runtime_plan)

    output:
    tuple val(sample_id), val('tile'), path("embeddings_${sample_id}_tile"), emit: tile_embeddings_dir
    tuple val(sample_id), val('inner_square'), path("embeddings_${sample_id}_inner_square"), emit: inner_square_embeddings_dir

    script:
    def force_full_flag = params.uni2_force_full_image ? '--force-full-image' : ''
    def save_tiles_flag = params.uni2_save_tiles ? '--save-tiles' : ''
    def tiles_root_path = "embeddings_${sample_id}_tile/${params.uni2_tiles_root}"
    def resolvedComputeDevice = TaskRuntime.device(runtime_plan)
    def resolvedHardwareProfile = TaskRuntime.profile(runtime_plan)
    def device_value = resolvedComputeDevice == 'gpu' ? 'cuda' : 'cpu'
    def task_mem_gb = task.memory ? Math.max(1, task.memory.toGiga() as int) : Math.max(1, runtime_plan.memory_budget_gb as int)
    def requested_batch = Math.max(1, params.uni2_batch as int)
    def encoder_batch_cap = ['uni2-h': 256, 'virchow': 16, 'virchow2': 16, 'phikon-v2': 64]
      .getOrDefault(params.uni2_encoder.toString().toLowerCase(), 1) as int
    def initial_batch = Math.min(requested_batch, encoder_batch_cap)
    if (device_value == 'cpu') {
      if (task_mem_gb <= 4) initial_batch = Math.min(initial_batch, 2)
      else if (task_mem_gb <= 8) initial_batch = Math.min(initial_batch, 4)
      else if (task_mem_gb <= 12) initial_batch = Math.min(initial_batch, 8)
      else if (task_mem_gb <= 16) initial_batch = Math.min(initial_batch, 16)
    }
    def configured_torch_threads = params.containsKey('uni2_torch_threads') ? params.uni2_torch_threads : 1
    def resolved_torch_threads = Math.max(1, Math.min(task.cpus as int, TaskRuntime.setting(runtime_plan, 'uni2_torch_threads', configured_torch_threads) as int))
    if (device_value == 'cpu' && task_mem_gb <= 4) {
      resolved_torch_threads = 1
    }
    def resolved_rows_per_csv = Math.max(1000, params.uni2_rows_per_csv as int)
    if (device_value == 'cpu' && task_mem_gb <= 4) {
      resolved_rows_per_csv = Math.min(resolved_rows_per_csv, 2000)
    }
    def paired_inner_square_mode = (params.uni2_paired_inner_square_mode ?: 'token_subset').toString()
    def observation_args = observation_mode == 'grid'
      ? "--objects-csv \"${observations_file}\" --observation-type grid"
      : '--observation-type cell'
    def uni2_script = "${projectDir}/${params.uni2_script}"
    def embeddingStorage = (params.uni2_embedding_storage ?: 'csv').toString()
    if (!(embeddingStorage in ['csv', 'binary'])) error "uni2_embedding_storage must be csv or binary"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    [uni2_script, "${projectDir}/bin/uni2_grid.py", "${projectDir}/bin/uni2_embedding_io.py"].each { codeDigest.update(new File(it).bytes) }
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    def token_env_file = params.hf_token_env_file ? (params.hf_token_env_file.toString().startsWith('/') ? params.hf_token_env_file : "${projectDir}/${params.hf_token_env_file}") : ''
    """
    set -euo pipefail
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    echo "[INFO] UNI2 shared code fingerprint: ${codeFingerprint}"
    python -c 'import json,math,sys; d=json.load(open(sys.argv[1])); x=float(d["mpp_x"]); y=float(d["mpp_y"]); (d.get("status")=="pass" and all(math.isfinite(v) and 0.01<=v<=10 for v in (x,y)) and math.isclose(x,y,rel_tol=1e-4)) or sys.exit("UNI2 requires a passed finite isotropic source-resolution report"); all(math.isclose(float(d[k]),(x+y)/2,rel_tol=1e-4) for k in ("source_mpp","source_mpp_x","source_mpp_y","effective_mpp") if d.get(k) is not None) or sys.exit("UNI2 source-resolution aliases conflict")' "${resolution_file}"

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
    HARDWARE_AUTO="${params.hardware_auto}"
    UNI2_AUTO_HARDWARE="${params.uni2_auto_hardware}"
    HARDWARE_PROFILE="${resolvedHardwareProfile}"
    UNI2_MAX_AUTO_BATCH="${params.uni2_max_auto_batch}"
    MODEL_BATCH_CAP=${encoder_batch_cap}
    if [[ "${device_value}" == "cuda" && "\$HARDWARE_AUTO" == "true" && "\$UNI2_AUTO_HARDWARE" == "true" ]]; then
      GPU_TOTAL_MIB=0
      GPU_FREE_MIB=0
      GPU_USABLE_MIB=0
      if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_SELECTOR="\${CELLPHENOTYPER_GPU_INDEX:-\${CUDA_VISIBLE_DEVICES:-}}"
        GPU_SELECTOR="\${GPU_SELECTOR%%,*}"
        [[ -n "\$GPU_SELECTOR" && "\$GPU_SELECTOR" != "-1" ]] || GPU_SELECTOR=0
        GPU_MEMORY="\$(nvidia-smi -i "\$GPU_SELECTOR" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
        GPU_FREE_MIB="\$(awk -F, '{gsub(/ /,"",\$1); print int(\$1)}' <<<"\$GPU_MEMORY")"
        GPU_TOTAL_MIB="\$(awk -F, '{gsub(/ /,"",\$2); print int(\$2)}' <<<"\$GPU_MEMORY")"
        GPU_FREE_MIB="\${GPU_FREE_MIB:-0}"
        GPU_TOTAL_MIB="\${GPU_TOTAL_MIB:-0}"
        GPU_RESERVE_MIB="\$(awk -v total="\$GPU_TOTAL_MIB" 'BEGIN {r=int(total*0.08); if(r<768)r=768; if(r>4096)r=4096; print r}')"
        GPU_USABLE_MIB="\$(awk -v free="\$GPU_FREE_MIB" -v reserve="\$GPU_RESERVE_MIB" 'BEGIN {v=free-reserve; print (v>0?v:0)}')"
      fi
      AUTO_BATCH=${requested_batch}
      case "\$HARDWARE_PROFILE" in
        aggressive)
          if [[ "\$GPU_USABLE_MIB" -ge 48000 ]]; then AUTO_BATCH=256
          elif [[ "\$GPU_USABLE_MIB" -ge 24000 ]]; then AUTO_BATCH=192
          elif [[ "\$GPU_USABLE_MIB" -ge 16000 ]]; then AUTO_BATCH=128
          elif [[ "\$GPU_USABLE_MIB" -ge 10000 ]]; then AUTO_BATCH=64
          else AUTO_BATCH=32
          fi
          ;;
        conservative)
          if [[ "\$GPU_USABLE_MIB" -ge 48000 ]]; then AUTO_BATCH=128
          elif [[ "\$GPU_USABLE_MIB" -ge 24000 ]]; then AUTO_BATCH=96
          elif [[ "\$GPU_USABLE_MIB" -ge 16000 ]]; then AUTO_BATCH=64
          elif [[ "\$GPU_USABLE_MIB" -ge 10000 ]]; then AUTO_BATCH=32
          else AUTO_BATCH=16
          fi
          ;;
        *)
          if [[ "\$GPU_USABLE_MIB" -ge 48000 ]]; then AUTO_BATCH=192
          elif [[ "\$GPU_USABLE_MIB" -ge 24000 ]]; then AUTO_BATCH=128
          elif [[ "\$GPU_USABLE_MIB" -ge 16000 ]]; then AUTO_BATCH=96
          elif [[ "\$GPU_USABLE_MIB" -ge 10000 ]]; then AUTO_BATCH=64
          else AUTO_BATCH=32
          fi
          ;;
      esac
      if [[ ${task_mem_gb} -le 16 && "\$AUTO_BATCH" -gt 64 ]]; then AUTO_BATCH=64; fi
      if [[ ${task_mem_gb} -le 24 && "\$AUTO_BATCH" -gt 128 ]]; then AUTO_BATCH=128; fi
      if [[ "\$AUTO_BATCH" -gt "\$UNI2_MAX_AUTO_BATCH" ]]; then AUTO_BATCH="\$UNI2_MAX_AUTO_BATCH"; fi
      if [[ "\$AUTO_BATCH" -gt "\$MODEL_BATCH_CAP" ]]; then AUTO_BATCH="\$MODEL_BATCH_CAP"; fi
      ATTEMPT_BATCH="\$AUTO_BATCH"
    fi
    echo "[INFO] Foundation encoder shared runtime tune: encoder=${params.uni2_encoder}, mem_gb=${task_mem_gb}, gpu_free_mib=\${GPU_FREE_MIB:-0}, gpu_usable_mib=\${GPU_USABLE_MIB:-0}, requested_batch=${requested_batch}, model_batch_cap=\$MODEL_BATCH_CAP, start_batch=\$ATTEMPT_BATCH, profile=\${HARDWARE_PROFILE}, torch_threads=${resolved_torch_threads}, rows_per_csv=${resolved_rows_per_csv}"

    while true; do
      mkdir -p "\$OUTDIR_TILE" "\$OUTDIR_INNER"

      ATTEMPT_ERR=".uni2_shared_attempt_batch_\${ATTEMPT_BATCH}.err"
      set +e
      python "${uni2_script}" \\
        --image "${image_tif}" \\
        --mask "${mask_tif}" \\
        --resolution-json "${resolution_file}" \\
        ${observation_args} \\
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
        --embedding-storage "${embeddingStorage}" \\
        --embedding-mode tile \\
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
    echo "[INFO] Process code cache fingerprint: ${task.ext.code_fingerprint}"
    echo "[INFO] Process directory cache fingerprint: ${task.ext.source_fingerprint}"
    mkdir -p "embeddings_${sample_id}_tile" "embeddings_${sample_id}_inner_square"
    touch "embeddings_${sample_id}_tile/embeddings_000000.csv"
    touch "embeddings_${sample_id}_inner_square/embeddings_000000.csv"
    printf '{"model_provenance":{"used_model":false}}\n' > "embeddings_${sample_id}_tile/.${params.uni2_encoder}_embedding_complete.json"
    printf '{"model_provenance":{"used_model":false}}\n' > "embeddings_${sample_id}_inner_square/.${params.uni2_encoder}_embedding_complete.json"
    """
}
