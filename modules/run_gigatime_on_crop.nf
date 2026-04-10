process RUN_GIGATIME_ON_CROP {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/02_gigatime/${sample_id}", mode: 'copy', overwrite: true

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.gigatime_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.gigatime_memory_gb as int))} GB" }
    time { params.gigatime_time as String }

    input:
    tuple val(sample_id), path(crop_tif)

    output:
    tuple val(sample_id), path("gigatime_${sample_id}"), emit: gigatime_dir
    tuple val(sample_id), path("gigatime_${sample_id}/gigatime_probs.ome.tif"), emit: gigatime_tif

    script:
    def gigatime_script = "${projectDir}/${params.gigatime_script}"
    def device_value = params.compute_device == 'gpu' ? 'cuda' : 'cpu'
    def token_env_file = params.hf_token_env_file ? (params.hf_token_env_file.toString().startsWith('/') ? params.hf_token_env_file : "${projectDir}/${params.hf_token_env_file}") : ''
    """
    set -euo pipefail

    TOKEN_ENV_FILE="${token_env_file}"
    TOKEN_VAR_NAME="${params.gigatime_hf_token_env_var_name ?: 'HF_GIGATIME'}"
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
    export HF_TOKEN="\$(printenv "\$TOKEN_VAR_NAME" || true)"
    if [[ -z "\$HF_TOKEN" ]]; then
      export HF_TOKEN="\$(printenv HF_GIGATIME || true)"
    fi
    if [[ -z "\$HF_TOKEN" ]]; then
      export HF_TOKEN="\$(printenv HF_TOKEN || true)"
    fi
    if [[ -z "\$HF_TOKEN" ]]; then
      export HF_TOKEN="\$(printenv ${params.hf_token_env_var_name ?: 'HF_UNI2'} || true)"
    fi
    echo "[INFO] GigaTIME HF cache env: HF_HOME=\$HF_HOME HF_HUB_CACHE=\$HF_HUB_CACHE HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-unset}"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    if [[ "${params.gpu_debug_diagnostics}" == "true" && "${params.compute_device}" == "gpu" ]]; then
      echo "[DEBUG] GigaTIME GPU diagnostics"
      nvidia-smi -L || true
      python - <<'PY'
try:
    import torch
    print(f"[DEBUG] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] cuda_device_0={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[DEBUG] Torch diagnostics unavailable: {exc}")
PY
    fi

    mkdir -p "gigatime_${sample_id}"

    python "${gigatime_script}" \\
      --image "${crop_tif}" \\
      --outdir "gigatime_${sample_id}" \\
      --repo-id "${params.gigatime_repo_id}" \\
      --page ${params.gigatime_page} \\
      --patch-size ${params.gigatime_patch_size} \\
      --stride ${params.gigatime_stride} \\
      --batch-size ${params.gigatime_batch_size} \\
      --device "${device_value}" \\
      --compression "${params.gigatime_output_compression}"
    """

    stub:
    """
    mkdir -p "gigatime_${sample_id}"
    touch "gigatime_${sample_id}/gigatime_probs.ome.tif"
    """
}
