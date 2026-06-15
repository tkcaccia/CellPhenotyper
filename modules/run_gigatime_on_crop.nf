process RUN_GIGATIME_ON_CROP {
    tag "${sample_id}"
    label 'compute_heavy'
    label 'gpu_capable'
    maxForks 1

    publishDir "${params.outdir_base}/05_gigatime/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "gigatime_${sample_id}"
    publishDir "${params.outdir_base}/05_gigatime/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true, pattern: "quantification_${sample_id}"

    cpus { Math.max(1, Math.min(params.max_cpus as int, params.gigatime_cpus as int)) }
    memory { "${Math.max(2, Math.min(params.max_memory_gb as int, params.gigatime_memory_gb as int))} GB" }
    time { params.gigatime_time as String }

    input:
    tuple val(sample_id), path(crop_tif), path(shift_json), path(nuclei_mask_tif), path(cyto_mask_tif)

    output:
    tuple val(sample_id), path("gigatime_${sample_id}"), emit: gigatime_dir
    tuple val(sample_id), path("quantification_${sample_id}"), emit: quant_dir

    script:
    def gigatime_script = "${projectDir}/${params.gigatime_script}"
    def device_value = params.compute_device == 'gpu' ? 'cuda' : 'cpu'
    def token_env_file = params.hf_token_env_file ? (params.hf_token_env_file.toString().startsWith('/') ? params.hf_token_env_file : "${projectDir}/${params.hf_token_env_file}") : ''
    def strict_target_flag = params.gigatime_strict_target_mpp ? '--strict-target-mpp' : ''
    def pyramid_flag = params.gigatime_output_pyramid ? '--pyramid' : ''
    def blockwise_flag = params.gigatime_blockwise ? '--blockwise' : ''
    def predictor_flag = params.gigatime_output_predictor ? '--predictor' : ''
    def skip_background_flag = params.gigatime_skip_background_blocks ? '--skip-background-blocks' : ''
    def jpg_save_tiles_flag = params.gigatime_jpg_save_tiles ? '--jpg-save-tiles' : ''
    def output_format = params.gigatime_output_format ?: 'ome_tiff'
    def auto_hardware_flag = params.gigatime_auto_hardware ? '--auto-hardware' : ''
    def task_mem_gb = task.memory ? Math.max(1, task.memory.toGiga() as int) : Math.max(1, params.max_memory_gb as int)
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
    mkdir -p "quantification_${sample_id}"

    python "${gigatime_script}" \\
      --image "${crop_tif}" \\
      --shift-json "${shift_json}" \\
      --outdir "gigatime_${sample_id}" \\
      --nuclei-mask "${nuclei_mask_tif}" \\
      --cyto-mask "${cyto_mask_tif}" \\
      --quant-dir "quantification_${sample_id}" \\
      --repo-id "${params.gigatime_repo_id}" \\
      --page ${params.gigatime_page} \\
      --patch-size ${params.gigatime_patch_size} \\
      --stride ${params.gigatime_stride} \\
      --batch-size ${params.gigatime_batch_size} \\
      ${auto_hardware_flag} \\
      --task-memory-gb ${task_mem_gb} \\
      --min-free-system-gb ${params.gigatime_min_free_system_gb} \\
      --device "${device_value}" \\
      --auto-threshold-mpix ${params.gigatime_auto_threshold_mpix} \\
      --max-side ${params.gigatime_max_side} \\
      --target-mpp ${params.gigatime_target_mpp} \\
      --output-format "${output_format}" \\
      --output-dtype "${params.gigatime_output_dtype}" \\
      --output-channels "${params.gigatime_output_channels ?: ''}" \\
      --jpg-markers "${params.gigatime_jpg_markers}" \\
      --jpg-quality ${params.gigatime_jpg_quality} \\
      --jpg-preview-max-side ${params.gigatime_jpg_preview_max_side} \\
      ${jpg_save_tiles_flag} \\
      --block-size ${params.gigatime_block_size} \\
      --skip-background-downsample ${params.gigatime_skip_background_downsample} \\
      --skip-background-min-fraction ${params.gigatime_skip_background_min_fraction} \\
      --skip-background-close-radius ${params.gigatime_skip_background_close_radius} \\
      --skip-background-min-obj-area ${params.gigatime_skip_background_min_obj_area} \\
      --skip-background-hole-area ${params.gigatime_skip_background_hole_area} \\
      --max-output-gib ${params.gigatime_max_output_gib} \\
      --disk-buffer-threshold-gib ${params.gigatime_disk_buffer_threshold_gib} \\
      ${strict_target_flag} \\
      ${pyramid_flag} \\
      ${blockwise_flag} \\
      ${predictor_flag} \\
      ${skip_background_flag} \\
      --compression "${params.gigatime_output_compression}"
    """

    stub:
    """
    mkdir -p "gigatime_${sample_id}"
    mkdir -p "quantification_${sample_id}"
    """
}
