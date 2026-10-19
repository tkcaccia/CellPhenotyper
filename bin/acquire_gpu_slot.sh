#!/usr/bin/env bash

# Source this file from a Nextflow beforeScript. Open flock descriptors remain
# attached to the task shell and are released automatically on exit or kill.
cellphenotyper_acquire_gpu_slot() {
  local lock_root="$1" required_gb="$2" reserve_gb="$3" token_gb="$4"
  local max_tasks="$5" poll_seconds="$6" timeout_seconds="$7"

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[ERROR] GPU scheduling requested but nvidia-smi is unavailable" >&2
    return 2
  }
  command -v flock >/dev/null 2>&1 || {
    echo "[ERROR] GPU scheduling requires flock (util-linux)" >&2
    return 2
  }
  mkdir -p "$lock_root"

  required_gb="$(awk -v x="$required_gb" 'BEGIN { print (x > 0 ? x : 1) }')"
  reserve_gb="$(awk -v x="$reserve_gb" 'BEGIN { print (x >= 0 ? x : 0) }')"
  token_gb="$(awk -v x="$token_gb" 'BEGIN { print (x > 0 ? x : 1) }')"
  max_tasks="${max_tasks:-0}"
  poll_seconds="${poll_seconds:-5}"
  timeout_seconds="${timeout_seconds:-172800}"

  local started now elapsed
  started="$(date +%s)"
  while true; do
    while IFS=',' read -r gpu_index total_mib free_mib; do
      gpu_index="${gpu_index//[[:space:]]/}"
      total_mib="${total_mib//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      [[ "$gpu_index" =~ ^[0-9]+$ && "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] || continue

      local total_gb free_gb usable_gb token_count required_tokens task_slots
      total_gb="$(awk -v x="$total_mib" 'BEGIN { printf "%.6f", x/1024 }')"
      free_gb="$(awk -v x="$free_mib" 'BEGIN { printf "%.6f", x/1024 }')"
      usable_gb="$(awk -v total="$total_gb" -v reserve="$reserve_gb" 'BEGIN { print total-reserve }')"
      token_count="$(awk -v usable="$usable_gb" -v unit="$token_gb" 'BEGIN { n=int(usable/unit); print (n>0?n:0) }')"
      required_tokens="$(awk -v req="$required_gb" -v unit="$token_gb" 'BEGIN { print int((req+unit-0.000001)/unit) }')"
      awk -v free="$free_gb" -v req="$required_gb" -v reserve="$reserve_gb" 'BEGIN { exit !(free >= req+reserve) }' || continue
      (( token_count >= required_tokens )) || continue

      task_slots="$max_tasks"
      if (( task_slots <= 0 )); then task_slots="$token_count"; fi

      local task_fd="" task_slot=-1 slot fd acquired=0
      for ((slot=0; slot<task_slots; slot++)); do
        exec {fd}>"$lock_root/gpu-${gpu_index}-task-${slot}.lock"
        if flock -n "$fd"; then
          task_fd="$fd"
          task_slot="$slot"
          break
        fi
        eval "exec ${fd}>&-"
      done
      (( task_slot >= 0 )) || continue

      local -a token_fds=()
      for ((slot=0; slot<token_count && acquired<required_tokens; slot++)); do
        exec {fd}>"$lock_root/gpu-${gpu_index}-memory-${slot}.lock"
        if flock -n "$fd"; then
          token_fds+=("$fd")
          ((acquired+=1))
        else
          eval "exec ${fd}>&-"
        fi
      done
      if (( acquired < required_tokens )); then
        for fd in "${token_fds[@]}"; do eval "exec ${fd}>&-"; done
        eval "exec ${task_fd}>&-"
        continue
      fi

      export CUDA_VISIBLE_DEVICES="$gpu_index"
      export CELLPHENOTYPER_GPU_INDEX="$gpu_index"
      export CELLPHENOTYPER_GPU_REQUIRED_GB="$required_gb"
      export CELLPHENOTYPER_GPU_TASK_LOCK_FD="$task_fd"
      export CELLPHENOTYPER_GPU_MEMORY_LOCK_FDS="${token_fds[*]}"
      echo "[INFO] GPU admission: gpu=${gpu_index} total_gb=${total_gb} free_gb=${free_gb} required_gb=${required_gb} memory_tokens=${required_tokens}/${token_count} task_slot=${task_slot}/${task_slots}"
      nvidia-smi --id="$gpu_index" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
      return 0
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | sort -t, -k3,3nr)

    now="$(date +%s)"
    elapsed=$((now-started))
    if (( elapsed >= timeout_seconds )); then
      echo "[ERROR] Timed out after ${elapsed}s waiting for a GPU with ${required_gb} GB free plus ${reserve_gb} GB reserve" >&2
      return 3
    fi
    if (( elapsed == 0 || elapsed % 60 < poll_seconds )); then
      echo "[INFO] Waiting for GPU admission: required_gb=${required_gb}, reserve_gb=${reserve_gb}, elapsed_s=${elapsed}" >&2
    fi
    sleep "$poll_seconds"
  done
}
