#!/usr/bin/env bash

# Resolve one inherited visibility constraint against the host NVML inventory.
# Documentation:
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
# Unset means no *host-side* constraint; explicit empty/none/void/-1 never does.
# UUID prefixes must be unique. We intentionally reject CUDA's invalid-token
# truncation syntax and MIG selectors: this allocator leases whole GPUs only.
cellphenotyper_gpu_visibility_indices() {
  local variable="$1" is_set="$2" restriction="$3" inventory="$4"
  local token gpu_index gpu_uuid total_mib free_mib matched count allowed=""
  local -a tokens=()
  if [[ -z "$is_set" || ( "$variable" == NVIDIA_VISIBLE_DEVICES && "$restriction" == all ) ]]; then
    while IFS=',' read -r gpu_index gpu_uuid total_mib free_mib; do
      printf '%s ' "$gpu_index"
    done <<< "$inventory"
    return 0
  fi
  case "$restriction" in
    ''|none|void|-1)
      echo "[ERROR] ${variable} explicitly disables GPU visibility" >&2
      return 2
      ;;
  esac
  if [[ "$restriction" == *[[:space:]]* || "$restriction" == ,* || "$restriction" == *, || "$restriction" == *,,* ]]; then
    echo "[ERROR] Unsupported ${variable} list; use unambiguous GPU indices or UUIDs" >&2
    return 2
  fi
  IFS=',' read -r -a tokens <<< "$restriction"
  for token in "${tokens[@]}"; do
    case "$token" in
      MIG-*)
        echo "[ERROR] ${variable} requests MIG; whole-GPU admission does not support MIG allocations" >&2
        return 2
        ;;
    esac
    if [[ ! "$token" =~ ^(0|[1-9][0-9]*)$ && ! "$token" =~ ^GPU-[0-9a-fA-F][0-9a-fA-F-]*$ ]]; then
      echo "[ERROR] Unsupported ${variable} selector '${token}'; invalid-token truncation is not supported" >&2
      return 2
    fi
    matched=""
    count=0
    while IFS=',' read -r gpu_index gpu_uuid total_mib free_mib; do
      if [[ "$token" == GPU-* && -z "$gpu_uuid" ]]; then
        echo "[ERROR] Cannot resolve ${variable} UUIDs with an incomplete GPU UUID inventory" >&2
        return 2
      fi
      if [[ "$token" == "$gpu_index" || ( "$token" == GPU-* && "$gpu_uuid" == "$token"* ) ]]; then
        matched="$gpu_index"
        ((count+=1))
      fi
    done <<< "$inventory"
    if (( count != 1 )); then
      echo "[ERROR] ${variable} selector '${token}' is unknown or ambiguous in the host GPU inventory" >&2
      return 2
    fi
    if [[ " $allowed " == *" $matched "* ]]; then
      echo "[ERROR] ${variable} contains duplicate GPU selectors" >&2
      return 2
    fi
    allowed+="${matched} "
  done
  printf '%s' "$allowed"
}

# Source this file from a Nextflow beforeScript. Open flock descriptors remain
# attached to the task shell and are released automatically on exit or kill.
# The existing dynamic descriptor allocation below requires Bash >= 4.1.
cellphenotyper_acquire_gpu_slot() {
  local lock_root="$1" required_gb="$2" reserve_gb="$3" token_gb="$4"
  local max_tasks="$5" poll_seconds="$6" timeout_seconds="$7"
  # Snapshot before the eventual single-device export. Never widen an inherited
  # scheduler/container allocation, including an explicitly empty allocation.
  local cuda_is_set="${CUDA_VISIBLE_DEVICES+x}" cuda_restriction="${CUDA_VISIBLE_DEVICES-}"
  local nvidia_is_set="${NVIDIA_VISIBLE_DEVICES+x}" nvidia_restriction="${NVIDIA_VISIBLE_DEVICES-}"

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[ERROR] GPU scheduling requested but nvidia-smi is unavailable" >&2
    return 2
  }
  command -v flock >/dev/null 2>&1 || {
    echo "[ERROR] GPU scheduling requires flock (util-linux)" >&2
    return 2
  }
  required_gb="$(awk -v x="$required_gb" 'BEGIN { print (x > 0 ? x : 1) }')"
  reserve_gb="$(awk -v x="$reserve_gb" 'BEGIN { print (x >= 0 ? x : 0) }')"
  token_gb="$(awk -v x="$token_gb" 'BEGIN { print (x > 0 ? x : 1) }')"
  max_tasks="${max_tasks:-0}"
  poll_seconds="${poll_seconds:-5}"
  timeout_seconds="${timeout_seconds:-172800}"

  local started now elapsed
  started="$(date +%s)"
  while true; do
    local inventory normalized="" indices="" uuids="" gpu_index gpu_uuid total_mib free_mib extra
    if ! inventory="$(nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits)"; then
      echo "[ERROR] Cannot obtain GPU inventory for visibility-aware admission" >&2
      return 2
    fi
    while IFS=',' read -r gpu_index gpu_uuid total_mib free_mib extra; do
      gpu_index="${gpu_index//[[:space:]]/}"
      gpu_uuid="${gpu_uuid//[[:space:]]/}"
      total_mib="${total_mib//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      if [[ -n "$extra" || ! "$gpu_index" =~ ^(0|[1-9][0-9]*)$ || ! "$total_mib" =~ ^[0-9]+$ || ! "$free_mib" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid or empty GPU inventory; refusing ambiguous admission" >&2
        return 2
      fi
      case "$gpu_uuid" in
        ''|'N/A'|'[N/A]'|'[NotSupported]') gpu_uuid="" ;;
        *)
          if [[ ! "$gpu_uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]; then
            echo "[ERROR] Invalid GPU UUID inventory; whole-GPU admission does not support MIG" >&2
            return 2
          fi
          ;;
      esac
      if [[ " $indices " == *" $gpu_index "* || ( -n "$gpu_uuid" && " $uuids " == *" $gpu_uuid "* ) ]]; then
        echo "[ERROR] Duplicate GPU inventory identity; refusing ambiguous admission" >&2
        return 2
      fi
      indices+="${gpu_index} "
      uuids+="${gpu_uuid} "
      normalized+="${gpu_index},${gpu_uuid},${total_mib},${free_mib}"$'\n'
    done <<< "$inventory"
    # Remove the final newline so a here-string does not create an empty row.
    normalized="${normalized%$'\n'}"
    local cuda_allowed nvidia_allowed eligible=""
    if ! cuda_allowed="$(cellphenotyper_gpu_visibility_indices CUDA_VISIBLE_DEVICES "$cuda_is_set" "$cuda_restriction" "$normalized")"; then return 2; fi
    if ! nvidia_allowed="$(cellphenotyper_gpu_visibility_indices NVIDIA_VISIBLE_DEVICES "$nvidia_is_set" "$nvidia_restriction" "$normalized")"; then return 2; fi
    while IFS=',' read -r gpu_index gpu_uuid total_mib free_mib; do
      if [[ " $cuda_allowed " == *" $gpu_index "* && " $nvidia_allowed " == *" $gpu_index "* ]]; then
        eligible+="${gpu_index},${gpu_uuid},${total_mib},${free_mib}"$'\n'
      fi
    done <<< "$normalized"
    if [[ -z "$eligible" ]]; then
      echo "[ERROR] CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES permit no common GPU; refusing to widen visibility" >&2
      return 2
    fi
    mkdir -p "$lock_root" || return 2
    while IFS=',' read -r gpu_index gpu_uuid total_mib free_mib; do

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

      # CUDA ordinals can be reordered/remapped at container boundaries. UUIDs
      # preserve device identity; keep NVML index-based locks for compatibility.
      export CUDA_VISIBLE_DEVICES="${gpu_uuid:-$gpu_index}"
      export CELLPHENOTYPER_GPU_INDEX="$gpu_index"
      export CELLPHENOTYPER_GPU_REQUIRED_GB="$required_gb"
      export CELLPHENOTYPER_GPU_TASK_LOCK_FD="$task_fd"
      export CELLPHENOTYPER_GPU_MEMORY_LOCK_FDS="${token_fds[*]}"
      echo "[INFO] GPU admission: gpu=${gpu_index} total_gb=${total_gb} free_gb=${free_gb} required_gb=${required_gb} memory_tokens=${required_tokens}/${token_count} task_slot=${task_slot}/${task_slots}"
      nvidia-smi --id="$gpu_index" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
      return 0
    done < <(printf '%s' "$eligible" | sort -t, -k4,4nr)

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
