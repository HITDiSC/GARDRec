#!/usr/bin/env bash

# Pick an idle physical GPU for the next process.
# Usage: GPU_INDEX=$(select_free_gpu auto)
# The returned value is the nvidia-smi index, suitable for CUDA_VISIBLE_DEVICES.
select_free_gpu() {
  local requested="${1:-auto}"
  local max_used_mib="${MAX_GPU_USED_MIB:-2048}"
  local max_util="${MAX_GPU_UTIL:-10}"
  local min_free_mib="${MIN_FREE_MIB:-20000}"
  local wait_sec="${GPU_WAIT_SEC:-30}"
  local max_wait_sec="${GPU_MAX_WAIT_SEC:-7200}"
  local avoid="${AVOID_GPU:-}"
  local start_ts
  local selected

  select_from() {
    local avoid_list="$1"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' \
        -v max_used="${max_used_mib}" \
        -v max_util="${max_util}" \
        -v min_free="${min_free_mib}" \
        -v avoid="${avoid_list}" '
          BEGIN {
            n = split(avoid, parts, ",")
            for (i = 1; i <= n; i++) {
              gsub(/^ +| +$/, "", parts[i])
              if (parts[i] != "") skip[parts[i]] = 1
            }
          }
          $1 ~ /^[0-9]+$/ &&
          $2 ~ /^[0-9]+$/ &&
          $3 ~ /^[0-9]+$/ &&
          $4 ~ /^[0-9]+$/ &&
          !($1 in skip) &&
          $2 <= max_used &&
          $4 <= max_util &&
          ($3 - $2) >= min_free {
            printf "%s %s %s\n", $2, $4, $1
          }
        ' |
      sort -n -k1,1 -k2,2 -k3,3 |
      head -n 1 |
      awk '{print $3}'
  }

  if [[ "${requested}" != "auto" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  start_ts=$(date +%s)
  while true; do
    selected=$(select_from "${avoid}")

    if [[ -z "${selected}" && -n "${avoid}" ]]; then
      echo "[gpu-select] no idle GPU outside AVOID_GPU=${avoid}; allowing avoided GPU" >&2
      selected=$(select_from "")
    fi

    if [[ -n "${selected}" ]]; then
      printf '%s\n' "${selected}"
      return 0
    fi

    if (( $(date +%s) - start_ts >= max_wait_sec )); then
      echo "ERROR: no idle GPU after ${max_wait_sec}s (need used<=${max_used_mib} MiB, util<=${max_util}%, free>=${min_free_mib} MiB)" >&2
      return 1
    fi
    echo "[gpu-select] no idle GPU yet; retrying in ${wait_sec}s" >&2
    sleep "${wait_sec}"
  done
}
