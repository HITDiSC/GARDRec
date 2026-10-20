#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/WY/GARDRec}
cd "${REPO_ROOT}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/gpu_utils.sh"

RUN_NAME=${RUN_NAME:-kragrec_qwen2_ml1m}
BASE_OUT=${BASE_OUTPUT_DIR:-output/${RUN_NAME}}
OUT=${OUTPUT_DIR:-output/${RUN_NAME}_ep5}
DATA=${DATA_DIR:-data/dataset/ML1M}
INDEX=${INDEX_DIR:-data/dataset/kg_ml1m_kragrec}
PY=${PY:-/mnt/sda/miniconda3/envs/GARDRec/bin/python}
DEPS=${DEPS_DIR:-baselines/kragrec_qwen2/.deps_gardrec}
GPU_MODE=${GPU:-auto}
BASE_CKPT=${BASE_CHECKPOINT:-${BASE_OUT}/checkpoint_epoch3.pth}
EVAL_RUN_NAME=${EVAL_RUN_NAME:-${RUN_NAME}_ep5_selected}

export PYTHONPATH="${DEPS}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
case "${PY}" in *tokenrec-ads*|*aaai*) echo "ERROR: refusing ${PY}"; exit 4;; esac
[[ "${OUT}" != "${BASE_OUT}" ]] || { echo "ERROR: output dir must differ"; exit 4; }
[[ -f "${BASE_CKPT}" ]] || { echo "ERROR: missing ${BASE_CKPT}"; exit 4; }

mkdir -p "${OUT}/epoch3_eval/results" "${OUT}/epoch4_eval" "${OUT}/epoch5_eval" "${OUT}/final_eval"
exec 9>"${OUT}/continue_ep5.lock"
flock -n 9 || { echo "already running"; exit 0; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

activate_gpu() {
  local selected
  selected=$(select_free_gpu "${GPU_MODE}") || exit 5
  GPU="${selected}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  log "selected GPU ${GPU} for $1"
}

eval_valid() {
  local epoch="$1" checkpoint="$2"
  mkdir -p "${OUT}/epoch${epoch}_eval"
  activate_gpu "epoch${epoch} validation"
  "${PY}" -u "${SCRIPT_DIR}/evaluate.py" --eval_file "${DATA}/kragrec_v2_valid.json" \
    --index_dir "${INDEX}" --output_dir "${OUT}/epoch${epoch}_eval" \
    --run_name "${RUN_NAME}_epoch${epoch}" --checkpoint "${checkpoint}" --device cuda
}
cp -f "${BASE_CKPT}" "${OUT}/checkpoint_epoch3.pth"
BASE_EP3="${BASE_OUT}/epoch3_eval/results/${RUN_NAME}_epoch3_kragrec_v2_valid.json"
EP3="${OUT}/epoch3_eval/results/${RUN_NAME}_epoch3_kragrec_v2_valid.json"
if [[ -f "${BASE_EP3}" && ! -f "${EP3}" ]]; then cp -f "${BASE_EP3}" "${EP3}"; fi
if [[ ! -f "${EP3}" ]]; then eval_valid 3 "${OUT}/checkpoint_epoch3.pth"; fi

activate_gpu "epoch4-epoch5 training"
log "resuming epoch4-epoch5 from ${OUT}/checkpoint_epoch3.pth"
GPU="${GPU}" EPOCHS=2 EPOCH_OFFSET=3 RESUME="${OUT}/checkpoint_epoch3.pth" \
  RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUT}" INDEX_DIR="${INDEX}" DATA_DIR="${DATA}" \
  PY="${PY}" DEPS_DIR="${DEPS}" bash "${SCRIPT_DIR}/run_server.sh"

log "epoch4/epoch5 training finished; evaluating valid checkpoints"
for EP in 4 5; do
  CKPT="${OUT}/checkpoint_epoch${EP}.pth"
  if [[ ! -f "${CKPT}" ]]; then log "missing ${CKPT}; skipping"; continue; fi
  eval_valid "${EP}" "${CKPT}"
done

log "selecting best checkpoint among epochs 3, 4, and 5 by validation ACC"
"${PY}" - "${OUT}" "${RUN_NAME}" <<'PY'
import json
import os
import shutil
import sys

out, run = sys.argv[1], sys.argv[2]
rows = []
for epoch in (3, 4, 5):
    checkpoint = os.path.join(out, "checkpoint_epoch{}.pth".format(epoch))
    result_path = os.path.join(
        out, "epoch{}_eval".format(epoch), "results",
        "{}_epoch{}_kragrec_v2_valid.json".format(run, epoch),
    )
    if not os.path.exists(checkpoint) or not os.path.exists(result_path):
        continue
    metrics = json.load(open(result_path, encoding="utf-8"))["metrics"]
    rows.append({
        "epoch": epoch,
        "checkpoint": checkpoint,
        "valid_acc": metrics["acc"],
        "valid_mrr": metrics["mrr"],
        "valid_recall@5": metrics["recall@5"],
    })
if not rows:
    raise SystemExit("SELECT_FAILED: no valid evaluation results")
rows.sort(key=lambda item: (item["valid_acc"], item["valid_mrr"]), reverse=True)
best = rows[0]
selected = os.path.join(out, "checkpoint_selected.pth")
shutil.copyfile(best["checkpoint"], selected)
with open(os.path.join(out, "selected_checkpoint.txt"), "w", encoding="utf-8") as handle:
    handle.write(selected + "\n")
with open(os.path.join(out, "checkpoint_selection.json"), "w", encoding="utf-8") as handle:
    json.dump({"selected": best, "candidates": rows}, handle, indent=2, ensure_ascii=False)
print(json.dumps({"selected": best, "candidates": rows}, indent=2, ensure_ascii=False), flush=True)
PY

SELECTED=$(cat "${OUT}/selected_checkpoint.txt")
log "selected checkpoint: ${SELECTED}"
for SPLIT in easy_test hard10_test; do
  activate_gpu "final ${SPLIT} evaluation"
  "${PY}" -u "${SCRIPT_DIR}/evaluate.py" --eval_file "${DATA}/kragrec_v2_${SPLIT}.json" \
    --index_dir "${INDEX}" --output_dir "${OUT}/final_eval" \
    --run_name "${EVAL_RUN_NAME}" --checkpoint "${SELECTED}" --device cuda
done
log "all done: ${OUT}"