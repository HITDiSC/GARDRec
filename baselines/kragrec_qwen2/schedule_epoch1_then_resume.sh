#!/usr/bin/env bash
set -euo pipefail

cd /data/WY/GARDRec

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/gpu_utils.sh"

RUN_NAME=${RUN_NAME:-kragrec_qwen2_ml1m}
OUT=${OUTPUT_DIR:-output/${RUN_NAME}}
DATA=${DATA_DIR:-data/dataset/ML1M}
INDEX=${INDEX_DIR:-data/dataset/kg_ml1m_kragrec}
PY=${PY:-/mnt/sda/miniconda3/envs/GARDRec/bin/python}
DEPS=${DEPS_DIR:-baselines/kragrec_qwen2/.deps_gardrec}
GPU_MODE=${GPU:-auto}
AVOID_GPU=${AVOID_GPU:-}

export PYTHONPATH="${DEPS}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
case "${PY}" in
  *tokenrec-ads*|*aaai*) echo "ERROR: refusing to schedule with ${PY}"; exit 4 ;;
esac
export PY

mkdir -p "${OUT}" "${OUT}/results" "${OUT}/epoch1_eval" "${OUT}/final_eval"

exec 9>"${OUT}/schedule_resume.lock"
if ! flock -n 9; then
  echo "another resume scheduler is already running"
  exit 0
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

detect_train_gpu() {
  local pid="$1"
  local uuid
  uuid=$(nvidia-smi --query-compute-apps=pid,gpu_uuid \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v pid="${pid}" '$1 == pid {print $2; exit}' || true)
  [[ -n "${uuid}" ]] || return 0
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null |
    awk -F', *' -v uuid="${uuid}" '$2 == uuid {print $1; exit}' || true
}

if [[ -z "${AVOID_GPU}" ]]; then
  EPOCH1_PID=$(pgrep -f "baselines/kragrec_qwen2/train.py.*${RUN_NAME}" | head -n 1 || true)
  if [[ -n "${EPOCH1_PID}" ]]; then
    AVOID_GPU=$(detect_train_gpu "${EPOCH1_PID}")
  fi
fi
export AVOID_GPU

activate_gpu() {
  local stage="$1"
  local selected
  if ! selected=$(select_free_gpu "${GPU_MODE}"); then
    log "ERROR: no idle GPU available for ${stage}"
    exit 5
  fi
  GPU="${selected}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  log "selected GPU ${GPU} for ${stage} (mode=${GPU_MODE})"
}

log "waiting for epoch 1 to finish (run=${RUN_NAME}, gpu_mode=${GPU_MODE}, avoid_gpu=${AVOID_GPU:-none})"
START_TS=$(date +%s)
while true; do
  STATUS=""
  if [[ -f "${OUT}/progress.json" ]]; then
    STATUS=$("${PY}" -c "import json; print(json.load(open('${OUT}/progress.json')).get('status',''))" 2>/dev/null || true)
  fi

  if [[ "${STATUS}" == "completed" && -f "${OUT}/checkpoint_best.pth" ]]; then
    log "epoch 1 reported completed"
    break
  fi

  if ! pgrep -f "baselines/kragrec_qwen2/train.py.*${RUN_NAME}" >/dev/null 2>&1; then
    if [[ -f "${OUT}/checkpoint_best.pth" ]]; then
      log "training process exited; checkpoint exists, continuing"
      break
    fi
    log "ABORT: training process exited without checkpoint_best.pth"
    exit 2
  fi

  if (( $(date +%s) - START_TS > 14400 )); then
    log "ABORT: timeout waiting for epoch 1"
    exit 3
  fi
  sleep 30
done

# Let the old training process release GPU memory.
for _ in $(seq 1 60); do
  if ! pgrep -f "baselines/kragrec_qwen2/train.py.*${RUN_NAME}" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
sleep 3

cp -f "${OUT}/checkpoint_best.pth" "${OUT}/checkpoint_epoch1.pth"
cp -f "${OUT}/progress.json" "${OUT}/progress_epoch1.json" 2>/dev/null || true
log "saved epoch1 checkpoint to ${OUT}/checkpoint_epoch1.pth"

log "checking evaluation data"
"${PY}" -u baselines/kragrec_qwen2/check_eval_data.py \
  --data_dir "${DATA}" \
  --stats "${DATA}/kragrec_v2_stats.json"

cd /data/WY/GARDRec
activate_gpu "epoch1 evaluation"
log "evaluating epoch1 on valid/easy/hard10"
for SPLIT in valid easy_test hard10_test; do
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA}/kragrec_v2_${SPLIT}.json" \
    --index_dir "${INDEX}" \
    --output_dir "${OUT}/epoch1_eval" \
    --run_name "${RUN_NAME}_epoch1" \
    --checkpoint "${OUT}/checkpoint_epoch1.pth" \
    --device cuda
done

log "validating epoch1 metrics"
"${PY}" - "${OUT}/epoch1_eval" "${RUN_NAME}" <<'PY'
import json
import math
import os
import sys

root, run = sys.argv[1], sys.argv[2]
splits = {
    "valid": "kragrec_v2_valid",
    "easy_test": "kragrec_v2_easy_test",
    "hard10_test": "kragrec_v2_hard10_test",
}
required = ["acc", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "mean_target_prob"]
for split, eval_name in splits.items():
    path = os.path.join(root, "results", "{}_epoch1_{}.json".format(run, eval_name))
    if not os.path.exists(path):
        raise SystemExit("METRICS_CHECK_FAILED: missing {}".format(path))
    payload = json.load(open(path, encoding="utf-8"))
    metrics = payload["metrics"]
    predictions = payload["predictions"]
    if metrics.get("num_samples") != 906 or len(predictions) != 906:
        raise SystemExit("METRICS_CHECK_FAILED: {} sample count".format(split))
    for key in required:
        value = float(metrics[key])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise SystemExit("METRICS_CHECK_FAILED: {} {}={}".format(split, key, value))
    if not (
        metrics["recall@10"] >= metrics["recall@5"]
        >= metrics["recall@3"] >= metrics["recall@1"]
    ):
        raise SystemExit("METRICS_CHECK_FAILED: {} recall is not monotonic".format(split))
    for index, item in enumerate(predictions):
        if len(item["scores"]) != 20:
            raise SystemExit("METRICS_CHECK_FAILED: {} row {} score count".format(split, index))
        if abs(sum(item["scores"]) - 1.0) > 1e-3:
            raise SystemExit("METRICS_CHECK_FAILED: {} row {} score sum".format(split, index))
        if not (1 <= int(item["rank"]) <= 20):
            raise SystemExit("METRICS_CHECK_FAILED: {} row {} rank".format(split, index))
    if metrics["acc"] < 0.05:
        raise SystemExit(
            "METRICS_CHECK_FAILED: {} acc={:.4f} is at/below random".format(
                split, metrics["acc"]
            )
        )
    print(
        "METRICS_OK {} acc={:.4f} r@1={:.4f} r@5={:.4f} mrr={:.4f}".format(
            split,
            metrics["acc"],
            metrics["recall@1"],
            metrics["recall@5"],
            metrics["mrr"],
        ),
        flush=True,
    )
print("EPOCH1_EVAL_REASONABLE", flush=True)
PY

log "epoch1 evaluation is reasonable; resuming for 2 more epochs (total 3)"
RESUME="${OUT}/checkpoint_epoch1.pth" \
EPOCHS=2 \
EPOCH_OFFSET=1 \
MODE=train \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUT}" \
GPU="${GPU_MODE}" \
bash baselines/kragrec_qwen2/run_server.sh

log "resumed training finished; evaluating epoch2/epoch3 on valid"
for EP in 2 3; do
  CKPT="${OUT}/checkpoint_epoch${EP}.pth"
  if [[ ! -f "${CKPT}" ]]; then
    log "missing ${CKPT}; skipping"
    continue
  fi
  mkdir -p "${OUT}/epoch${EP}_eval"
  activate_gpu "epoch${EP} validation"
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA}/kragrec_v2_valid.json" \
    --index_dir "${INDEX}" \
    --output_dir "${OUT}/epoch${EP}_eval" \
    --run_name "${RUN_NAME}_epoch${EP}" \
    --checkpoint "${CKPT}" \
    --device cuda
done

log "selecting best checkpoint by validation ACC"
"${PY}" - "${OUT}" "${RUN_NAME}" <<'PY'
import json
import os
import shutil
import sys

out, run = sys.argv[1], sys.argv[2]
candidates = [(1, os.path.join(out, "checkpoint_epoch1.pth"))]
for epoch in (2, 3):
    candidates.append((epoch, os.path.join(out, "checkpoint_epoch{}.pth".format(epoch))))

rows = []
for epoch, path in candidates:
    if not os.path.exists(path):
        continue
    result_dir = os.path.join(out, "epoch{}_eval".format(epoch))
    result_path = os.path.join(
        result_dir, "results", "{}_epoch{}_kragrec_v2_valid.json".format(run, epoch)
    )
    if not os.path.exists(result_path):
        continue
    payload = json.load(open(result_path, encoding="utf-8"))
    metrics = payload["metrics"]
    rows.append(
        {
            "epoch": epoch,
            "checkpoint": path,
            "valid_acc": metrics["acc"],
            "valid_mrr": metrics["mrr"],
            "valid_recall@5": metrics["recall@5"],
        }
    )

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

log "evaluating selected checkpoint on easy/hard10"
for SPLIT in easy_test hard10_test; do
  activate_gpu "final ${SPLIT} evaluation"
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA}/kragrec_v2_${SPLIT}.json" \
    --index_dir "${INDEX}" \
    --output_dir "${OUT}/final_eval" \
    --run_name "${RUN_NAME}_selected" \
    --checkpoint "${SELECTED}" \
    --device cuda
done

log "all done"
