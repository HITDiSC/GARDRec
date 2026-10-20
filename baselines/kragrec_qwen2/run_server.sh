#!/usr/bin/env bash
set -euo pipefail

cd /data/WY/GARDRec

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/gpu_utils.sh"

# Training/eval use our own GARDRec environment plus a project-local
# pip --target directory.  Do not use tokenrec-ads or aaai.
PY=${PY:-/mnt/sda/miniconda3/envs/GARDRec/bin/python}
INDEX_PY=${INDEX_PY:-${PY}}
DEPS_DIR=${DEPS_DIR:-baselines/kragrec_qwen2/.deps_gardrec}
export PYTHONPATH="${DEPS_DIR}:${PYTHONPATH:-}"
case "${PY}" in
  *tokenrec-ads*|*aaai*) echo "ERROR: refusing to run with ${PY}"; exit 4 ;;
esac
case "${INDEX_PY}" in
  *tokenrec-ads*|*aaai*) echo "ERROR: refusing to build index with ${INDEX_PY}"; exit 4 ;;
esac
GPU=${GPU:-auto}
MODE=${MODE:-all}
RUN_NAME=${RUN_NAME:-kragrec_qwen2_ml1m}
INDEX_DIR=${INDEX_DIR:-data/dataset/kg_ml1m_kragrec}
DATA_DIR=${DATA_DIR:-data/dataset/ML1M}
OUTPUT_DIR=${OUTPUT_DIR:-output/${RUN_NAME}}
EPOCHS=${EPOCHS:-1}
EPOCH_OFFSET=${EPOCH_OFFSET:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-5}
MAX_TXT_LEN=${MAX_TXT_LEN:-768}
ADAPTIVE_RATIO=${ADAPTIVE_RATIO:-5}
FORCE_INDEX=${FORCE_INDEX:-0}
FORCE_DATA=${FORCE_DATA:-0}
RESUME=${RESUME:-}

if [[ "${GPU}" == "auto" ]]; then
  GPU=$(select_free_gpu auto)
  echo "auto-selected GPU=${GPU}"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/results"

echo "GPU=${GPU} MODE=${MODE} RUN_NAME=${RUN_NAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

build_index() {
  if [[ "${FORCE_INDEX}" == "1" || ! -f "${INDEX_DIR}/movie_id_to_node.json" ]]; then
    "${INDEX_PY}" -u baselines/kragrec_qwen2/build_kg_index.py --out_dir "${INDEX_DIR}"
  else
    echo "KG index already exists: ${INDEX_DIR}"
  fi
}

build_data() {
  if [[ "${FORCE_DATA}" == "1" || ! -f "${DATA_DIR}/kragrec_v2_hard10_test.json" ]]; then
    "${PY}" -u baselines/kragrec_qwen2/prepare_data.py \
      --input data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json \
      --hard10 baselines/ml1m/v2_hard10_train.json \
      --out_dir "${DATA_DIR}"
  else
    echo "K-RagRec data already exists: ${DATA_DIR}"
  fi
}

train() {
  local extra=()
  if [[ -n "${RESUME}" ]]; then
    extra=(--checkpoint "${RESUME}")
  fi
  "${PY}" -u baselines/kragrec_qwen2/train.py \
    --train_file "${DATA_DIR}/kragrec_v2_train.json" \
    --index_dir "${INDEX_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_epochs "${EPOCHS}" \
    --epoch_offset "${EPOCH_OFFSET}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max_txt_len "${MAX_TXT_LEN}" \
    --adaptive_ratio "${ADAPTIVE_RATIO}" \
    --device cuda \
    "${extra[@]}"
}

evaluate_all() {
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA_DIR}/kragrec_v2_easy_test.json" \
    --index_dir "${INDEX_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --checkpoint "${OUTPUT_DIR}/checkpoint_best.pth" \
    --max_txt_len "${MAX_TXT_LEN}" \
    --adaptive_ratio "${ADAPTIVE_RATIO}" \
    --device cuda
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA_DIR}/kragrec_v2_hard10_test.json" \
    --index_dir "${INDEX_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --checkpoint "${OUTPUT_DIR}/checkpoint_best.pth" \
    --max_txt_len "${MAX_TXT_LEN}" \
    --adaptive_ratio "${ADAPTIVE_RATIO}" \
    --device cuda
}

smoke() {
  "${PY}" -u baselines/kragrec_qwen2/train.py \
    --train_file "${DATA_DIR}/kragrec_v2_train.json" \
    --index_dir "${INDEX_DIR}" \
    --output_dir "${OUTPUT_DIR}_smoke" \
    --run_name "${RUN_NAME}_smoke" \
    --num_epochs 1 --limit_train 16 --batch_size 1 --grad_accum_steps 4 \
    --max_txt_len "${MAX_TXT_LEN}" --device cuda
  "${PY}" -u baselines/kragrec_qwen2/evaluate.py \
    --eval_file "${DATA_DIR}/kragrec_v2_hard10_test.json" \
    --index_dir "${INDEX_DIR}" \
    --output_dir "${OUTPUT_DIR}_smoke" \
    --run_name "${RUN_NAME}_smoke" \
    --checkpoint "${OUTPUT_DIR}_smoke/checkpoint_best.pth" \
    --limit_eval 8 --max_txt_len "${MAX_TXT_LEN}" --device cuda
}

case "${MODE}" in
  index) build_index ;;
  data) build_data ;;
  smoke) build_index; build_data; smoke ;;
  train) build_index; build_data; train ;;
  eval) evaluate_all ;;
  all) build_index; build_data; train; evaluate_all ;;
  *) echo "Unknown MODE=${MODE}"; exit 2 ;;
esac
