#!/usr/bin/env bash
set -euo pipefail

cd /data/WY/GARDRec

PY=${PY:-/mnt/sda/miniconda3/envs/GARDRec/bin/python}
DEPS_DIR=${DEPS_DIR:-baselines/kragrec_qwen2/.deps_gardrec}

mkdir -p "${DEPS_DIR}"
"${PY}" -m pip install --target "${DEPS_DIR}" --no-deps --no-cache-dir "torch_geometric==2.7.0"

echo "Installed project-local dependencies into ${DEPS_DIR}"
echo "Use: export PYTHONPATH=${DEPS_DIR}:\${PYTHONPATH:-}"