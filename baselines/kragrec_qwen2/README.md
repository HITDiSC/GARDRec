# K-RagRec-Qwen2 (ML-1M) baseline

This directory is a server-compatible adaptation of the K-RagRec baseline
(https://github.com/Sjay-Wang/K-ragrec) for the GARDRec comparison.

## Naming and scope

The original K-RagRec release depends on a Freebase graph index and
`sentence-transformers/all-roberta-large-v1`, neither of which is available on
the current GPU server.  This variant keeps the K-RagRec decision pipeline
(graph retrieval -> GNN graph encoder -> soft graph token -> frozen LLM) but
replaces the retrieval backend with assets that already exist on the server:

* KG: Neo4j database `kg-ml-1m` (43k nodes / 103k edges)
* item features: `output/ml1m/rank_save/gcn_embeddings.json` (256-d)
* LLM: `/data/WY/models/Qwen2-7B-Instruct`

Every artifact is named `kragrec_qwen2_*` / `kragrec_qwen2_ml1m` to avoid
confusing it with the original K-RagRec numbers.  Results from this directory
are **K-RagRec-Qwen2 with a server KG/GCN retrieval backend**, not a verbatim
reproduction of the paper's Freebase + SBERT configuration.

## Files

* `config.py`         - CLI arguments
* `gnn.py`            - GNN encoders (copied from K-RagRec)
* `graph_llm.py`      - Qwen2-compatible GraphLLM (Qwen2 chat template, no torch_scatter)
* `retrieve.py`       - GCN/KG graph retrieval (no SBERT / Freebase)
* `data_utils.py`     - GARDRec JSON -> K-RagRec JSON and prompt construction
* `prepare_data.py`   - build train/valid/easy-test/hard10-test JSONs
* `build_kg_index.py` - export `kg-ml-1m` and build PyG graph files
* `train.py`          - train GNN + projector with frozen Qwen2
* `evaluate.py`       - evaluate ACC / R@1/3/5/10 / MRR
* `setup_deps.sh`     - project-local `pip --target` setup (GARDRec + .deps_gardrec)
* `run_server.sh`     - one-command server workflow

## Server commands

```bash
cd /data/WY/GARDRec
bash baselines/kragrec_qwen2/run_server.sh
```

By default the script uses GPU 7, one epoch, batch size 1 and gradient
accumulation 4.  Override with environment variables:

```bash
GPU=7 EPOCHS=1 MODE=all bash baselines/kragrec_qwen2/run_server.sh
GPU=7 MODE=smoke bash baselines/kragrec_qwen2/run_server.sh
MODE=eval RUN_NAME=kragrec_qwen2_ml1m bash baselines/kragrec_qwen2/run_server.sh
```

Equivalent manual commands:

```bash
PY=/mnt/sda/miniconda3/envs/GARDRec/bin/python
INDEX_PY=${PY}
DEPS=baselines/kragrec_qwen2/.deps_gardrec
export PYTHONPATH="${DEPS}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7

# one-time project-local dependency directory (does not modify GARDRec/tokenrec-ads/aaai)
bash baselines/kragrec_qwen2/setup_deps.sh

# index export only: GARDRec has the neo4j driver
$INDEX_PY baselines/kragrec_qwen2/build_kg_index.py \
  --out_dir data/dataset/kg_ml1m_kragrec

$PY baselines/kragrec_qwen2/prepare_data.py \
  --input data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json \
  --hard10 baselines/ml1m/v2_hard10_train.json \
  --out_dir data/dataset/ML1M

$PY baselines/kragrec_qwen2/train.py \
  --train_file data/dataset/ML1M/kragrec_v2_train.json \
  --index_dir data/dataset/kg_ml1m_kragrec \
  --num_epochs 1 --batch_size 1 --grad_accum_steps 4 --device cuda

$PY baselines/kragrec_qwen2/evaluate.py \
  --eval_file data/dataset/ML1M/kragrec_v2_easy_test.json \
  --index_dir data/dataset/kg_ml1m_kragrec \
  --checkpoint output/kragrec_qwen2_ml1m/checkpoint_best.pth --device cuda
$PY baselines/kragrec_qwen2/evaluate.py \
  --eval_file data/dataset/ML1M/kragrec_v2_hard10_test.json \
  --index_dir data/dataset/kg_ml1m_kragrec \
  --checkpoint output/kragrec_qwen2_ml1m/checkpoint_best.pth --device cuda
```

## Runtime artifacts

* `output/kragrec_qwen2_ml1m/progress.json` - live progress for the CC GUI panel
* `output/kragrec_qwen2_ml1m/train_log.jsonl` - per-log-step history
* `output/kragrec_qwen2_ml1m/checkpoint_best.pth` - trainable GNN + projector only
* `output/kragrec_qwen2_ml1m/checkpoint_last.pth` - last epoch
* `output/kragrec_qwen2_ml1m/results/*.json` - metrics and per-sample predictions

The checkpoint stores only `graph_encoder.*` and `projector.*`; the frozen
Qwen2 weights are loaded from `--llm_model_path` during evaluation.  This keeps
each checkpoint around tens of MB instead of several GB.

`build_kg_index.py` uses the `GARDRec` conda environment only because that
environment has the `neo4j` Python driver.  Training and evaluation use the
our own `GARDRec` environment plus a project-local `pip --target` directory
(`baselines/kragrec_qwen2/.deps_gardrec`) that contains only `torch_geometric`
(installed with `--no-deps`).  `tokenrec-ads` and `aaai` are neither modified
nor used by this baseline.
`torch_scatter` is not required: `graph_llm.py` uses a small pure-PyTorch
`scatter_mean` implementation.

## Split

The split follows the GARDRec v2 ML-1M protocol in file order:

* train: rows `[0, 7248)`
* valid: rows `[7248, 8154)`
* easy test: rows `[8154, 9060)` with original easy candidates
* hard10 test: rows `[8154, 9060)` with 10 hard + 9 easy negatives

This is an 80/10/10 split.  It is also exactly the test partition of the older
`--train-ratio 0.9` protocol: that protocol computes `test_size =
round(0.1 * 9060) = 906`, so its test rows are also `[8154, 9060)`.  The only
difference is that the 80/10/10 protocol carves the preceding 906 rows out of
the old training prefix as validation.  `prepare_data.py` writes
`same_test_as_9010`, `test_range`, and `test_sample_idx_sha256` into
`kragrec_v2_stats.json` so this can be checked directly.

Training and validation samples use the easy candidates; only the test
candidates change in Hard10.  The first 8,154 easy and Hard10 rows are compared
for exact equality (`train_valid_prefix_unchanged` in the stats file).

## Random seed

The first run uses `--seed 9` (the original K-RagRec default).  Add more seeds
later only after the first baseline result is verified.