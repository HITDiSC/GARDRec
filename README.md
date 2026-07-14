GCN Recommendation — Exported Project

Overview
--------

This repository contains an exported, self-contained subset of the original GCN-based recommendation research code. It is organized to make the experiment reproducible and easy to upload to a GitHub repository for sharing or archival.

This export focuses on:

- Preparing and enriching ranking training samples with graph retrieval signals (Neo4j) and GCN embeddings.
- Training and evaluating a retrieval-augmented ranker that fuses graph embeddings with an LLM-based ranking backbone.

Contents
--------

- `gcn_recommend.py`: Main training and evaluation entrypoint. Loads preprocessed retrieval details, builds datasets, constructs the ranker and trainer, then runs training and evaluation.
- `preprocess_retrieval_augmented_samples.py`: Generates offline retrieval augmentation artifacts (history retrieval embeddings and candidate match features) using Neo4j and GCN embeddings.
- `preprocess_retrieval_augmented_samples_patch.py`: Adds interaction feature construction (co-occurrence, transition, user/item stats) to retrieval details and outputs a versioned file.
- `components/`: Reusable modules extracted from the original project. Key modules:
  - `components/config.py`: Logging and configuration helpers.
  - `components/data.py`: Input resolution, loading and attach/retrieve utilities.
  - `components/utils.py`: Prompt builders, token id helpers and sample construction (`build_ranking_samples_from_input`).
  - `components/model.py`: Dataset builders and tokenizer helpers.
  - `components/llm_graph_ranker.py`: Model implementation that fuses GCN projector + LLM.
  - `components/trainer.py`: Training loop and evaluation helpers.
  - `components/neo4j_utils.py`: Neo4j helper functions to read item embeddings.
  - `components/kg_retrieval_enhancement.py`: Retrieval augmentation builder (graph co-occurrence and attribute overlap features).
- `neo4j_connect.py`: Minimal Neo4j connection wrapper used by some exported utilities.
- `gcn_detection.py`: Model projector and minimal Neo4j config used by the LLM projector.
- `use_llm.py`: Lightweight LLM call wrapper (exported placeholder).
- `notify_local.py`: Local notification utility (placeholder).
- `.env.example`: Example environment variables file for DB and API credentials.
- `requirements.txt`: Minimal dependency list to run the exported scripts.
- `run.sh` / `run.bat`: Example run scripts for UNIX and Windows.

Quick Start
-----------

Prerequisites

- Python 3.8+ (3.9/3.10 recommended)
- GPU with sufficient memory if you plan to run LLM training/inference (LLM model sizes vary)
- Neo4j database containing precomputed node embeddings (optional for full preprocessing)

1. Create a Python environment (conda recommended):

```bash
conda create -n gcnrec python=3.10 -y
conda activate gcnrec
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your Neo4j and API credentials:

```bash
cp .env.example .env
# edit .env with your values
```

3. Prepare data and embeddings

- Place dataset files under `./data/` according to the expected layout (see `components/data.py`). Example:
  - `./data/dataset/ML20M/parse_data/train.json`
  - `./data/dataset/AmazonBook/parse_data/train.json`
- Ensure Neo4j contains item nodes with `original_id` and `gcn_embedding` or `embedding` properties if you plan to run preprocessing.

4. Preprocess retrieval augmentation (optional, recommended before training):

```bash
python preprocess_retrieval_augmented_samples.py --domain ml20m
# then add interaction features
python preprocess_retrieval_augmented_samples_patch.py --domain ml20m
```

5. Train and evaluate the ranker

```bash
python gcn_recommend.py --domain ml20m --model shakechen/Llama-2-7b-chat-hf
```

Notes and Implementation Details
--------------------------------

- The exported `components` folder contains simplified and self-contained implementations of the original project's helper modules. Some methods are reduced to minimal or stub behavior to keep the export compact and portable.
- `preprocess_retrieval_augmented_samples.py` depends on Neo4j for graph-based augmentation. If Neo4j is not available, the script will still run but will produce empty or default retrieval embeddings.
- The ranking dataset unifies `history_movies`/`history_books` naming to be domain-agnostic.

Troubleshooting
---------------

- If `preprocess_retrieval_augmented_samples.py` reports missing Neo4j connection errors, check `.env` and ensure `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` are correct.
- If LLM model loading fails due to memory or model availability, try a smaller LLM model or run on a GPU with more memory.
