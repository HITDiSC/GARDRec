import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import logger
from .utils import build_ranking_prompt_with_mode, compute_label_token_ids, compute_answer_prompt_ids


DEFAULT_INTERACTION_FEATURE_NAMES = [
    "gsim",
    "lmax",
    "first_norm",
    "user_avg_feat",
    "user_cnt_feat",
    "item_avg_feat",
    "item_cnt_feat",
    "user_rated_flag",
    "co_feat",
    "trans_feat",
    "genre_feat",
    "rank_feat",
    "co_trans_mul",
    "co_trans_absdiff",
]


def resolve_input_file(domain: str) -> str:
    if domain == "amazonbook":
        candidate_paths = [
            "./data/dataset/AmazonBook/parse_data/train.json"
        ]
    elif domain == "ml20m":
        candidate_paths = [
            "./data/dataset/ML20M/parse_data/train.json",
        ]
    elif domain == "ml1m":
        candidate_paths = [
            "./data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json",
            "./data/dataset/ML1M/parse_data/train.json",
        ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find ranking samples file in expected locations.")


def resolve_retrieval_details_file(domain: str) -> str:
    if domain == "amazonbook":
        candidate_paths = ["./output/amazonbook/rank_save/retrieval_augmented_details.json"]
    elif domain == "ml20m":
        candidate_paths = ["./output/ml20m/rank_save/retrieval_augmented_details.json"]
    elif domain == "ml1m":
        candidate_paths = ["./output/ml1m/rank_save/retrieval_augmented_details.json"]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return candidate_paths[0]


def load_input_entries(input_file: str) -> List[Dict]:
    with open(input_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_retrieval_details(detail_file: str) -> Tuple[Dict[str, Dict], List[str]]:
    with open(detail_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    detail_map = payload.get("details_by_sample_idx", payload)
    normalized_map = {str(key): value for key, value in detail_map.items()}
    interaction_feature_names = payload.get("interaction_feature_names")
    if isinstance(interaction_feature_names, list) and interaction_feature_names:
        names = [str(item) for item in interaction_feature_names]
    else:
        inferred_dim = 0
        for value in normalized_map.values():
            raw = value.get("interaction_features") if isinstance(value, dict) else None
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] > 0:
                inferred_dim = int(arr.shape[1])
                break
        if inferred_dim <= 0:
            names = []
        elif inferred_dim <= len(DEFAULT_INTERACTION_FEATURE_NAMES):
            names = DEFAULT_INTERACTION_FEATURE_NAMES[:inferred_dim]
        else:
            names = list(DEFAULT_INTERACTION_FEATURE_NAMES)
            names.extend(
                f"interaction_extra_{i}"
                for i in range(inferred_dim - len(DEFAULT_INTERACTION_FEATURE_NAMES))
            )
    return normalized_map, names


def select_interaction_features_by_whitelist(
    interaction_features,
    interaction_feature_names: List[str],
    whitelist: List[str],
):
    if interaction_features is None:
        return None
    arr = np.asarray(interaction_features, dtype=np.float32)
    if arr.ndim != 2:
        return None
    selected_indices = [
        idx for idx, name in enumerate(interaction_feature_names)
        if name in whitelist and idx < arr.shape[1]
    ]
    if not selected_indices:
        return np.zeros((arr.shape[0], 0), dtype=np.float32)
    return arr[:, selected_indices]


def attach_retrieval_details(
    samples: List[Dict],
    retrieval_detail_map: Dict[str, Dict],
    interaction_feature_names: Optional[List[str]] = None,
    interaction_feature_whitelist: Optional[List[str]] = None,
) -> List[Dict]:
    missing_detail_count = 0
    interaction_feature_names = interaction_feature_names or []
    interaction_feature_whitelist = interaction_feature_whitelist or []
    for sample in samples:
        sample_idx = sample.get("sample_idx")
        if sample_idx is None:
            missing_detail_count += 1
            continue
        detail = retrieval_detail_map.get(str(sample_idx))
        if detail is None:
            missing_detail_count += 1
            continue
        sample["history_retrieval_embedding"] = detail.get("history_retrieval_embedding")
        sample["candidate_match_features"] = detail.get("candidate_match_features")
        sample["interaction_features"] = select_interaction_features_by_whitelist(
            detail.get("interaction_features"),
            interaction_feature_names,
            interaction_feature_whitelist,
        )
    if missing_detail_count:
        logger.warning("Missing retrieval details for %s samples", missing_detail_count)
    return samples


def infer_interaction_feature_dim(samples: List[Dict], num_candidates: int) -> int:
    for sample in samples:
        features = sample.get("interaction_features")
        if features is None:
            continue
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        return int(arr.shape[1])
    logger.warning(
        "No interaction_features found in samples, fallback to dim=0. "
        "Run preprocess_retrieval_augmented_samples_patch.py to generate enhanced details file."
    )
    return 0
