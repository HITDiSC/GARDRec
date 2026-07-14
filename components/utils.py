import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from typing import Dict


def get_movie_mapping(movie_data: List[Dict]) -> Dict[str, Dict]:
    return {str(movie.get("original_id")): movie for movie in movie_data}


def load_input_entries(input_file: str) -> List[Dict]:
    with open(input_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_ranking_samples_from_input(
    entries: List[Dict],
    gcn_embedding_dict: Dict,
    num_history: int = 10,
    num_candidates: int = 20,
) -> List[Dict]:
    samples = []
    skipped_missing_target = 0
    skipped_short_candidates = 0
    skipped_empty_history = 0

    def _get_valid_movie(item_dict: Dict, gcn_embedding_dict: Dict, target_id: Optional[str] = None) -> Dict:
        item_id = str(item_dict.get("original_id"))
        if item_id not in gcn_embedding_dict:
            valid_ids = list(gcn_embedding_dict.keys())
            if target_id is not None and target_id in valid_ids:
                valid_ids.remove(target_id)
            if valid_ids:
                item_id = str(np.random.choice(valid_ids))
        return {
            "original_id": item_id,
            "title": item_dict.get("title", "Unknown"),
        }

    for entry_index, entry in enumerate(tqdm(entries, desc="Filtering ranking samples")):
        raw_history = entry.get("history_movies", entry.get("history_books", []))
        raw_candidates = entry.get("candidate_movies", entry.get("candidate_books", []))
        raw_target = entry.get("target_movie", entry.get("target_book", {}))
        sample_idx = entry.get("sample_idx", entry_index)
        target_id = (
            str(raw_target.get("original_id"))
            if raw_target.get("original_id") is not None
            else None
        )

        filtered_history = [
            _get_valid_movie(item, gcn_embedding_dict, target_id=target_id)
            for item in raw_history
        ]
        filtered_candidates = [
            _get_valid_movie(item, gcn_embedding_dict, target_id=target_id)
            for item in raw_candidates
        ]

        if not filtered_history:
            skipped_empty_history += 1
            continue
        if len(filtered_history) > num_history:
            filtered_history = filtered_history[-num_history:]
        if len(filtered_candidates) < num_candidates:
            skipped_short_candidates += 1
            continue

        filtered_candidates = filtered_candidates[:num_candidates]
        candidate_ids = [str(item.get("original_id")) for item in filtered_candidates]
        if target_id is None or candidate_ids.count(target_id) != 1:
            skipped_missing_target += 1
            continue

        target_idx = candidate_ids.index(target_id)
        samples.append(
            {
                "sample_idx": sample_idx,
                "history_movies": filtered_history,
                "candidate_movies": filtered_candidates,
                "target_idx": target_idx,
            }
        )

    return samples


def build_ranking_prompt_with_mode(
    history_movies: List[Dict],
    candidate_movies: List[Dict],
    prompt_mode: str = "structure_only",
) -> str:
    if prompt_mode == "minimal":
        return (
            "Graph embeddings for watched history are followed by graph embeddings for candidate movies. "
            "Answer: choose one label from A to T."
        )

    if prompt_mode == "structure_only":
        return (
            "The appended graph embeddings are structured in two segments. "
            "First comes the watched history embeddings. "
            "Second comes the candidate movie embeddings in fixed order from A to T. "
            "Answer: choose one label from A to T."
        )

    if prompt_mode == "with_titles":
        history_text = "; ".join(
            f"{index + 1}. {movie.get('title', 'Unknown')}"
            for index, movie in enumerate(history_movies)
        )
        candidate_parts = []
        for index, movie in enumerate(candidate_movies):
            label = ascii_uppercase[index]
            candidate_parts.append(f"{label}. {movie.get('title', 'Unknown')}")
        candidate_text = "; ".join(candidate_parts)
        return (
            "The appended graph embeddings are structured in two segments. "
            "First comes the watched history embeddings. "
            "Second comes the candidate movie embeddings in fixed order from A to T. "
            f"Watched history: {history_text}. "
            f"Candidates: {candidate_text}. "
        )

    if prompt_mode == "text_only":
        history_text = "; ".join(
            f"{index + 1}. {movie.get('title', 'Unknown')}"
            for index, movie in enumerate(history_movies)
        )
        candidate_parts = []
        for index, movie in enumerate(candidate_movies):
            label = ascii_uppercase[index]
            candidate_parts.append(f"{label}. {movie.get('title', 'Unknown')}")
        candidate_text = "; ".join(candidate_parts)
        return (
            "Recommend the best next movie using only movie titles. "
            f"Watched history: {history_text}. "
            f"Candidates: {candidate_text}. "
            "Answer: choose one label from A to T."
        )

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def compute_answer_prompt_ids(tokenizer) -> List[int]:
    answer_prompt_text = "\nAnswer:"
    answer_prompt_ids = tokenizer(answer_prompt_text, add_special_tokens=False)["input_ids"]
    if not answer_prompt_ids:
        raise ValueError("Answer prompt tokenizes to empty sequence")
    return answer_prompt_ids


def compute_label_token_ids(tokenizer, num_candidates: int = 20) -> List[int]:
    answer_prompt_text = "\nAnswer:"
    answer_prefix_ids = tokenizer(answer_prompt_text, add_special_tokens=False)["input_ids"]
    if not answer_prefix_ids:
        raise ValueError("Answer prefix tokenizes to empty sequence")
    label_ids = []
    for i in range(num_candidates):
        label = ascii_uppercase[i]
        ids = tokenizer(f"{answer_prompt_text} {label}", add_special_tokens=False)["input_ids"]
        if len(ids) <= len(answer_prefix_ids):
            raise ValueError(f"Label '{label}' does not appear after answer prefix")
        suffix_ids = ids[len(answer_prefix_ids):]
        if len(suffix_ids) != 1:
            raise ValueError(
                f"Label '{label}' is tokenized into {len(suffix_ids)} tokens after answer prefix: {suffix_ids}"
            )
        label_ids.append(suffix_ids[0])
    return label_ids


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
    if not interaction_feature_names:
        return arr
    selected_indices = [
        idx for idx, name in enumerate(interaction_feature_names)
        if name in whitelist and idx < arr.shape[1]
    ]
    if not selected_indices:
        return np.zeros((arr.shape[0], 0), dtype=np.float32)
    return arr[:, selected_indices]


def infer_interaction_feature_dim(samples: List[Dict], num_candidates: int) -> int:
    for sample in samples:
        features = sample.get("interaction_features")
        if features is None:
            continue
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        return int(arr.shape[1])
    return 0
