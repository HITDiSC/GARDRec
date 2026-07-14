from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modelscope import AutoModelForCausalLM
from torch.utils.data import Dataset

from .config import logger
from .utils import build_ranking_prompt_with_mode


def resolve_model_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def prepare_tokenizer(model_name: str):
    from modelscope import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(f"Tokenizer for {model_name} has neither pad_token nor eos_token nor unk_token.")
    tokenizer.padding_side = "right"
    return tokenizer


def build_text_only_prompt_and_candidate_spans(
    history_movies: List[Dict],
    candidate_movies: List[Dict],
    tokenizer,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    prefix_ids = tokenizer(
        "Recommend the best next movie using only movie titles. Watched history: ",
        add_special_tokens=False,
    )["input_ids"]
    input_ids = list(prefix_ids)

    history_text = "; ".join(
        f"{index + 1}. {movie.get('title', 'Unknown')}"
        for index, movie in enumerate(history_movies)
    )
    history_ids = tokenizer(history_text, add_special_tokens=False)["input_ids"]
    input_ids.extend(history_ids)

    separator_ids = tokenizer(
        ". Candidates: ",
        add_special_tokens=False,
    )["input_ids"]
    input_ids.extend(separator_ids)

    candidate_spans: List[Tuple[int, int]] = []
    for index, movie in enumerate(candidate_movies):
        label = ascii_uppercase[index]
        candidate_text = f"{label}. {movie.get('title', 'Unknown')}"
        candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
        span_start = len(input_ids)
        input_ids.extend(candidate_ids)
        span_end = len(input_ids)
        candidate_spans.append((span_start, span_end))
        if index < len(candidate_movies) - 1:
            input_ids.extend(tokenizer("; ", add_special_tokens=False)["input_ids"])

    suffix_ids = tokenizer(
        ". Answer: choose one label from A to T.",
        add_special_tokens=False,
    )["input_ids"]
    input_ids.extend(suffix_ids)
    return input_ids, candidate_spans


def build_with_titles_prompt_and_candidate_spans(
    history_movies: List[Dict],
    candidate_movies: List[Dict],
    tokenizer,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    prefix_text = (
        "The appended graph embeddings are structured in two segments. "
        "First comes the watched history embeddings. "
        "Second comes the candidate movie embeddings in fixed order from A to T. "
        "Watched history: "
    )
    input_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    history_text = "; ".join(
        f"{index + 1}. {movie.get('title', 'Unknown')}"
        for index, movie in enumerate(history_movies)
    )
    input_ids.extend(tokenizer(history_text, add_special_tokens=False)["input_ids"])
    input_ids.extend(tokenizer(". Candidates: ", add_special_tokens=False)["input_ids"])

    candidate_spans: List[Tuple[int, int]] = []
    for index, movie in enumerate(candidate_movies):
        label = ascii_uppercase[index]
        candidate_text = f"{label}. {movie.get('title', 'Unknown')}"
        candidate_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
        span_start = len(input_ids)
        input_ids.extend(candidate_ids)
        span_end = len(input_ids)
        candidate_spans.append((span_start, span_end))
        if index < len(candidate_movies) - 1:
            input_ids.extend(tokenizer("; ", add_special_tokens=False)["input_ids"])

    input_ids.extend(tokenizer(".", add_special_tokens=False)["input_ids"])
    return input_ids, candidate_spans


class GCNRecommendationDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        gcn_embedding_dict: Dict,
        embedding_dim: int,
        match_feature_dim: int,
        max_prompt_length: int = 256,
        num_history: int = 10,
        num_candidates: int = 20,
        interaction_feature_dim: int = 0,
        prompt_mode: str = "structure_only",
    ):
        self.data = data
        self.match_feature_dim = match_feature_dim
        self.tokenizer = tokenizer
        self.gcn_embedding_dict = gcn_embedding_dict
        self.embedding_dim = embedding_dim
        self.max_prompt_length = max_prompt_length
        self.num_history = num_history
        self.num_candidates = num_candidates
        self.interaction_feature_dim = interaction_feature_dim
        self.prompt_mode = prompt_mode

    def __len__(self):
        return len(self.data)

    def _pad_prompt(self, input_ids: List[int]):
        input_ids = input_ids[:self.max_prompt_length]
        padding_len = self.max_prompt_length - len(input_ids)
        padded = [self.tokenizer.pad_token_id] * padding_len + input_ids
        attention_mask = [0] * padding_len + [1] * len(input_ids)
        return padded, attention_mask

    def _history_embeddings(self, history_movies: List[Dict]):
        embeddings = []
        history_mask = []
        for movie in history_movies[:self.num_history]:
            movie_id = str(movie.get("original_id"))
            emb_list = self.gcn_embedding_dict.get(movie_id)
            if emb_list is None:
                emb_array = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                emb_array = np.asarray(emb_list, dtype=np.float32)
            embeddings.append(emb_array)
            history_mask.append(1)
        while len(embeddings) < self.num_history:
            embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            history_mask.append(0)
        return np.stack(embeddings, axis=0), history_mask

    def _candidate_embeddings(self, candidate_movies: List[Dict]):
        embeddings = []
        for movie in candidate_movies[:self.num_candidates]:
            movie_id = str(movie.get("original_id"))
            emb_list = self.gcn_embedding_dict.get(movie_id)
            if emb_list is None:
                emb_array = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                emb_array = np.asarray(emb_list, dtype=np.float32)
            embeddings.append(emb_array)
        return np.stack(embeddings, axis=0)

    def _candidate_text_mask(self, candidate_spans: List[Tuple[int, int]], prompt_token_length: int):
        text_mask = np.zeros((self.num_candidates, self.max_prompt_length), dtype=np.float32)
        for index, (start, end) in enumerate(candidate_spans[:self.num_candidates]):
            clipped_start = min(start, self.max_prompt_length)
            clipped_end = min(end, self.max_prompt_length)
            if clipped_end > clipped_start:
                text_mask[index, clipped_start:clipped_end] = 1.0
        return text_mask

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.prompt_mode == "text_only":
            prompt_ids, candidate_spans = build_text_only_prompt_and_candidate_spans(
                sample["history_movies"],
                sample["candidate_movies"],
                self.tokenizer,
            )
            candidate_text_mask = self._candidate_text_mask(candidate_spans, len(prompt_ids))
        elif self.prompt_mode == "with_titles":
            prompt_ids, candidate_spans = build_with_titles_prompt_and_candidate_spans(
                sample["history_movies"],
                sample["candidate_movies"],
                self.tokenizer,
            )
            candidate_text_mask = self._candidate_text_mask(candidate_spans, len(prompt_ids))
        else:
            prompt_text = build_ranking_prompt_with_mode(
                sample["history_movies"],
                sample["candidate_movies"],
                prompt_mode=self.prompt_mode,
            )
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            candidate_text_mask = np.zeros((self.num_candidates, self.max_prompt_length), dtype=np.float32)
        input_ids, attention_mask = self._pad_prompt(prompt_ids)
        history_embeddings, history_mask = self._history_embeddings(sample["history_movies"])
        candidate_embeddings = self._candidate_embeddings(sample["candidate_movies"])
        history_retrieval_embedding = np.asarray(
            sample.get("history_retrieval_embedding", np.zeros(self.embedding_dim, dtype=np.float32)),
            dtype=np.float32,
        )
        candidate_match_features = np.asarray(
            sample.get(
                "candidate_match_features",
                np.zeros((self.num_candidates, self.match_feature_dim), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        interaction_features = np.asarray(
            sample.get(
                "interaction_features",
                np.zeros((self.num_candidates, self.interaction_feature_dim), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if self.interaction_feature_dim > 0:
            if interaction_features.ndim != 2:
                interaction_features = np.zeros((self.num_candidates, self.interaction_feature_dim), dtype=np.float32)
            if interaction_features.shape[0] < self.num_candidates:
                pad = np.zeros((self.num_candidates - interaction_features.shape[0], interaction_features.shape[1]), dtype=np.float32)
                interaction_features = np.concatenate([interaction_features, pad], axis=0)
            interaction_features = interaction_features[: self.num_candidates]
            if interaction_features.shape[1] < self.interaction_feature_dim:
                pad = np.zeros((interaction_features.shape[0], self.interaction_feature_dim - interaction_features.shape[1]), dtype=np.float32)
                interaction_features = np.concatenate([interaction_features, pad], axis=1)
            interaction_features = interaction_features[:, : self.interaction_feature_dim]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "history_embeddings": torch.tensor(history_embeddings, dtype=torch.float32),
            "history_mask": torch.tensor(history_mask, dtype=torch.long),
            "history_retrieval_embedding": torch.tensor(history_retrieval_embedding, dtype=torch.float32),
            "candidate_embeddings": torch.tensor(candidate_embeddings, dtype=torch.float32),
            "candidate_match_features": torch.tensor(candidate_match_features, dtype=torch.float32),
            "interaction_features": torch.tensor(interaction_features, dtype=torch.float32),
            "candidate_text_mask": torch.tensor(candidate_text_mask, dtype=torch.float32),
            "labels": torch.tensor(sample["target_idx"], dtype=torch.long),
        }
