#!/usr/bin/env python3
"""Build hard-negative candidate sets for ranking evaluation.

The original benchmark uses 19 uniform-random negatives per instance.  This
script keeps the same 1-positive / 19-negative format but replaces a chosen
number of negatives with content-similar but unobserved items.  Item similarity
comes from metadata (movie genres / book categories) and titles, not from the
recommender being evaluated, so the hard-negative construction is independent
of the model's own embeddings.

By default the query is built only from the user's history.  The target is used
only after the hard-negative pool has been fixed, which avoids test-label
leakage.  The remaining easy negatives are sampled from the original 19
negatives, making Easy and Hard evaluations paired at the sample level.

Only test rows are changed.  Training and validation rows and their candidates
are copied verbatim, so model training and early stopping are identical to the
original protocol and any metric change is attributable to the test candidates.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

try:
    import orjson

    def json_loads(text):
        return orjson.loads(text)
except ImportError:
    def json_loads(text):
        return json.loads(text)


def detect_fields(row: Dict) -> Tuple[str, str, str]:
    history_key = next((key for key in ("history_movies", "history_books", "history") if key in row), None)
    candidate_key = next((key for key in ("candidate_movies", "candidate_books", "candidates") if key in row), None)
    target_key = next((key for key in ("target_movie", "target_book", "target") if key in row), None)
    if history_key is None or candidate_key is None or target_key is None:
        raise KeyError(f"Cannot detect fields. Available keys: {sorted(row.keys())}")
    return history_key, candidate_key, target_key


def original_id(value) -> str:
    if isinstance(value, dict):
        for key in ("original_id", "item_id", "id"):
            if key in value:
                return str(value[key])
        raise KeyError(f"Item object has no id field: {sorted(value.keys())}")
    return str(value)


def flatten_categories(value) -> Set[str]:
    out: Set[str] = set()
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.add(text)
        return out
    if isinstance(value, dict):
        for item in value.values():
            out |= flatten_categories(item)
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out |= flatten_categories(item)
    return out


def load_entries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(domain: str, path: str, needed_items: Set[str]):
    titles: Dict[str, str] = {}
    categories: Dict[str, Set[str]] = defaultdict(set)
    if domain == "ml1m":
        with open(path, "r", encoding="latin-1") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("::")
                if len(parts) < 3:
                    continue
                item_id, title, genres = parts[0], parts[1], parts[2]
                if item_id not in needed_items:
                    continue
                titles[item_id] = title
                categories[item_id] |= {g for g in genres.split("|") if g and g != "(no genres listed)"}
    elif domain == "ml20m":
        import pandas as pd

        frame = pd.read_csv(path, usecols=["movieId", "title", "genres"], dtype={"movieId": str})
        for row in frame.itertuples(index=False):
            item_id = str(row.movieId)
            if item_id not in needed_items:
                continue
            titles[item_id] = str(row.title)
            categories[item_id] |= {g for g in str(row.genres).split("|") if g and g != "(no genres listed)"}
    elif domain == "amazonbook":
        seen = 0
        with open(path, "rb") as handle:
            for line in handle:
                seen += 1
                if seen % 2_000_000 == 0:
                    print(f"  metadata lines={seen:,}", flush=True)
                try:
                    row = json_loads(line)
                except Exception:
                    continue
                item_id = str(row.get("asin") or "")
                if item_id not in needed_items:
                    continue
                titles[item_id] = str(row.get("title") or "")
                cats = flatten_categories(row.get("category"))
                cats |= flatten_categories(row.get("categories"))
                cats |= flatten_categories(row.get("main_cat"))
                categories[item_id] |= cats
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return titles, categories


def load_observed_histories(domain: str, path: Optional[str], users: Set[str]) -> Dict[str, Set[str]]:
    observed: Dict[str, Set[str]] = defaultdict(set)
    if not path:
        return observed
    print(f"Loading full observed histories from {path}", flush=True)
    if domain == "ml1m":
        with open(path, "r", encoding="latin-1") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("::")
                if len(parts) < 2:
                    continue
                user, item = str(parts[0]), str(parts[1])
                if user in users:
                    observed[user].add(item)
    elif domain == "ml20m":
        import pandas as pd

        frame = pd.read_csv(path, usecols=["userId", "movieId"], dtype={"userId": str, "movieId": str})
        for user, item in zip(frame["userId"], frame["movieId"]):
            if user in users:
                observed[user].add(item)
    elif domain == "amazonbook":
        seen = 0
        with open(path, "rb") as handle:
            for line in handle:
                seen += 1
                if seen % 2_000_000 == 0:
                    print(f"  review lines={seen:,}", flush=True)
                try:
                    row = json_loads(line)
                except Exception:
                    continue
                user = str(row.get("reviewerID") or "")
                item = str(row.get("asin") or "")
                if user in users and item:
                    observed[user].add(item)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    print(f"Loaded observed histories for {len(observed)} users", flush=True)
    return observed


def build_title_index(item_ids: Sequence[str], titles: Dict[str, str]):
    texts = [titles.get(item_id) or item_id for item_id in item_ids]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, normalize(matrix)


def build_category_index(item_ids: Sequence[str], categories: Dict[str, Set[str]]):
    labels = [sorted(categories.get(item_id, set())) for item_id in item_ids]
    if not any(labels):
        return None, None
    encoder = MultiLabelBinarizer(sparse_output=True)
    matrix = encoder.fit_transform(labels)
    return encoder, normalize(matrix)


def choose_easy_negatives(
    count: int,
    all_items: Sequence[str],
    blocked: Set[str],
    selected: Set[str],
    rng: random.Random,
) -> List[str]:
    out: List[str] = []
    attempts = 0
    max_attempts = max(10_000, count * 1_000)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        item = rng.choice(all_items)
        if item in blocked or item in selected:
            continue
        out.append(item)
        selected.add(item)
    if len(out) < count:
        for item in all_items:
            if item not in blocked and item not in selected:
                out.append(item)
                selected.add(item)
                if len(out) == count:
                    break
    if len(out) < count:
        raise RuntimeError("Not enough candidate items to fill negatives")
    return out


def choose_popularity_matched_hard(
    pool: Sequence[str],
    target_bins: Sequence[int],
    popularity_bin,
    rng: random.Random,
) -> List[str]:
    by_bin: Dict[int, List[str]] = defaultdict(list)
    for item in pool:
        by_bin[popularity_bin(item)].append(item)

    chosen: List[str] = []
    used: Set[str] = set()
    remaining = Counter(target_bins)
    for target_bin, count in sorted(
        remaining.items(), key=lambda pair: (len(by_bin.get(pair[0], [])), pair[0])
    ):
        bin_options = [item for item in by_bin.get(target_bin, []) if item not in used]
        rank_limit = max(50, count * 10)
        if len(bin_options) > count:
            options = rng.sample(bin_options[:rank_limit], count)
        else:
            options = bin_options[:count]
        for item in options:
            chosen.append(item)
            used.add(item)
        missing = count - len(options)
        if not missing:
            continue
        available_bins = sorted(
            (abs(bin_id - target_bin), bin_id)
            for bin_id, items in by_bin.items()
            if any(item not in used for item in items)
        )
        for _, bin_id in available_bins:
            options = [item for item in by_bin[bin_id] if item not in used][:missing]
            for item in options:
                chosen.append(item)
                used.add(item)
            missing -= len(options)
            if missing <= 0:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build content-based hard-negative ranking candidates")
    parser.add_argument("--data", required=True)
    parser.add_argument("--domain", required=True, choices=["ml1m", "ml20m", "amazonbook"])
    parser.add_argument("--metadata", required=True, help="movies.dat / movies.csv / meta_Books.json")
    parser.add_argument("--raw-ratings", default=None, help="ratings.dat / ratings.csv / Books_5.json")
    parser.add_argument(
        "--observed-json",
        default=None,
        help="Optional JSON mapping user_id to the list of all observed item ids",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--hard-count", type=int, default=10, help="Number of hard negatives (max 19)")
    parser.add_argument("--topk", type=int, default=2000, help="Retrieve this many similar items before sampling")
    parser.add_argument("--title-weight", type=float, default=0.6)
    parser.add_argument("--category-weight", type=float, default=0.4)
    parser.add_argument(
        "--query-mode",
        choices=["history", "target_history"],
        default="history",
        help="Use history only (default, no test-label leakage) or target plus history",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process the first N samples")
    parser.add_argument(
        "--catalog-file",
        default=None,
        help="Optional JSON list of allowed original item ids for hard-negative retrieval",
    )
    args = parser.parse_args()

    if not 0 <= args.hard_count <= 19:
        raise ValueError("--hard-count must be between 0 and 19")

    rng = random.Random(args.seed)
    entries = load_entries(args.data)
    if args.limit:
        entries = entries[: args.limit]
    history_key, candidate_key, target_key = detect_fields(entries[0])

    raw_entry_count = len(entries)
    filtered_entries = []
    dropped_short_history = 0
    dropped_bad_candidates = 0
    for row in entries:
        history = [original_id(x) for x in row.get(history_key, [])]
        candidates = [original_id(x) for x in row.get(candidate_key, [])]
        target = original_id(row.get(target_key))
        if not history:
            dropped_short_history += 1
            continue
        if len(candidates) != 20 or candidates.count(target) != 1:
            dropped_bad_candidates += 1
            continue
        filtered_entries.append(row)
    entries = filtered_entries
    if not entries:
        raise RuntimeError("No valid evaluation samples remain after filtering")
    print(
        f"Filtered samples: kept={len(entries)}/{raw_entry_count} "
        f"dropped_short_history={dropped_short_history} "
        f"dropped_bad_candidates={dropped_bad_candidates}",
        flush=True,
    )
    total_samples = len(entries)
    test_size = int(round(total_samples * (1.0 - args.train_ratio)))
    val_size = int(round(total_samples * args.val_ratio))
    train_size = total_samples - test_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("The configured split must leave non-empty partitions")
    eval_entries = entries[train_size + val_size :]
    print(
        f"Hard protocol split: train={train_size} unchanged, "
        f"valid={val_size} unchanged, test={test_size} hardened",
        flush=True,
    )

    item_ids: Set[str] = set()
    users: Set[str] = set()
    for row in entries:
        users.add(str(row.get("user_id")))
        item_ids.update(original_id(x) for x in row.get(history_key, []))
        item_ids.update(original_id(x) for x in row.get(candidate_key, []))
        item_ids.add(original_id(row.get(target_key)))
    item_list = sorted(item_ids)
    if args.catalog_file:
        with open(args.catalog_file, "r", encoding="utf-8") as handle:
            allowed = {str(item) for item in json.load(handle)}
        item_list = [item for item in item_list if item in allowed]
        print(
            f"Restricted catalog to {len(item_list)} items from {args.catalog_file}",
            flush=True,
        )
    item_to_index = {item: idx for idx, item in enumerate(item_list)}
    print(f"Samples={len(entries)} users={len(users)} unique_items={len(item_list)}", flush=True)

    train_popularity: Counter = Counter()
    for row in entries[:train_size]:
        for item in row.get(history_key, []):
            train_popularity[original_id(item)] += 1
        train_popularity[original_id(row.get(target_key))] += 1
    positive_counts = np.asarray([count for count in train_popularity.values() if count > 0], dtype=np.float64)

    def popularity_bin(item: str) -> int:
        count = train_popularity.get(item, 0)
        if count <= 10:
            return count
        return 11 + int(np.floor(np.log2(count / 10.0)))

    print(
        f"Training popularity bins: items={len(train_popularity)} "
        f"positive_items={positive_counts.size} scheme=exact_to_10_then_log2",
        flush=True,
    )

    print("Loading metadata", flush=True)
    titles, categories = load_metadata(args.domain, args.metadata, item_ids)
    missing_titles = sum(1 for item in item_list if item not in titles)
    missing_cats = sum(1 for item in item_list if not categories.get(item))
    print(
        f"Metadata coverage: titles={len(item_list)-missing_titles}/{len(item_list)} "
        f"categories={len(item_list)-missing_cats}/{len(item_list)}",
        flush=True,
    )

    if args.observed_json:
        with open(args.observed_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        observed = defaultdict(set)
        for user, items in payload.items():
            user = str(user)
            if user in users:
                observed[user].update(str(item) for item in items)
        print(f"Loaded observed histories for {len(observed)} users from {args.observed_json}", flush=True)
        observed_source = "observed_json"
    else:
        observed = load_observed_histories(args.domain, args.raw_ratings, users)
        observed_source = "raw_all_interactions" if args.raw_ratings else "json_history_plus_target"

    print("Building title TF-IDF index", flush=True)
    vectorizer, title_matrix = build_title_index(item_list, titles)
    category_encoder, category_matrix = build_category_index(item_list, categories)
    if category_matrix is None:
        print("No category metadata; using title similarity only", flush=True)
        category_weight = 0.0
    else:
        category_weight = args.category_weight

    output_rows: List[Dict] = [dict(row) for row in entries[: train_size + val_size]]
    hard_similarities: List[float] = []
    easy_similarities: List[float] = []
    hard_popularities: List[int] = []
    easy_popularities: List[int] = []
    popularity_bin_matches = 0
    insufficient = 0
    easier_count = 19 - args.hard_count

    for start in range(0, len(eval_entries), args.batch_size):
        batch = eval_entries[start : start + args.batch_size]
        query_texts = []
        query_category_rows = []
        for row in batch:
            history = [original_id(x) for x in row.get(history_key, [])]
            query_item_ids = list(history)
            if args.query_mode == "target_history":
                query_item_ids.insert(0, original_id(row.get(target_key)))
            query_texts.append(" ".join(titles.get(item, item) for item in query_item_ids))
            query_ids = [item_to_index[item] for item in query_item_ids if item in item_to_index]
            if category_matrix is not None and query_ids:
                query_category_rows.append(np.asarray(category_matrix[query_ids].mean(axis=0)).ravel())
            else:
                query_category_rows.append(None)

        query_title = vectorizer.transform(query_texts)
        title_scores = (query_title @ title_matrix.T).toarray().astype(np.float32)
        if category_matrix is not None and any(row is not None for row in query_category_rows):
            valid_rows = [row if row is not None else np.zeros(category_matrix.shape[1], dtype=np.float32) for row in query_category_rows]
            category_scores = np.asarray(np.vstack(valid_rows) @ category_matrix.T, dtype=np.float32)
        else:
            category_scores = np.zeros_like(title_scores)

        scores = args.title_weight * title_scores + category_weight * category_scores
        for row_index, row in enumerate(batch):
            user = str(row.get("user_id"))
            target = original_id(row.get(target_key))
            history = [original_id(x) for x in row.get(history_key, [])]
            if observed:
                blocked = set(observed.get(user, set()))
                blocked.update(history)
            else:
                blocked = set(history)
            blocked.add(target)

            original_negatives = []
            seen_original = set()
            for item in [original_id(x) for x in row.get(candidate_key, [])]:
                if item != target and item not in seen_original:
                    original_negatives.append(item)
                    seen_original.add(item)
            hard_blocked = blocked | seen_original

            blocked_indices = [item_to_index[item] for item in hard_blocked if item in item_to_index]
            unmasked_scores = scores[row_index].copy()
            row_scores = scores[row_index].copy()
            if blocked_indices:
                row_scores[blocked_indices] = -np.inf
            finite = np.isfinite(row_scores)

            easy_pool = list(original_negatives)
            rng.shuffle(easy_pool)
            easy_items = easy_pool[:easier_count]
            replaced_easy = easy_pool[easier_count:]
            if len(replaced_easy) < args.hard_count:
                matched_selected = set(easy_items) | set(replaced_easy)
                replaced_easy.extend(
                    choose_easy_negatives(
                        args.hard_count - len(replaced_easy),
                        item_list,
                        hard_blocked,
                        matched_selected,
                        rng,
                    )
                )
            target_bins = [popularity_bin(item) for item in replaced_easy[: args.hard_count]]

            if not finite.any():
                hard_items: List[str] = []
            else:
                k = min(args.topk, int(finite.sum()))
                candidate_indices = np.argpartition(-row_scores, k - 1)[:k]
                candidate_indices = candidate_indices[np.isfinite(row_scores[candidate_indices])]
                if candidate_indices.size:
                    candidate_indices = candidate_indices[np.argsort(-row_scores[candidate_indices])]
                hard_pool = [item_list[idx] for idx in candidate_indices]
                hard_items = choose_popularity_matched_hard(
                    hard_pool, target_bins, popularity_bin, rng
                )
                if len(hard_items) < args.hard_count:
                    insufficient += 1
                if hard_items:
                    sims = [float(row_scores[item_to_index[item]]) for item in hard_items]
                    hard_similarities.append(float(np.mean(sims)))

            selected = set(hard_items) | set(easy_items)
            if len(easy_items) < easier_count:
                easy_items.extend(
                    choose_easy_negatives(
                        easier_count - len(easy_items), item_list, blocked, selected, rng
                    )
                )
            if replaced_easy:
                easy_similarities.append(
                    float(np.mean([unmasked_scores[item_to_index[item]] for item in replaced_easy]))
                )
                easy_popularities.extend(train_popularity.get(item, 0) for item in replaced_easy)
            if hard_items:
                hard_popularities.extend(train_popularity.get(item, 0) for item in hard_items)
            if Counter(popularity_bin(item) for item in hard_items) == Counter(
                popularity_bin(item) for item in replaced_easy
            ):
                popularity_bin_matches += 1
            negatives = hard_items + easy_items
            rng.shuffle(negatives)
            target_idx = rng.randrange(20)
            negatives.insert(target_idx, target)
            if len(negatives) != 20 or len(set(negatives)) != 20 or negatives.count(target) != 1:
                raise RuntimeError("Candidate construction failed")

            new_row = dict(row)
            new_row[candidate_key] = [
                {"original_id": item, "title": titles.get(item, item)} for item in negatives
            ]
            new_row[target_key] = {"original_id": target, "title": titles.get(target, target)}
            new_row["target_idx"] = target_idx
            new_row["hard_negative_count"] = len(hard_items)
            new_row["hard_negative_ids"] = list(hard_items)
            new_row["easy_negative_ids"] = list(easy_items)
            new_row["replaced_easy_ids"] = list(replaced_easy)
            new_row["hard_popularity_bins"] = [popularity_bin(item) for item in hard_items]
            new_row["replaced_easy_popularity_bins"] = [
                popularity_bin(item) for item in replaced_easy
            ]
            output_rows.append(new_row)

        print(
            f"Processed test rows {min(start + args.batch_size, len(eval_entries))}/"
            f"{len(eval_entries)}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=False)

    stats = {
        "data": args.data,
        "output": args.output,
        "domain": args.domain,
        "samples": len(output_rows),
        "train_samples_unchanged": train_size,
        "validation_samples_unchanged": val_size,
        "evaluation_samples_hardened": len(eval_entries),
        "test_samples_hardened": test_size,
        "unique_items": len(item_list),
        "catalog_file": args.catalog_file,
        "hard_count": args.hard_count,
        "easy_count": easier_count,
        "topk": args.topk,
        "seed": args.seed,
        "query_mode": args.query_mode,
        "title_weight": args.title_weight,
        "category_weight": category_weight,
        "observed_source": observed_source,
        "observed_json": args.observed_json,
        "raw_samples": raw_entry_count,
        "dropped_short_history": dropped_short_history,
        "dropped_bad_candidates": dropped_bad_candidates,
        "easy_source": "original_test_negatives_then_catalog",
        "popularity_control": "training_popularity_bin_matched",
        "metadata_title_coverage": (len(item_list) - missing_titles) / max(1, len(item_list)),
        "metadata_category_coverage": (len(item_list) - missing_cats) / max(1, len(item_list)),
        "mean_hard_similarity": float(np.mean(hard_similarities)) if hard_similarities else 0.0,
        "mean_easy_similarity": float(np.mean(easy_similarities)) if easy_similarities else 0.0,
        "mean_similarity_gap": (
            float(np.mean(hard_similarities) - np.mean(easy_similarities))
            if hard_similarities and easy_similarities
            else 0.0
        ),
        "mean_hard_popularity": float(np.mean(hard_popularities)) if hard_popularities else 0.0,
        "mean_replaced_easy_popularity": (
            float(np.mean(easy_popularities)) if easy_popularities else 0.0
        ),
        "hard_popularity_gap": (
            float(np.mean(hard_popularities) - np.mean(easy_popularities))
            if hard_popularities and easy_popularities
            else 0.0
        ),
        "popularity_bin_match_rate": popularity_bin_matches / max(1, len(eval_entries)),
        "insufficient_hard_samples": insufficient,
    }
    stats_path = os.path.splitext(args.output)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
