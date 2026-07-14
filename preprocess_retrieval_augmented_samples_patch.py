import argparse
import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

from components.data import load_input_entries
from components.kg_retrieval_enhancement import RetrievalAugmentationBuilder
from components.neo4j_utils import get_item_gcn_embedding
from components import get_neo4j_config
from components.utils import build_ranking_samples_from_input
from components.two_stage_recall_test import load_ratings_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_NAMES = [
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

# 这里为导出脚本保留预处理补丁的核心函数（简化版）

def resolve_versioned_output_file(retrieval_details_file: str) -> str:
    base_dir = os.path.dirname(retrieval_details_file) or "."
    base_name = os.path.basename(retrieval_details_file)
    stem, ext = os.path.splitext(base_name)
    normalized_stem = re.sub(r"_v\d+$", "", stem)
    pattern = re.compile(rf"^{re.escape(normalized_stem)}_v(\d+){re.escape(ext)}$")
    max_version = 0
    try:
        for existing_name in os.listdir(base_dir):
            match = pattern.match(existing_name)
            if match:
                max_version = max(max_version, int(match.group(1)))
    except FileNotFoundError:
        os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{normalized_stem}_v{max_version + 1}{ext}")


def ensure_sample_idx(entries: List[Dict]) -> List[Dict]:
    for idx, entry in enumerate(entries):
        if entry.get("sample_idx") is None:
            entry["sample_idx"] = idx
    return entries


def l2_normalize_vec(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm <= 0:
        return v
    return v / norm


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da <= 0 or db <= 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def update_historical_cooccurrence(co_map: Dict[str, Dict[str, int]], hist: List[Dict]) -> int:
    hist_ids = [str(m.get("original_id") or "") for m in hist]
    uniq = list(dict.fromkeys([item_id for item_id in hist_ids if item_id]))
    local_max = 0
    n = len(uniq)
    for left_idx in range(n):
        left_id = uniq[left_idx]
        for right_idx in range(left_idx + 1, n):
            right_id = uniq[right_id]
            left_bucket = co_map.setdefault(left_id, {})
            right_bucket = co_map.setdefault(right_id, {})
            left_bucket[right_id] = left_bucket.get(right_id, 0) + 1
            right_bucket[left_id] = right_bucket.get(left_id, 0) + 1
            local_max = max(local_max, left_bucket[right_id], right_bucket[left_id])
    return local_max


def compute_historical_co_feature(candidate_id: str, hist: List[Dict], historical_co_map: Dict[str, Dict[str, int]], historical_co_max: int) -> float:
    if not candidate_id or not hist or not historical_co_map:
        return 0.0
    co_vals = []
    for movie in hist:
        hist_id = str(movie.get("original_id") or "")
        if not hist_id:
            continue
        candidate_bucket = historical_co_map.get(candidate_id)
        hist_bucket = historical_co_map.get(hist_id)
        value = None
        if candidate_bucket is not None and hist_id in candidate_bucket:
            value = candidate_bucket[hist_id]
        elif hist_bucket is not None and candidate_id in hist_bucket:
            value = hist_bucket[candidate_id]
        if value is not None:
            co_vals.append(float(value))
    if not co_vals or historical_co_max <= 0:
        return 0.0
    return float(np.log1p(sum(co_vals) / len(co_vals))) / float(np.log1p(historical_co_max))


def update_historical_user_items(user_history_map: Dict[str, set], user_id: Optional[str], hist: List[Dict]):
    if not user_id:
        return
    history_items = user_history_map.setdefault(user_id, set())
    for movie in hist:
        item_id = str(movie.get("original_id") or "")
        if item_id:
            history_items.add(item_id)


def compute_historical_user_rated_flag(candidate_id: str, user_id: Optional[str], hist: List[Dict], user_history_map: Dict[str, set]) -> float:
    if not candidate_id:
        return 0.0
    current_hist_ids = {str(movie.get("original_id") or "") for movie in hist if movie.get("original_id") is not None}
    if candidate_id in current_hist_ids:
        return 1.0
    if not user_id:
        return 0.0
    historical_items = user_history_map.get(user_id)
    return 1.0 if historical_items and candidate_id in historical_items else 0.0


def update_historical_transitions(transition_map: Dict[str, Dict[str, int]], hist: List[Dict]) -> int:
    seq = [str(m.get("original_id") or "") for m in hist if m.get("original_id") is not None]
    seq = [item_id for item_id in seq if item_id]
    if len(seq) < 2:
        return 0
    added = 0
    for left_id, right_id in zip(seq[:-1], seq[1:]):
        left_bucket = transition_map.setdefault(left_id, {})
        left_bucket[right_id] = left_bucket.get(right_id, 0) + 1
        added += 1
    return added


def compute_historical_transition_feat(candidate_id: str, hist: List[Dict], historical_transition_map: Dict[str, Dict[str, int]]) -> float:
    if not candidate_id or not hist or not historical_transition_map:
        return 0.0
    seq = [str(m.get("original_id") or "") for m in hist if m.get("original_id") is not None]
    seq = [item_id for item_id in seq if item_id]
    if not seq:
        return 0.0
    last_ids = seq[-1:]
    total = 0.0
    for hist_id in last_ids:
        trans_bucket = historical_transition_map.get(hist_id, {})
        total_out = float(sum(trans_bucket.values()))
        if total_out > 0:
            total += float(trans_bucket.get(candidate_id, 0)) / total_out
    return total / len(last_ids) if last_ids else 0.0


def build_interaction_features_for_samples(
    samples: List[Dict],
    entries_by_idx: Dict[str, Dict],
    gcn_embedding_dict: Dict[str, List[float]],
    user_stats: Dict,
    item_stats: Dict,
    user_item_set: set,
    max_user_count: int,
    max_item_count: int,
    retrieval_details_map: Dict[str, Dict],
    num_candidates: int = 20,
    initial_state: Optional[Dict] = None,
    update_state: bool = True,
    return_state: bool = False,
):
    out_map = {}
    if initial_state is None:
        initial_state = {}
    historical_co_map = initial_state.get("historical_co_map", {})
    historical_co_max = int(initial_state.get("historical_co_max", 0))
    historical_user_items = initial_state.get("historical_user_items", {})
    historical_transition_map = initial_state.get("historical_transition_map", {})

    for s in samples:
        sid = str(s.get("sample_idx"))
        hist = s.get("history_movies", [])
        cand_list = s.get("candidate_movies", [])[:num_candidates]

        first_scores = [float(item.get("score") or 0.0) for item in cand_list]
        min_first = min(first_scores) if first_scores else 0.0
        max_first = max(first_scores) if first_scores else 0.0
        denom_first = max_first - min_first if abs(max_first - min_first) > 1e-12 else 1.0

        centroid = None
        history_vecs = []
        if hist:
            vecs = [np.asarray(gcn_embedding_dict.get(str(m.get("original_id")), np.zeros(1)), dtype=np.float32) for m in hist if m.get("original_id") is not None]
            if vecs:
                weights = [1.0 + float(i + 1) / len(vecs) for i in range(len(vecs))]
                total_w = sum(weights)
                accum = np.zeros_like(vecs[0], dtype=np.float32)
                for v, w in zip(vecs, weights):
                    accum += v * float(w)
                centroid = l2_normalize_vec(accum / total_w)
                history_vecs = [l2_normalize_vec(v) for v in vecs]

        retrieval_detail = retrieval_details_map.get(sid, {})
        match_feats = retrieval_detail.get("candidate_match_features")

        entry = entries_by_idx.get(sid)
        user_id = str(entry.get("user_id")) if entry and entry.get("user_id") is not None else None

        features_rows = []
        for i, c in enumerate(cand_list):
            oid = str(c.get("original_id") or "")
            cand_emb = gcn_embedding_dict.get(oid)
            if cand_emb is not None:
                cand_emb = np.asarray(cand_emb, dtype=np.float32)
                cand_emb_norm = l2_normalize_vec(cand_emb)
                gsim = cosine_sim(centroid, cand_emb_norm) if centroid is not None else 0.0
                lmax = max((cosine_sim(hv, cand_emb_norm) for hv in history_vecs), default=0.0)
            else:
                gsim = 0.0
                lmax = 0.0
            raw_first = float(c.get("score") or 0.0)
            first_norm = (raw_first - min_first) / denom_first if denom_first else 0.0
            ustat = user_stats.get(user_id, {}) if user_id is not None else {}
            user_avg = float(ustat.get("avg", 0.0)) / 5.0
            user_cnt_feat = np.log1p(float(ustat.get("count", 0))) / (np.log1p(float(max_user_count)) if max_user_count > 0 else 1.0)
            istat = item_stats.get(oid, {})
            item_avg = float(istat.get("avg", 0.0)) / 5.0
            item_cnt_feat = np.log1p(float(istat.get("count", 0))) / (np.log1p(float(max_item_count)) if max_item_count > 0 else 1.0)
            user_rated_flag = compute_historical_user_rated_flag(oid, user_id, hist, historical_user_items)
            co_feat = compute_historical_co_feature(oid, hist, historical_co_map, historical_co_max)
            trans_feat = compute_historical_transition_feat(oid, hist, historical_transition_map)
            genre_feat = 0.0
            if match_feats is not None and i < len(match_feats):
                row = match_feats[i]
                if isinstance(row, (list, tuple)) and len(row) > 2:
                    genre_feat = float(row[2])
            rank_feat = float(i) / float(max(1, len(cand_list) - 1)) if len(cand_list) > 1 else 0.0
            co_trans_mul = co_feat * trans_feat
            co_trans_absdiff = abs(co_feat - trans_feat)
            features_rows.append([
                gsim,
                lmax,
                first_norm,
                user_avg,
                user_cnt_feat,
                item_avg,
                item_cnt_feat,
                user_rated_flag,
                co_feat,
                trans_feat,
                genre_feat,
                rank_feat,
                co_trans_mul,
                co_trans_absdiff,
            ])
        out_map[sid] = features_rows
        if update_state:
            historical_co_max = max(historical_co_max, update_historical_cooccurrence(historical_co_map, hist))
            update_historical_user_items(historical_user_items, user_id, hist)
            update_historical_transitions(historical_transition_map, hist)

    if return_state:
        return out_map, {
            "historical_co_map": historical_co_map,
            "historical_co_max": historical_co_max,
            "historical_user_items": historical_user_items,
            "historical_transition_map": historical_transition_map,
        }
    return out_map


def main():
    parser = argparse.ArgumentParser(description="Add interaction features to retrieval details file")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--retrieval-details", default=None)
    parser.add_argument("--ratings-path", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--split-first-prefix", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--freeze-eval-prefix", action="store_true")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--domain", type=str, default="amazonbook", choices=["amazonbook", "ml20m", "ml1m"])
    args = parser.parse_args()

    input_file = args.input_file or ("./data/dataset/AmazonBook/parse_data/train.json" if args.domain == "amazonbook" else "./data/dataset/ML20M/parse_data/train.json" if args.domain == "ml20m" else "./data/dataset/ML1M/parse_data/train.json")
    retrieval_details_file = args.retrieval_details or ("./output/amazonbook/rank_save/retrieval_augmented_details.json" if args.domain == "amazonbook" else "./output/ml20m/rank_save/retrieval_augmented_details.json" if args.domain == "ml20m" else "./output/ml1m/rank_save/retrieval_augmented_details.json")
    output_file = args.output_file
    ratings_path = args.ratings_path or ("./data/movielens/ml-1m/ratings.dat" if args.domain == "ml1m" else "./data/movielens/ml-1m/ratings.dat")

    if not os.path.exists(retrieval_details_file):
        raise FileNotFoundError(f"Retrieval details file not found: {retrieval_details_file}")

    logger.info("Loading GCN embeddings")
    gcn_embedding_dict, _ = get_item_gcn_embedding(args.domain)

    logger.info("Loading entries and ranking samples")
    entries = ensure_sample_idx(load_input_entries(input_file))
    entries_by_idx = {str(e.get("sample_idx")): e for e in entries}
    samples = build_ranking_samples_from_input(entries, gcn_embedding_dict, num_history=10, num_candidates=args.num_candidates)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    ratings_stats = load_ratings_stats(ratings_path)
    if len(ratings_stats) == 6:
        user_stats, item_stats, user_item_set, max_user_count, max_item_count, _ = ratings_stats
    else:
        user_stats, item_stats, user_item_set, max_user_count, max_item_count = ratings_stats

    with open(retrieval_details_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    details_map = payload.get("details_by_sample_idx", payload)

    if args.split_first_prefix:
        train_size = int(len(samples) * args.train_ratio)
        train_size = max(1, min(train_size, len(samples)))
        train_samples = samples[:train_size]
        eval_samples = samples[train_size:]
        train_map, prefix_state = build_interaction_features_for_samples(
            train_samples,
            entries_by_idx,
            gcn_embedding_dict,
            user_stats,
            item_stats,
            user_item_set,
            max_user_count,
            max_item_count,
            details_map,
            num_candidates=args.num_candidates,
            initial_state=None,
            update_state=True,
            return_state=True,
        )
        eval_map = build_interaction_features_for_samples(
            eval_samples,
            entries_by_idx,
            gcn_embedding_dict,
            user_stats,
            item_stats,
            user_item_set,
            max_user_count,
            max_item_count,
            details_map,
            num_candidates=args.num_candidates,
            initial_state=prefix_state,
            update_state=not args.freeze_eval_prefix,
            return_state=False,
        )
        interaction_map = {**train_map, **(eval_map if isinstance(eval_map, dict) else eval_map[0])}
    else:
        interaction_map = build_interaction_features_for_samples(
            samples,
            entries_by_idx,
            gcn_embedding_dict,
            user_stats,
            item_stats,
            user_item_set,
            max_user_count,
            max_item_count,
            details_map,
            num_candidates=args.num_candidates,
            initial_state=None,
            update_state=True,
            return_state=False,
        )
        if isinstance(interaction_map, tuple):
            interaction_map = interaction_map[0]

    missing = 0
    for sid, feats in interaction_map.items():
        if sid not in details_map:
            missing += 1
            continue
        details_map[sid]["interaction_features"] = feats

    if output_file is None:
        output_file = resolve_versioned_output_file(retrieval_details_file)

    payload_out = dict(payload)
    payload_out["details_by_sample_idx"] = details_map
    payload_out["interaction_feature_names"] = FEATURE_NAMES
    payload_out["feature_construction_mode"] = {
        "split_first_prefix": bool(args.split_first_prefix),
        "train_ratio": float(args.train_ratio),
        "freeze_eval_prefix": bool(args.freeze_eval_prefix),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, ensure_ascii=False, indent=2)

    logger.info("Saved patched retrieval details to %s", output_file)

if __name__ == "__main__":
    main()
