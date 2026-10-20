import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from components.data import load_input_entries
from components.kg_retrieval_enhancement import RetrievalAugmentationBuilder
from components.neo4j_utils import get_item_gcn_embedding
from components import get_neo4j_config
from components.utils import build_ranking_samples_from_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.ERROR)

def resolve_output_file(domain: str) -> str:
    if domain == "amazonbook":
        return "./output/amazonbook/rank_save/retrieval_augmented_details.json"
    elif domain == "ml20m":
        return "./output/ml20m/rank_save/retrieval_augmented_details.json"
    elif domain == "ml1m":
        return "./output/ml1m/rank_save/retrieval_augmented_details.json"
    raise ValueError(f"Unknown domain: {domain}")

def ensure_sample_idx(entries: List[Dict]) -> List[Dict]:
    for idx, entry in enumerate(entries):
        if entry.get("sample_idx") is None:
            entry["sample_idx"] = idx
    return entries

def serialize_retrieval_detail(sample: Dict) -> Dict:
    sample_idx = sample.get("sample_idx")
    if sample_idx is None:
        raise ValueError("sample_idx is required for retrieval detail serialization")
    return {
        "sample_idx": sample_idx,
        "history_retrieval_embedding": sample["history_retrieval_embedding"].tolist(),
        "candidate_match_features": sample["candidate_match_features"].tolist(),
    }

def build_and_save_augmented_samples(
    domain: str = "amazonbook",
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    num_history: int = 10,
    num_candidates: int = 20,
    num_samples: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> str:
    input_file = input_file or ("./data/dataset/AmazonBook/parse_data/train.json" if domain == "amazonbook" else "./data/dataset/ML20M/parse_data/train.json" if domain == "ml20m" else "./data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json")
    output_file = output_file or resolve_output_file(domain)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"[{domain.upper()}] Loading item GCN embeddings for offline retrieval preprocessing")
    gcn_embedding_dict, embedding_dim = get_item_gcn_embedding(domain=domain, limit=None)
    entries = ensure_sample_idx(load_input_entries(input_file))
    samples = build_ranking_samples_from_input(
        entries,
        gcn_embedding_dict,
        num_history=num_history,
        num_candidates=num_candidates,
    )
    if num_samples is not None:
        samples = samples[:num_samples]

    neo4j_conn = None
    try:
        from components.neo4j_utils import Neo4jConnection
        neo4j_conn = Neo4jConnection(**get_neo4j_config(domain))
    except Exception:
        pass
    retrieval_builder = RetrievalAugmentationBuilder(
        neo4j_conn=neo4j_conn,
        gcn_embedding_dict=gcn_embedding_dict,
        embedding_dim=embedding_dim,
        num_candidates=num_candidates,
        domain=domain,
    )

    if max_workers is None:
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        max_workers = min(32, cpu_count * 5)

    logger.info("Enriching %s samples using %s workers", len(samples), max_workers)
    augmented_samples = []
    completed = 0
    total = len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(retrieval_builder.enrich_sample, sample): i for i, sample in enumerate(samples)}
        for fut in as_completed(future_to_idx):
            try:
                res = fut.result()
                augmented_samples.append(res)
            except Exception as e:
                logger.exception("Error enriching a sample: %s", e)
            completed += 1
            if completed % 50 == 0 or completed == total:
                logger.info("Preprocess progress: %s/%s", completed, total)

    serialized_details = [serialize_retrieval_detail(sample) for sample in augmented_samples]
    details_by_sample_idx = {
        str(item["sample_idx"]): {
            "history_retrieval_embedding": item["history_retrieval_embedding"],
            "candidate_match_features": item["candidate_match_features"],
        }
        for item in serialized_details
    }

    payload = {
        "input_file": input_file,
        "num_history": num_history,
        "num_candidates": num_candidates,
        "num_samples": len(serialized_details),
        "details_by_sample_idx": details_by_sample_idx,
    }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info("Saved retrieval details for %s samples to %s", len(serialized_details), output_file)
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KG Retrieval Samples")
    parser.add_argument("--domain", type=str, default="amazonbook", choices=["amazonbook", "ml20m", "ml1m"])
    args = parser.parse_args()
    build_and_save_augmented_samples(domain=args.domain)
