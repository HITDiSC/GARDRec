# Copied minimal version of kg_retrieval_enhancement.py for export
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def get_domain_config(domain: str) -> dict:
    domain = domain.lower()
    if domain == "amazonbook":
        return {
            "MAIN_NODE_LABEL": "Book",
            "MATCH_FEATURE_NAMES": [
                "total_overlap_count",
                "total_overlap_ratio",
                "category_overlap_count",
            ],
            "RELATION_TO_BUCKET": {},
        }
    else:
        return {
            "MAIN_NODE_LABEL": "Movie",
            "MATCH_FEATURE_NAMES": [
                "total_overlap_count",
                "total_overlap_ratio",
                "genre_overlap_count",
            ],
            "RELATION_TO_BUCKET": {},
        }

class RetrievalAugmentationBuilder:
    def __init__(self, neo4j_conn, gcn_embedding_dict: Dict[str, List[float]], embedding_dim: int, num_candidates: int, domain: str = "amazonbook"):
        self.neo4j_conn = neo4j_conn
        self.gcn_embedding_dict = gcn_embedding_dict
        self.embedding_dim = embedding_dim
        self.num_candidates = num_candidates
    def enrich_samples(self, samples: List[Dict], show_progress: bool = True) -> List[Dict]:
        return samples
