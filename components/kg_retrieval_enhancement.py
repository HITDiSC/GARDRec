import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_domain_config(domain: str) -> dict:
    domain = domain.lower()
    if domain == "amazonbook":
        return {
            "MAIN_NODE_LABEL": "Book",
            "COMMON_NODE_TYPE_TO_BUCKET": {
                "Category": "category",
                "Brand": "brand",
                "Book": "also_buy",
            },
            "MATCH_FEATURE_NAMES": [
                "total_overlap_count",
                "total_overlap_ratio",
                "category_overlap_count",
                "brand_overlap_count",
                "also_buy_overlap_count",
            ],
            "RELATION_TO_BUCKET": {
                "BELONGS_TO_CATEGORY": "category",
                "HAS_BRAND": "brand",
                "ALSO_BUY": "also_buy",
            },
        }
    else:
        return {
            "MAIN_NODE_LABEL": "Movie",
            "COMMON_NODE_TYPE_TO_BUCKET": {
                "Genre": "genre",
                "Company": "company",
                "Director": "director",
                "Actor": "actor",
                "Keyword": "keyword",
                "Writer": "writer",
            },
            "MATCH_FEATURE_NAMES": [
                "total_overlap_count",
                "total_overlap_ratio",
                "genre_overlap_count",
                "director_overlap_count",
                "actor_overlap_count",
                "keyword_overlap_count",
                "writer_overlap_count",
                "company_overlap_count",
            ],
            "RELATION_TO_BUCKET": {
                "BELONGS_TO_GENRE": "genre",
                "PRODUCED_BY": "company",
                "DIRECTED": "director",
                "ACTED_IN": "actor",
                "HAS_KEYWORD": "keyword",
                "WROTE": "writer",
            },
        }


class RetrievalAugmentationBuilder:
    def __init__(
        self,
        neo4j_conn,
        gcn_embedding_dict: Dict[str, List[float]],
        embedding_dim: int,
        num_candidates: int,
        domain: str = "amazonbook",
        min_common_history: int = 2,
        common_node_limit: int = 20,
        related_movie_limit: int = 32,
    ):
        self.neo4j_conn = neo4j_conn
        self.gcn_embedding_dict = gcn_embedding_dict
        self.embedding_dim = embedding_dim
        self.num_candidates = num_candidates
        self.min_common_history = min_common_history
        self.common_node_limit = common_node_limit
        self.related_movie_limit = related_movie_limit
        self.domain = domain.lower()
        self.config = get_domain_config(self.domain)
        self.original_to_node_id, self.node_to_original_id = self._load_node_mappings()
        self.history_common_cache: Dict[Tuple[str, ...], List[Dict]] = {}
        self.common_node_neighbor_cache: Dict[str, List[str]] = {}
        self.movie_attribute_cache: Dict[str, Dict[str, set]] = {}
        self._lock = threading.Lock()

    def _load_node_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        label = self.config["MAIN_NODE_LABEL"]
        query = f"""
        MATCH (m:{label})
        WHERE m.original_id IS NOT NULL AND m.node_id IS NOT NULL
        RETURN m.original_id AS original_id, m.node_id AS node_id
        """
        records = self.neo4j_conn.execute_query(query)
        original_to_node_id = {str(record["original_id"]): str(record["node_id"]) for record in records}
        node_to_original_id = {str(record["node_id"]): str(record["original_id"]) for record in records}
        logger.info("Loaded %s %s node mappings", len(original_to_node_id), label)
        return original_to_node_id, node_to_original_id

    def _history_signature(self, history_items: List[Dict]) -> Tuple[str, ...]:
        history_ids = [str(item.get("original_id")) for item in history_items if item.get("original_id") is not None]
        return tuple(sorted(history_ids))

    def _history_node_ids(self, history_items: List[Dict]) -> List[str]:
        history_node_ids = []
        for item in history_items:
            original_id = str(item.get("original_id"))
            node_id = self.original_to_node_id.get(original_id)
            if node_id is not None:
                history_node_ids.append(node_id)
        return history_node_ids

    def _get_history_common_nodes(self, history_items: List[Dict]) -> List[Dict]:
        signature = self._history_signature(history_items)
        cached = self.history_common_cache.get(signature)
        if cached is not None:
            return cached
        history_node_ids = self._history_node_ids(history_items)
        if len(history_node_ids) < self.min_common_history:
            self.history_common_cache[signature] = []
            return []
        common_nodes = self.neo4j_conn.find_common_nodes(
            history_node_ids,
            min_connected_movies=self.min_common_history,
        )[: self.common_node_limit]
        self.history_common_cache[signature] = common_nodes
        return common_nodes

    def _get_neighbor_item_ids_for_common_node(self, node_id: str) -> List[str]:
        cached = self.common_node_neighbor_cache.get(node_id)
        if cached is not None:
            return cached

        label = self.config["MAIN_NODE_LABEL"]
        query = f"""
        MATCH (attr {{node_id: $node_id}})--(m:{label})
        WHERE m.original_id IS NOT NULL 
          AND (m.embedding IS NOT NULL OR m.hybrid_embedding IS NOT NULL OR m.gcn_embedding IS NOT NULL)
        RETURN DISTINCT m.original_id AS original_id
        LIMIT $limit
        """
        records = self.neo4j_conn.execute_query(query, parameters={"node_id": node_id, "limit": self.related_movie_limit})
        item_ids = [str(record["original_id"]) for record in records]
        self.common_node_neighbor_cache[node_id] = item_ids
        return item_ids

    def _build_history_retrieval_embedding(self, common_nodes: List[Dict]) -> np.ndarray:
        if not common_nodes:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        weighted_embeddings = []
        weights = []
        for node in common_nodes:
            neighbor_ids = self._get_neighbor_item_ids_for_common_node(str(node.get("node_id")))
            neighbor_embeddings = [
                np.asarray(self.gcn_embedding_dict[item_id], dtype=np.float32)
                for item_id in neighbor_ids
                if item_id in self.gcn_embedding_dict
            ]
            if not neighbor_embeddings:
                continue
            node_embedding = np.mean(np.stack(neighbor_embeddings, axis=0), axis=0)
            weight = float(node.get("connected_movie_count", 1.0))
            weighted_embeddings.append(node_embedding * weight)
            weights.append(weight)
        if not weighted_embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.sum(np.stack(weighted_embeddings, axis=0), axis=0) / max(sum(weights), 1.0)

    def _get_candidate_attribute_sets(self, original_id: str) -> Dict[str, set]:
        original_id = str(original_id)
        cached = self.movie_attribute_cache.get(original_id)
        if cached is not None:
            return cached

        node_id = self.original_to_node_id.get(original_id)
        empty_sets = {bucket: set() for bucket in self.config["COMMON_NODE_TYPE_TO_BUCKET"].values()}
        if node_id is None:
            self.movie_attribute_cache[original_id] = empty_sets
            return empty_sets

        label = self.config["MAIN_NODE_LABEL"]
        rel_types = "', '".join(self.config["RELATION_TO_BUCKET"].keys())
        query = f"""
        MATCH (m:{label} {{node_id: $node_id}})-[r]-(attr)
        WHERE type(r) IN ['{rel_types}']
        WITH type(r) AS relation_type, COLLECT(DISTINCT attr.node_id) AS attr_node_ids
        RETURN relation_type, attr_node_ids
        """
        records = self.neo4j_conn.execute_query(query, parameters={"node_id": node_id})
        attribute_sets = {bucket: set() for bucket in self.config["COMMON_NODE_TYPE_TO_BUCKET"].values()}
        for record in records:
            bucket = self.config["RELATION_TO_BUCKET"].get(record["relation_type"])
            if bucket is None:
                continue
            attribute_sets[bucket].update(str(nid) for nid in record["attr_node_ids"])
        self.movie_attribute_cache[original_id] = attribute_sets
        return attribute_sets

    def _build_common_node_buckets(self, common_nodes: List[Dict]) -> Dict[str, set]:
        common_buckets = {bucket: set() for bucket in self.config["COMMON_NODE_TYPE_TO_BUCKET"].values()}
        for node in common_nodes:
            bucket = self.config["COMMON_NODE_TYPE_TO_BUCKET"].get(str(node.get("node_type")))
            if bucket is None:
                continue
            common_buckets[bucket].add(str(node.get("node_id")))
        return common_buckets

    def _build_candidate_match_features(self, candidate_items: List[Dict], common_nodes: List[Dict]) -> np.ndarray:
        common_buckets = self._build_common_node_buckets(common_nodes)
        total_common_nodes = sum(len(node_ids) for node_ids in common_buckets.values())
        feature_rows = []
        match_names = self.config["MATCH_FEATURE_NAMES"]
        dim = len(match_names)
        for item in candidate_items[: self.num_candidates]:
            attribute_sets = self._get_candidate_attribute_sets(str(item.get("original_id")))
            overlaps = {
                bucket: len(attribute_sets[bucket].intersection(common_buckets[bucket]))
                for bucket in common_buckets
            }
            total_overlap = sum(overlaps.values())
            overlap_ratio = float(total_overlap) / float(total_common_nodes) if total_common_nodes else 0.0
            feature_row = [float(total_overlap), overlap_ratio]
            for feat in match_names[2:]:
                bucket_name = feat.replace("_overlap_count", "")
                feature_row.append(float(overlaps.get(bucket_name, 0)))
            feature_rows.append(feature_row)
        while len(feature_rows) < self.num_candidates:
            feature_rows.append([0.0] * dim)
        return np.asarray(feature_rows, dtype=np.float32)

    def enrich_sample(self, sample: Dict) -> Dict:
        raw_history = sample.get("history_movies", sample.get("history_books", []))
        raw_candidates = sample.get("candidate_movies", sample.get("candidate_books", []))
        common_nodes = self._get_history_common_nodes(raw_history)
        enriched_sample = dict(sample)
        enriched_sample["history_common_nodes"] = common_nodes
        enriched_sample["history_retrieval_embedding"] = self._build_history_retrieval_embedding(common_nodes)
        enriched_sample["candidate_match_features"] = self._build_candidate_match_features(raw_candidates, common_nodes)
        return enriched_sample

    def enrich_samples(self, samples: List[Dict], show_progress: bool = True) -> List[Dict]:
        iterator = tqdm(samples, desc="Building retrieval enhancements") if show_progress else samples
        return [self.enrich_sample(sample) for sample in iterator]
