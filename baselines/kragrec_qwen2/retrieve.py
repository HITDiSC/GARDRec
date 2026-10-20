import json
import os

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphRetrieval:
    """KG retrieval used by the server-side K-RagRec-Qwen2 baseline.

    The original K-RagRec code retrieves Freebase subgraphs with SBERT
    embeddings.  This implementation uses the server assets instead:

    * topology: Neo4j database ``kg-ml-1m`` exported by ``build_kg_index.py``
    * node features: 256-d ML-1M GCN item embeddings
    * query features: the same GCN embeddings (no external text encoder)
    """

    def __init__(self, index_dir, movie_path=None):
        self.index_dir = index_dir
        self.movie_path = movie_path or index_dir

        self.G = self._load_graph(os.path.join(index_dir, "graphs", "0.pt"))
        self.G1 = self._load_graph(os.path.join(index_dir, "graphs", "layer2_embeddings_W.pt"))
        self.G.x = self.G.x.float()
        self.G.edge_attr = self.G.edge_attr.float()
        self.G1.x = self.G1.x.float()

        self.x_first = F.normalize(self.G.x, dim=1)
        self.x_second = F.normalize(self.G1.x, dim=1)

        self.item_embeddings = torch.load(
            os.path.join(index_dir, "item_embeddings.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.item_embeddings = {
            str(key): value.float() for key, value in self.item_embeddings.items()
        }

        with open(os.path.join(index_dir, "movie_id_to_node.json"), "r", encoding="utf-8") as handle:
            self.movie_id_to_node = {str(k): int(v) for k, v in json.load(handle).items()}

        self.movie_id_to_name = {}
        name_path = os.path.join(index_dir, "movies_id_name.txt")
        if os.path.exists(name_path):
            with open(name_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        self.movie_id_to_name[parts[0]] = parts[1]

        self.sorted_item_ids, self.popularity_rank = self._load_popularity()

        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(range(int(self.G.num_nodes)))
        edges = self.G.edge_index.t().tolist()
        self.Graph.add_edges_from(edges)

        self.edge_to_index = {}
        for index, (src, dst) in enumerate(edges):
            self.edge_to_index[(int(src), int(dst))] = index

        print(
            "KG index: {} nodes, {} edges, {} movie embeddings, {} popularity items".format(
                int(self.G.num_nodes),
                int(self.G.edge_index.shape[1]),
                len(self.item_embeddings),
                len(self.popularity_rank),
            ),
            flush=True,
        )

    @staticmethod
    def _load_graph(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "x" in payload:
            return Data(
                x=payload["x"],
                edge_index=payload["edge_index"],
                edge_attr=payload["edge_attr"],
                num_nodes=int(payload["num_nodes"]),
            )
        return payload

    def _load_popularity(self):
        path = os.path.join(self.index_dir, "ratings_45.txt")
        if not os.path.exists(path):
            path = os.path.join(self.movie_path, "ratings_45.txt")

        counts = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        item = str(parts[1])
                        counts[item] = counts.get(item, 0) + 1

        if not counts:
            counts = {str(item): 1 for item in self.item_embeddings}
        sorted_items = sorted(counts, key=lambda item: (counts[item], item))
        rank = {item: index for index, item in enumerate(sorted_items)}
        return sorted_items, rank

    def whether_retrieval(self, watching_list, k):
        known = []
        for item in watching_list:
            item = str(item)
            if item in self.popularity_rank and item not in known:
                known.append(item)
        known.sort(key=lambda item: self.popularity_rank[item])
        k = int(k)
        if k <= 0:
            return known
        return known[:k]

    def encode_history(self, item_ids):
        vectors = [
            self.item_embeddings[str(item)]
            for item in item_ids
            if str(item) in self.item_embeddings
        ]
        if not vectors:
            return None
        vector = torch.stack(vectors, dim=0).mean(dim=0)
        return F.normalize(vector, dim=0)

    @staticmethod
    def _topk_nodes(features, query, k):
        if query is None or features.shape[0] == 0:
            return []
        k = min(int(k), int(features.shape[0]))
        scores = features @ query
        return torch.topk(scores, k=k, largest=True).indices.tolist()

    @staticmethod
    def _rerank(features, query, nodes, k):
        if not nodes:
            return []
        if query is None:
            return list(dict.fromkeys(nodes))[: int(k)]
        node_tensor = torch.tensor(nodes, dtype=torch.long)
        scores = features.index_select(0, node_tensor) @ query
        order = torch.argsort(scores, descending=True)[: int(k)].tolist()
        return [nodes[index] for index in order]

    def retrieval_topk(self, input_text, retrieve_movies_list, topk_nodes=3, topk_rerank_nodes=5):
        del input_text  # kept for API compatibility; the graph query is the history mean
        first_nodes = []
        second_nodes = []

        for movie_id in retrieve_movies_list:
            movie_id = str(movie_id)
            node = self.movie_id_to_node.get(movie_id)
            embedding = self.item_embeddings.get(movie_id)
            if node is None or embedding is None:
                continue
            query = F.normalize(embedding, dim=0)
            first_nodes.extend(self._topk_nodes(self.x_first, query, topk_nodes))
            second_nodes.extend(self._topk_nodes(self.x_second, query, topk_nodes))

        first_nodes = list(dict.fromkeys(first_nodes))
        second_nodes = list(dict.fromkeys(second_nodes))
        global_query = self.encode_history(retrieve_movies_list)
        first_nodes = self._rerank(self.x_first, global_query, first_nodes, topk_rerank_nodes)
        second_nodes = self._rerank(self.x_second, global_query, second_nodes, topk_rerank_nodes)

        graphs = []
        for node in first_nodes + second_nodes:
            subgraph = self.get_first_order_subgraph(node)
            if subgraph is not None and subgraph.num_nodes > 0:
                graphs.append(subgraph)
        return graphs

    def get_first_order_subgraph(self, node):
        node = int(node)
        if node not in self.Graph:
            return None
        neighbors = sorted(int(item) for item in self.Graph.neighbors(node))
        sub_nodes = [node] + [item for item in neighbors if item != node]
        node_index = {old: new for new, old in enumerate(sub_nodes)}

        sub_edges = []
        sub_attrs = []
        for neighbor in sub_nodes[1:]:
            for src, dst in ((node, neighbor), (neighbor, node)):
                edge_index = self.edge_to_index.get((src, dst))
                if edge_index is None:
                    continue
                sub_edges.append([node_index[src], node_index[dst]])
                sub_attrs.append(self.G.edge_attr[edge_index])

        if not sub_edges:
            # Keep a valid PyG graph for isolated nodes.
            sub_edges = [[0, 0]]
            sub_attrs = [torch.zeros_like(self.G.x[0])]

        edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(sub_attrs, dim=0).float()
        return Data(
            x=self.G.x[sub_nodes].clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(sub_nodes),
        )