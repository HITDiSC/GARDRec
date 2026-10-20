import argparse
import csv
import json
import os

import numpy as np
import torch
from neo4j import GraphDatabase


def _norm(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def main():
    parser = argparse.ArgumentParser("Build K-RagRec KG/GCN retrieval index from kg-ml-1m")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default="Wangyong123")
    parser.add_argument("--database", type=str, default="kg-ml-1m")
    parser.add_argument("--gcn_embeddings", type=str,
                        default="output/ml1m/rank_save/gcn_embeddings.json")
    parser.add_argument("--input", type=str,
                        default="data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json")
    parser.add_argument("--out_dir", type=str, default="data/dataset/kg_ml1m_kragrec")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "nodes"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "edges"), exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session(database=args.database) as session:
        node_rows = list(session.run("""
            MATCH (n)
            RETURN elementId(n) AS eid, labels(n)[0] AS label,
                   coalesce(n.node_id, '') AS node_id,
                   coalesce(n.original_id, '') AS original_id,
                   coalesce(n.name, n.title, n.node_id, '') AS text
        """))
        edge_rows = list(session.run("""
            MATCH (a)-[r]->(b)
            RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS rel
        """))
    driver.close()

    eid_to_idx = {}
    labels = []
    node_ids = []
    original_ids = []
    texts = []
    for idx, row in enumerate(node_rows):
        eid_to_idx[row["eid"]] = idx
        labels.append(row["label"])
        node_ids.append(str(row["node_id"]))
        original_ids.append(str(row["original_id"]))
        texts.append(str(row["text"]))

    n_nodes = len(node_rows)
    src_list = []
    dst_list = []
    rel_list = []
    for row in edge_rows:
        src = eid_to_idx.get(row["src"])
        dst = eid_to_idx.get(row["dst"])
        if src is None or dst is None:
            continue
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(str(row["rel"]))

    with open(args.gcn_embeddings, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    emb_map = payload.get("embeddings", payload)
    dim = int(payload.get("dim", len(next(iter(emb_map.values())))))
    print("nodes", n_nodes, "edges", len(src_list), "gcn items", len(emb_map), "dim", dim)

    x = np.zeros((n_nodes, dim), dtype=np.float32)
    movie_mask = np.zeros(n_nodes, dtype=bool)
    for idx, (label, oid) in enumerate(zip(labels, original_ids)):
        if label == "Movie" and oid in emb_map:
            x[idx] = np.asarray(emb_map[oid], dtype=np.float32)
            movie_mask[idx] = True

    src_arr = np.asarray(src_list, dtype=np.int64)
    dst_arr = np.asarray(dst_list, dtype=np.int64)
    for _ in range(2):
        sums = np.zeros((n_nodes, dim), dtype=np.float32)
        counts = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(sums, src_arr, x[dst_arr])
        np.add.at(counts, src_arr, 1.0)
        mean = sums / np.maximum(counts[:, None], 1.0)
        missing = ~movie_mask
        x[missing] = mean[missing]
    x = _norm(x)

    edge_attr = 0.5 * (x[src_arr] + x[dst_arr])
    edge_attr = _norm(edge_attr)
    edge_index = torch.tensor(
        np.concatenate([np.stack([src_arr, dst_arr], axis=0),
                        np.stack([dst_arr, src_arr], axis=0)], axis=1),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(np.concatenate([edge_attr, edge_attr], axis=0), dtype=torch.float32)
    torch.save(
        {
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": n_nodes,
            "format": "raw_tensors_v1",
        },
        os.path.join(args.out_dir, "graphs", "0.pt"),
    )

    x2 = x.copy()
    for _ in range(2):
        sums = np.zeros_like(x2)
        counts = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(sums, src_arr, x2[dst_arr])
        np.add.at(counts, src_arr, 1.0)
        neigh = sums / np.maximum(counts[:, None], 1.0)
        x2 = _norm(0.5 * x2 + 0.5 * neigh)
    torch.save(
        {
            "x": torch.tensor(x2, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": n_nodes,
            "format": "raw_tensors_v1",
        },
        os.path.join(args.out_dir, "graphs", "layer2_embeddings_W.pt"),
    )

    item_embeddings = {}
    movies_id_name = []
    movie_id_to_node = {}
    for node_index, (label, oid, text) in enumerate(zip(labels, original_ids, texts)):
        if label == "Movie" and oid in emb_map:
            item_embeddings[oid] = torch.tensor(np.asarray(emb_map[oid], dtype=np.float32))
            movies_id_name.append((oid, text))
            movie_id_to_node[oid] = node_index
    torch.save(item_embeddings, os.path.join(args.out_dir, "item_embeddings.pt"))
    with open(os.path.join(args.out_dir, "movies_id_name.txt"), "w", encoding="utf-8") as handle:
        for oid, text in movies_id_name:
            handle.write(f"{oid}\t{text}\n")
    with open(os.path.join(args.out_dir, "movie_id_to_node.json"), "w", encoding="utf-8") as handle:
        json.dump(movie_id_to_node, handle, ensure_ascii=False)

    with open(args.input, "r", encoding="utf-8") as handle:
        inputs = json.load(handle)
    with open(os.path.join(args.out_dir, "ratings_45.txt"), "w", encoding="utf-8") as handle:
        for row in inputs:
            user_id = str(row.get("user_id", ""))
            for item in row.get("history_movies", row.get("history_books", [])):
                handle.write(f"{user_id}\t{item.get('original_id')}\n")

    with open(os.path.join(args.out_dir, "nodes", "all_nodes.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "node_attr"])
        for idx, text in enumerate(texts):
            writer.writerow([idx, text])
    with open(os.path.join(args.out_dir, "edges", "all_edges.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "edge_attr", "dst"])
        for src, rel, dst in zip(src_list, rel_list, dst_list):
            writer.writerow([src, rel, dst])

    meta = {
        "nodes": n_nodes,
        "edges": len(src_list),
        "dim": dim,
        "movie_nodes": int(movie_mask.sum()),
        "gcn_items": len(emb_map),
        "database": args.database,
        "features": "gcn_embeddings_256",
        "graph": "kg-ml-1m",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()