#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


@dataclass
class Sample:
    user: int
    history: List[int]
    candidates: List[int]
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _detect_fields(row: Dict) -> Tuple[str, str, str]:
    history_key = next((key for key in ('history_movies', 'history_books', 'history') if key in row), None)
    candidate_key = next((key for key in ('candidate_movies', 'candidate_books', 'candidates') if key in row), None)
    target_key = next((key for key in ('target_movie', 'target_book', 'target') if key in row), None)
    if history_key is None or candidate_key is None or target_key is None:
        raise KeyError(
            'Could not detect history/candidate/target fields. '
            f'Available keys: {sorted(row.keys())}'
        )
    return history_key, candidate_key, target_key


def _original_id(value) -> str:
    if isinstance(value, dict):
        for key in ('original_id', 'item_id', 'id'):
            if key in value:
                return str(value[key])
        raise KeyError(f'Item object has no id field: {sorted(value.keys())}')
    return str(value)


def load_and_encode(path: str) -> Tuple[List[Sample], Dict[str, int]]:
    with open(path, 'r', encoding='utf-8') as handle:
        raw = json.load(handle)
    parsed = []
    dropped_short_history = 0
    dropped_bad_candidates = 0
    history_key = candidate_key = target_key = None
    for row in raw:
        if history_key is None:
            history_key, candidate_key, target_key = _detect_fields(row)
        history = [_original_id(item) for item in row.get(history_key, [])]
        candidates = [_original_id(item) for item in row.get(candidate_key, [])]
        target_id = _original_id(row.get(target_key))
        if not history:
            dropped_short_history += 1
            continue
        if len(candidates) != 20 or candidates.count(target_id) != 1:
            dropped_bad_candidates += 1
            continue
        parsed.append({
            'user': str(row.get('user_id')),
            'history': history[-10:],
            'candidates': candidates,
            'label': candidates.index(target_id),
        })
    item_ids = sorted({item for row in parsed for item in row['history'] + row['candidates']})
    user_ids = sorted({row['user'] for row in parsed})
    item_to_id = {item: idx + 1 for idx, item in enumerate(item_ids)}
    user_to_id = {user: idx for idx, user in enumerate(user_ids)}
    samples = [
        Sample(
            user=user_to_id[row['user']],
            history=[PAD_ID] * (10 - len(row['history'])) + [item_to_id[item] for item in row['history']],
            candidates=[item_to_id[item] for item in row['candidates']],
            label=int(row['label']),
        )
        for row in parsed
    ]
    stats = {
        'raw_rows': len(raw),
        'kept_rows': len(samples),
        'dropped_short_history': dropped_short_history,
        'dropped_bad_candidates': dropped_bad_candidates,
        'users': len(user_to_id),
        'items': len(item_to_id),
        'fields': {
            'history': history_key,
            'candidates': candidate_key,
            'target': target_key,
        },
    }
    return samples, stats


def split_samples(
    samples: Sequence[Sample], train_ratio: float = 0.9, val_ratio: float = 0.1
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    total = len(samples)
    test_size = int(round(total * (1.0 - train_ratio)))
    val_size = int(round(total * val_ratio))
    train_size = total - test_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError('The split must leave non-empty train, validation, and test sets')
    return (
        list(samples[:train_size]),
        list(samples[train_size : train_size + val_size]),
        list(samples[train_size + val_size :]),
    )


def build_user_positives(
    samples: Sequence[Sample], include_target: bool = True
) -> Tuple[Set[Tuple[int, int]], DefaultDict[int, Set[int]]]:
    edges: Set[Tuple[int, int]] = set()
    user_pos: DefaultDict[int, Set[int]] = defaultdict(set)
    for sample in samples:
        items = set(sample.history)
        if include_target:
            items.add(sample.candidates[sample.label])
        for item in items:
            edges.add((sample.user, item))
            user_pos[sample.user].add(item)
    return edges, user_pos


def build_graph_data(
    train: Sequence[Sample], valid: Sequence[Sample], test: Sequence[Sample]
) -> Tuple[Set[Tuple[int, int]], DefaultDict[int, Set[int]]]:
    # Histories are model inputs at test time, so graph-based baselines may use them
    # transductively. Test targets are never added to the training graph.
    train_edges, user_pos = build_user_positives(train, include_target=True)
    valid_history_edges, _ = build_user_positives(valid, include_target=False)
    test_history_edges, _ = build_user_positives(test, include_target=False)
    edges = train_edges | valid_history_edges | test_history_edges
    return edges, user_pos


def item_popularity(edges: Iterable[Tuple[int, int]]) -> Counter:
    counts: Counter = Counter()
    for _, item in edges:
        counts[item] += 1
    return counts


@torch.no_grad()
def rank_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    ranks = (order == labels.unsqueeze(1)).float().argmax(dim=-1) + 1
    ranks_f = ranks.float()
    dcg = torch.where(ranks <= 5, 1.0 / torch.log2(ranks_f + 1.0), torch.zeros_like(ranks_f))
    return {
        'top1_acc': float((ranks <= 1).float().mean().item()),
        'top3_acc': float((ranks <= 3).float().mean().item()),
        'top5_acc': float((ranks <= 5).float().mean().item()),
        'ndcg5': float(dcg.mean().item()),
        'mrr': float((1.0 / ranks_f).mean().item()),
    }


def score_matrix(model, samples: Sequence[Sample], device: torch.device) -> torch.Tensor:
    users = torch.tensor([s.user for s in samples], dtype=torch.long, device=device)
    histories = torch.tensor([s.history for s in samples], dtype=torch.long, device=device)
    candidates = torch.tensor([s.candidates for s in samples], dtype=torch.long, device=device)
    with torch.no_grad():
        return model.score_candidates(users, histories, candidates)


def evaluate(model, samples: Sequence[Sample], device: torch.device) -> Dict[str, float]:
    scores = score_matrix(model, samples, device)
    labels = torch.tensor([s.label for s in samples], dtype=torch.long, device=device)
    return rank_metrics(scores, labels)


def sample_negatives(
    users: torch.Tensor,
    user_pos: DefaultDict[int, Set[int]],
    negative_pool: torch.Tensor,
) -> torch.Tensor:
    pool = negative_pool.to(users.device)
    negatives = pool[torch.randint(0, pool.numel(), users.shape, device=users.device)]
    for idx in range(users.numel()):
        user = int(users[idx].item())
        while int(negatives[idx].item()) in user_pos[user]:
            negatives[idx] = pool[int(torch.randint(0, pool.numel(), (1,), device=users.device).item())]
    return negatives


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=PAD_ID)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        with torch.no_grad():
            self.item_emb.weight[PAD_ID].zero_()

    def score_candidates(self, users: torch.Tensor, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(users)
        item_vec = self.item_emb(candidates)
        return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, layers: int = 2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=PAD_ID)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        with torch.no_grad():
            self.item_emb.weight[PAD_ID].zero_()
        self.register_buffer('adj_indices', torch.empty(2, 0, dtype=torch.long), persistent=False)
        self.register_buffer('adj_values', torch.empty(0), persistent=False)
        self.adj_shape: Tuple[int, int] = (n_users + n_items, n_users + n_items)

    def set_graph(self, edges: Iterable[Tuple[int, int]]) -> None:
        edge_set = set(edges)
        degree_users = Counter(user for user, _ in edge_set)
        degree_items = Counter(item for _, item in edge_set)
        rows: List[int] = []
        cols: List[int] = []
        values: List[float] = []
        for user, item in edge_set:
            item_node = self.n_users + item - 1
            weight = 1.0 / math.sqrt(degree_users[user] * degree_items[item])
            rows.extend((user, item_node))
            cols.extend((item_node, user))
            values.extend((weight, weight))
        self.adj_indices = torch.tensor([rows, cols], dtype=torch.long)
        self.adj_values = torch.tensor(values, dtype=torch.float32)

    def _propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight[1:]], dim=0)
        outputs = [all_emb]
        adj = torch.sparse_coo_tensor(
            self.adj_indices,
            self.adj_values,
            self.adj_shape,
            device=all_emb.device,
        ).coalesce()
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            outputs.append(all_emb)
        final = torch.stack(outputs, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    def score_candidates(self, users: torch.Tensor, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user_vec, item_vec = self._propagate()
        return (user_vec[users].unsqueeze(1) * item_vec[candidates - 1]).sum(dim=-1)


def train_bpr(
    model,
    train: Sequence[Sample],
    valid: Sequence[Sample],
    test: Sequence[Sample],
    edges: Sequence[Tuple[int, int]],
    user_pos: DefaultDict[int, Set[int]],
    negative_pool: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_tensor = torch.tensor(list(edges), dtype=torch.long)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate(model, valid, device)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(edge_tensor.size(0))
        total_loss = 0.0
        batches = 0
        for start in range(0, edge_tensor.size(0), batch_size):
            batch = edge_tensor[perm[start : start + batch_size]].to(device)
            users, positives = batch[:, 0], batch[:, 1]
            negatives = sample_negatives(users, user_pos, negative_pool)
            if isinstance(model, LightGCN):
                user_vec, item_vec = model._propagate()
                pos_scores = (user_vec[users] * item_vec[positives - 1]).sum(dim=-1)
                neg_scores = (user_vec[users] * item_vec[negatives - 1]).sum(dim=-1)
            else:
                user_vec = model.user_emb(users)
                pos_scores = (user_vec * model.item_emb(positives)).sum(dim=-1)
                neg_scores = (user_vec * model.item_emb(negatives)).sum(dim=-1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        metrics = evaluate(model, valid, device)
        if metrics['top1_acc'] > best_metrics['top1_acc']:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:03d} loss={total_loss / max(1, batches):.4f} "
                f"top1={metrics['top1_acc']:.4f} ndcg5={metrics['ndcg5']:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return evaluate(model, test, device), {
        'best_epoch': float(best_epoch),
        'best_val_top1': float(best_metrics['top1_acc']),
    }


class PopularityBaseline:
    def __init__(self, counts: Counter, n_items: int):
        self.scores = torch.zeros(n_items + 1)
        for item, count in counts.items():
            self.scores[item] = math.log1p(count)

    def score_candidates(self, users: torch.Tensor, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        self.scores = self.scores.to(candidates.device)
        return self.scores[candidates]


class RandomBaseline:
    def score_candidates(self, users: torch.Tensor, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.rand(candidates.shape, device=candidates.device)


def run_one(
    name: str,
    seed: int,
    train: Sequence[Sample],
    valid: Sequence[Sample],
    test: Sequence[Sample],
    n_users: int,
    n_items: int,
    edges: Sequence[Tuple[int, int]],
    user_pos: DefaultDict[int, Set[int]],
    counts: Counter,
    negative_pool: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    set_seed(seed)
    start = time.time()
    if name == 'random':
        model = RandomBaseline()
        metrics = evaluate(model, test, device)
    elif name == 'popularity':
        model = PopularityBaseline(counts, n_items)
        metrics = evaluate(model, test, device)
    elif name in {'bprmf', 'lightgcn'}:
        if name == 'bprmf':
            model = BPRMF(n_users, n_items, dim=args.dim).to(device)
        else:
            model = LightGCN(n_users, n_items, dim=args.dim, layers=args.lightgcn_layers).to(device)
            model.set_graph(edges)
        metrics, extra = train_bpr(
            model=model,
            train=train,
            valid=valid,
            test=test,
            edges=edges,
            user_pos=user_pos,
            negative_pool=negative_pool,
            device=device,
            epochs=args.graph_epochs,
            batch_size=args.graph_batch_size,
            lr=args.graph_lr,
            weight_decay=args.graph_weight_decay,
        )
        metrics.update(extra)
    elif name == 'sasrec':
        model = SASRec(
            n_items=n_items + 1,
            d_model=args.dim,
            max_len=11,
            layers=args.seq_layers,
            heads=args.seq_heads,
            dropout=args.seq_dropout,
        ).to(device)
        metrics, extra = train_sasrec(
            model=model,
            train=train,
            valid=valid,
            test=test,
            n_items=n_items,
            device=device,
            epochs=args.seq_epochs,
            batch_size=args.seq_batch_size,
            lr=args.seq_lr,
            weight_decay=args.seq_weight_decay,
        )
        metrics.update(extra)
    elif name == 'bert4rec':
        model = BERT4Rec(
            n_items=n_items + 1,
            d_model=args.dim,
            max_len=11,
            layers=args.seq_layers,
            heads=args.seq_heads,
            dropout=args.seq_dropout,
        ).to(device)
        metrics, extra = train_bert4rec(
            model=model,
            train=train,
            valid=valid,
            test=test,
            n_items=n_items,
            device=device,
            epochs=args.seq_epochs,
            batch_size=args.seq_batch_size,
            lr=args.seq_lr,
            weight_decay=args.seq_weight_decay,
        )
        metrics.update(extra)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    metrics['seed'] = float(seed)
    metrics['seconds'] = time.time() - start
    return metrics


def summarize(all_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    keys = ('top1_acc', 'top3_acc', 'top5_acc', 'ndcg5', 'mrr', 'seconds')
    for name, rows in all_results.items():
        summary[name] = {}
        for key in keys:
            values = [row[key] for row in rows]
            summary[name][f'{key}_mean'] = statistics.mean(values)
            summary[name][f'{key}_std'] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
    headers = ['Model', 'Top-1', 'Top-3', 'Top-5', 'NDCG@5', 'MRR', 'Seconds']
    print('\n' + ' | '.join(headers))
    print(' | '.join(['---'] * len(headers)))
    for name in sorted(summary):
        row = summary[name]
        print(
            ' | '.join(
                [
                    name,
                    f"{row['top1_acc_mean']:.4f}+/-{row['top1_acc_std']:.4f}",
                    f"{row['top3_acc_mean']:.4f}+/-{row['top3_acc_std']:.4f}",
                    f"{row['top5_acc_mean']:.4f}+/-{row['top5_acc_std']:.4f}",
                    f"{row['ndcg5_mean']:.4f}+/-{row['ndcg5_std']:.4f}",
                    f"{row['mrr_mean']:.4f}+/-{row['mrr_std']:.4f}",
                    f"{row['seconds_mean']:.1f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Non-LLM recommendation baselines')
    parser.add_argument('--data', type=str, default=os.path.join('data', 'ml1m_train.json'))
    parser.add_argument('--tag', type=str, default=None, help='Output file prefix; defaults to the data filename')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['random', 'popularity', 'bprmf', 'lightgcn', 'sasrec', 'bert4rec'],
        choices=['random', 'popularity', 'bprmf', 'lightgcn', 'sasrec', 'bert4rec'],
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--graph-epochs', type=int, default=60)
    parser.add_argument('--graph-batch-size', type=int, default=2048)
    parser.add_argument('--graph-lr', type=float, default=1e-3)
    parser.add_argument('--graph-weight-decay', type=float, default=1e-5)
    parser.add_argument('--lightgcn-layers', type=int, default=2)
    parser.add_argument('--seq-epochs', type=int, default=40)
    parser.add_argument('--seq-batch-size', type=int, default=256)
    parser.add_argument('--seq-lr', type=float, default=1e-3)
    parser.add_argument('--seq-weight-decay', type=float, default=1e-4)
    parser.add_argument('--seq-layers', type=int, default=2)
    parser.add_argument('--seq-heads', type=int, default=4)
    parser.add_argument('--seq-dropout', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default=os.path.join('baselines', 'ml1m', 'results'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    samples, stats = load_and_encode(args.data)
    train, valid, test = split_samples(samples, args.train_ratio, args.val_ratio)
    train_edges, _ = build_user_positives(train, include_target=True)
    edges, user_pos = build_graph_data(train, valid, test)
    counts = item_popularity(train_edges)
    negative_pool = torch.tensor(sorted({item for _, item in edges}), dtype=torch.long)
    n_users = max(sample.user for sample in samples) + 1
    n_items = max(max(sample.history + sample.candidates) for sample in samples)
    print('Dataset stats: ' + json.dumps(stats, ensure_ascii=False), flush=True)
    print(
        f"Split: train={len(train)} valid={len(valid)} test={len(test)} "
        f"users={n_users} items={n_items} "
        f"train_edges={len(train_edges)} graph_edges={len(edges)} "
        f"negative_pool={negative_pool.numel()}",
        flush=True,
    )
    all_results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for seed in args.seeds:
        for name in args.models:
            print(f"\nRunning {name} seed={seed}", flush=True)
            metrics = run_one(
                name=name,
                seed=seed,
                train=train,
                valid=valid,
                test=test,
                n_users=n_users,
                n_items=n_items,
                edges=edges,
                user_pos=user_pos,
                counts=counts,
                negative_pool=negative_pool,
                device=device,
                args=args,
            )
            all_results[name].append(metrics)
            print('  ' + json.dumps(metrics, ensure_ascii=False), flush=True)
    summary = summarize(all_results)
    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or os.path.splitext(os.path.basename(args.data))[0].removesuffix('_train')
    output = {
        'protocol': {
            'data': args.data,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'candidate_size': 20,
            'history_size': 10,
            'seeds': args.seeds,
            'stats': stats,
            'graph_protocol': 'train targets + all input histories; test targets excluded',
            'graph_edges': len(edges),
            'negative_pool': negative_pool.numel(),
        },
        'runs': all_results,
        'summary': summary,
    }
    json_path = os.path.join(args.output_dir, f'{tag}_baselines.json')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    csv_path = os.path.join(args.output_dir, f'{tag}_baselines.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'top1_acc_mean', 'top1_acc_std', 'top3_acc_mean', 'top3_acc_std', 'top5_acc_mean', 'top5_acc_std', 'ndcg5_mean', 'ndcg5_std', 'mrr_mean', 'mrr_std', 'seconds_mean'])
        for name in sorted(summary):
            row = summary[name]
            writer.writerow([
                name,
                row['top1_acc_mean'], row['top1_acc_std'],
                row['top3_acc_mean'], row['top3_acc_std'],
                row['top5_acc_mean'], row['top5_acc_std'],
                row['ndcg5_mean'], row['ndcg5_std'],
                row['mrr_mean'], row['mrr_std'], row['seconds_mean'],
            ])
    print_table(summary)
    print(f"\nSaved {json_path} and {csv_path}", flush=True)


class SequenceEncoder(nn.Module):
    causal: bool = False
    mask_token: bool = False

    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        max_len: int = 11,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        vocab_size = n_items + (2 if self.mask_token else 1)
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_ID].zero_()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0)
        hidden = self.item_emb(sequences) + self.pos_emb(positions)
        hidden = self.dropout(hidden)
        padding_mask = sequences.eq(PAD_ID)
        mask = None
        if self.causal:
            length = sequences.size(1)
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=sequences.device), diagonal=1)
        hidden = self.encoder(hidden, mask=mask, src_key_padding_mask=padding_mask)
        return self.norm(hidden)

    def item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_emb.weight[1 : self.n_items + 1].T

    def score_candidates(self, users: torch.Tensor, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(histories)[:, -1]
        item_vectors = self.item_emb.weight[candidates]
        return (hidden.unsqueeze(1) * item_vectors).sum(dim=-1)


class SASRec(SequenceEncoder):
    causal = True


class BERT4Rec(SequenceEncoder):
    causal = False
    mask_token = True


def make_train_sequences(train: Sequence[Sample], max_history: int) -> torch.Tensor:
    rows = []
    for sample in train:
        history = sample.history[-max_history:]
        sequence = [PAD_ID] * (max_history - len(history)) + history + [sample.candidates[sample.label]]
        rows.append(sequence)
    return torch.tensor(rows, dtype=torch.long)


def train_sasrec(
    model: SASRec,
    train: Sequence[Sample],
    valid: Sequence[Sample],
    test: Sequence[Sample],
    n_items: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sequences = make_train_sequences(train, model.max_len - 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate(model, valid, device)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(sequences.size(0))
        total_loss = 0.0
        batches = 0
        for start in range(0, sequences.size(0), batch_size):
            batch = sequences[perm[start : start + batch_size]]
            inputs = batch[:, :-1]
            labels = batch[:, 1:]
            hidden = model.encode(inputs)
            logits = model.item_logits(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        metrics = evaluate(model, valid, device)
        if metrics['top1_acc'] > best_metrics['top1_acc']:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:03d} loss={total_loss / max(1, batches):.4f} "
                f"top1={metrics['top1_acc']:.4f} ndcg5={metrics['ndcg5']:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return evaluate(model, test, device), {
        'best_epoch': float(best_epoch),
        'best_val_top1': float(best_metrics['top1_acc']),
    }


def make_masked_batch(sequences: torch.Tensor, n_items: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_id = n_items + 1
    inputs = sequences.clone()
    labels = torch.full_like(sequences, -100)
    for row in range(sequences.size(0)):
        valid = (sequences[row] != PAD_ID).nonzero(as_tuple=False).view(-1)
        if valid.numel() == 0:
            continue
        mask_count = max(1, int(round(0.15 * valid.numel())))
        selected = valid[torch.randperm(valid.numel(), device=sequences.device)[:mask_count]]
        labels[row, selected] = sequences[row, selected]
        random_values = torch.rand(selected.numel(), device=sequences.device)
        mask_positions = selected[random_values < 0.8]
        random_positions = selected[(random_values >= 0.8) & (random_values < 0.9)]
        inputs[row, mask_positions] = mask_id
        inputs[row, random_positions] = torch.randint(1, n_items + 1, (random_positions.numel(),), device=sequences.device)
    return inputs, labels


def train_bert4rec(
    model: BERT4Rec,
    train: Sequence[Sample],
    valid: Sequence[Sample],
    test: Sequence[Sample],
    n_items: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sequences = make_train_sequences(train, model.max_len - 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate(model, valid, device)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(sequences.size(0))
        total_loss = 0.0
        batches = 0
        for start in range(0, sequences.size(0), batch_size):
            batch = sequences[perm[start : start + batch_size]]
            inputs, labels = make_masked_batch(batch, n_items)
            hidden = model.encode(inputs)
            logits = model.item_logits(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        metrics = evaluate(model, valid, device)
        if metrics['top1_acc'] > best_metrics['top1_acc']:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:03d} loss={total_loss / max(1, batches):.4f} "
                f"top1={metrics['top1_acc']:.4f} ndcg5={metrics['ndcg5']:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return evaluate(model, test, device), {
        'best_epoch': float(best_epoch),
        'best_val_top1': float(best_metrics['top1_acc']),
    }


if __name__ == '__main__':
    main()
