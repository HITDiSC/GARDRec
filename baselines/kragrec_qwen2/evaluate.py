import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import parse_args
from data_utils import LETTERS, letter_to_index, load_json
from graph_llm import GraphLLM
from retrieve import GraphRetrieval
from train import build_batch, row_sequence, slice_rows


def main():
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_json(args.eval_file)
    rows = slice_rows(rows, args.eval_start, args.eval_end, args.limit_eval)
    if not rows:
        raise RuntimeError("No evaluation rows selected")

    print("Loading retrieval index from", args.index_dir, flush=True)
    retrieval = GraphRetrieval(args.index_dir)
    print("Loading graph LLM", flush=True)
    model = GraphLLM(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_trainable_state_dict(state)
    model.eval()
    print("loaded checkpoint:", args.checkpoint, flush=True)

    predictions = []
    started = time.time()
    total = len(rows)
    for start in tqdm(range(0, total, args.eval_batch_size), desc="evaluate"):
        batch_rows = rows[start:start + args.eval_batch_size]
        batch = build_batch(batch_rows, retrieval, args)
        scores = model.score_options(batch)
        predicted = scores.argmax(dim=-1).tolist()

        for offset, row in enumerate(batch_rows):
            target_idx = row.get("target_idx")
            if target_idx is None:
                target_idx = letter_to_index(row.get("output", "A"))
            target_idx = int(target_idx)
            row_scores = scores[offset]
            target_score = float(row_scores[target_idx])
            rank = int((row_scores > row_scores[target_idx]).sum().item()) + 1
            pred_idx = int(predicted[offset])
            predictions.append(
                {
                    "sample_idx": row.get("sample_idx"),
                    "user_id": row.get("user_id"),
                    "target_idx": target_idx,
                    "target_letter": LETTERS[target_idx],
                    "pred_idx": pred_idx,
                    "pred_letter": LETTERS[pred_idx],
                    "rank": rank,
                    "target_prob": target_score,
                    "scores": [round(float(value), 8) for value in row_scores.tolist()],
                }
            )

        done = min(start + args.eval_batch_size, total)
        if done % max(1, args.eval_batch_size * 50) == 0 or done == total:
            correct = sum(1 for item in predictions if item["rank"] == 1)
            print(
                "evaluated {}/{} acc {:.4f} elapsed {:.1f}s".format(
                    done, total, correct / max(1, len(predictions)), time.time() - started
                ),
                flush=True,
            )

    count = len(predictions)
    metrics = {
        "num_samples": count,
        "acc": sum(1 for item in predictions if item["rank"] == 1) / max(1, count),
        "recall@1": sum(1 for item in predictions if item["rank"] <= 1) / max(1, count),
        "recall@3": sum(1 for item in predictions if item["rank"] <= 3) / max(1, count),
        "recall@5": sum(1 for item in predictions if item["rank"] <= 5) / max(1, count),
        "recall@10": sum(1 for item in predictions if item["rank"] <= 10) / max(1, count),
        "mrr": sum(1.0 / item["rank"] for item in predictions) / max(1, count),
        "mean_target_prob": sum(item["target_prob"] for item in predictions) / max(1, count),
        "elapsed_sec": round(time.time() - started, 2),
    }

    eval_name = os.path.splitext(os.path.basename(args.eval_file))[0]
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    result_path = os.path.join(
        args.output_dir, "results", "{}_{}.json".format(args.run_name, eval_name)
    )
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": args.run_name,
                "eval_file": args.eval_file,
                "checkpoint": args.checkpoint,
                "metrics": metrics,
                "predictions": predictions,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print("result saved to", result_path, flush=True)


if __name__ == "__main__":
    main()