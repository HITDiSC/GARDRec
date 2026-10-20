#!/usr/bin/env python3
"""Summarize paired Easy versus hard-negative baseline results."""

import argparse
import csv
import json
import os


METRICS = ("top1_acc", "top3_acc", "top5_acc", "ndcg5", "mrr")


def load_summary(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["summary"]


def mean(summary, model, metric):
    return float(summary.get(model, {}).get(f"{metric}_mean", 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy", required=True, help="Easy-protocol baseline JSON")
    parser.add_argument("--hard", required=True, help="Hard-protocol baseline JSON")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    easy = load_summary(args.easy)
    hard = load_summary(args.hard)
    models = args.models or sorted(set(easy) & set(hard))

    rows = []
    for model in models:
        if model not in easy or model not in hard:
            raise KeyError(f"Model {model!r} is missing from one result file")
        row = {"model": model}
        for metric in METRICS:
            easy_value = mean(easy, model, metric)
            hard_value = mean(hard, model, metric)
            row[f"easy_{metric}"] = easy_value
            row[f"hard_{metric}"] = hard_value
            row[f"delta_{metric}"] = hard_value - easy_value
            row[f"relative_{metric}"] = (
                (hard_value - easy_value) / easy_value if easy_value else 0.0
            )
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("| Model | Easy Top-1 | Hard Top-1 | Delta | Relative |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['easy_top1_acc']:.4f} | "
            f"{row['hard_top1_acc']:.4f} | {row['delta_top1_acc']:+.4f} | "
            f"{100.0 * row['relative_top1_acc']:+.2f}% |"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
