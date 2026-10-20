#!/usr/bin/env python3
"""Build the cross-dataset Easy versus Hard10 summary table."""

import csv
import json
import os

PAIRS = [
    ("ML-1M", "baselines/ml1m/results_80_10_10/ml1m_baselines.json",
     "baselines/ml1m/results_hard10/ml1m_hard10_baselines.json"),
    ("ML-20M", "baselines/ml20m/results/ml20m_baselines.json",
     "baselines/ml20m/results_hard10/ml20m_hard10_baselines.json"),
    ("Amazon-Book", "baselines/amazonbook/results/amazonbook_baselines.json",
     "baselines/amazonbook/results_hard10/amazonbook_hard10_baselines.json"),
]
MODELS = ["popularity", "lightgcn", "sasrec", "bert4rec"]
METRICS = ["top1_acc", "top3_acc", "top5_acc", "ndcg5", "mrr"]
OUT_DIR = "baselines/results_hard_negative"


def main():
    rows = []
    for dataset, easy_path, hard_path in PAIRS:
        easy = json.load(open(easy_path, encoding="utf-8"))["summary"]
        hard = json.load(open(hard_path, encoding="utf-8"))["summary"]
        for model in MODELS:
            row = {"dataset": dataset, "model": model}
            for metric in METRICS:
                easy_value = easy[model][f"{metric}_mean"]
                hard_value = hard[model][f"{metric}_mean"]
                row[f"easy_{metric}"] = easy_value
                row[f"hard_{metric}"] = hard_value
                row[f"delta_{metric}"] = hard_value - easy_value
                row[f"rel_{metric}"] = (hard_value - easy_value) / easy_value
            rows.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Easy vs Hard10 结果汇总",
        "",
        "协议：1 正 + 19 原始均匀负例（Easy） vs 1 正 + 10 内容难负例 + 9 原始均匀负例（Hard10）。",
        "训练集与验证集完全不变，仅测试候选集变化。",
        "",
    ]
    for dataset, easy_path, hard_path in PAIRS:
        subset = [row for row in rows if row["dataset"] == dataset]
        lines += [
            f"## {dataset}",
            "",
            "| 模型 | Easy Top-1 | Hard10 Top-1 | Δ Top-1 | 相对变化 | Easy NDCG@5 | Hard10 NDCG@5 | Δ NDCG@5 | Easy MRR | Hard10 MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in subset:
            lines.append(
                "| {model} | {e:.4f} | {h:.4f} | {d:+.4f} | {r:+.2f}% | {en:.4f} | {hn:.4f} | {dn:+.4f} | {em:.4f} | {hm:.4f} |".format(
                    model=row["model"], e=row["easy_top1_acc"], h=row["hard_top1_acc"],
                    d=row["delta_top1_acc"], r=100 * row["rel_top1_acc"],
                    en=row["easy_ndcg5"], hn=row["hard_ndcg5"], dn=row["delta_ndcg5"],
                    em=row["easy_mrr"], hm=row["hard_mrr"],
                )
            )
        lines.append("")
    lines += ["## 结果来源", ""]
    for dataset, easy_path, hard_path in PAIRS:
        lines.append(f"- {dataset}: Easy = `{easy_path}`; Hard10 = `{hard_path}`")
    lines.append("")
    open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()