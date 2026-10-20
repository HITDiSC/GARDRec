import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import dump_json, load_json, to_kragrec_entry


def main():
    parser = argparse.ArgumentParser("Build K-RagRec-format ML-1M files")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--hard10", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/dataset/ML1M")
    parser.add_argument("--train_end", type=int, default=7248)
    parser.add_argument("--valid_end", type=int, default=8154)
    args = parser.parse_args()

    inputs = load_json(args.input)
    by_idx = {int(row["sample_idx"]): row for row in inputs if "sample_idx" in row}
    hard_rows = load_json(args.hard10)

    easy_all = []
    hard_all = []
    missing = 0
    for row in hard_rows:
        sid = int(row["sample_idx"])
        if sid not in by_idx:
            missing += 1
            continue
        easy_all.append(to_kragrec_entry(by_idx[sid]))
        hard_all.append(to_kragrec_entry(row))

    total = min(len(easy_all), len(hard_all))
    train_end = min(args.train_end, total)
    valid_end = min(args.valid_end, total)

    splits = {
        "kragrec_v2_train.json": easy_all[:train_end],
        "kragrec_v2_valid.json": easy_all[train_end:valid_end],
        "kragrec_v2_easy_test.json": easy_all[valid_end:total],
        "kragrec_v2_hard10_test.json": hard_all[valid_end:total],
        "kragrec_v2_all_easy.json": easy_all[:total],
        "kragrec_v2_all_hard10.json": hard_all[:total],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(args.out_dir, name)
        dump_json(path, rows)
        print(name, len(rows), path)

    test_size_9010 = int(round(total * 0.1))
    test_start_9010 = total - test_size_9010
    aligned = all(
        easy_all[index].get("sample_idx") == hard_all[index].get("sample_idx")
        for index in range(total)
    )
    prefix_unchanged = all(
        easy_all[index] == hard_all[index] for index in range(valid_end)
    )
    test_rows = easy_all[valid_end:total]
    test_hash = hashlib.sha256(
        "\n".join(str(row.get("sample_idx", "")) for row in test_rows).encode("utf-8")
    ).hexdigest()
    test_target_hash = hashlib.sha256(
        "\n".join(
            "{}:{}".format(row.get("sample_idx", ""), row.get("target_idx", ""))
            for row in test_rows
        ).encode("utf-8")
    ).hexdigest()
    stats = {
        "input": args.input,
        "hard10": args.hard10,
        "raw_input": len(inputs),
        "hard10_rows": len(hard_rows),
        "matched": total,
        "missing_from_input": missing,
        "train": len(splits["kragrec_v2_train.json"]),
        "valid": len(splits["kragrec_v2_valid.json"]),
        "easy_test": len(splits["kragrec_v2_easy_test.json"]),
        "hard10_test": len(splits["kragrec_v2_hard10_test.json"]),
        "output_dir": args.out_dir,
        "split_protocol": "80/10/10 in file order",
        "train_range": [0, train_end],
        "valid_range": [train_end, valid_end],
        "test_range": [valid_end, total],
        "test_size_9010": test_size_9010,
        "test_start_9010": test_start_9010,
        "same_test_as_9010": bool(valid_end == test_start_9010),
        "hard10_aligned_by_sample_idx": bool(aligned),
        "train_valid_prefix_unchanged": bool(prefix_unchanged),
        "test_sample_idx_sha256": test_hash,
        "test_target_sha256": test_target_hash,
    }
    dump_json(os.path.join(args.out_dir, "kragrec_v2_stats.json"), stats)
    with open(os.path.join(args.out_dir, "kragrec_v2_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()