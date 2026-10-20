import argparse
import hashlib
import json
import os
import re
import sys

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPTION_RE = re.compile(r"^([A-T])\. ", re.MULTILINE)


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message):
    print("DATA_CHECK_FAILED: " + message, flush=True)
    sys.exit(2)


def check_rows(name, rows, expected):
    if len(rows) != expected:
        fail("{} has {} rows, expected {}".format(name, len(rows), expected))
    target_counts = {}
    for index, row in enumerate(rows):
        for key in ("input", "questions", "output", "sequence_ids", "target_id", "target_idx"):
            if key not in row:
                fail("{} row {} is missing {}".format(name, index, key))
        target_idx = int(row["target_idx"])
        if target_idx < 0 or target_idx >= 20:
            fail("{} row {} target_idx={} out of range".format(name, index, target_idx))
        if str(row["output"]) != LETTERS[target_idx]:
            fail(
                "{} row {} output={} does not match target_idx={}".format(
                    name, index, row["output"], target_idx
                )
            )
        options = OPTION_RE.findall(str(row["questions"]))
        if options != list(LETTERS[:20]):
            fail(
                "{} row {} option letters are not A-T exactly once: {}".format(
                    name, index, options
                )
            )
        sequence = row["sequence_ids"]
        if isinstance(sequence, str):
            sequence = json.loads(sequence)
        if not isinstance(sequence, list) or not sequence:
            fail("{} row {} has an empty sequence".format(name, index))
        target_counts[target_idx] = target_counts.get(target_idx, 0) + 1
    if len(target_counts) < 5:
        fail("{} target positions collapsed to {}".format(name, target_counts))
    return target_counts


def main():
    parser = argparse.ArgumentParser("Validate K-RagRec-Qwen2 evaluation data")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stats", type=str, default="")
    parser.add_argument("--expected_size", type=int, default=906)
    args = parser.parse_args()

    stats_path = args.stats or os.path.join(args.data_dir, "kragrec_v2_stats.json")
    stats = load(stats_path)
    for key in (
        "same_test_as_9010",
        "hard10_aligned_by_sample_idx",
        "train_valid_prefix_unchanged",
    ):
        if stats.get(key) is not True:
            fail("stats.{} is not true".format(key))
    if stats.get("test_range") != [8154, 9060]:
        fail("stats.test_range={} is not [8154, 9060]".format(stats.get("test_range")))
    if stats.get("test_size_9010") != args.expected_size:
        fail("stats.test_size_9010={} is not {}".format(stats.get("test_size_9010"), args.expected_size))

    valid = load(os.path.join(args.data_dir, "kragrec_v2_valid.json"))
    easy = load(os.path.join(args.data_dir, "kragrec_v2_easy_test.json"))
    hard10 = load(os.path.join(args.data_dir, "kragrec_v2_hard10_test.json"))

    valid_targets = check_rows("valid", valid, args.expected_size)
    easy_targets = check_rows("easy_test", easy, args.expected_size)
    hard_targets = check_rows("hard10_test", hard10, args.expected_size)

    easy_ids = [str(row["sample_idx"]) for row in easy]
    hard_ids = [str(row["sample_idx"]) for row in hard10]
    if easy_ids != hard_ids:
        fail("easy_test and hard10_test sample_idx order differs")

    test_hash = hashlib.sha256("\n".join(easy_ids).encode("utf-8")).hexdigest()
    if test_hash != stats.get("test_sample_idx_sha256"):
        fail(
            "test sample_idx hash mismatch: {} vs {}".format(
                test_hash, stats.get("test_sample_idx_sha256")
            )
        )

    summary = {
        "valid": len(valid),
        "easy_test": len(easy),
        "hard10_test": len(hard10),
        "test_range": stats.get("test_range"),
        "same_test_as_9010": stats.get("same_test_as_9010"),
        "test_sample_idx_sha256": test_hash,
        "valid_target_positions": len(valid_targets),
        "easy_target_positions": len(easy_targets),
        "hard10_target_positions": len(hard_targets),
    }
    print("DATA_CHECK_OK", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()