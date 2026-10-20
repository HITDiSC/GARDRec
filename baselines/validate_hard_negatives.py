import json, sys

SPECS = {
    "ml1m": ("data/ml1m_train.json", "baselines/ml1m/hard10_train.json"),
    "ml20m": ("data/ml20m_train.json", "baselines/ml20m/hard10_train.json"),
    "amazonbook": ("data/amazonbook_train.json", "baselines/amazonbook/hard10_train.json"),
}
FIELDS = {
    "ml1m": ("history_movies", "candidate_movies", "target_movie"),
    "ml20m": ("history_movies", "candidate_movies", "target_movie"),
    "amazonbook": ("history_books", "candidate_books", "target_book"),
}

def oid(x):
    for k in ("original_id", "id", "item_id"):
        if k in x: return str(x[k])
    raise KeyError(x)

ok_all = True
for ds, (orig_p, hard_p) in SPECS.items():
    raw = json.load(open(orig_p, encoding="utf-8"))
    hard = json.load(open(hard_p, encoding="utf-8"))
    hf, cf, tf = FIELDS[ds]
    entries, dropped = [], 0
    for row in raw:
        cands = [oid(x) for x in row[cf]]
        if len(cands) != 20 or cands.count(oid(row[tf])) != 1:
            dropped += 1; continue
        entries.append(row)
    n = len(entries)
    test_size = int(round(n * 0.1)); val_size = int(round(n * 0.1))
    train_size = n - test_size - val_size
    identical = all(entries[i].get(f) == hard[i].get(f)
                    for i in range(train_size + val_size) for f in (hf, cf, tf))
    bad = mismatch = overlap = leak_new = 0
    for i in range(train_size + val_size, n):
        row = hard[i]; orig = entries[i]
        cands = [oid(x) for x in row[cf]]; target = oid(row[tf])
        if len(cands) != 20 or len(set(cands)) != 20 or cands.count(target) != 1:
            bad += 1; continue
        hn = [str(x) for x in row["hard_negative_ids"]]
        en = [str(x) for x in row["easy_negative_ids"]]
        hist = {oid(x) for x in row[hf]}
        if len(hn) != 10 or len(en) != 9 or set(hn) | set(en) != set(cands) - {target}:
            mismatch += 1
        if set(hn) & set(en): overlap += 1
        orig_overlap = {oid(x) for x in orig[cf]} & hist
        new_overlap = (set(hn) | set(en)) & hist - orig_overlap
        if new_overlap or target in hn or target in en:
            leak_new += 1
    print(f"{ds}: rows={n} (dropped {dropped}) train={train_size} valid={val_size} test={test_size} | "
          f"train+valid identical={'yes' if identical else 'NO'} | "
          f"bad_candidates={bad} partition_mismatch={mismatch} hard_easy_overlap={overlap} "
          f"builder_introduced_history_leak={leak_new}")
    if not identical or bad or mismatch or overlap or leak_new: ok_all = False
sys.exit(0 if ok_all else 1)