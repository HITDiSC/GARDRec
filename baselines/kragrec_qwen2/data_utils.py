import json
import os

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _title(item):
    return str(item.get("title", "Unknown")).strip()


def history_text(history):
    parts = []
    for item in history:
        parts.append(f"{_title(item)} [{item.get('original_id')}]")
    return "; ".join(parts)


def options_text(candidates):
    lines = []
    for idx, item in enumerate(candidates):
        lines.append(f"{LETTERS[idx]}. {_title(item)}")
    return "\n".join(lines)


def to_kragrec_entry(entry, candidates=None):
    history = entry.get("history_movies", entry.get("history_books", []))
    cands = candidates if candidates is not None else entry.get(
        "candidate_movies", entry.get("candidate_books", [])
    )
    target = entry.get("target_movie", entry.get("target_book", {}))
    target_id = str(target.get("original_id"))
    candidate_ids = [str(x.get("original_id")) for x in cands]
    if target_id not in candidate_ids:
        raise ValueError(f"target {target_id} not in candidates")
    target_idx = candidate_ids.index(target_id)
    if target_idx >= len(LETTERS):
        raise ValueError("too many candidates for A-Z letters")
    return {
        "user_id": str(entry.get("user_id", "")),
        "sample_idx": entry.get("sample_idx", entry.get("sample_id")),
        "input": history_text(history),
        "questions": options_text(cands),
        "output": LETTERS[target_idx],
        "sequence_ids": json.dumps([str(x.get("original_id")) for x in history]),
        "target_id": target_id,
        "target_idx": target_idx,
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
def letter_to_index(letter):
    return LETTERS.index(str(letter).strip())


def build_prompt(entry):
    history = str(entry.get("input", "")).strip()
    options = str(entry.get("questions", "")).strip()
    return (
        "Given the user's watching history, select the movie that is most "
        "likely to interest the user from the options.\n"
        "Watching history: {}\n"
        "Options:\n{}\n"
        "Answer with the option letter only (A to T).\n"
        "Answer:"
    ).format(history, options)