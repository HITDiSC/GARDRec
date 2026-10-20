import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def load_ratings_stats(ratings_path: Optional[str], allowed_pairs: Optional[set] = None):
    if not ratings_path or not os.path.exists(ratings_path):
        return {}, {}, set(), 0, 0

    user_sum = {}
    user_count = {}
    user_last = {}
    item_sum = {}
    item_count = {}
    user_item = set()

    with open(ratings_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '::' in line:
                parts = line.split('::')
            else:
                parts = line.split()
            if len(parts) < 3:
                continue
            try:
                user = str(int(parts[0]))
                item = str(int(parts[1]))
                rating = float(parts[2])
                ts = int(parts[3]) if len(parts) > 3 else 0
            except Exception:
                continue
            if allowed_pairs is not None and (user, item) not in allowed_pairs:
                continue
            user_sum[user] = user_sum.get(user, 0.0) + rating
            user_count[user] = user_count.get(user, 0) + 1
            user_last[user] = max(user_last.get(user, 0), ts)
            item_sum[item] = item_sum.get(item, 0.0) + rating
            item_count[item] = item_count.get(item, 0) + 1
            user_item.add((user, item))

    user_stats = {
        u: {'avg': s / float(user_count[u]), 'count': user_count[u], 'last_ts': user_last.get(u, 0)}
        for u, s in user_sum.items()
    }
    item_stats = {
        it: {'avg': s / float(item_count[it]), 'count': item_count[it]}
        for it, s in item_sum.items()
    }
    max_user_count = max(user_count.values()) if user_count else 0
    max_item_count = max(item_count.values()) if item_count else 0
    max_ts = max(user_last.values()) if user_last else 0
    return user_stats, item_stats, user_item, max_user_count, max_item_count, int(max_ts)
