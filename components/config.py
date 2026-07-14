from string import ascii_uppercase
import os
import time
from typing import Dict, List
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_INTERACTION_FEATURE_NAMES = [
    "gsim",
    "lmax",
    "first_norm",
    "user_avg_feat",
    "user_cnt_feat",
    "item_avg_feat",
    "item_cnt_feat",
    "user_rated_flag",
    "co_feat",
    "trans_feat",
    "genre_feat",
    "rank_feat",
    "co_trans_mul",
    "co_trans_absdiff",
]

SAFE_INTERACTION_FEATURE_WHITELIST = [
    "gsim",
    "lmax",
    "first_norm",
    "user_avg_feat",
    "user_cnt_feat",
    "item_avg_feat",
    "item_cnt_feat",
    "co_feat",
    "genre_feat",
    "rank_feat",
]

class LocalTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = time.localtime(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s

formatter = LocalTimeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("gcn_recommend")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logging.getLogger("neo4j").setLevel(logging.ERROR)


def setup_file_logging(log_dir: str, run_tag: str, model_tag: str) -> Dict[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    run_log_path = os.path.join(log_dir, f"{model_tag}_{run_tag}.log")
    metrics_log_path = os.path.join(log_dir, f"{model_tag}_{run_tag}_metrics.jsonl")

    existing_paths = {
        getattr(handler, "baseFilename", None)
        for handler in logger.handlers
        if hasattr(handler, "baseFilename")
    }
    if os.path.abspath(run_log_path) not in {os.path.abspath(path) for path in existing_paths if path}:
        fh = RotatingFileHandler(
            run_log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("File logging enabled. run_log=%s metrics_log=%s", run_log_path, metrics_log_path)
    return {
        "run_log_path": run_log_path,
        "metrics_log_path": metrics_log_path,
    }


def build_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
