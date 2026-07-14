import logging
import os
import subprocess
from typing import Optional

LOGGER = logging.getLogger("notify_local_sender")
if not LOGGER.handlers:
    handler = logging.FileHandler("notify_local_sender.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

DEFAULT_NOTIFY_URL = os.getenv("LOCAL_NOTIFY_URL", "http://127.0.0.1:8765")
DEFAULT_TITLE = os.getenv("LOCAL_NOTIFY_TITLE", "Remote Task Notification")

def send_notification(message: str, title: Optional[str] = None, url: Optional[str] = None) -> bool:
    return True

def send_email_notification(message: str, title: str) -> bool:
    return True
