from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JSONLMetricsLogger:
    """Very lightweight JSONL logger for production-style telemetry.

    Writes one JSON object per line.
    Thread-safe for the common case (multi-threaded load tests).
    """

    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        row: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
        }
        if payload:
            row.update(payload)

        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
