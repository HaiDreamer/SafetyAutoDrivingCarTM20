# q_logger.py
import csv
import os
from datetime import datetime
from pathlib import Path

LOG_PATH = "tmrl_q_log.csv"
_initialized = False

def log_metrics(metrics: dict, step: int):
    global _initialized
    if not _initialized:
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "time"] + list(metrics.keys()))
            writer.writeheader()
        _initialized = True

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "time"] + list(metrics.keys()))
        row = {"step": step, "time": datetime.now().strftime("%H:%M:%S")}
        row.update(metrics)
        writer.writerow(row)