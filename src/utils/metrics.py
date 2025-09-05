from __future__ import annotations
from typing import Dict, Any
import csv
import json


class MetricsLogger:
    def __init__(self, json_path: str | None, csv_path: str | None) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        if self.csv_path:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["timestamp", "frame_id", "active_count", "total_ids"])
            self._csv_writer.writeheader()
        self._json_records: list[Dict[str, Any]] = []

    def log(self, timestamp: float, frame_id: int, active_count: int, total_ids: int):
        row = {
            "timestamp": float(timestamp),
            "frame_id": int(frame_id),
            "active_count": int(active_count),
            "total_ids": int(total_ids),
        }
        if self._csv_writer:
            self._csv_writer.writerow(row)
        if self.json_path:
            self._json_records.append(row)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self.json_path:
            with open(self.json_path, "w") as f:
                json.dump(self._json_records, f, indent=2)

