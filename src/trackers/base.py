from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import itertools
import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    score: float


class BaseTracker:
    name: str = "base"

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Track]:
        raise NotImplementedError


class SimpleSORT(BaseTracker):
    """Basit IoU tabanlı, sabit ömürlü takip (fallback).
    Not: Bu, gerçek SORT/OC-SORT değildir fakat arayüz uyumlu ve hafiftir.
    """

    name = "simple_sort"

    def __init__(self, max_age: int = 10, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Track]:
        # Yaşlandır
        for t in self._tracks.values():
            t["age"] += 1

        # Eşleme: greedy IoU
        unmatched_trk_ids = list(self._tracks.keys())
        assigned_trk = set()
        outputs: List[Track] = []

        for (x1, y1, x2, y2, s) in detections:
            best_iou = 0.0
            best_id = None
            for tid in unmatched_trk_ids:
                if tid in assigned_trk:
                    continue
                iou = self._iou(self._tracks[tid]["bbox"], (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is not None and best_iou >= self.iou_threshold:
                self._tracks[best_id]["bbox"] = (x1, y1, x2, y2)
                self._tracks[best_id]["age"] = 0
                assigned_trk.add(best_id)
                outputs.append(Track(best_id, (x1, y1, x2, y2), s))
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": (x1, y1, x2, y2), "age": 0}
                assigned_trk.add(tid)
                outputs.append(Track(tid, (x1, y1, x2, y2), s))

        # Eski izleri sil
        to_del = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
        for tid in to_del:
            del self._tracks[tid]

        # Takipte kalan ama bu karede görülmeyenler döndürülmez
        return outputs

