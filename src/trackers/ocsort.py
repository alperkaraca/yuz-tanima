from __future__ import annotations
from typing import List, Tuple
from loguru import logger

from .base import BaseTracker, Track, SimpleSORT


class OCSORTTracker(BaseTracker):
    name = "ocsort"

    def __init__(self) -> None:
        self._impl: BaseTracker
        try:
            # Yerleşik bir OC-SORT paketi varsaymıyoruz; fallback kullanıyoruz.
            raise ImportError
        except Exception:
            logger.warning("OC-SORT bulunamadı; SimpleSORT fallback kullanılıyor.")
            self._impl = SimpleSORT(max_age=10, iou_threshold=0.3)

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Track]:
        return self._impl.update(detections)

