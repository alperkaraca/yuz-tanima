from __future__ import annotations
from typing import List, Tuple


class Detection:
    """Basit tespit yapısı: (x1, y1, x2, y2, score)."""

    def __init__(self, bbox: Tuple[int, int, int, int], score: float) -> None:
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.score = float(score)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class BaseFaceDetector:
    """Tüm yüz dedektörleri için arayüz."""

    name: str = "base"

    def detect(self, image) -> List[Detection]:  # image: np.ndarray (BGR)
        raise NotImplementedError

