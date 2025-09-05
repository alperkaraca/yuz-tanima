from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


class HeatmapAccumulator:
    """Yüz merkezlerinden birikimli yoğunluk haritası üretir."""

    def __init__(self, height: int, width: int, sigma: float = 8.0) -> None:
        self.height = int(height)
        self.width = int(width)
        self.map = np.zeros((self.height, self.width), dtype=np.float32)
        self.sigma = float(sigma)

    def add_bbox(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self.map[cy, cx] += 1.0

    def render(self, normalize: bool = True) -> np.ndarray:
        hm = gaussian_filter(self.map, sigma=self.sigma)
        if normalize and hm.max() > 0:
            hm = hm / hm.max()
        return hm

    def to_color(self) -> np.ndarray:
        hm = self.render(normalize=True)
        hm_uint8 = (hm * 255).astype(np.uint8)
        import cv2

        color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        return color

