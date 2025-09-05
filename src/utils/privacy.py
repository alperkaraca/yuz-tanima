from __future__ import annotations
from typing import Tuple
import cv2


def gaussian_blur_face(image, bbox: Tuple[int, int, int, int], ksize: int = 15):
    """Yüz alanını Gauss bulanıklaştırma ile anonimize et.
    ksize tek olmalı ve pozitif olmalı.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1] - 1, x2)
    y2 = min(image.shape[0] - 1, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = max(3, ksize | 1)  # tek sayıya yuvarla
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    image[y1:y2, x1:x2] = blurred

