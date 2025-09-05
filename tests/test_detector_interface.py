from __future__ import annotations
import numpy as np
import cv2

from src.detectors.retinaface import RetinaFaceDetector


def test_retinaface_interface():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    det = RetinaFaceDetector(score_threshold=0.1, min_face=10)
    res = det.detect(img)
    # Çalışmalı ve liste döndürmeli
    assert isinstance(res, list)
    # bbox/score tipleri uygun olmalı
    if res:
        d = res[0]
        assert hasattr(d, "to_xyxy")
        x1, y1, x2, y2 = d.to_xyxy()
        assert isinstance(x1, int)

