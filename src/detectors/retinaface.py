from __future__ import annotations
from typing import List
from loguru import logger
import cv2
import numpy as np

from .base import BaseFaceDetector, Detection


class RetinaFaceDetector(BaseFaceDetector):
    """RetinaFace adapteri.
    Not: Bağımlılık/weights yoksa, OpenCV Haar Cascade'e otomatik düşer.
    """

    name = "retinaface"

    def __init__(self, score_threshold: float = 0.5, min_face: int = 24) -> None:
        self.score_threshold = float(score_threshold)
        self.min_face = int(min_face)
        self._impl = None
        self._fallback = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        try:
            import retinaface  # type: ignore

            self._impl = retinaface
            logger.info("RetinaFace bulundu, onu kullanacağız.")
        except Exception:
            logger.warning("RetinaFace paketi yok; Haar Cascade fallback kullanılacak.")

    def detect(self, image) -> List[Detection]:
        h, w = image.shape[:2]
        dets: List[Detection] = []
        if self._impl is not None:
            # Bu kısım sadece paket kuruluysa çalışır. Basit skor eşiği uygula.
            try:
                faces = self._impl.RetinaFace.detect_faces(image)
                if isinstance(faces, dict):
                    for _, f in faces.items():
                        (x1, y1), (x2, y2) = f.get("facial_area", ((0, 0), (0, 0)))
                        score = float(f.get("score", 1.0))
                        if score >= self.score_threshold:
                            x1 = int(np.clip(x1, 0, w - 1))
                            y1 = int(np.clip(y1, 0, h - 1))
                            x2 = int(np.clip(x2, 0, w - 1))
                            y2 = int(np.clip(y2, 0, h - 1))
                            dets.append(Detection((x1, y1, x2, y2), score))
                return dets
            except Exception as e:
                logger.error(f"RetinaFace çalıştırılamadı, fallback'e dönüyoruz: {e}")

        # Fallback: Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._fallback.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face, self.min_face))
        for (x, y, w0, h0) in faces:
            x1, y1, x2, y2 = x, y, x + w0, y + h0
            dets.append(Detection((x1, y1, x2, y2), 0.99))
        return dets

