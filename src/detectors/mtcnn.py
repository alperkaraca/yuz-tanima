from __future__ import annotations
from typing import List
from loguru import logger
import cv2

from .base import BaseFaceDetector, Detection


class MTCNNDetector(BaseFaceDetector):
    """MTCNN adapteri; paket yoksa Haar Cascade'e düşer."""

    name = "mtcnn"

    def __init__(self, score_threshold: float = 0.5, min_face: int = 24) -> None:
        self.score_threshold = float(score_threshold)
        self.min_face = int(min_face)
        self._impl = None
        self._fallback = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        try:
            from mtcnn import MTCNN  # type: ignore

            self._impl = MTCNN()
            logger.info("MTCNN bulundu, onu kullanacağız.")
        except Exception:
            logger.warning("MTCNN paketi yok; Haar Cascade fallback kullanılacak.")

    def detect(self, image) -> List[Detection]:
        dets: List[Detection] = []
        if self._impl is not None:
            try:
                res = self._impl.detect_faces(image)
                for r in res:
                    x, y, w, h = r.get("box", [0, 0, 0, 0])
                    score = float(r.get("confidence", 1.0))
                    if score >= self.score_threshold and w > 0 and h > 0:
                        dets.append(Detection((x, y, x + w, y + h), score))
                return dets
            except Exception as e:
                logger.error(f"MTCNN çalıştırılamadı, fallback'e dönüyoruz: {e}")

        # Fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._fallback.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face, self.min_face))
        for (x, y, w0, h0) in faces:
            dets.append(Detection((x, y, x + w0, y + h0), 0.99))
        return dets

