from __future__ import annotations
from typing import Tuple, List
import cv2


def draw_bbox(image, bbox: Tuple[int, int, int, int], color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text: str, org: Tuple[int, int], color=(255, 255, 255)):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overlay_tracks(image, tracks: List[Tuple[int, int, int, int, int, float]], fps: float, active_count: int, total_count: int):
    for (tid, x1, y1, x2, y2, score) in tracks:
        draw_bbox(image, (x1, y1, x2, y2), (0, 200, 0), 2)
        draw_text(image, f"ID {tid} {score:.2f}", (x1, max(0, y1 - 5)))
    draw_text(image, f"Active: {active_count}", (10, 20))
    draw_text(image, f"Total: {total_count}", (10, 40))
    draw_text(image, f"FPS: {fps:.1f}", (10, 60))

