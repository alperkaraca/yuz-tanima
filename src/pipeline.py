from __future__ import annotations
from typing import Dict, Tuple, List
from dataclasses import dataclass
from loguru import logger
import time
import os
import yaml
import cv2

from .detectors.base import BaseFaceDetector, Detection
from .detectors.retinaface import RetinaFaceDetector
from .detectors.mtcnn import MTCNNDetector
from .trackers.base import BaseTracker, Track
from .trackers.ocsort import OCSORTTracker
from .trackers.deepsort import DeepSORTTracker
from .utils.video_io import open_video, get_props, open_writer
from .utils.fps import FPSMeter
from .utils.draw import overlay_tracks
from .utils.heatmap import HeatmapAccumulator
from .utils.privacy import gaussian_blur_face
from .utils.metrics import MetricsLogger


def build_detector(name: str, score_threshold: float, min_face: int) -> BaseFaceDetector:
    name = name.lower()
    if name == "retinaface":
        return RetinaFaceDetector(score_threshold=score_threshold, min_face=min_face)
    elif name == "mtcnn":
        return MTCNNDetector(score_threshold=score_threshold, min_face=min_face)
    else:
        logger.warning(f"Bilinmeyen dedektör '{name}', RetinaFace fallback kullanılacak.")
        return RetinaFaceDetector(score_threshold=score_threshold, min_face=min_face)


def build_tracker(name: str) -> BaseTracker:
    name = name.lower()
    if name == "ocsort":
        return OCSORTTracker()
    elif name == "deepsort":
        return DeepSORTTracker()
    else:
        logger.warning(f"Bilinmeyen takipçi '{name}', OC-SORT fallback kullanılacak.")
        return OCSORTTracker()


@dataclass
class PipelineConfig:
    source: str
    output_video: str | None
    output_metrics_json: str | None
    output_metrics_csv: str | None
    output_heatmap: str | None
    detector: str
    tracker: str
    score_threshold: float
    nms_threshold: float
    min_face_size: int
    blur: bool
    blur_level: int
    heatmap: bool
    frame_skip: int
    half_res: bool


def load_config(path: str) -> PipelineConfig:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    v = y.get("video", {})
    p = y.get("processing", {})
    return PipelineConfig(
        source=v.get("source"),
        output_video=v.get("output_video"),
        output_metrics_json=v.get("output_metrics_json"),
        output_metrics_csv=v.get("output_metrics_csv"),
        output_heatmap=v.get("output_heatmap"),
        detector=p.get("detector", "retinaface"),
        tracker=p.get("tracker", "ocsort"),
        score_threshold=float(p.get("score_threshold", 0.5)),
        nms_threshold=float(p.get("nms_threshold", 0.4)),
        min_face_size=int(p.get("min_face_size", 24)),
        blur=bool(p.get("blur", False)),
        blur_level=int(p.get("blur_level", 15)),
        heatmap=bool(p.get("heatmap", True)),
        frame_skip=int(p.get("frame_skip", 1)),
        half_res=bool(p.get("half_res", False)),
    )


def process_video(cfg: PipelineConfig):
    logger.info(f"Kaynak: {cfg.source}")
    cap = open_video(cfg.source)
    w, h, fps_in, total_frames = get_props(cap)
    logger.info(f"Video boyutu: {w}x{h} @ {fps_in:.1f} FPS, toplam {total_frames}")

    write_out = None
    if cfg.output_video:
        os.makedirs(os.path.dirname(cfg.output_video), exist_ok=True)
        write_out = open_writer(cfg.output_video, fps_in, w, h)

    metrics = MetricsLogger(cfg.output_metrics_json, cfg.output_metrics_csv)
    if cfg.output_metrics_json:
        os.makedirs(os.path.dirname(cfg.output_metrics_json), exist_ok=True)
    if cfg.output_metrics_csv:
        os.makedirs(os.path.dirname(cfg.output_metrics_csv), exist_ok=True)

    detector = build_detector(cfg.detector, cfg.score_threshold, cfg.min_face_size)
    tracker = build_tracker(cfg.tracker)

    heatmap = None
    if cfg.heatmap:
        out_h, out_w = h, w
        heatmap = HeatmapAccumulator(out_h, out_w, sigma=8.0)
        if cfg.output_heatmap:
            os.makedirs(os.path.dirname(cfg.output_heatmap), exist_ok=True)

    fpsm = FPSMeter()
    total_unique_ids = set()
    frame_id = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            proc_frame = frame
            scale = 1.0
            if cfg.half_res:
                proc_frame = cv2.resize(frame, (w // 2, h // 2))
                scale = 0.5

            # Frame skipping: sadece her N. karede tespit yap, diğerlerinde önceki sonuçları kullanma yerine basit tekrar takip uygulanır.
            do_detect = (frame_id % max(1, cfg.frame_skip) == 0)

            detections: List[Detection] = []
            if do_detect:
                detections = detector.detect(proc_frame)

            det_xyxys: List[Tuple[int, int, int, int, float]] = []
            for d in detections:
                x1, y1, x2, y2 = d.to_xyxy()
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                det_xyxys.append((x1, y1, x2, y2, d.score))

            tracks: List[Track] = tracker.update(det_xyxys)

            # Isı haritası için merkezleri ekle
            if heatmap is not None:
                for tr in tracks:
                    heatmap.add_bbox(tr.bbox)

            # Bulanıklaştırma
            if cfg.blur:
                for tr in tracks:
                    gaussian_blur_face(frame, tr.bbox, ksize=max(3, cfg.blur_level))

            # Overlay çiz
            fps = fpsm.tick()
            for tr in tracks:
                total_unique_ids.add(tr.track_id)
            overlay_tracks(
                frame,
                [(tr.track_id, *tr.bbox, tr.score) for tr in tracks],
                fps=fps,
                active_count=len(tracks),
                total_count=len(total_unique_ids),
            )

            # Çıktı yaz
            if write_out is not None:
                write_out.write(frame)

            # Metrikler
            now = time.time() - t0
            metrics.log(now, frame_id, active_count=len(tracks), total_ids=len(total_unique_ids))

        # Isı haritası kaydet
        if heatmap is not None and cfg.output_heatmap:
            hm_color = heatmap.to_color()
            cv2.imwrite(cfg.output_heatmap, hm_color)

    finally:
        metrics.close()
        cap.release()
        if write_out is not None:
            write_out.release()

