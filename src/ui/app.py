from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st

# Streamlit, script klasörünü çalışma dizini yapabilir. 'src' paketini
# güvenle bulabilmek için proje kökünü sys.path'e ekleyelim.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, process_video


st.set_page_config(page_title="Kimliksiz Yüz Analizi", layout="centered")
st.title("Kimliksiz Yüz Tespiti, Takibi ve Isı Haritası")
st.caption("Yerel videonuzu yükleyin, seçenekleri belirleyin ve çalıştırın.")


with st.sidebar:
    st.header("Ayarlar")
    detector = st.selectbox("Dedektör", ["retinaface", "mtcnn"], index=0)
    tracker = st.selectbox("Takipçi", ["ocsort", "deepsort"], index=0)
    blur = st.checkbox("Yüzleri bulanıklaştır", value=True)
    heatmap = st.checkbox("Isı haritası üret", value=True)
    score_threshold = st.slider("Skor eşiği", 0.1, 0.9, 0.5, 0.05)
    min_face_size = st.slider("Min yüz boyutu (px)", 16, 80, 24, 2)
    frame_skip = st.slider("Her N karede tespit", 1, 5, 1, 1)
    half_res = st.checkbox("Yarı çözünürlükte işle")


uploaded = st.file_uploader("Video yükleyin (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"]) 
run_btn = st.button("Çalıştır")

status = st.empty()
result_video = st.empty()
result_heatmap = st.empty()
result_metrics = st.empty()


def _save_uploaded(file) -> Optional[str]:
    if file is None:
        return None
    os.makedirs("data/input_videos", exist_ok=True)
    dst = os.path.join("data", "input_videos", f"uploaded_{int(time.time())}.mp4")
    with open(dst, "wb") as f:
        f.write(file.read())
    return dst


if run_btn:
    if uploaded is None:
        st.error("Lütfen bir video yükleyin.")
        st.stop()

    src_path = _save_uploaded(uploaded)
    if src_path is None:
        st.error("Video kaydedilemedi.")
        st.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cfg = PipelineConfig(
        source=src_path,
        output_video=os.path.join(out_dir, f"annotated_{timestamp}.mp4"),
        output_metrics_json=os.path.join(out_dir, f"metrics_{timestamp}.json"),
        output_metrics_csv=os.path.join(out_dir, f"metrics_{timestamp}.csv"),
        output_heatmap=os.path.join(out_dir, f"heatmap_{timestamp}.png"),
        detector=detector,
        tracker=tracker,
        score_threshold=float(score_threshold),
        nms_threshold=0.4,
        min_face_size=int(min_face_size),
        blur=bool(blur),
        blur_level=15,
        heatmap=bool(heatmap),
        frame_skip=int(frame_skip),
        half_res=bool(half_res),
    )

    status.info("İşleme başladı, lütfen bekleyin...")
    try:
        process_video(cfg)
    except Exception as e:
        status.error(f"Hata: {e}")
        st.stop()

    status.success("Tamamlandı ✅")

    if os.path.exists(cfg.output_video):
        st.subheader("Annotasyonlu Video")
        with open(cfg.output_video, "rb") as f:
            result_video.video(f.read())
        st.download_button("Videoyu indir", data=open(cfg.output_video, "rb").read(), file_name=os.path.basename(cfg.output_video))

    if cfg.output_heatmap and os.path.exists(cfg.output_heatmap):
        st.subheader("Isı Haritası")
        result_heatmap.image(cfg.output_heatmap)
        st.download_button("Isı haritasını indir", data=open(cfg.output_heatmap, "rb").read(), file_name=os.path.basename(cfg.output_heatmap))

    if cfg.output_metrics_json and os.path.exists(cfg.output_metrics_json):
        st.subheader("Metrikler")
        st.code(cfg.output_metrics_json)
        result_metrics.download_button(
            "Metrik JSON indir",
            data=open(cfg.output_metrics_json, "rb").read(),
            file_name=os.path.basename(cfg.output_metrics_json),
        )
