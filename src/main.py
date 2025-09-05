from __future__ import annotations
import typer
from typing import Optional
from loguru import logger
import os

from .pipeline import PipelineConfig, process_video, load_config


app = typer.Typer(help="Yüz tespiti/ takibi/ sayımı/ ısı haritası (kimliksiz)")


def _execute(
    source: str,
    detector: str,
    tracker: str,
    blur: str,
    heatmap: str,
    save_annotated: Optional[str],
    save_metrics: Optional[str],
    save_metrics_csv: Optional[str],
    heatmap_png: Optional[str],
    score_threshold: float,
    min_face_size: int,
    frame_skip: int,
    half_res: str,
    config: Optional[str],
):
    """Aynı iş mantığını hem alt komut hem de kök komut için kullan."""
    if config:
        cfg = load_config(config)
        # CLI ile override
        cfg.source = source or cfg.source
        cfg.detector = detector or cfg.detector
        cfg.tracker = tracker or cfg.tracker
        cfg.blur = (blur.lower() == "on") if blur else cfg.blur
        cfg.heatmap = (heatmap.lower() == "on") if heatmap else cfg.heatmap
        cfg.output_video = save_annotated or cfg.output_video
        cfg.output_metrics_json = save_metrics or cfg.output_metrics_json
        cfg.output_metrics_csv = save_metrics_csv or cfg.output_metrics_csv
        cfg.output_heatmap = heatmap_png or cfg.output_heatmap
        cfg.score_threshold = score_threshold or cfg.score_threshold
        cfg.min_face_size = min_face_size or cfg.min_face_size
        cfg.frame_skip = frame_skip or cfg.frame_skip
        cfg.half_res = (half_res.lower() == "on") if half_res else cfg.half_res
    else:
        cfg = PipelineConfig(
            source=source,
            output_video=save_annotated,
            output_metrics_json=save_metrics,
            output_metrics_csv=save_metrics_csv,
            output_heatmap=heatmap_png,
            detector=detector,
            tracker=tracker,
            score_threshold=score_threshold,
            nms_threshold=0.4,
            min_face_size=min_face_size,
            blur=(blur.lower() == "on"),
            blur_level=15,
            heatmap=(heatmap.lower() == "on"),
            frame_skip=frame_skip,
            half_res=(half_res.lower() == "on"),
        )

    logger.info("Pipeline başlatılıyor...")
    process_video(cfg)


@app.command()
def run(
    source: str = typer.Option(..., help="Giriş video yolu"),
    detector: str = typer.Option("retinaface", help="Dedektör: retinaface | mtcnn"),
    tracker: str = typer.Option("ocsort", help="Takipçi: ocsort | deepsort"),
    blur: str = typer.Option("off", help="Yüz blur: on/off"),
    heatmap: str = typer.Option("on", help="Isı haritası: on/off"),
    save_annotated: Optional[str] = typer.Option(None, help="Annotasyonlu video çıkışı"),
    save_metrics: Optional[str] = typer.Option(None, help="JSON metrik çıkışı"),
    save_metrics_csv: Optional[str] = typer.Option(None, help="CSV metrik çıkışı"),
    heatmap_png: Optional[str] = typer.Option(None, help="Isı haritası PNG"),
    score_threshold: float = typer.Option(0.5, help="Skor eşiği"),
    min_face_size: int = typer.Option(24, help="Min yüz boyutu (px)"),
    frame_skip: int = typer.Option(1, help="Her N karede tespit"),
    half_res: str = typer.Option("off", help="Yarı çözünürlük: on/off"),
    config: Optional[str] = typer.Option(None, help="YAML config yolu (opsiyonel)"),
):
    """Komut satırı arayüzü (alt komut)."""
    _execute(
        source,
        detector,
        tracker,
        blur,
        heatmap,
        save_annotated,
        save_metrics,
        save_metrics_csv,
        heatmap_png,
        score_threshold,
        min_face_size,
        frame_skip,
        half_res,
        config,
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    source: Optional[str] = typer.Option(None, help="Giriş video yolu"),
    detector: str = typer.Option("retinaface", help="Dedektör: retinaface | mtcnn"),
    tracker: str = typer.Option("ocsort", help="Takipçi: ocsort | deepsort"),
    blur: str = typer.Option("off", help="Yüz blur: on/off"),
    heatmap: str = typer.Option("on", help="Isı haritası: on/off"),
    save_annotated: Optional[str] = typer.Option(None, help="Annotasyonlu video çıkışı"),
    save_metrics: Optional[str] = typer.Option(None, help="JSON metrik çıkışı"),
    save_metrics_csv: Optional[str] = typer.Option(None, help="CSV metrik çıkışı"),
    heatmap_png: Optional[str] = typer.Option(None, help="Isı haritası PNG"),
    score_threshold: float = typer.Option(0.5, help="Skor eşiği"),
    min_face_size: int = typer.Option(24, help="Min yüz boyutu (px)"),
    frame_skip: int = typer.Option(1, help="Her N karede tespit"),
    half_res: str = typer.Option("off", help="Yarı çözünürlük: on/off"),
    config: Optional[str] = typer.Option(None, help="YAML config yolu (opsiyonel)"),
):
    """Kök komut: 'run' yazmadan çağrılmayı da destekler."""
    if ctx.invoked_subcommand is None:
        if not source:
            # Alt komut verilmediyse ve source yoksa yardım gösterelim
            typer.echo(ctx.get_help())
            raise typer.Exit(code=1)
        _execute(
            source,
            detector,
            tracker,
            blur,
            heatmap,
            save_annotated,
            save_metrics,
            save_metrics_csv,
            heatmap_png,
            score_threshold,
            min_face_size,
            frame_skip,
            half_res,
            config,
        )


if __name__ == "__main__":
    app()
