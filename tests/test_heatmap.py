from __future__ import annotations
from src.utils.heatmap import HeatmapAccumulator


def test_heatmap_accumulate():
    hm = HeatmapAccumulator(100, 100, sigma=2.0)
    hm.add_bbox((10, 10, 20, 20))
    hm.add_bbox((10, 10, 20, 20))
    img = hm.render()
    assert img.shape == (100, 100)
    assert img.max() > 0.0

