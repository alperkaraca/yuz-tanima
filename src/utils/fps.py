from __future__ import annotations
import time


class FPSMeter:
    """Basit FPS ölçer."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.frames = 0
        self.fps = 0.0

    def tick(self) -> float:
        self.frames += 1
        now = time.time()
        dt = now - self.t0
        if dt >= 1.0:
            self.fps = self.frames / dt
            self.frames = 0
            self.t0 = now
        return self.fps

