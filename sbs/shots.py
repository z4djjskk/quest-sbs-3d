from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import compute_hist_hsv, hist_correlation


@dataclass
class ShotDetector:
    cut_threshold: float
    min_len_frames: int
    max_len_frames: int

    last_hist: np.ndarray | None = None
    frames_since_cut: int = 0

    def update(self, frame_bgr: np.ndarray) -> bool:
        self.frames_since_cut += 1
        if self.last_hist is None:
            self.last_hist = compute_hist_hsv(frame_bgr)
            return False

        hist = compute_hist_hsv(frame_bgr)
        corr = hist_correlation(self.last_hist, hist)
        self.last_hist = hist

        if self.frames_since_cut < self.min_len_frames:
            return False
        if self.frames_since_cut >= self.max_len_frames:
            self.frames_since_cut = 0
            return True
        if corr < self.cut_threshold:
            self.frames_since_cut = 0
            return True
        return False

    def reset(self) -> None:
        self.last_hist = None
        self.frames_since_cut = 0