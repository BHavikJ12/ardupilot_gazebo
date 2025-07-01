"""MOSSE Tracker implementation
================================
Author: ChatGPT (OpenAI)
Date  : 2025‑06‑30

This script contains a minimal yet faithful Python implementation of the
MOSSE (Minimum Output Sum of Squared Error) visual tracker introduced by
Bolme et al., CVPR 2010.

Dependencies
------------
- Python >= 3.7
- numpy
- opencv‑python (cv2)

Install:
    pip install numpy opencv-python

Run the demo on your webcam:
    python mosse_tracker.py --source 0

Run the demo on a video file:
    python mosse_tracker.py --source path/to/video.mp4

Press **SPACE** to (re‑)select the initial bounding box.
Press **ESC**   to quit.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import cv2
import numpy as np

# ----------------------------- Utility Functions ----------------------------- #

def _preprocess(img: np.ndarray) -> np.ndarray:
    """Apply log transform & standardization (as in the MOSSE paper)."""
    img = np.log(img.astype(np.float32) + 1.0)
    img -= img.mean()
    std = img.std() + 1e-5
    return img / std


def _gaussian_2d(shape: Tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """Create 2‑D Gaussian label function with peak at the center."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    return np.exp(-0.5 * (((y - cy) ** 2 + (x - cx) ** 2) / sigma**2))


def _random_warp(img: np.ndarray) -> np.ndarray:
    """Apply a small random affine warp for data augmentation."""
    h, w = img.shape
    theta = (np.random.randn() * 360) * 0.0  # tiny rotations disabled (can enable)
    trans = 0.1 * np.array([w, h]) * np.random.randn(2)
    mat = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), trans[0]],
                    [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta)), trans[1]]], dtype=np.float32)
    return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT101)


# ----------------------------- MOSSE Functions ------------------------------- #

def train_filter(patch: np.ndarray, num_aug: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train initial MOSSE filter.

    Returns (H, A, B) where H = A / B is the correlation filter,
    and A, B are numerator / denominator for online updates.
    """
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    g = _gaussian_2d(patch.shape)
    G = np.fft.fft2(g)

    A = np.zeros_like(G, dtype=np.complex64)
    B = np.zeros_like(G, dtype=np.complex64)

    for _ in range(num_aug):
        img = _random_warp(patch)
        F = np.fft.fft2(_preprocess(img))
        A += G * np.conj(F)
        B += F * np.conj(F)

    H = A / (B + 1e-5)  # avoid divide‑by‑zero
    return H, A, B


def update_filter(A: np.ndarray, B: np.ndarray, F: np.ndarray, G: np.ndarray, eta: float = 0.125) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update numerator A and denominator B using learning rate eta."""
    A_new = (1 - eta) * A + eta * (G * np.conj(F))
    B_new = (1 - eta) * B + eta * (F * np.conj(F))
    H_new = A_new / (B_new + 1e-5)
    return H_new, A_new, B_new


def correlate(H: np.ndarray, patch: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Correlate patch with filter H; return response map and peak coordinates."""
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    F = np.fft.fft2(_preprocess(patch))
    R = np.fft.ifft2(H * F).real
    max_loc = np.unravel_index(np.argmax(R), R.shape)
    return R, max_loc[::-1]  # return (x, y)


# ----------------------------- Tracker Class --------------------------------- #

class MOSSETracker:
    """Lightweight MOSSE tracker object."""

    def __init__(self):
        self.pos = None  # (x, y) center of target
        self.size = None  # (w, h)
        self.H = self.A = self.B = None
        self.G = None  # desired output (frequency domain)
        self.window = None

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        self.pos = (x + w // 2, y + h // 2)
        self.size = (w, h)
        patch = self._get_patch(frame, self.pos, self.size)
        self.H, self.A, self.B = train_filter(patch)
        self.G = np.fft.fft2(_gaussian_2d((h, w)))
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def _get_patch(self, frame: np.ndarray, pos: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
        x, y = pos
        w, h = size
        xs = int(x - w // 2)
        ys = int(y - h // 2)
        patch = frame[ys:ys + h, xs:xs + w].copy()
        if patch.shape[0] != h or patch.shape[1] != w:  # pad if out‑of‑bounds
            patch = cv2.copyMakeBorder(patch, 0, h - patch.shape[0], 0, w - patch.shape[1], cv2.BORDER_REPLICATE)
        return patch

    def update(self, frame: np.ndarray, eta: float = 0.125) -> Tuple[int, int, int, int, float]:
        x, y = self.pos
        w, h = self.size
        patch = self._get_patch(frame, self.pos, self.size)
        R, (dx, dy) = correlate(self.H, patch * self.window[..., None])
        x += dx - w // 2
        y += dy - h // 2
        self.pos = (x, y)

        # update filter with new patch
        patch = self._get_patch(frame, self.pos, self.size)
        F = np.fft.fft2(_preprocess(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))
        self.H, self.A, self.B = update_filter(self.A, self.B, F, self.G, eta)

        psr = self._calc_psr(R)
        return x - w // 2, y - h // 2, w, h, psr

    @staticmethod
    def _calc_psr(resp: np.ndarray, k: int = 11) -> float:
        """Peak‑to‑Sidelobe Ratio for monitoring tracker confidence."""
        h, w = resp.shape
        cy, cx = np.unravel_index(np.argmax(resp), resp.shape)
        peak = resp[cy, cx]
        side = resp.copy()
        side[max(0, cy - k):min(h, cy + k + 1), max(0, cx - k):min(w, cx + k + 1)] = 0
        mean = side.mean()
        std = side.std() + 1e-5
        return float((peak - mean) / std)


# ----------------------------- Demo Code ------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="MOSSE tracker demo (press SPACE to select ROI)")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam or path to video file)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source {args.source}")

    tracker = MOSSETracker()
    init_once = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not init_once:
            cv2.putText(frame, "Press SPACE to select ROI", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            x, y, w, h, psr = tracker.update(frame)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(frame, f"PSR: {psr:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("MOSSE Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE: select ROI
            roi = cv2.selectROI("MOSSE Tracker", frame, fromCenter=False, showCrosshair=True)
            if roi[2] > 0 and roi[3] > 0:
                tracker.init(frame, roi)
                init_once = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
