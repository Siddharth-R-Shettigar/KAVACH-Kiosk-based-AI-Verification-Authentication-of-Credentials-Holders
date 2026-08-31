from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class LivenessConfig:
    # Face patch is resized to this size before analysis so thresholds
    # are comparable across different input resolutions.
    resize_dim: tuple[int, int] = (256, 256)

    # FFT: fraction of the spectrum radius considered "high frequency".
    # Frequencies beyond this radial fraction (from the DC center) are
    # summed to measure high-frequency energy.
    fft_high_freq_start: float = 0.35

    # FFT: energy ratio above which we suspect a moire/dot-matrix
    # pattern (screen or print), calibrated empirically.
    fft_spoof_ratio_threshold: float = 0.05

    # Color: HSV Value-channel peak concentration above which we
    # suspect a discrete/quantized backlight (screen) or flat print.
    hsv_peak_concentration_threshold: float = 0.05


    # Color: minimum Cb/Cr variance expected from a naturally lit,
    # live face. Displays tend to compress this variance.
    ycbcr_min_chroma_variance: float = 15.0

    # Weights for combining the two sub-scores into the final score.
    fft_weight: float = 0.55
    color_weight: float = 0.45


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class LivenessResult:
    spoof_score: float          # 0.0 (live) -> 1.0 (spoofed)
    fft_score: float            # sub-score from frequency analysis
    color_score: float          # sub-score from color analysis
    verdict: str                # "live", "suspicious", or "spoof"

    def is_live(self, accept_threshold: float = 0.4) -> bool:
        return self.spoof_score < accept_threshold


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #

class LivenessDetector:
    """
    Stateless scorer: call `score(face_bgr)` per frame/capture.
    Designed to run on a single still face patch (e.g. the frame the
    kiosk chooses after face detection + alignment).
    """

    def __init__(self, config: LivenessConfig | None = None):
        self.cfg = config or LivenessConfig()

    # -- public API -------------------------------------------------- #

    def score(self, face_bgr: np.ndarray) -> LivenessResult:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("face_bgr is empty")

        face = cv2.resize(face_bgr, self.cfg.resize_dim, interpolation=cv2.INTER_AREA)

        fft_score = self._fft_high_frequency_score(face)
        color_score = self._color_inconsistency_score(face)

        combined = (
            self.cfg.fft_weight * fft_score
            + self.cfg.color_weight * color_score
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        verdict = self._verdict_from_score(combined)

        return LivenessResult(
            spoof_score=combined,
            fft_score=fft_score,
            color_score=color_score,
            verdict=verdict,
        )

    # -- FFT / high-frequency texture analysis ------------------------ #

    def _fft_high_frequency_score(self, face_bgr: np.ndarray) -> float:
        """
        Returns a 0-1 score: how strongly the image exhibits regular
        high-frequency energy consistent with print dot-matrices or
        screen sub-pixel/moire grids.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Mild normalization so lighting doesn't dominate the spectrum.
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)

        spectrum = fftshift(fft2(gray))
        magnitude = np.abs(spectrum)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Build a radial distance map from the DC (zero-frequency) center.
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_radius = np.sqrt(cy ** 2 + cx ** 2)
        norm_radius = radius / (max_radius + 1e-6)

        high_freq_mask = norm_radius >= self.cfg.fft_high_freq_start

        total_energy = magnitude.sum() + 1e-6
        high_freq_energy = magnitude[high_freq_mask].sum()
        high_freq_ratio = high_freq_energy / total_energy

        # Regularity check: real skin texture has diffuse high-frequency
        # energy; print/screen artifacts concentrate energy into sharp,
        # periodic peaks. Measure peakiness via the ratio of max bin
        # energy to the mean energy within the high-frequency band.
        hf_values = magnitude[high_freq_mask]
        peakiness = (hf_values.max() / (hf_values.mean() + 1e-6)) if hf_values.size else 0.0
        peakiness_norm = float(np.clip(peakiness / 50.0, 0.0, 1.0))  # empirical scale

        # Combine ratio-above-threshold and peakiness into a single 0-1 score.
        ratio_component = float(
            np.clip(
                (high_freq_ratio - self.cfg.fft_spoof_ratio_threshold)
                / (1.0 - self.cfg.fft_spoof_ratio_threshold + 1e-6),
                0.0,
                1.0,
            )
        )

        fft_score = float(np.clip(0.6 * ratio_component + 0.4 * peakiness_norm, 0.0, 1.0))
        return fft_score

    # -- color space inconsistency analysis --------------------------- #

    def _color_inconsistency_score(self, face_bgr: np.ndarray) -> float:
        """
        Returns a 0-1 score: how strongly the color statistics resemble
        a display/print reproduction rather than a naturally lit,
        directly-captured face.
        """
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        ycbcr = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)

        v_channel = hsv[:, :, 2].astype(np.float32)
        cr_channel = ycbcr[:, :, 1].astype(np.float32)
        cb_channel = ycbcr[:, :, 2].astype(np.float32)

        # 1) HSV Value histogram peak concentration.
        # Displays/prints tend to clip or quantize luminance into a few
        # dominant bins; live skin under ambient light shows a broader,
        # smoother Value histogram.
        hist_v, _ = np.histogram(v_channel, bins=32, range=(0, 255), density=True)
        hist_v = hist_v / (hist_v.sum() + 1e-6)
        peak_concentration = float(hist_v.max())

        peak_component = float(
            np.clip(
                (peak_concentration - self.cfg.hsv_peak_concentration_threshold)
                / (1.0 - self.cfg.hsv_peak_concentration_threshold + 1e-6),
                0.0,
                1.0,
            )
        )

        # 2) Chroma variance in YCbCr.
        # Real skin under natural/ambient light shows richer, more
        # variable chroma (subtle blood-flow, shading, texture) than a
        # screen or flat print reproducing a fixed color gamut.
        chroma_variance = float(cb_channel.var() + cr_channel.var()) / 2.0
        variance_component = float(
            np.clip(
                1.0 - (chroma_variance / (self.cfg.ycbcr_min_chroma_variance + 1e-6)),
                0.0,
                1.0,
            )
        )

        color_score = float(np.clip(0.5 * peak_component + 0.5 * variance_component, 0.0, 1.0))
        return color_score

    # -- helpers -------------------------------------------------------- #

    @staticmethod
    def _verdict_from_score(score: float) -> str:
        if score < 0.35:
            return "live"
        if score < 0.65:
            return "suspicious"
        return "spoof"


# --------------------------------------------------------------------------- #
# CLI / manual test entry point
# --------------------------------------------------------------------------- #

def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Score a face image for liveness/spoofing.")
    parser.add_argument("image_path", help="Path to a cropped face-patch image (jpg/png).")
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.05,
        help="Spoof score below this value is treated as live (default: 0.4).",
    )
    args = parser.parse_args()

    face = cv2.imread(args.image_path)
    if face is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    detector = LivenessDetector()
    result = detector.score(face)

    print(f"spoof_score : {result.spoof_score:.3f}")
    print(f"fft_score   : {result.fft_score:.3f}")
    print(f"color_score : {result.color_score:.3f}")
    print(f"verdict     : {'live' if result.is_live(args.accept_threshold)else 'spoof'}")
    print(f"is_live     : {result.is_live(args.accept_threshold)}")


if __name__ == "__main__":
    _main()
def run_liveness_detection(image_input) -> dict:
    """
    Standard entry-point function for kavach_engine.py integration.
    Accepts an image path (str) or pre-loaded BGR image array (np.ndarray).
    """
    try:
        # 1. Handle image loading
        if isinstance(image_input, str):
            face_bgr = cv2.imread(image_input)
            if face_bgr is None:
                return {
                    "detector_name": "liveness_analysis",
                    "score": 0.5,
                    "confidence": "low",
                    "explanation": f"Unable to read image at path: {image_input}",
                    "status": "failed"
                }
        else:
            face_bgr = image_input

        # 2. Run Detector Logic
        detector = LivenessDetector()
        res = detector.score(face_bgr)

        # 3. Determine confidence based on extreme scores
        confidence = "high" if (res.spoof_score > 0.75 or res.spoof_score < 0.25) else "medium"

        # 4. Return Standard KAVACH Payload Dictionary
        return {
            "detector_name": "liveness_analysis",
            "score": round(res.spoof_score, 3),
            "confidence": confidence,
            "explanation": f"Spoof verdict: '{res.verdict}'. FFT high-freq energy: {res.fft_score:.2f}, Color inconsistency: {res.color_score:.2f}.",
            "status": "passed" if res.verdict == "live" else "flagged"
        }

    except Exception as e:
        return {
            "detector_name": "liveness_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Liveness execution error: {str(e)}",
            "status": "failed"
        }    


