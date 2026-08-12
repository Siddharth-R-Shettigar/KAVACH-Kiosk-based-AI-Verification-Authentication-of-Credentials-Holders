import json
import sys
import os

from detectors.exif_detector import run_exif_detector
from detectors.c2pa_detector import run_c2pa_detector
from detectors.ela_detector import run_ela_detector
from detectors.sherloq_noise import run_sherloq_noise
from detectors.jpeg_ghost import run_jpeg_ghost_detector
from detectors.phash_detector import run_phash_detector
from detectors.histogram_detector import run_histogram_detector
from detectors.frequency_detector import run_frequency_detector
from detectors.hf_ai_detector import run_hf_ai_detector
from detectors.copy_move_detector import run_copy_move_detector
from detectors.blur_detector import run_blur_detector
from detectors.cfa_detector import run_cfa_detector
from detectors.resampling_detector import run_resampling_detector
from detectors.illuminant_detector import run_illuminant_detector
from detectors.quantization_detector import run_quantization_detector

def analyze_media(image_path):
    if not os.path.exists(image_path):
        return {"error": f"File {image_path} not found."}

    ext = os.path.splitext(image_path)[1].lower()
    is_jpeg = ext in ['.jpg', '.jpeg']

    # Define detectors with domain importance weights
    # Structure: (function, weight, requires_jpeg)
    detector_pipeline = [
        (run_exif_detector, 1.5, False),
        (run_c2pa_detector, 0.0, False),  # Neutral when missing; excluded from weighted mean
        (run_cfa_detector, 2.0, False),   # High weight: physical sensor signature
        (run_hf_ai_detector, 2.0, False), # High weight: deep feature classification
        (run_resampling_detector, 1.5, False),
        (run_ela_detector, 1.0, False),
        (run_sherloq_noise, 1.0, False),
        (run_histogram_detector, 0.8, False),
        (run_frequency_detector, 0.8, False),
        (run_copy_move_detector, 1.0, False),
        (run_blur_detector, 0.8, False),
        (run_illuminant_detector, 1.0, False),
        (run_phash_detector, 0.0, False),  # Hash generation only
        (run_jpeg_ghost_detector, 1.2, True),
        (run_quantization_detector, 1.2, True)
    ]

    results = []
    weighted_score_sum = 0.0
    total_weight = 0.0

    for detector_fn, weight, req_jpeg in detector_pipeline:
        if req_jpeg and not is_jpeg:
            continue  # Skip JPEG-specific detectors for PNG/WEBP files

        res = detector_fn(image_path)
        results.append(res)

        if weight > 0.0 and "score" in res:
            weighted_score_sum += res["score"] * weight
            total_weight += weight

    final_synthetic_score = weighted_score_sum / max(total_weight, 1.0)
    
    synthetic_prob = int(round(final_synthetic_score * 100))
    authentic_prob = max(0, 100 - synthetic_prob)

    return {
        "engine": "VEDA (Verifiable Evidence & Digital Authenticity)",
        "file_analyzed": os.path.basename(image_path),
        "probabilities": {
            "authentic_capture": f"{authentic_prob}%",
            "synthetic_ai_generated": f"{synthetic_prob}%"
        },
        "overall_confidence": "high",
        "active_detectors_evaluated": len(results),
        "detector_signals": results
    }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    report = analyze_media(target)
    print(json.dumps(report, indent=2))