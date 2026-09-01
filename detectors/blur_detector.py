import json
import sys
import cv2
import numpy as np

def run_blur_detector(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to read image file.")

        # Compute variance of Laplacian to measure overall sharpness
        focus_measure = cv2.Laplacian(img, cv2.CV_64F).var()

        # Low focus measure (< 80) flags heavy blur / unnatural smoothing
        if focus_measure < 50.0:
            score = 0.85
            exp = f"High artificial blur detected (sharpness metric: {round(focus_measure, 2)})."
        elif focus_measure > 2500.0:
            score = 0.75
            exp = f"Excessive edge sharpening detected (sharpness metric: {round(focus_measure, 2)})."
        else:
            score = 0.15
            exp = f"Standard physical lens focus distribution (sharpness metric: {round(focus_measure, 2)})."

        return {
            "detector_name": "blur_sharpness_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": exp,
            "status": "flagged" if score >= 0.4 else "passed"
        }
    except Exception as e:
        return {
            "detector_name": "blur_sharpness_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Blur analysis failed: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    # Switching default test image to phone camera capture
    target_image = sys.argv[1] if len(sys.argv) > 1 else "20260804_073446.jpg"
    result = run_blur_detector(target_image)
    print(json.dumps(result, indent=2))