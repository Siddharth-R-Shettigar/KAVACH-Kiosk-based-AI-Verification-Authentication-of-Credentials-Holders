import json
import sys
import cv2
import numpy as np

def run_sherloq_noise(image_path):
    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image file.")

        # Estimate high-frequency noise using Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Unusually low noise variance (< 100) often correlates with smooth AI renders/denoised images
        # Unusually high variance (> 1500) can indicate heavy processing or synthetic grain
        if laplacian_var < 100:
            score = 0.8
            exp = f"Low noise variance ({round(laplacian_var, 2)}). Unnaturally smooth surfaces detected."
        elif laplacian_var > 2000:
            score = 0.75
            exp = f"High noise variance ({round(laplacian_var, 2)}). Significant digital grain or sharpening detected."
        else:
            score = 0.2
            exp = f"Standard physical noise distribution detected (variance: {round(laplacian_var, 2)})."

        return {
            "detector_name": "sherloq_noise_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": exp
        }

    except Exception as e:
        return {
            "detector_name": "sherloq_noise_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Noise evaluation failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_sherloq_noise(target_image)
    print(json.dumps(result, indent=2))