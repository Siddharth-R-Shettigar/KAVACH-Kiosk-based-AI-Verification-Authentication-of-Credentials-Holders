import json
import sys
import cv2
import numpy as np

def run_resampling_detector(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to read image file.")

        # Compute second derivative to highlight pixel interpolation cycles
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        
        # Calculate derivative variance across rows and columns to spot periodic ripples
        row_var = np.var(np.diff(laplacian, axis=1), axis=0)
        col_var = np.var(np.diff(laplacian, axis=0), axis=1)

        resampling_metric = float((np.std(row_var) + np.std(col_var)) / 2.0)

        # Extreme periodic spikes (> 450) or completely flat cycles (< 10) indicate resampling/scaling
        if resampling_metric > 450.0 or resampling_metric < 10.0:
            score = 0.8
            exp = f"Periodic interpolation artifacts detected (metric: {round(resampling_metric, 2)}). Image shows signs of AI upscaling or non-native resampling."
        else:
            score = 0.2
            exp = f"Standard organic pixel continuity verified (metric: {round(resampling_metric, 2)})."

        return {
            "detector_name": "resampling_interpolation_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": exp
        }

    except Exception as e:
        return {
            "detector_name": "resampling_interpolation_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Resampling evaluation failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Merry Christmas.png"
    result = run_resampling_detector(target_image)
    print(json.dumps(result, indent=2))