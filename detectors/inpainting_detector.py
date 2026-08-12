import json
import sys
import cv2
import numpy as np

def run_inpainting_detector(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute laplacian spatial variance across 8x8 blocks
        h, w = gray.shape
        block_size = 16
        block_variances = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                patch = gray[y:y+block_size, x:x+block_size]
                var = cv2.Laplacian(patch, cv2.CV_64F).var()
                block_variances.append(var)

        block_variances = np.array(block_variances)
        
        # High localized variance spread indicates regional AI inpainting / smooth patching
        std_dev = np.std(block_variances)
        mean_var = np.mean(block_variances)
        ratio = std_dev / (mean_var + 1e-5)

        score = 0.80 if ratio > 2.2 else 0.15

        return {
            "detector_name": "inpainting_trace_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": f"Inpainting edge boundary dispersion ratio computed at {round(float(ratio), 2)}. " +
                           ("Suspicious localized generative inpainting or region patching detected." if score > 0.5 else "Uniform pixel edge continuity verified across image patches.")
        }

    except Exception as e:
        return {
            "detector_name": "inpainting_trace_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Inpainting analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_inpainting_detector(target), indent=2))