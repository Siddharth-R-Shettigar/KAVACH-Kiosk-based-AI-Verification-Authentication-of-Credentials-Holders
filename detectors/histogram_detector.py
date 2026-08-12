import json
import sys
import cv2
import numpy as np

def run_histogram_detector(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file.")
            
        chans = cv2.split(img)
        entropies = []
        
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            hist_norm = hist_norm[hist_norm > 0]
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            entropies.append(entropy)
            
        mean_entropy = float(np.mean(entropies))
        score = 0.75 if mean_entropy < 4.5 or mean_entropy > 7.8 else 0.15
        
        return {
            "detector_name": "histogram_color_forensics",
            "score": score,
            "confidence": "medium",
            "explanation": f"Color channel entropy computed at {round(mean_entropy, 2)} bits." +
                           (" Potential color space anomaly detected." if score > 0.5 else " Standard natural RGB channel distribution.")
        }
    except Exception as e:
        return {
            "detector_name": "histogram_color_forensics",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Histogram analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_histogram_detector(target_image)
    print(json.dumps(result, indent=2))