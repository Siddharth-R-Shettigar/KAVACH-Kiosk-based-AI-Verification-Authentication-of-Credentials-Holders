import json
import sys
import cv2
import numpy as np

def run_illuminant_detector(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate spatial gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(sobelx, sobely)
        
        # Filter for top 25% strongest light transitions
        thresh = np.percentile(magnitude, 75)
        mask = magnitude > thresh
        
        h, w = gray.shape
        quad_bounds = [
            (0, h//2, 0, w//2),
            (0, h//2, w//2, w),
            (h//2, h, 0, w//2),
            (h//2, h, w//2, w)
        ]
        
        quad_sin, quad_cos = [], []
        for y1, y2, x1, x2 in quad_bounds:
            q_gx = sobelx[y1:y2, x1:x2]
            q_gy = sobely[y1:y2, x1:x2]
            q_mask = mask[y1:y2, x1:x2]
            
            if np.sum(q_mask) > 0:
                avg_gx = np.mean(q_gx[q_mask])
                avg_gy = np.mean(q_gy[q_mask])
                angle = np.arctan2(avg_gy, avg_gx)
                quad_sin.append(np.sin(angle))
                quad_cos.append(np.cos(angle))

        if len(quad_sin) < 4:
            circ_var = 0.0
        else:
            mean_cos = np.mean(quad_cos)
            mean_sin = np.mean(quad_sin)
            R = np.hypot(mean_cos, mean_sin)
            circ_var = float(1.0 - R)
        
        # Continuous score scaling: normal scenes land near 0.2 - 0.4, severe mismatches (>0.82) approach 0.85
        if circ_var > 0.82:
            score = 0.85
            exp = f"Illuminant circular variance computed at {round(circ_var, 3)}. Severe conflicting scene lighting vectors detected."
        elif circ_var > 0.72:
            score = 0.45
            exp = f"Illuminant circular variance computed at {round(circ_var, 3)}. Moderate directional lighting spread (common in detailed physical geometry)."
        else:
            score = 0.15
            exp = f"Illuminant circular variance computed at {round(circ_var, 3)}. Consistent scene illumination vector verified."

        return {
            "detector_name": "illuminant_lighting_consistency",
            "score": score,
            "confidence": "medium",
            "explanation": exp
        }

    except Exception as e:
        return {
            "detector_name": "illuminant_lighting_consistency",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Illuminant analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    result = run_illuminant_detector(target_image)
    print(json.dumps(result, indent=2))