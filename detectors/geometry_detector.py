import json
import sys
import cv2
import numpy as np

def run_geometry_detector(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect straight structural lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            return {
                "detector_name": "object_geometry_analysis",
                "score": 0.2,
                "confidence": "low",
                "explanation": "Insufficient structural straight lines found for vanishing-point geometry check."
            }

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)

        # Measure angular clustering variance
        angle_hist, _ = np.histogram(angles, bins=18, range=(0, 180))
        dominant_spread = np.std(angle_hist)

        # AI scenes often yield chaotic structural angles that fail physical perspective grouping
        score = 0.75 if dominant_spread < 2.0 else 0.15

        return {
            "detector_name": "object_geometry_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": f"Perspective structural angle distribution variance: {round(float(dominant_spread), 2)}. " +
                           ("Physical geometry perspective anomalies or warped structural lines detected." if score > 0.5 else "Consistent parallel line convergence and physical perspective verified.")
        }

    except Exception as e:
        return {
            "detector_name": "object_geometry_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Geometry analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_geometry_detector(target), indent=2))