import json
import sys
import os
import cv2
import numpy as np
from PIL import Image

def run_jpeg_ghost_detector(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        ghost_scores = []
        
        # Test re-compression differences across quality levels (50 to 95)
        for q in range(50, 100, 10):
            temp_path = f"temp_ghost_{q}.jpg"
            img.save(temp_path, "JPEG", quality=q)
            resaved = Image.open(temp_path).convert('RGB')
            
            diff = np.mean(np.abs(np.array(img, dtype=np.float32) - np.array(resaved, dtype=np.float32)))
            ghost_scores.append(diff)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Calculate variance across ghost curves
        ghost_variance = float(np.var(ghost_scores))
        
        # Low variance across quality steps indicates non-JPEG or flat synthetic compression
        score = 0.85 if ghost_variance < 5.0 else 0.15
        
        return {
            "detector_name": "jpeg_ghost_analysis",
            "score": round(score, 2),
            "confidence": "medium",
            "explanation": f"JPEG Ghost variance computed at {round(ghost_variance, 2)}. " +
                           ("Suspicious uniform compression grid detected." if ghost_variance < 5.0 else "Normal multi-stage JPEG compression curve.")
        }
    except Exception as e:
        return {
            "detector_name": "jpeg_ghost_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"JPEG Ghost check failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_jpeg_ghost_detector(target_image)
    print(json.dumps(result, indent=2))