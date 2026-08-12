import json
import sys
import cv2
import numpy as np

def run_copy_move_detector(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to read image file.")

        # Extract ORB keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) < 2:
            return {
                "detector_name": "copy_move_forgery",
                "score": 0.1,
                "confidence": "low",
                "explanation": "Insufficient feature keypoints found for clone analysis."
            }

        # Match keypoints against themselves to find duplicate regions
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=2)

        cloned_points = 0
        for match in matches:
            if len(match) > 1:
                m, n = match[0], match[1]
                # Filter out self-matches and check distance ratio
                if m.distance < 0.65 * n.distance and m.queryIdx != m.trainIdx:
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    # Ignore keypoints that are virtually adjacent
                    if np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]) > 30:
                        cloned_points += 1

        score = round(min(cloned_points / 15.0, 1.0), 2)

        return {
            "detector_name": "copy_move_forgery",
            "score": score,
            "confidence": "medium",
            "explanation": f"Found {cloned_points} suspicious duplicated feature clusters." +
                           (" Potential copy-move editing detected." if score > 0.4 else " No duplicate region cloning detected.")
        }

    except Exception as e:
        return {
            "detector_name": "copy_move_forgery",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Copy-move evaluation failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_copy_move_detector(target_image)
    print(json.dumps(result, indent=2))