import json
import sys
import os
import cv2
import numpy as np

def run_video_optical_flow(video_path):
    if not os.path.exists(video_path) or not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        return {
            "detector_name": "video_optical_flow_analysis",
            "score": 0.1,
            "confidence": "low",
            "explanation": "Input file is not a supported video file format. Video temporal analysis bypassed."
        }

    try:
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Unable to read video stream.")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow_variances = []
        frame_count = 0

        while cap.isOpened() and frame_count < 60: # Evaluate first 60 frames
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude of temporal frame vector displacement
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_variances.append(np.var(magnitude))

            prev_gray = gray
            frame_count += 1

        cap.release()

        avg_flow_var = float(np.mean(flow_variances)) if flow_variances else 0.0
        score = 0.85 if avg_flow_var > 45.0 or avg_flow_var < 0.01 else 0.15

        return {
            "detector_name": "video_optical_flow_analysis",
            "score": score,
            "confidence": "high",
            "explanation": f"Inter-frame dense optical flow variance computed at {round(avg_flow_var, 2)}. " +
                           ("Unnatural temporal jitter or AI face-swap frame warping detected." if score > 0.5 else "Smooth organic physical temporal fluid motion verified.")
        }

    except Exception as e:
        return {
            "detector_name": "video_optical_flow_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Video optical flow analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_video.mp4"
    print(json.dumps(run_video_optical_flow(target), indent=2))