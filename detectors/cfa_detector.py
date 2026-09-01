import json
import sys
import cv2
import numpy as np

def run_cfa_detector(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image file.")

        # Convert image to float representation
        img_float = img.astype(np.float32)
        green = img_float[:, :, 1]

        # Calculate high-pass Laplacian filter across Green channel to extract demosaicing residuals
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        filtered_green = cv2.filter2D(green, -1, kernel)

        # Compute 2D Fourier Transform of residual grid
        f = np.fft.fft2(filtered_green)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Measure peak energy at expected Bayer frequency locations
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Check energy ratio at half-sampling frequency (Bayer grid signatures)
        grid_energy = (
            magnitude[center_h, 0] + 
            magnitude[0, center_w] + 
            magnitude[center_h, center_w // 2]
        ) / 3.0
        
        total_energy = np.mean(magnitude) + 1e-8
        cfa_ratio = float(grid_energy / total_energy)

        # Real cameras show strong periodic Bayer peaks (ratio > 12.0)
        # AI images lack physical demosaicing grid residuals
        score = 0.15 if cfa_ratio > 10.0 else 0.85

        return {
            "detector_name": "cfa_demosaicing_analysis",
            "score": score,
            "confidence": "medium",
            "explanation": f"Bayer CFA grid residual ratio computed at {round(cfa_ratio, 2)}." +
                           (" Periodic camera sensor demosaicing pattern detected." if score < 0.5 else " Lacks physical camera Bayer filter demosaicing signature."),
            "status": "flagged" if score >= 0.4 else "passed"                           
        }

    except Exception as e:
        return {
            "detector_name": "cfa_demosaicing_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"CFA demosaicing evaluation failed: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    result = run_cfa_detector(target_image)
    print(json.dumps(result, indent=2))