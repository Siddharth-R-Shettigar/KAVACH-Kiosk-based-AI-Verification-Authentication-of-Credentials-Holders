import json
import sys
import cv2
import numpy as np

def run_frequency_detector(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to read image file.")
            
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        magnitude_spectrum_high = magnitude_spectrum.copy()
        magnitude_spectrum_high[crow-30:crow+30, ccol-30:ccol+30] = 0
        
        high_freq_power = float(np.mean(magnitude_spectrum_high))
        score = 0.8 if high_freq_power > 80.0 else 0.2
        
        return {
            "detector_name": "frequency_domain_fft",
            "score": score,
            "confidence": "medium",
            "explanation": f"FFT spectral power calculated at {round(high_freq_power, 2)} dB." +
                           (" High-frequency grid noise detected." if score > 0.5 else " Standard continuous natural spectrum."),
            "status": "flagged" if score >= 0.4 else "passed"
        }
    except Exception as e:
        return {
            "detector_name": "frequency_domain_fft",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Frequency domain transform failed: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_frequency_detector(target_image)
    print(json.dumps(result, indent=2))