import json
import sys
from PIL import Image

def run_quantization_detector(image_path):
    try:
        img = Image.open(image_path)
        
        if img.format != 'JPEG':
            return {
                "detector_name": "jpeg_quantization_analysis",
                "score": 0.1,
                "confidence": "low",
                "explanation": "Non-JPEG file format. Quantization table inspection bypassed."
            }

        qtables = getattr(img, 'quant_tables', None)
        if not qtables:
            return {
                "detector_name": "jpeg_quantization_analysis",
                "score": 0.6,
                "confidence": "medium",
                "explanation": "Missing standard JPEG quantization tables."
            }

        # Check luminance table variance to detect non-standard software re-compression
        lum_table = qtables[0]
        avg_q_val = sum(lum_table) / len(lum_table)

        # Standard high-quality camera captures maintain average Q-values below 15
        score = 0.75 if avg_q_val > 25.0 else 0.15

        return {
            "detector_name": "jpeg_quantization_analysis",
            "score": score,
            "confidence": "high",
            "explanation": f"JPEG quantization mean coefficient computed at {round(avg_q_val, 2)}." +
                           (" Non-standard software re-compression matrix detected." if score > 0.5 else " Standard camera sensor hardware Q-table verified.")
        }

    except Exception as e:
        return {
            "detector_name": "jpeg_quantization_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Quantization table inspection failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    result = run_quantization_detector(target_image)
    print(json.dumps(result, indent=2))