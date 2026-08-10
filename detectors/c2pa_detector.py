import json
import sys
import c2pa

def run_c2pa_detector(image_path):
    try:
        # try_create returns None if no manifest is embedded
        reader = c2pa.Reader.try_create(image_path)
        
        if reader is None:
            return {
                "detector_name": "c2pa",
                "score": 0.5,
                "confidence": "low",
                "explanation": "No C2PA Content Credentials manifest found (typical for standard or unsigned media)."
            }
        
        manifest_data = json.loads(reader.json())
        manifest_str = json.dumps(manifest_data).lower()
        
        # Look for explicit synthetic/AI generation markers in the manifest
        if any(keyword in manifest_str for keyword in ["digitalcreation", "trainedalgorithmicmedia", "openai", "dall-e"]):
            return {
                "detector_name": "c2pa",
                "score": 0.95,
                "confidence": "high",
                "explanation": "Valid C2PA Content Credentials found: File explicitly signed as AI/synthetic creation."
            }
        
        return {
            "detector_name": "c2pa",
            "score": 0.1,
            "confidence": "high",
            "explanation": "Valid C2PA Content Credentials found certifying authentic media/capture."
        }

    except Exception as e:
        return {
            "detector_name": "c2pa",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"C2PA evaluation bypassed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_c2pa_detector(target_image)
    print(json.dumps(result, indent=2))