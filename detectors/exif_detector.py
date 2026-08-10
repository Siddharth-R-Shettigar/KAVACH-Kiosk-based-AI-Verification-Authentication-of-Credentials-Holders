import json
import sys
import exiftool

def run_exif_detector(image_path):
    try:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(image_path)[0]
        
        # Check if standard camera EXIF tags exist
        has_camera_data = "EXIF:Make" in metadata or "EXIF:DateTimeOriginal" in metadata
        
        return {
            "detector_name": "exiftool",
            "score": 0.1 if has_camera_data else 0.85, # Higher score = more suspicious / likely synthetic
            "confidence": "high",
            "explanation": "Original camera EXIF data detected." if has_camera_data else "No camera EXIF metadata found; image may be AI-generated or stripped."
        }
    except Exception as e:
        return {
            "detector_name": "exiftool",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Could not process metadata: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    result = run_exif_detector(target_image)
    print(json.dumps(result, indent=2))