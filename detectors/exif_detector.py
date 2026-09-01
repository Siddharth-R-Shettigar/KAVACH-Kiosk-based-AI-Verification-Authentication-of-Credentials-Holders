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
            "score": 0.15 if has_camera_data else 0.35,
            "confidence": "medium" if has_camera_data else "low",
            "explanation": (
                "Original camera EXIF data detected."
                if has_camera_data
                else "No camera EXIF metadata found. Common on scans and shared files; weak signal only."
            ),
            "status": "passed" if has_camera_data else "passed"
        }
    except Exception as e:
        return {
            "detector_name": "exiftool",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Could not process metadata: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    result = run_exif_detector(target_image)
    print(json.dumps(result, indent=2))