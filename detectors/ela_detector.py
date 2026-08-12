import json
import sys
import os
import tempfile
from PIL import Image, ImageChops, ImageStat

def run_ela_detector(image_path, quality=95):
    temp_filename = None
    try:
        original = Image.open(image_path).convert("RGB")

        fd, temp_filename = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        original.save(temp_filename, "JPEG", quality=quality)
        resaved = Image.open(temp_filename)
        ela_image = ImageChops.difference(original, resaved)

        stat = ImageStat.Stat(ela_image)
        mean_diff = sum(stat.mean) / len(stat.mean)

        score = round(min(mean_diff / 15.0, 1.0), 2)

        return {
            "detector_name": "ela_compression",
            "score": score,
            "confidence": "medium",
            "explanation": f"Error Level Analysis computed pixel difference variance of {round(mean_diff, 2)}."
        }

    except Exception as e:
        return {
            "detector_name": "ela_compression",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"ELA calculation failed: {str(e)}"
        }
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_ela_detector(target_image)
    print(json.dumps(result, indent=2))