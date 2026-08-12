import json
import sys
import imagehash
from PIL import Image

def run_phash_detector(image_path):
    try:
        img = Image.open(image_path)
        
        phash_val = str(imagehash.phash(img))
        dhash_val = str(imagehash.dhash(img))
        ahash_val = str(imagehash.average_hash(img))
        
        return {
            "detector_name": "perceptual_hashing",
            "score": 0.1,
            "confidence": "high",
            "explanation": f"Generated perceptual signatures for duplicate cross-referencing.",
            "hashes": {
                "phash": phash_val,
                "dhash": dhash_val,
                "ahash": ahash_val
            }
        }
    except Exception as e:
        return {
            "detector_name": "perceptual_hashing",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Perceptual hash computation failed: {str(e)}"
        }

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "Group_6.png"
    result = run_phash_detector(target_image)
    print(json.dumps(result, indent=2))