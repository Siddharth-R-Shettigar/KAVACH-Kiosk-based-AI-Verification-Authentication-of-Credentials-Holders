# kavach/document_classifier.py

import cv2
import numpy as np
from PIL import Image

# This tells us what kind of document we are looking at.
# We look at the shape (aspect ratio) and key text clues found by OCR.

DOCUMENT_TYPES = {
    "passport":  {"width_ratio": (1.3, 1.5), "keywords": ["passport", "nationality", "mrz", "p<"]},
    "visa":      {"width_ratio": (1.3, 1.6), "keywords": ["visa", "entry", "stay", "valid for"]},
    "id_card":   {"width_ratio": (1.5, 1.7), "keywords": ["identity", "aadhaar", "pan", "voter"]},
    "licence":   {"width_ratio": (1.5, 1.7), "keywords": ["driving", "licence", "license", "transport"]},
}

def classify_document(image_path: str, ocr_text: str = "") -> dict:
    """
    Given a document image path and any OCR text already extracted,
    returns what type of document it most likely is.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"doc_type": "unknown", "confidence": 0.0, "explanation": "Could not read image"}

        h, w = img.shape[:2]
        ratio = w / h if h > 0 else 1.0
        ocr_lower = ocr_text.lower()

        scores = {}
        for doc_type, rules in DOCUMENT_TYPES.items():
            score = 0.0
            min_r, max_r = rules["width_ratio"]

            # Check shape
            if min_r <= ratio <= max_r:
                score += 0.4

            # Check keywords found in OCR text
            for kw in rules["keywords"]:
                if kw in ocr_lower:
                    score += 0.2

            scores[doc_type] = min(score, 1.0)

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < 0.2:
            best_type = "unknown"

        return {
            "doc_type": best_type,
            "confidence": round(best_score, 2),
            "all_scores": scores,
            "explanation": f"Classified as {best_type} based on shape ratio {ratio:.2f} and keyword matches."
        }

    except Exception as e:
        return {
            "doc_type": "unknown",
            "confidence": 0.0,
            "explanation": f"Error during classification: {str(e)}"
        }


if __name__ == "__main__":
    # Quick test — replace with your image path
    result = classify_document("test_passport.jpg", ocr_text="PASSPORT Nationality Indian MRZ")
    print(result)
