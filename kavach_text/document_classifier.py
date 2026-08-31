# kavach/document_classifier.py

import re
import cv2
from typing import Dict, Any

# Document definitions: aspect ratios (normalized) and keyword patterns
DOCUMENT_TYPES = {
    "passport": {
        "width_ratio": (1.3, 1.55),
        "keywords": [r"\bpassport\b", r"\bnationality\b", r"\bmrz\b", r"p<"]
    },
    "visa": {
        "width_ratio": (1.3, 1.6),
        "keywords": [r"\bvisa\b", r"\bentry\b", r"\bstay\b", r"valid\s+for"]
    },
    "id_card": {
        "width_ratio": (1.5, 1.7),
        "keywords": [r"\bidentity\b", r"\baadhaar\b", r"\bpan\b", r"\bvoter\b", r"\belectoral\b"]
    },
    "licence": {
        "width_ratio": (1.5, 1.7),
        "keywords": [r"\bdriving\b", r"\blicence\b", r"\blicense\b", r"\btransport\b", r"\bdriver\b"]
    },
}

def classify_document(image_path: str, ocr_text: str = "") -> Dict[str, Any]:
    """
    Given a document image path and OCR text, returns the predicted document type,
    confidence score, and a breakdown of scores.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "doc_type": "unknown",
                "confidence": 0.0,
                "explanation": "Could not read image file or path is invalid."
            }

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return {
                "doc_type": "unknown",
                "confidence": 0.0,
                "explanation": "Invalid image dimensions."
            }

        # Orientation-agnostic aspect ratio
        norm_ratio = max(w, h) / min(w, h)
        ocr_lower = ocr_text.lower()

        scores = {}
        for doc_type, rules in DOCUMENT_TYPES.items():
            score = 0.0
            min_r, max_r = rules["width_ratio"]

            # Aspect ratio match
            if min_r <= norm_ratio <= max_r:
                score += 0.35

            # Regex keyword matching
            matched_keywords = 0
            for pattern in rules["keywords"]:
                if re.search(pattern, ocr_lower):
                    matched_keywords += 1

            # Scale keyword contribution
            score += min(matched_keywords * 0.25, 0.65)
            scores[doc_type] = round(min(score, 1.0), 2)

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < 0.3:
            best_type = "unknown"

        return {
            "doc_type": best_type,
            "confidence": best_score,
            "all_scores": scores,
            "explanation": f"Classified as {best_type} (aspect ratio: {norm_ratio:.2f}, confidence: {best_score})."
        }

    except Exception as e:
        return {
            "doc_type": "unknown",
            "confidence": 0.0,
            "explanation": f"Error during classification: {str(e)}"
        }


if __name__ == "__main__":
    result = classify_document(
        "/workspaces/VEDA-Verifiable-Evidence-Digital-Authenticity/test_images/fake/U.S._passport_card.jpg",
        ocr_text="PASSPORT Nationality Indian MRZ P<IND"
    )
    print(result)