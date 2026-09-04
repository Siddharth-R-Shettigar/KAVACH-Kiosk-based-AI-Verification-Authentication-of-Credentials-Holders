# kavach/document_classifier.py

import re
import cv2
from typing import Dict, Any

DOCUMENT_TYPES = {
    "passport": {
        "keywords": [
            r"\bpassport\b",
            r"\bnationality\b",
            r"p<ind",
            r"p<",
            r"\brepublic\s+of\s+india\b",
            r"\bicao\b",
            r"\bmachine\s+readable\b",
        ],
    },
    "aadhaar": {
        "keywords": [
            r"\baadhaar\b",
            r"\buidai\b",
            r"\bunique\s+identification\b",
            r"\baadhaar\s+number\b",
            r"\byour\s+aadhaar\b",
            r"\benrollment\s+no\b",
            r"\bvid\b",
        ],
    },
    "pan": {
        "keywords": [
            r"\bpermanent\s+account\s+number\b",
            r"\bincome\s+tax\s+department\b",
            r"\bincome.tax\b",
            r"\bpan\b",
            r"\bgovt\.?\s+of\s+india\b",
        ],
    },
    "voter_id": {
        "keywords": [
            r"\belection\s+commission\b",
            r"\bepic\b",
            r"\belectoral\b",
            r"\bvoter\b",
            r"\belectors?\s+photo\b",
        ],
    },
    "driving_licence": {
        "keywords": [
            r"\bdriving\s+licen[cs]e\b",
            r"\bdriving\s+licence\b",
            r"\bdl\s*no\b",
            r"\bmotor\s+vehicles?\s+act\b",
            r"\btransport\s+department\b",
            r"\bsarathi\b",
        ],
    },
    "visa": {
        "keywords": [
            r"\bvisa\b",
            r"\bport\s+of\s+entry\b",
            r"\bvalid\s+for\s+stay\b",
            r"\bnumber\s+of\s+entries\b",
            r"\bplace\s+of\s+issue\b",
        ],
    },
}

# How much each keyword match contributes to the score
KEYWORD_WEIGHT = 0.22

# Minimum score to make a call — below this → unknown
MIN_SCORE = 0.3


def classify_document(image_path: str, ocr_text: str = "") -> Dict[str, Any]:
    try:
        # We still read the image to confirm it's valid,
        # but we no longer use aspect ratio for scoring.
        img = cv2.imread(image_path)
        if img is None:
            return {
                "detector_name": "document_classifier",
                "doc_type": "unknown",
                "score": 0.0,
                "confidence": "low",
                "explanation": "Could not read image file.",
                "status": "failed",
            }

        ocr_lower = ocr_text.lower()
        scores: Dict[str, float] = {}
        keyword_hits: Dict[str, list] = {}

        for doc_type, rules in DOCUMENT_TYPES.items():
            matched = [p for p in rules["keywords"] if re.search(p, ocr_lower)]
            # Cap at 1.0, but even 2-3 strong keyword hits should be decisive
            score = round(min(len(matched) * KEYWORD_WEIGHT, 1.0), 3)
            scores[doc_type] = score
            keyword_hits[doc_type] = matched

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < MIN_SCORE:
            best_type = "unknown"

        if best_score >= 0.65:
            conf = "high"
        elif best_score >= 0.4:
            conf = "medium"
        else:
            conf = "low"

        return {
            "detector_name": "document_classifier",
            "doc_type": best_type,
            "score": float(best_score),
            "confidence": conf,
            "all_scores": scores,
            "keyword_hits": {k: v for k, v in keyword_hits.items() if v},
            "explanation": f"Classified as '{best_type}' (score {best_score}). "
                           f"Matched {len(keyword_hits.get(best_type, []))} keyword(s).",
            "status": "passed" if best_type != "unknown" else "unavailable",
        }

    except Exception as e:
        return {
            "detector_name": "document_classifier",
            "doc_type": "unknown",
            "score": 0.0,
            "confidence": "low",
            "explanation": f"Error: {str(e)}",
            "status": "failed",
        }


if __name__ == "__main__":
    import json

    test_cases = [
        ("passport",         "PASSPORT Republic of India Nationality IND MRZ P<IND ICAO machine readable"),
        ("aadhaar",          "Aadhaar UIDAI Unique Identification Authority of India VID enrollment no"),
        ("pan",              "Permanent Account Number Income Tax Department Govt. of India"),
        ("voter_id",         "Election Commission of India EPIC Electors Photo Identity Card voter"),
        ("driving_licence",  "Driving Licence DL No Motor Vehicles Act Transport Department Sarathi"),
        ("visa",             "VISA port of entry valid for stay number of entries place of issue"),
        ("unknown",          "some random text that matches nothing at all"),
    ]

    print(f"{'Expected':<20} {'Got':<20} {'Score':<8} {'Hits'}")
    print("-" * 70)
    for expected, ocr in test_cases:
        # pass a dummy path — image check will fail gracefully for non-existent files
        result = classify_document("dummy.jpg", ocr_text=ocr)
        got = result["doc_type"]
        score = result["score"]
        hits = sum(len(v) for v in result.get("keyword_hits", {}).values())
        status = "✓" if got == expected else "✗"
        print(f"{status} {expected:<18} {got:<20} {score:<8} {hits} keyword(s)")