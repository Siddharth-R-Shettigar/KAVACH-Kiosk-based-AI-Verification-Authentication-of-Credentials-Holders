# kavach/qr_barcode_checker.py

import cv2
from pyzbar.pyzbar import decode
import json


def read_qr_barcodes(image_path: str) -> list:
    """
    Reads all QR codes and barcodes from an image.
    Returns a list of decoded string values.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        decoded_objects = decode(img)
        results = []
        for obj in decoded_objects:
            try:
                data = obj.data.decode("utf-8")
            except Exception:
                data = obj.data.decode("latin-1", errors="replace")
            results.append({
                "type": obj.type,
                "data": data,
                "rect": {
                    "x": obj.rect.left,
                    "y": obj.rect.top,
                    "w": obj.rect.width,
                    "h": obj.rect.height
                }
            })
        return results

    except Exception as e:
        return []


def check_qr_consistency(image_path: str, ocr_fields: dict) -> dict:
    """
    Reads QR/barcode data from the document and compares it
    to the OCR-extracted visible text fields.
    """
    qr_data = read_qr_barcodes(image_path)

    if not qr_data:
        return {
            "status": "unavailable",
            "score": 0.5,
            "confidence": 0.3,
            "qr_found": False,
            "explanation": "No QR code or barcode found in document. This is normal for some document types."
        }

    mismatches = []
    comparisons = []

    for qr_item in qr_data:
        raw = qr_item["data"]

        # Try to parse as JSON (some Indian documents encode JSON in QR)
        try:
            qr_parsed = json.loads(raw)
        except Exception:
            qr_parsed = {}

        # Compare name if present
        if "name" in qr_parsed:
            ocr_name = (ocr_fields.get("surname", "") + " " + ocr_fields.get("given_names", "")).strip().upper()
            qr_name = str(qr_parsed["name"]).upper().replace("<", " ").strip()
            match = qr_name in ocr_name or ocr_name in qr_name
            comparisons.append({"field": "name", "qr": qr_name, "ocr": ocr_name, "match": match})
            if not match:
                mismatches.append("name")

        # Compare DOB if present
        if "dob" in qr_parsed:
            qr_dob = str(qr_parsed["dob"]).replace("-", "").replace("/", "")
            ocr_dob = str(ocr_fields.get("dob", "")).replace("-", "").replace("/", "")
            match = qr_dob == ocr_dob
            comparisons.append({"field": "dob", "qr": qr_dob, "ocr": ocr_dob, "match": match})
            if not match:
                mismatches.append("dob")

    score = 1.0 - (len(mismatches) / max(len(comparisons), 1)) if comparisons else 1.0

    return {
        "status": "passed" if not mismatches else "flagged",
        "score": round(score, 2),
        "confidence": 0.85,
        "qr_found": True,
        "qr_count": len(qr_data),
        "comparisons": comparisons,
        "mismatches": mismatches,
        "explanation": (
            "QR/barcode data matches visible document fields." if not mismatches
            else f"QR/barcode mismatches in: {', '.join(mismatches)}"
        )
    }


if __name__ == "__main__":
    import json as _json
    result = check_qr_consistency("test_passport.jpg", {"surname": "MUKHERJEE", "dob": "800101"})
    print(_json.dumps(result, indent=2))