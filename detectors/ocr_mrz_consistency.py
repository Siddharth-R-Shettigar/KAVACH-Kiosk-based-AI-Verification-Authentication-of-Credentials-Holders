# kavach/ocr_mrz_consistency.py

def normalize(text: str) -> str:
    """Strip, uppercase, remove spaces and filler characters."""
    return text.upper().replace("<", "").replace(" ", "").strip()


def compare_fields(ocr_value: str, mrz_value: str, field_name: str) -> dict:
    ocr_n = normalize(ocr_value)
    mrz_n = normalize(mrz_value)

    if not ocr_n or not mrz_n:
        return {
            "field": field_name,
            "status": "unavailable",
            "ocr": ocr_value,
            "mrz": mrz_value,
            "explanation": f"One or both values missing for {field_name}"
        }

    match = ocr_n == mrz_n
    return {
        "field": field_name,
        "status": "passed" if match else "flagged",
        "ocr": ocr_value,
        "mrz": mrz_value,
        "explanation": f"{field_name}: OCR='{ocr_value}' MRZ='{mrz_value}' → {'MATCH' if match else 'MISMATCH'}"
    }


def check_ocr_mrz_consistency(ocr_fields: dict, mrz_fields: dict) -> dict:
    """
    Compares extracted OCR fields against MRZ-decoded fields.

    ocr_fields: dict with keys like 'passport_number', 'dob', 'expiry', 'surname', 'nationality'
    mrz_fields: dict from mrz_parser.parse_mrz()
    """
    comparisons = []
    mismatches = []

    fields_to_check = [
        ("passport_number", "passport_number"),
        ("dob",             "dob"),
        ("expiry",          "expiry"),
        ("surname",         "surname"),
        ("nationality",     "nationality"),
    ]

    for ocr_key, mrz_key in fields_to_check:
        ocr_val = ocr_fields.get(ocr_key, "")
        mrz_val = mrz_fields.get(mrz_key, "")

        result = compare_fields(str(ocr_val), str(mrz_val), ocr_key)
        comparisons.append(result)

        if result["status"] == "flagged":
            mismatches.append(ocr_key)

    score = 1.0 - (len(mismatches) / max(len(comparisons), 1))

    return {
        "status": "passed" if not mismatches else "flagged",
        "score": round(score, 2),
        "confidence": 0.9,
        "mismatches": mismatches,
        "comparisons": comparisons,
        "explanation": (
            "All OCR and MRZ fields match." if not mismatches
            else f"Mismatches found in: {', '.join(mismatches)}"
        )
    }


if __name__ == "__main__":
    ocr = {
        "passport_number": "A1234567",
        "dob": "800101",
        "expiry": "251231",
        "surname": "MUKHERJEE",
        "nationality": "IND"
    }
    mrz = {
        "passport_number": "A1234568",  # deliberately wrong
        "dob": "800101",
        "expiry": "251231",
        "surname": "MUKHERJEE",
        "nationality": "IND"
    }
    import json
    print(json.dumps(check_ocr_mrz_consistency(ocr, mrz), indent=2))