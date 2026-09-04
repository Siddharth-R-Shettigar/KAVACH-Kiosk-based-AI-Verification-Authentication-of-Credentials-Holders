# detectors/national_id_validator.py

import re


def verhoeff_check(number: str) -> bool:
    d = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8],
        [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],
        [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0]
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0],
        [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5],
        [7,0,4,6,9,1,3,2,5,8]
    ]
    inv = [0,4,3,2,1,9,8,7,6,5]

    c = 0
    digits = [int(x) for x in reversed(number)]
    for i, digit in enumerate(digits):
        c = d[c][p[i % 8][digit]]
    return c == 0


def validate_aadhaar(aadhaar: str) -> dict:
    clean = re.sub(r'[\s-]', '', aadhaar)

    if not re.match(r'^\d{12}$', clean):
        return {
            "id_type": "aadhaar",
            "status": "flagged",
            "score": 0.0,
            "verdict": "INVALID",
            "value": aadhaar,
            "explanation": "Aadhaar must be exactly 12 digits."
        }

    if clean[0] in ('0', '1'):
        return {
            "id_type": "aadhaar",
            "status": "flagged",
            "score": 0.2,
            "verdict": "INVALID",
            "value": aadhaar,
            "explanation": "Aadhaar first digit cannot be 0 or 1."
        }

    valid = verhoeff_check(clean)

    return {
        "id_type": "aadhaar",
        "status": "passed" if valid else "flagged",
        "score": 1.0 if valid else 0.3,
        "verdict": "VALID" if valid else "SUSPICIOUS / INVALID",
        "value": aadhaar,
        "explanation": (
            "Aadhaar number and checksum are valid."
            if valid else
            "Aadhaar checksum failed — number is likely fabricated."
        )
    }


def validate_pan(pan: str) -> dict:
    clean = pan.upper().strip()
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'

    if not re.match(pattern, clean):
        return {
            "id_type": "pan",
            "status": "flagged",
            "score": 0.0,
            "verdict": "INVALID",
            "value": pan,
            "explanation": "PAN format invalid. Expected: 5 letters + 4 digits + 1 letter."
        }

    valid_types = {'P', 'C', 'H', 'F', 'A', 'T', 'B', 'L', 'J', 'G'}
    type_char = clean[3]

    if type_char not in valid_types:
        return {
            "id_type": "pan",
            "status": "flagged",
            "score": 0.4,
            "verdict": "SUSPICIOUS",
            "value": pan,
            "explanation": f"4th character '{type_char}' is not a valid taxpayer type."
        }

    return {
        "id_type": "pan",
        "status": "passed",
        "score": 1.0,
        "verdict": "VALID",
        "value": pan,
        "explanation": f"PAN format is valid. Taxpayer type: '{type_char}'."
    }


def validate_national_id(id_type: str, id_value: str) -> dict:
    id_type = id_type.lower().strip()

    if id_type == "aadhaar":
        return validate_aadhaar(id_value)
    elif id_type == "pan":
        return validate_pan(id_value)
    else:
        return {
            "id_type": id_type,
            "status": "unavailable",
            "score": 0.0,
            "verdict": "UNSUPPORTED",
            "value": id_value,
            "explanation": f"No validator for ID type '{id_type}'."
        }