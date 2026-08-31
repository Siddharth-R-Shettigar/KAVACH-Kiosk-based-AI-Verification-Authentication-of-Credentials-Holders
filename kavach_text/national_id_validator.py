# kavach/national_id_validator.py

import re


# --- Aadhaar ---

def verhoeff_check(number: str) -> bool:
    """
    Verhoeff algorithm — used by Aadhaar for its check digit.
    Returns True if the number is valid.
    """
    # Multiplication table
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
    # Permutation table
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
    # Inverse table
    inv = [0,4,3,2,1,9,8,7,6,5]

    c = 0
    digits = [int(x) for x in reversed(number)]
    for i, digit in enumerate(digits):
        c = d[c][p[i % 8][digit]]
    return c == 0


def validate_aadhaar(aadhaar: str) -> dict:
    """Validates an Aadhaar number (12 digits)."""
    clean = re.sub(r'[\s-]', '', aadhaar)

    if not re.match(r'^\d{12}$', clean):
        return {
            "id_type": "aadhaar",
            "status": "flagged",
            "value": aadhaar,
            "explanation": "Aadhaar must be exactly 12 digits."
        }

    # First digit cannot be 0 or 1
    if clean[0] in ('0', '1'):
        return {
            "id_type": "aadhaar",
            "status": "flagged",
            "value": aadhaar,
            "explanation": "Aadhaar first digit cannot be 0 or 1."
        }

    valid_checksum = verhoeff_check(clean)

    return {
        "id_type": "aadhaar",
        "status": "passed" if valid_checksum else "flagged",
        "value": aadhaar,
        "explanation": (
            "Aadhaar number format and checksum are valid."
            if valid_checksum
            else "Aadhaar checksum is INVALID — number may be fabricated."
        )
    }


# --- PAN ---

def validate_pan(pan: str) -> dict:
    """
    Validates an Indian PAN card number.
    Format: 5 letters + 4 digits + 1 letter
    4th letter encodes taxpayer type, 5th letter is surname initial.
    """
    clean = pan.upper().strip()
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'

    if not re.match(pattern, clean):
        return {
            "id_type": "pan",
            "status": "flagged",
            "value": pan,
            "explanation": f"PAN '{pan}' format invalid. Expected: 5 letters + 4 digits + 1 letter."
        }

    valid_types = {'P', 'C', 'H', 'F', 'A', 'T', 'B', 'L', 'J', 'G'}
    type_char = clean[3]

    if type_char not in valid_types:
        return {
            "id_type": "pan",
            "status": "flagged",
            "value": pan,
            "explanation": f"PAN 4th character '{type_char}' is not a valid taxpayer type."
        }

    return {
        "id_type": "pan",
        "status": "passed",
        "value": pan,
        "explanation": f"PAN '{pan}' format is valid. Taxpayer type: '{type_char}'."
    }


# --- Router ---

def validate_national_id(id_type: str, id_value: str) -> dict:
    """
    Main entry point. Pass id_type as 'aadhaar' or 'pan'.
    """
    id_type = id_type.lower().strip()

    if id_type == "aadhaar":
        return validate_aadhaar(id_value)
    elif id_type == "pan":
        return validate_pan(id_value)
    else:
        return {
            "id_type": id_type,
            "status": "unavailable",
            "value": id_value,
            "explanation": f"No validator implemented for ID type '{id_type}'."
        }


if __name__ == "__main__":
    import json
    
    # Test valid PAN structure (Individual 'P')
    print(json.dumps(validate_national_id("pan", "ABCPE1234F"), indent=2))