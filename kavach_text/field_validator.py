# kavach/field_validator.py

import re
from datetime import date

def parse_mrz_date(yymmdd: str, is_expiry: bool = False) -> date | None:
    """
    Convert YYMMDD to a date object.
    - For DOB: assumes 2000s for years <= current_year % 100, else 1900s.
    - For Expiry: assumes 2000s for years up to ~50 years ahead, else 1900s.
    """
    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        current_yy = date.today().year % 100

        if is_expiry:
            # Passport expiry is typically within 10–20 years into the future.
            # If yy <= current_yy + 50, it belongs to the 2000s.
            year = 2000 + yy if yy <= (current_yy + 50) else 1900 + yy
        else:
            # DOB cannot be in the future.
            year = 2000 + yy if yy <= current_yy else 1900 + yy

        return date(year, mm, dd)
    except Exception:
        return None


def validate_passport_number(number: str) -> dict:
    """Indian passport numbers: 1 letter + 7 digits."""
    pattern = r'^[A-Z][0-9]{7}$'
    ok = bool(re.match(pattern, number.upper().replace(" ", "")))
    return {
        "field": "passport_number",
        "status": "passed" if ok else "flagged",
        "value": number,
        "explanation": f"Passport number '{number}' format {'valid' if ok else 'INVALID (expected: 1 letter + 7 digits)'}."
    }


def validate_dob(dob_yymmdd: str) -> dict:
    """DOB must be a real past date."""
    parsed = parse_mrz_date(dob_yymmdd, is_expiry=False)
    if parsed is None:
        return {"field": "dob", "status": "flagged", "value": dob_yymmdd,
                "explanation": f"DOB '{dob_yymmdd}' could not be parsed."}

    today = date.today()
    if parsed >= today:
        return {"field": "dob", "status": "flagged", "value": dob_yymmdd,
                "explanation": f"DOB {parsed} is in the future — invalid."}

    age = (today - parsed).days // 365
    if age > 120:
        return {"field": "dob", "status": "flagged", "value": dob_yymmdd,
                "explanation": f"DOB implies age {age} — implausible."}

    return {"field": "dob", "status": "passed", "value": str(parsed),
            "explanation": f"DOB {parsed} is valid (age ~{age})."}


def validate_expiry(expiry_yymmdd: str) -> dict:
    """Expiry must be a real date (past is allowed but flagged as expired)."""
    parsed = parse_mrz_date(expiry_yymmdd, is_expiry=False)
    if parsed is None:
        return {"field": "expiry", "status": "flagged", "value": expiry_yymmdd,
                "explanation": f"Expiry '{expiry_yymmdd}' could not be parsed."}

    today = date.today()
    if parsed < today:
        return {"field": "expiry", "status": "flagged", "value": str(parsed),
                "explanation": f"Document EXPIRED on {parsed}."}

    return {"field": "expiry", "status": "passed", "value": str(parsed),
            "explanation": f"Document valid until {parsed}."}


def validate_sex(sex: str) -> dict:
    valid = sex.upper() in ("M", "F", "X", "<")
    return {
        "field": "sex",
        "status": "passed" if valid else "flagged",
        "value": sex,
        "explanation": f"Sex field '{sex}' is {'valid' if valid else 'INVALID'}."
    }


def validate_country_code(code: str) -> dict:
    """Country codes are 3 uppercase letters."""
    ok = bool(re.match(r'^[A-Z]{3}$', code.replace("<", "").strip()))
    return {
        "field": "country_code",
        "status": "passed" if ok else "flagged",
        "value": code,
        "explanation": f"Country code '{code}' is {'valid' if ok else 'INVALID'}."
    }


def validate_document_fields(mrz_fields: dict, doc_type: str = "passport") -> dict:
    """
    Runs all field validators on the given MRZ-extracted fields.
    Returns a summary with individual field results.
    """
    results = []

    if doc_type == "passport":
        results.append(validate_passport_number(mrz_fields.get("passport_number", "")))
        results.append(validate_dob(mrz_fields.get("dob", "")))
        results.append(validate_expiry(mrz_fields.get("expiry", "")))
        results.append(validate_sex(mrz_fields.get("sex", "")))
        results.append(validate_country_code(mrz_fields.get("country_code", "")))

    failures = [r for r in results if r["status"] == "flagged"]
    score = 1.0 - (len(failures) / max(len(results), 1))

    return {
        "status": "passed" if not failures else "flagged",
        "score": round(score, 2),
        "confidence": 0.95,
        "field_results": results,
        "failures": [r["field"] for r in failures],
        "explanation": (
            "All document fields valid." if not failures
            else f"Field validation failures: {', '.join(r['field'] for r in failures)}"
        )
    }


if __name__ == "__main__":
    mrz = {
        "passport_number": "A1234567",
        "dob": "800101",
        "expiry": "251231",
        "sex": "M",
        "country_code": "IND"
    }
    import json
    print(json.dumps(validate_document_fields(mrz), indent=2))