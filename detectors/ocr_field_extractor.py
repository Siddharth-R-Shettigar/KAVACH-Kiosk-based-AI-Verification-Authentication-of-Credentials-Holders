# detectors/ocr_field_extractor.py
# Pulls structured fields out of raw OCR token list for non-MRZ documents

import re
from datetime import datetime

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}


def normalize_date(raw: str) -> str:
    """
    Convert human-readable date to YYMMDD.
    Handles: 'JAN 1981', '30 NOV 2009', '29 NOV 2019'
    Returns '' if unparseable.
    """
    raw = raw.upper().strip()

    # Format: MON YYYY (DOB) → just YYMM01 approximate
    m = re.match(r'^([A-Z]{3})\s+(\d{4})$', raw)
    if m:
        mon = MONTH_MAP.get(m.group(1), "01")
        yy = m.group(2)[2:]
        return f"{yy}{mon}01"

    # Format: DD MON YYYY
    m = re.match(r'^(\d{1,2})\s+([A-Z]{3})\s+(\d{4})$', raw)
    if m:
        dd = m.group(1).zfill(2)
        mon = MONTH_MAP.get(m.group(2), "01")
        yy = m.group(3)[2:]
        return f"{yy}{mon}{dd}"

    return ""


def extract_fields_from_ocr(ocr_result: dict) -> dict:
    """
    Extracts structured document fields from the OCR token list.
    Works for documents without MRZ (passport cards, ID cards, licences).

    Returns a dict compatible with what mrz_parser would return in 'fields'.
    """
    fields_list = ocr_result.get("fields", [])
    lines = ocr_result.get("lines", [])
    extracted = {}

    texts = [f["text"].strip() for f in fields_list]
    full = " ".join(texts).upper()

    # ── Surname ──────────────────────────────────────────────────────────────
    for i, t in enumerate(texts):
        if "SURNAME" in t.upper() and i + 1 < len(texts):
            # Next token after label is the value
            candidate = texts[i + 1].upper().strip()
            if candidate and not any(skip in candidate for skip in ["GIVEN", "NAME", "SEX", "DATE"]):
                extracted["surname"] = candidate
                break

    # ── Given names ──────────────────────────────────────────────────────────
    for i, t in enumerate(texts):
        if "GIVEN" in t.upper() and i + 1 < len(texts):
            candidate = texts[i + 1].upper().strip()
            if candidate and len(candidate) > 1:
                extracted["given_names"] = candidate
                break

    # ── Sex ──────────────────────────────────────────────────────────────────
    for i, t in enumerate(texts):
        if t.upper() == "SEX" and i + 1 < len(texts):
            val = texts[i + 1].upper().strip()
            if val in ("M", "F", "X"):
                extracted["sex"] = val
                break
        # Sometimes M appears directly next to sex label
        if t.upper() in ("M", "F") and i > 0 and "SEX" in texts[i-1].upper():
            extracted["sex"] = t.upper()
            break

    # ── DOB ──────────────────────────────────────────────────────────────────
    # Look for pattern like "JAN 1981" or "01 JAN 1981"
    dob_pattern = re.search(
        r'\b([A-Z]{3}\s+\d{4}|\d{1,2}\s+[A-Z]{3}\s+\d{4})\b', full
    )
    if dob_pattern:
        extracted["dob"] = normalize_date(dob_pattern.group(1))

    # ── Expiry ───────────────────────────────────────────────────────────────
    # Look for "Expires On DD MON YYYY" — find the date after "EXPIRES"
    expires_idx = next((i for i, t in enumerate(texts) if "EXPIRE" in t.upper()), None)
    if expires_idx is not None:
        # Check next 1-2 tokens for a date
        for j in range(expires_idx + 1, min(expires_idx + 3, len(texts))):
            candidate = texts[j].upper().strip()
            d = normalize_date(candidate)
            if d:
                extracted["expiry"] = d
                break
            # Handle "29 NOV 20" (truncated) — just store what we have
            if re.match(r'\d{1,2}\s+[A-Z]{3}', candidate):
                extracted["expiry"] = candidate  # partial, better than nothing
                break

    # ── Nationality / Country code ────────────────────────────────────────────
    for i, t in enumerate(texts):
        if "NATIONALITY" in t.upper() and i + 1 < len(texts):
            val = texts[i + 1].upper().strip()
            if re.match(r'^[A-Z]{2,3}$', val):
                extracted["country_code"] = val
                extracted["nationality"] = val
                break

    # ── Passport / Card number ────────────────────────────────────────────────
    for i, t in enumerate(texts):
        if any(kw in t.upper() for kw in ["PASSPORT CARD NO", "CARD NO", "PASSPORT NO"]):
            if i + 1 < len(texts):
                val = texts[i + 1].replace(" ", "").upper()
                extracted["passport_number"] = val
                break
    # Fallback: look for pattern like C03005988
    if "passport_number" not in extracted:
        for t in texts:
            if re.match(r'^[A-Z]\d{7,8}$', t.replace(" ", "").upper()):
                extracted["passport_number"] = t.upper()
                break

    return extracted