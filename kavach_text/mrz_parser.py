# kavach/mrz_parser.py
# ICAO 9303 MRZ parser and check digit validator

MRZ_WEIGHTS = [7, 3, 1]

CHAR_VALUES = {str(i): i for i in range(10)}
CHAR_VALUES.update({chr(c): c - 55 for c in range(65, 91)})  # A=10, B=11, ... Z=35
CHAR_VALUES['<'] = 0


def compute_check_digit(s: str) -> int:
    """
    Compute the ICAO 9303 check digit for a given string.
    Each character is multiplied by weights cycling 7,3,1.
    Result mod 10 is the check digit.
    """
    total = 0
    for i, ch in enumerate(s.upper()):
        val = CHAR_VALUES.get(ch, 0)
        total += val * MRZ_WEIGHTS[i % 3]
    return total % 10


def parse_mrz(mrz_lines: list) -> dict:
    """
    Parses MRZ lines from a passport (TD3 format — 2 lines of 44 chars).
    Returns all extracted fields and whether check digits are valid.
    """
    if not mrz_lines or len(mrz_lines) < 2:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "explanation": "MRZ lines not found or incomplete.",
            "fields": {}
        }

    # Clean up lines — remove spaces, ensure uppercase
    line1 = mrz_lines[0].replace(" ", "").upper()
    line2 = mrz_lines[1].replace(" ", "").upper()

    # Pad or trim to 44 characters (TD3 standard)
    line1 = line1[:44].ljust(44, '<')
    line2 = line2[:44].ljust(44, '<')

    issues = []
    fields = {}

    try:
        # --- Line 1 ---
        fields["doc_type"]       = line1[0]       # P = passport
        fields["country_code"]   = line1[2:5]
        raw_name                 = line1[5:44]
        name_parts               = raw_name.split("<<")
        fields["surname"]        = name_parts[0].replace("<", " ").strip() if len(name_parts) > 0 else ""
        fields["given_names"]    = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""

        # --- Line 2 ---
        fields["passport_number"]     = line2[0:9].replace("<", "")
        check_passport                = int(line2[9])
        fields["nationality"]         = line2[10:13]
        fields["dob"]                 = line2[13:19]   # YYMMDD
        check_dob                     = int(line2[19])
        fields["sex"]                 = line2[20]
        fields["expiry"]              = line2[21:27]   # YYMMDD
        check_expiry                  = int(line2[27])
        fields["personal_number"]     = line2[28:42].replace("<", "")
        check_personal                = int(line2[42])
        check_composite               = int(line2[43])

        # --- Check digit validation ---
        computed_passport = compute_check_digit(line2[0:9])
        if computed_passport != check_passport:
            issues.append(f"Passport number check digit FAIL (expected {computed_passport}, got {check_passport})")

        computed_dob = compute_check_digit(line2[13:19])
        if computed_dob != check_dob:
            issues.append(f"DOB check digit FAIL (expected {computed_dob}, got {check_dob})")

        computed_expiry = compute_check_digit(line2[21:27])
        if computed_expiry != check_expiry:
            issues.append(f"Expiry check digit FAIL (expected {computed_expiry}, got {check_expiry})")

        composite_string = line2[0:10] + line2[13:20] + line2[21:43]
        computed_composite = compute_check_digit(composite_string)
        if computed_composite != check_composite:
            issues.append(f"Composite check digit FAIL (expected {computed_composite}, got {check_composite})")

        fields["check_digit_results"] = {
            "passport_number": "PASS" if computed_passport == check_passport else "FAIL",
            "dob":             "PASS" if computed_dob == check_dob else "FAIL",
            "expiry":          "PASS" if computed_expiry == check_expiry else "FAIL",
            "composite":       "PASS" if computed_composite == check_composite else "FAIL",
        }

        all_pass = len(issues) == 0
        score = 1.0 if all_pass else max(0.0, 1.0 - (len(issues) * 0.25))

        return {
            "status": "passed" if all_pass else "flagged",
            "score": round(score, 2),
            "confidence": 0.95,
            "fields": fields,
            "issues": issues,
            "explanation": "All MRZ check digits valid." if all_pass else f"MRZ issues: {'; '.join(issues)}"
        }

    except Exception as e:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "fields": fields,
            "issues": [str(e)],
            "explanation": f"MRZ parsing error: {str(e)}"
        }


if __name__ == "__main__":
    # Example MRZ from a test passport
    line1 = "P<INDMUKHERJEE<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<"
    line2 = "A1234567<3IND8001011M2512315<<<<<<<<<<<<<<<4"
    result = parse_mrz([line1, line2])
    import json
    print(json.dumps(result, indent=2))