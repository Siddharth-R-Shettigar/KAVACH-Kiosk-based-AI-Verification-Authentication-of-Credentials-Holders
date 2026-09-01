# kavach/mrz_parser.py
# ICAO 9303 MRZ parser and check digit validator

from datetime import datetime

MRZ_WEIGHTS = [7, 3, 1]

CHAR_VALUES = {str(i): i for i in range(10)}
CHAR_VALUES.update({chr(c): c - 55 for c in range(65, 91)})  # A=10, B=11, ... Z=35
CHAR_VALUES["<"] = 0

ALLOWED_MRZ_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")

# Number of independent check-digit tests performed, used for scoring.
TOTAL_CHECK_DIGIT_TESTS = 5


def compute_check_digit(s: str) -> int:
    """
    Compute the ICAO 9303 check digit for a given string.
    Each character is multiplied by weights cycling 7,3,1.
    Result mod 10 is the check digit.
    """
    total = 0

    for i, ch in enumerate(s.upper()):
        if ch not in CHAR_VALUES:
            raise ValueError(f"Invalid MRZ character: '{ch}'")

        val = CHAR_VALUES[ch]
        total += val * MRZ_WEIGHTS[i % 3]

    return total % 10


def _validate_date(yymmdd: str) -> bool:
    """
    Confirms a YYMMDD MRZ date field is an actual calendar date.
    Does not attempt century disambiguation — only checks the
    day/month/year digits form a real date.
    """
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return False

    try:
        datetime.strptime(yymmdd, "%y%m%d")
        return True
    except ValueError:
        return False


def parse_mrz(mrz_lines: list) -> dict:
    """
    Parses MRZ lines from a passport (TD3 format — 2 lines of 44 chars).
    Returns all extracted fields and whether check digits are valid.
    """

    # ---------------------------------------------------------
    # Check that both MRZ lines exist
    # ---------------------------------------------------------
    if not mrz_lines or len(mrz_lines) < 2:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "explanation": "MRZ lines not found or incomplete.",
            "fields": {},
            "issues": ["Two MRZ lines are required."]
        }

    # Clean up lines — remove spaces, ensure uppercase
    line1 = mrz_lines[0].replace(" ", "").upper()
    line2 = mrz_lines[1].replace(" ", "").upper()

    # ---------------------------------------------------------
    # TD3 passport MRZ requires exactly 44 characters per line.
    # Do not silently pad incomplete OCR output.
    # ---------------------------------------------------------
    if len(line1) != 44 or len(line2) != 44:
        issues = []

        if len(line1) != 44:
            issues.append(
                f"MRZ line 1 has {len(line1)} characters; expected 44."
            )

        if len(line2) != 44:
            issues.append(
                f"MRZ line 2 has {len(line2)} characters; expected 44."
            )

        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "explanation": "MRZ lines are incomplete or invalid length.",
            "fields": {},
            "issues": issues
        }

    issues = []
    fields = {}

    # -----------------------------------------------------
    # Validate MRZ characters BEFORE attempting any check-digit
    # math. compute_check_digit() raises on an invalid character,
    # which would otherwise jump straight to the outer except
    # block and discard these detailed, position-by-position
    # messages in favor of one vague exception string.
    # -----------------------------------------------------
    for line_number, line in enumerate([line1, line2], start=1):
        for position, ch in enumerate(line, start=1):
            if ch not in ALLOWED_MRZ_CHARS:
                issues.append(
                    f"Invalid MRZ character '{ch}' "
                    f"at line {line_number}, position {position}."
                )

    if issues:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "fields": fields,
            "issues": issues,
            "explanation": (
                f"MRZ contains invalid characters: {'; '.join(issues)}"
            )
        }

    try:
        # -----------------------------------------------------
        # Line 1
        # -----------------------------------------------------

        fields["doc_type"] = line1[0]       # P = passport
        fields["country_code"] = line1[2:5]

        if fields["doc_type"] != "P":
            issues.append(
                f"Unexpected document type '{fields['doc_type']}' "
                f"(expected 'P' for a TD3 passport MRZ)."
            )

        raw_name = line1[5:44]
        name_parts = raw_name.split("<<", 1)

        fields["surname"] = (
            name_parts[0].replace("<", " ").strip()
            if len(name_parts) > 0
            else ""
        )

        fields["given_names"] = (
            name_parts[1].replace("<", " ").strip()
            if len(name_parts) > 1
            else ""
        )

        # -----------------------------------------------------
        # Line 2
        # -----------------------------------------------------

        fields["passport_number"] = line2[0:9].replace("<", "")
        check_passport = int(line2[9])

        fields["nationality"] = line2[10:13]

        fields["dob"] = line2[13:19]   # YYMMDD
        check_dob = int(line2[19])

        fields["sex"] = line2[20]

        fields["expiry"] = line2[21:27]   # YYMMDD
        check_expiry = int(line2[27])

        fields["personal_number"] = line2[28:42].replace("<", "")

        # Position 43 may be a digit OR '<' when the optional
        # personal-data field is completely unused.
        check_personal_char = line2[42]

        check_composite = int(line2[43])

        # -----------------------------------------------------
        # Check digit validation
        # -----------------------------------------------------

        # Passport number
        computed_passport = compute_check_digit(line2[0:9])

        if computed_passport != check_passport:
            issues.append(
                f"Passport number check digit FAIL "
                f"(expected {computed_passport}, got {check_passport})"
            )

        # Date of birth
        computed_dob = compute_check_digit(line2[13:19])

        if computed_dob != check_dob:
            issues.append(
                f"DOB check digit FAIL "
                f"(expected {computed_dob}, got {check_dob})"
            )

        # Expiry date
        computed_expiry = compute_check_digit(line2[21:27])

        if computed_expiry != check_expiry:
            issues.append(
                f"Expiry check digit FAIL "
                f"(expected {computed_expiry}, got {check_expiry})"
            )

        # -----------------------------------------------------
        # Calendar-date sanity checks
        # (check digits only prove internal consistency, not
        # that the digits form a real date)
        # -----------------------------------------------------

        dob_is_valid_date = _validate_date(fields["dob"])
        if not dob_is_valid_date:
            issues.append(
                f"DOB field '{fields['dob']}' is not a valid calendar date."
            )

        expiry_is_valid_date = _validate_date(fields["expiry"])
        if not expiry_is_valid_date:
            issues.append(
                f"Expiry field '{fields['expiry']}' is not a valid calendar date."
            )

        # -----------------------------------------------------
        # Personal / optional data check digit
        # -----------------------------------------------------

        personal_data = line2[28:42]

        computed_personal = compute_check_digit(personal_data)

        if (
            check_personal_char == "<"
            and set(personal_data) == {"<"}
        ):
            # ICAO 9303 allows '<' here when the optional
            # personal-data field is completely unused.
            personal_pass = True

        elif check_personal_char.isdigit():
            check_personal = int(check_personal_char)

            personal_pass = (
                computed_personal == check_personal
            )

            if not personal_pass:
                issues.append(
                    f"Personal number check digit FAIL "
                    f"(expected {computed_personal}, "
                    f"got {check_personal})"
                )

        else:
            personal_pass = False

            issues.append(
                f"Invalid personal-data check digit "
                f"'{check_personal_char}'"
            )

        # -----------------------------------------------------
        # Composite check digit
        # -----------------------------------------------------

        composite_string = (
            line2[0:10]
            + line2[13:20]
            + line2[21:43]
        )

        computed_composite = compute_check_digit(
            composite_string
        )

        if computed_composite != check_composite:
            issues.append(
                f"Composite check digit FAIL "
                f"(expected {computed_composite}, got {check_composite})"
            )

        # -----------------------------------------------------
        # Store results
        # -----------------------------------------------------

        fields["check_digit_results"] = {
            "passport_number":
                "PASS"
                if computed_passport == check_passport
                else "FAIL",

            "dob":
                "PASS"
                if computed_dob == check_dob
                else "FAIL",

            "expiry":
                "PASS"
                if computed_expiry == check_expiry
                else "FAIL",

            "personal_number":
                "PASS"
                if personal_pass
                else "FAIL",

            "composite":
                "PASS"
                if computed_composite == check_composite
                else "FAIL",
        }

        fields["date_validity_results"] = {
            "dob": "PASS" if dob_is_valid_date else "FAIL",
            "expiry": "PASS" if expiry_is_valid_date else "FAIL",
        }

        # -----------------------------------------------------
        # Score
        # -----------------------------------------------------
        # Score reflects ALL checks performed — check digits,
        # calendar-date validity, and doc-type correctness —
        # so it can never disagree with "status"/"issues".

        check_results = fields["check_digit_results"].values()
        passed_checks = sum(result == "PASS" for result in check_results)

        date_results = fields["date_validity_results"].values()
        passed_dates = sum(result == "PASS" for result in date_results)

        doc_type_ok = fields["doc_type"] == "P"

        passed_total = passed_checks + passed_dates + int(doc_type_ok)
        total_total = TOTAL_CHECK_DIGIT_TESTS + len(date_results) + 1

        score = passed_total / total_total

        all_pass = len(issues) == 0

        # Confidence scales down slightly as more issues pile up,
        # rather than always reporting the same fixed value
        # regardless of how flagged the result is.
        confidence = round(max(0.5, 0.95 - 0.05 * len(issues)), 2)

        return {
            "status": "passed" if all_pass else "flagged",
            "score": round(score, 2),
            "confidence": confidence,
            "fields": fields,
            "issues": issues,
            "explanation": (
                "All MRZ check digits valid."
                if all_pass
                else f"MRZ issues: {'; '.join(issues)}"
            )
        }

    except (ValueError, TypeError, IndexError) as e:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "fields": fields,
            "issues": [str(e)],
            "explanation": f"MRZ parsing error: {str(e)}"
        }


if __name__ == "__main__":
    # Synthetic TD3 MRZ for testing.
    # Not a real person's passport.

    line1 = "P<INDMUKHERJEE<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<"
    line2 = "A1234567<6IND8001014M2512314<<<<<<<<<<<<<<04"

    result = parse_mrz([line1, line2])

    import json
    print(json.dumps(result, indent=2))

    print("\n--- Test: invalid character (should not lose detail) ---")
    bad_line2 = "A123456#<6IND8001014M2512314<<<<<<<<<<<<<<04"
    print(json.dumps(parse_mrz([line1, bad_line2]), indent=2))

    print("\n--- Test: invalid calendar date (month 13) ---")
    bad_date_line2 = "A1234567<6IND8013014M2512314<<<<<<<<<<<<<<04"
    print(json.dumps(parse_mrz([line1, bad_date_line2]), indent=2))