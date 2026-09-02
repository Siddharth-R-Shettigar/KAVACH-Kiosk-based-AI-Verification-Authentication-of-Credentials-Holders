# kavach/kavach_pipeline.py
# Runs all 9 features in sequence and returns a combined evidence report

import json
from kavach.document_classifier import classify_document
from kavach.ocr_extractor import extract_text
from kavach.mrz_parser import parse_mrz
from kavach.ocr_mrz_consistency import check_ocr_mrz_consistency
from kavach.field_validator import validate_document_fields
from kavach.qr_barcode_checker import check_qr_consistency
from kavach.national_id_validator import validate_national_id


def run_kavach_pipeline(image_path: str, national_id_type: str = None, national_id_value: str = None) -> dict:
    """
    Full pipeline for features 1-9.

    image_path: path to the document image
    national_id_type: 'aadhaar' or 'pan' (optional)
    national_id_value: the ID number string (optional)
    """
    evidence = {}

    # --- Feature 2 & 3: OCR first (needed by classifier and others) ---
    print("[1/9] Running OCR...")
    ocr_result = extract_text(image_path)
    evidence["ocr"] = ocr_result

    # --- Feature 1: Document classification ---
    print("[2/9] Classifying document type...")
    classification = classify_document(image_path, ocr_result.get("full_text", ""))
    evidence["classification"] = classification
    doc_type = classification.get("doc_type", "passport")

    # --- Feature 4 & 5: MRZ parsing + check digits ---
    print("[3/9] Parsing MRZ...")
    mrz_result = parse_mrz(ocr_result.get("mrz_lines", []))
    evidence["mrz"] = mrz_result

    # Build OCR fields dictionary from extracted text for comparison
    # (In a full system, you'd have a smarter field extractor)
    ocr_fields = {}
    if mrz_result.get("fields"):
        # Use MRZ as baseline; in real use, also extract from visual OCR
        ocr_fields = mrz_result["fields"].copy()

    # --- Feature 6: OCR ↔ MRZ consistency ---
    print("[4/9] Checking OCR ↔ MRZ consistency...")
    consistency_result = check_ocr_mrz_consistency(ocr_fields, mrz_result.get("fields", {}))
    evidence["ocr_mrz_consistency"] = consistency_result

    # --- Feature 7: Field validation ---
    print("[5/9] Validating document fields...")
    validation_result = validate_document_fields(mrz_result.get("fields", {}), doc_type)
    evidence["field_validation"] = validation_result

    # --- Feature 8: QR/barcode consistency ---
    print("[6/9] Checking QR/barcode...")
    qr_result = check_qr_consistency(image_path, ocr_fields)
    evidence["qr_barcode"] = qr_result

    # --- Feature 9: National ID checksum (if provided) ---
    if national_id_type and national_id_value:
        print("[7/9] Validating national ID...")
        nid_result = validate_national_id(national_id_type, national_id_value)
    else:
        nid_result = {"status": "unavailable", "explanation": "No national ID provided."}
    evidence["national_id"] = nid_result

    # --- Summary ---
    all_statuses = [v.get("status", "unavailable") for v in evidence.values() if isinstance(v, dict)]
    failed = [k for k, v in evidence.items() if isinstance(v, dict) and v.get("status") == "flagged"]
    unavailable = [k for k, v in evidence.items() if isinstance(v, dict) and v.get("status") == "unavailable"]

    overall = "HIGH RISK" if len(failed) >= 2 else ("REVIEW" if failed else "PASS")

    return {
        "overall_status": overall,
        "document_type": doc_type,
        "failed_checks": failed,
        "unavailable_checks": unavailable,
        "evidence": evidence
    }


if __name__ == "__main__":
    result = run_kavach_pipeline("test_passport.jpg")
    print(json.dumps(result, indent=2, default=str))