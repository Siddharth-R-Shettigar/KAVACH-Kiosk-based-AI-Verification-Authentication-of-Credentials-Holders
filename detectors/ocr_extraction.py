# kavach/ocr_extractor.py

import os
import re
import cv2
import easyocr
import argparse
import numpy as np
from PIL import Image

_reader = None


def get_reader(gpu: bool = False):
    """
    Initializes and caches the EasyOCR reader singleton.
    """
    global _reader
    if _reader is None:
        # Load English model by default
        _reader = easyocr.Reader(['en'], gpu=gpu)
    return _reader


def clean_mrz_text(raw_text: str) -> str:
    """
    Normalizes common OCR misrecognitions of MRZ chevrons and characters.
    """
    cleaned = raw_text.upper().replace(" ", "")
    # Common OCR substitutions for MRZ chevrons '<'
    cleaned = re.sub(r'[«\(\{\[\<\_\-\–\—\~]', '<', cleaned)
    # Strip any remaining characters outside standard ICAO Doc 9303 alphabet
    cleaned = re.sub(r'[^A-Z0-9<]', '', cleaned)
    return cleaned


def group_text_into_lines(fields: list, y_threshold: int = 12) -> list:
    """
    Groups fragmented bounding boxes that fall on approximately the same vertical line
    and sorts them from left to right.
    """
    if not fields:
        return []

    # Sort boxes primarily by vertical position (y1), secondarily by horizontal (x1)
    sorted_fields = sorted(fields, key=lambda f: (f["bounding_box"]["y1"], f["bounding_box"]["x1"]))
    
    lines = []
    current_line = [sorted_fields[0]]

    for field in sorted_fields[1:]:
        prev_field = current_line[-1]
        prev_y_center = (prev_field["bounding_box"]["y1"] + prev_field["bounding_box"]["y2"]) / 2.0
        curr_y_center = (field["bounding_box"]["y1"] + field["bounding_box"]["y2"]) / 2.0

        if abs(curr_y_center - prev_y_center) <= y_threshold:
            current_line.append(field)
        else:
            # Sort current line horizontally left-to-right
            current_line.sort(key=lambda f: f["bounding_box"]["x1"])
            lines.append(current_line)
            current_line = [field]

    if current_line:
        current_line.sort(key=lambda f: f["bounding_box"]["x1"])
        lines.append(current_line)

    # Merge token segments into unified line strings
    reconstructed_lines = []
    for line in lines:
        line_text = " ".join(f["text"] for f in line).strip()
        if line_text:
            reconstructed_lines.append(line_text)

    return reconstructed_lines


def extract_mrz_candidates(reconstructed_lines: list) -> list:
    """
    Extracts candidate MRZ lines matching ICAO TD1 (30 chars), TD2 (36 chars),
    or TD3 (44 chars) structure with tolerance for minor length variations.
    """
    mrz_candidates = []

    for line in reconstructed_lines:
        cleaned = clean_mrz_text(line)
        
        # Check if string has substantial length and high density of chevrons/uppercase tokens
        if len(cleaned) >= 28:
            chevron_count = cleaned.count('<')
            # Real MRZ lines contain multiple chevrons or start with standard doc prefixes (P<, I<, C<, V<)
            if chevron_count >= 2 or cleaned.startswith(('P<', 'I<', 'C<', 'V<', 'A<')):
                mrz_candidates.append(cleaned)

    return mrz_candidates


def preprocess_for_ocr(image_path: str, scale_factor: float = 2.0):
    """
    Enhances blurry images before sending them to EasyOCR.
    Returns the processed image and the scale factor used.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image at {image_path}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale the image to give OCR more pixels for blurry edges
    upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Apply a sharpening kernel
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    return sharpened, scale_factor


def extract_text(image_path: str, gpu: bool = False) -> dict:
    """
    Extracts text regions, aggregates segmented lines, and isolates MRZ lines.
    """
    if not os.path.exists(image_path):
        return {
            "status": "failed",
            "score": 0.0,
            "confidence": 0.0,
            "full_text": "",
            "lines": [],
            "fields": [],
            "mrz_lines": [],
            "explanation": f"File does not exist: {image_path}"
        }

    try:
        # Preprocess the image
        processed_img, scale = preprocess_for_ocr(image_path, scale_factor=2.0)
        
        reader = get_reader(gpu=gpu)
        results = reader.readtext(processed_img)

        fields = []
        for (bbox, text, confidence) in results:
            text_clean = text.strip()
            if not text_clean:
                continue

            # Scale coordinates back down to match the original image size
            x_coords = [pt[0] / scale for pt in bbox]
            y_coords = [pt[1] / scale for pt in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))

            fields.append({
                "text": text_clean,
                "confidence": round(float(confidence), 3),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

        # Reconstruct fragmented horizontal boxes into coherent lines
        reconstructed_lines = group_text_into_lines(fields)
        full_text = " ".join(reconstructed_lines)

        # Detect candidate MRZ lines
        mrz_candidates = extract_mrz_candidates(reconstructed_lines)

        avg_confidence = (
            round(float(np.mean([f["confidence"] for f in fields])), 3)
            if fields else 0.0
        )

        return {
            "status": "passed",
            "score": 1.0 if fields else 0.0,
            "confidence": avg_confidence,
            "full_text": full_text,
            "lines": reconstructed_lines,
            "fields": fields,
            "mrz_lines": mrz_candidates,
            "explanation": f"Extracted {len(fields)} tokens across {len(reconstructed_lines)} lines. Found {len(mrz_candidates)} MRZ candidates."
        }

    except Exception as e:
        return {
            "status": "failed",
            "score": 0.0,
            "confidence": 0.0,
            "full_text": "",
            "lines": [],
            "fields": [],
            "mrz_lines": [],
            "explanation": f"OCR extraction error: {str(e)}"
        }


def draw_bounding_boxes(image_path: str, ocr_result: dict, output_path: str = "ocr_annotated.jpg") -> None:
    """
    Draws detected word bounding boxes and confidence overlays for visual verification.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    for field in ocr_result.get("fields", []):
        bb = field["bounding_box"]
        conf = field["confidence"]
        
        # Green for high confidence (> 0.7), Amber for medium/low
        color = (0, 255, 0) if conf >= 0.70 else (0, 165, 255)
        
        cv2.rectangle(img, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]), color, 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (bb["x1"], max(bb["y1"] - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, img)
    print(f"Annotated visualization saved to: {output_path}")


if __name__ == "__main__":
    # Setup argparse for robust terminal inputs
    parser = argparse.ArgumentParser(description="Extract text and MRZ from an image using EasyOCR.")
    parser.add_argument(
        "-i", "--image", 
        required=True, 
        help="Path to the target image (e.g., test_images/sample.jpg)"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU acceleration for EasyOCR"
    )
    
    args = parser.parse_args()

    # Pass the arguments to the extraction function
    result = extract_text(args.image, gpu=args.gpu)
    
    print("\n--- OCR EXTRACTION SUMMARY ---")
    if result["status"] == "failed":
        print(f"Extraction Failed: {result['explanation']}")
    else:
        print("Document Text Summary:")
        for idx, line in enumerate(result.get("lines", []), 1):
            print(f"  Line {idx:02d}: {line}")
            
        print("\nMRZ Candidate Lines Found:", result.get("mrz_lines", []))
        print(f"Total Tokens Extracted: {len(result.get('fields', []))}")
        print(f"Average Confidence: {result.get('confidence', 0.0)}")
        
        # Save visual debug output using the original image path
        draw_bounding_boxes(args.image, result)