# kavach/ocr_extractor.py

import os
import re
import cv2
import argparse
import numpy as np
from paddleocr import PaddleOCR

_reader = None

def get_reader(gpu: bool = False):
    """
    Initializes and caches the PaddleOCR reader singleton (v2.8.1).
    """
    global _reader
    if _reader is None:
        _reader = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            use_gpu=gpu, 
            show_log=False
        )
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
            # Real MRZ lines contain multiple chevrons or start with standard doc prefixes (P<, I<, C<, V<, A<)
            if chevron_count >= 2 or cleaned.startswith(('P<', 'I<', 'C<', 'V<', 'A<')):
                mrz_candidates.append(cleaned)

    return mrz_candidates


def extract_text(image_path: str, gpu: bool = False) -> dict:
    """
    Extracts text regions using PaddleOCR, aggregates segmented lines, and isolates MRZ lines.
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
        reader = get_reader(gpu=gpu)
        
        # PaddleOCR handles standard file paths seamlessly and performs internal optimizations
        # ocr() returns a list of results. result[0] contains the actual text blocks.
        raw_results = reader.ocr(image_path, cls=True)
        
        fields = []
        # If text is detected, raw_results[0] will be populated
        if raw_results and raw_results[0]:
            for element in raw_results[0]:
                bbox_polygon = element[0] # Returns 4 coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text = element[1][0]
                confidence = element[1][1]

                text_clean = text.strip()
                if not text_clean:
                    continue

                # Convert polygon back to standard x1, y1, x2, y2 bounding box for downstream logic
                x_coords = [pt[0] for pt in bbox_polygon]
                y_coords = [pt[1] for pt in bbox_polygon]
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
    parser = argparse.ArgumentParser(description="Extract text and MRZ from an image using PaddleOCR.")
    parser.add_argument(
        "-i", "--image", 
        required=True, 
        help="Path to the target image (e.g., test_images/sample.jpg)"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU acceleration for PaddleOCR"
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
        
        # Save visual debug output 
        draw_bounding_boxes(args.image, result)