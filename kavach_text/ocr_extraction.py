# kavach/ocr_extractor.py

import easyocr
import cv2
import numpy as np
from PIL import Image

# We create the reader once so it doesn't reload the model every time
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        # English only for documents; add 'hi' for Hindi if needed
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_text(image_path: str) -> dict:
    """
    Extracts all text from a document image.
    Returns each detected word with its bounding box and confidence score.
    """
    try:
        reader = get_reader()
        results = reader.readtext(image_path)

        fields = []
        full_text_lines = []

        for (bbox, text, confidence) in results:
            # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))

            fields.append({
                "text": text.strip(),
                "confidence": round(float(confidence), 3),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })
            full_text_lines.append(text.strip())

        full_text = " ".join(full_text_lines)

        # Find the MRZ lines — they are long lines of uppercase letters/numbers/chevrons
        mrz_lines = [
            f["text"] for f in fields
            if len(f["text"]) >= 30 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for c in f["text"].replace(" ", ""))
        ]

        return {
            "status": "passed",
            "score": 1.0,
            "confidence": 1.0,
            "full_text": full_text,
            "fields": fields,
            "mrz_lines": mrz_lines,
            "explanation": f"Extracted {len(fields)} text regions. Found {len(mrz_lines)} potential MRZ lines."
        }

    except Exception as e:
        return {
            "status": "unavailable",
            "score": 0.0,
            "confidence": 0.0,
            "full_text": "",
            "fields": [],
            "mrz_lines": [],
            "explanation": f"OCR failed: {str(e)}"
        }


def draw_bounding_boxes(image_path: str, ocr_result: dict, output_path: str = "ocr_annotated.jpg"):
    """
    Optional: saves a copy of the image with bounding boxes drawn on it.
    Useful for debugging.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    for field in ocr_result.get("fields", []):
        bb = field["bounding_box"]
        color = (0, 255, 0) if field["confidence"] > 0.7 else (0, 165, 255)
        cv2.rectangle(img, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]), color, 2)
        cv2.putText(img, f"{field['confidence']:.2f}", (bb["x1"], bb["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    result = extract_text("test_passport.jpg")
    print("Full text:", result["full_text"])
    print("MRZ lines found:", result["mrz_lines"])
    print("Field count:", len(result["fields"]))
    draw_bounding_boxes("test_passport.jpg", result)