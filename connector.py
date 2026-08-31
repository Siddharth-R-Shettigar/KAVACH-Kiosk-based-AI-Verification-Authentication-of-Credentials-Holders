# connector.py
from bridge import get_all_scores
from analyst import get_verdict


def analyze_image(image_path, preprocess=None):
    print(f"Starting analysis: {image_path}")

    # Get real scores from Siddharth's detection engine
    scores = get_all_scores(image_path)

    print("Scores collected. Sending to AI reasoning layer...")

    # Pass to your AI reasoning layer
    verdict = get_verdict(scores, preprocess)
    return verdict


if __name__ == "__main__":
    import json
    import os
    import sys
    default_path = os.path.join("test_images", "real", "20260804_073446.jpg")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    result = analyze_image(path)
    print(json.dumps(result, indent=2))