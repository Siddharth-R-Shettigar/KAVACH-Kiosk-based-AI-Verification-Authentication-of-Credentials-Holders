# bridge.py
# Connects the KAVACH engine to the AI reasoning layer

import sys
import os

from kavach_engine import analyze_media # type: ignore


def get_all_scores(image_path):
    """
    Calls Siddharth's detection engine.
    Converts his output into the dictionary analyst.py expects.
    """
    try:
        report = analyze_media(image_path)

        if "error" in report:
            return _fallback_scores(f"KAVACH engine error: {report['error']}")

        # Build lookup by detector name
        signals = report.get("detector_signals", [])
        lookup = {}
        for signal in signals:
            name = signal.get("detector_name", "")
            lookup[name] = signal

        # Map his detectors to your 6 slots
        # Using his actual detector names from the output
        metadata    = _extract(lookup, "exiftool",
                               "ExifTool detector unavailable.")

        ela         = _extract(lookup, "ela_compression",
                               "ELA detector unavailable.")

        frequency   = _extract(lookup, "frequency_domain_fft",
                               "Frequency detector unavailable.")

        trufor      = _extract(lookup, "hf_vision_transformer",
                               "HuggingFace detector unavailable.")

        provenance  = _extract(lookup, "c2pa",
                               "C2PA detector unavailable.")

        pixel_stats = _extract(lookup, "histogram_color_forensics",
                               "Histogram detector unavailable.")

        return {
            "metadata":             metadata["score"],
            "metadata_details":     metadata["details"],

            "ela":                  ela["score"],
            "ela_details":          ela["details"],

            "frequency":            frequency["score"],
            "frequency_details":    frequency["details"],

            "trufor":               trufor["score"],
            "trufor_details":       trufor["details"],

            "provenance":           provenance["score"],
            "provenance_details":   provenance["details"],

            "pixel_stats":          pixel_stats["score"],
            "pixel_stats_details":  pixel_stats["details"],
        }

    except Exception as e:
        return _fallback_scores(f"Bridge failed: {str(e)}")


def _extract(lookup, detector_name, fallback_detail):
    """
    Pulls score and explanation from one detector.
    Converts his 0.0-1.0 scale to your 0-100 scale.
    Low confidence detectors return neutral 50.
    """
    signal = lookup.get(detector_name)

    if signal is None:
        return {"score": 50, "details": fallback_detail}

    # His score: 0.0 = authentic, 1.0 = synthetic
    # Your score: 0 = clean, 100 = suspicious
    # Same direction — just multiply by 100
    raw_score = signal.get("score", 0.5)
    converted = int(round(raw_score * 100))

    explanation = signal.get("explanation", fallback_detail)

    # If confidence is low treat as neutral
    # This prevents skipped detectors from influencing verdict
    if signal.get("confidence") == "low":
        return {
            "score": 50,
            "details": f"Low confidence — {explanation}"
        }

    return {
        "score":   converted,
        "details": explanation
    }


def _fallback_scores(reason):
    """
    Returns neutral scores when something goes wrong.
    Never crashes. Always returns something usable.
    """
    neutral = {"score": 50, "details": reason}
    return {
        "metadata":             50,
        "metadata_details":     reason,
        "ela":                  50,
        "ela_details":          reason,
        "frequency":            50,
        "frequency_details":    reason,
        "trufor":               50,
        "trufor_details":       reason,
        "provenance":           50,
        "provenance_details":   reason,
        "pixel_stats":          50,
        "pixel_stats_details":  reason,
    }


# Test directly
if __name__ == "__main__":
    default_path = os.path.join("test_images", "real", "20260804_073446.jpg")
    test_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    result = get_all_scores(test_path)
    import json
    print(json.dumps(result, indent=2))