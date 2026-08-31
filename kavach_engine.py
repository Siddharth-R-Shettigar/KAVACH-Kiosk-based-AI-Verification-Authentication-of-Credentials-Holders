import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def _safe_import(module_path, func_name):
    """Import a detector function without crashing the whole engine if it fails."""
    try:
        module = __import__(module_path, fromlist=[func_name])
        return getattr(module, func_name)
    except Exception as e:
        print(f"[WARNING] Could not load {module_path}.{func_name}: {e}", file=sys.stderr)
        return None

run_exif_detector = _safe_import("detectors.exif_detector", "run_exif_detector")
run_c2pa_detector = _safe_import("detectors.c2pa_detector", "run_c2pa_detector")
run_ela_detector = _safe_import("detectors.ela_detector", "run_ela_detector")
run_jpeg_ghost_detector = _safe_import("detectors.jpeg_ghost", "run_jpeg_ghost_detector")
run_phash_detector = _safe_import("detectors.phash_detector", "run_phash_detector")
run_histogram_detector = _safe_import("detectors.histogram_detector", "run_histogram_detector")
run_frequency_detector = _safe_import("detectors.frequency_detector", "run_frequency_detector")
run_hf_ai_detector = _safe_import("detectors.hf_ai_detector", "run_hf_ai_detector")
run_copy_move_detector = _safe_import("detectors.copy_move_detector", "run_copy_move_detector")
run_blur_detector = _safe_import("detectors.blur_detector", "run_blur_detector")
run_cfa_detector = _safe_import("detectors.cfa_detector", "run_cfa_detector")
run_resampling_detector = _safe_import("detectors.resampling_detector", "run_resampling_detector")
run_illuminant_detector = _safe_import("detectors.illuminant_detector", "run_illuminant_detector")
run_quantization_detector = _safe_import("detectors.quantization_detector", "run_quantization_detector")
run_geometry_detector = _safe_import("detectors.geometry_detector", "run_geometry_detector")
run_weather_detector = _safe_import("detectors.weather_detector", "run_weather_detector")
run_inpainting_detector = _safe_import("detectors.inpainting_detector", "run_inpainting_detector")
run_vision_llm_inspector = _safe_import("detectors.vision_llm_inspector", "run_vision_llm_inspector")
run_reverse_search = _safe_import("detectors.reverse_search_detector", "run_reverse_search")
run_audio_spoof_detector = _safe_import("detectors.audio_spoof_detector", "run_audio_spoof_detector")
run_video_optical_flow = _safe_import("detectors.video_optical_flow", "run_video_optical_flow")
run_face_verification = _safe_import("detectors.face_verification_engine", "run_face_verification")


def analyze_media(image_path):
    if not os.path.exists(image_path):
        return {"error": f"File {image_path} not found."}

    ext = os.path.splitext(image_path)[1].lower()
    is_jpeg = ext in ['.jpg', '.jpeg']

    # (function, weight, requires_jpeg)
    # NOTE: weights below are a starting point, not calibrated. See Fix 6 (calibration_test.py)
    # before trusting these numbers for a real demo.
    detector_pipeline = [
        (run_exif_detector, 1.5, False),
        (run_c2pa_detector, 0.0, False),          # neutral when missing; excluded from weighted mean
        (run_cfa_detector, 2.0, False),
        (run_hf_ai_detector, 2.0, False),
        (run_resampling_detector, 1.5, False),
        (run_ela_detector, 1.0, False),
        (run_histogram_detector, 0.8, False),
        (run_frequency_detector, 0.8, False),
        (run_copy_move_detector, 1.0, False),
        (run_blur_detector, 0.8, False),
        (run_illuminant_detector, 1.0, False),
        (run_phash_detector, 0.0, False),         # hash generation only, not a fakeness signal
        (run_jpeg_ghost_detector, 1.2, True),
        (run_quantization_detector, 1.2, True),
        (run_geometry_detector, 0.7, False),
        (run_weather_detector, 0.0, False),       # informational only for now — see note in Fix 5
        (run_inpainting_detector, 1.0, False),
        (run_vision_llm_inspector, 1.5, False),   # requires GEMINI_API_KEY
        (run_reverse_search, 0.5, False),         # requires TINEYE_API_KEY
    ]

    results = []
    weighted_score_sum = 0.0
    total_weight = 0.0

    for detector_fn, weight, req_jpeg in detector_pipeline:
        if detector_fn is None:
            continue  # this detector's module failed to import — skip it, don't crash
        if req_jpeg and not is_jpeg:
            continue

        try:
            res = detector_fn(image_path)
        except Exception as e:
            res = {
                "detector_name": getattr(detector_fn, "__name__", "unknown_detector"),
                "score": 0.5,
                "confidence": "low",
                "explanation": f"Detector crashed during execution: {e}"
            }

        results.append(res)

        if weight > 0.0 and "score" in res and res.get("confidence") != "low":
            weighted_score_sum += res["score"] * weight
            total_weight += weight

    final_synthetic_score = weighted_score_sum / max(total_weight, 1.0)
    synthetic_prob = int(round(final_synthetic_score * 100))
    authentic_prob = max(0, 100 - synthetic_prob)

    return {
        "engine": "KAVACH (Verifiable Evidence & Digital Authenticity)",
        "file_analyzed": os.path.basename(image_path),
        "probabilities": {
            "authentic_capture": f"{authentic_prob}%",
            "synthetic_ai_generated": f"{synthetic_prob}%"
        },
        "overall_confidence": "high",
        "active_detectors_evaluated": len(results),
        "detector_signals": results
    }

def analyze_video(video_path):
    if not os.path.exists(video_path):
        return {"error": f"File {video_path} not found."}
    if run_video_optical_flow is None:
        return {"error": "video_optical_flow detector failed to load."}

    result = run_video_optical_flow(video_path)
    score = result.get("score", 0.5)
    synthetic_prob = int(round(score * 100))
    authentic_prob = max(0, 100 - synthetic_prob)

    return {
        "engine": "KAVACH (Verifiable Evidence & Digital Authenticity)",
        "file_analyzed": os.path.basename(video_path),
        "probabilities": {
            "authentic_capture": f"{authentic_prob}%",
            "synthetic_ai_generated": f"{synthetic_prob}%"
        },
        "overall_confidence": result.get("confidence", "low"),
        "active_detectors_evaluated": 1,
        "detector_signals": [result]
    }


def analyze_audio(audio_path):
    if not os.path.exists(audio_path):
        return {"error": f"File {audio_path} not found."}
    if run_audio_spoof_detector is None:
        return {"error": "audio_spoof_detector failed to load."}

    result = run_audio_spoof_detector(audio_path)
    score = result.get("score", 0.5)
    synthetic_prob = int(round(score * 100))
    authentic_prob = max(0, 100 - synthetic_prob)

    return {
        "engine": "KAVACH (Verifiable Evidence & Digital Authenticity)",
        "file_analyzed": os.path.basename(audio_path),
        "probabilities": {
            "authentic_capture": f"{authentic_prob}%",
            "synthetic_ai_generated": f"{synthetic_prob}%"
        },
        "overall_confidence": result.get("confidence", "low"),
        "active_detectors_evaluated": 1,
        "detector_signals": [result]
    }


def analyze_file(path):
    """Routes any uploaded file to the right pipeline based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov"}
    audio_exts = {".wav"}  # add ".mp3" here only if you do the MP3 upgrade below

    if ext in image_exts:
        return analyze_media(path)
    elif ext in video_exts:
        return analyze_video(path)
    elif ext in audio_exts:
        return analyze_audio(path)
    else:
        return {"error": f"Unsupported file type: {ext}"}
    

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    report = analyze_file(target)

    try:
        from llm_fusion import generate_human_summary
        report["human_summary"] = generate_human_summary(report)
    except Exception as e:
        report["human_summary"] = f"(AI summary unavailable: {e})"

    print(json.dumps(report, indent=2))

    # Save a copy of every analysis to disk, one file per case
    os.makedirs("case_logs", exist_ok=True)
    safe_name = os.path.splitext(os.path.basename(target))[0]
    log_path = os.path.join("case_logs", f"{safe_name}_report.json")
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved case log to {log_path}", file=sys.stderr)
