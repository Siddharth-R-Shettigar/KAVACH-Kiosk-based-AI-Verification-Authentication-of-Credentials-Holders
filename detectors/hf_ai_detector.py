import os
from PIL import Image
from transformers import pipeline

# Ensemble of lightweight, pre-trained AI image detectors
MODEL_CHECKPOINTS = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector"
]

def run_hf_ai_detector(image_path):
    if not os.path.exists(image_path):
        return {
            "detector_name": "hf_vision_transformer",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"File {image_path} not found."
        }

    try:
        image = Image.open(image_path).convert("RGB")
        scores = []

        for model_name in MODEL_CHECKPOINTS:
            try:
                pipe = pipeline("image-classification", model=model_name)
                predictions = pipe(image)
                
                # Extract score for fake/ai-generated label
                fake_score = 0.0
                for pred in predictions:
                    label = pred["label"].lower()
                    if "artificial" in label or "fake" in label or "ai" in label or "generator" in label:
                        fake_score = max(fake_score, float(pred["score"]))
                
                scores.append(fake_score)
            except Exception:
                continue

        if not scores:
            return {
                "detector_name": "hf_vision_transformer",
                "score": 0.5,
                "confidence": "low",
                "explanation": "Could not process image through Hugging Face pipelines."
            }

        avg_synthetic_score = float(sum(scores) / len(scores))

        return {
            "detector_name": "hf_vision_transformer",
            "score": round(avg_synthetic_score, 2),
            "confidence": "high",
            "explanation": f"Ensemble of {len(scores)} Hugging Face Vision Transformers evaluated synthetic probability at {int(avg_synthetic_score * 100)}%."
        }

    except Exception as e:
        return {
            "detector_name": "hf_vision_transformer",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"HF Vision Transformer error: {str(e)}"
        }