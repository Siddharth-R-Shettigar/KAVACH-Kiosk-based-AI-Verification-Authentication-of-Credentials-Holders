import json
import sys
import os
import base64
import requests

def run_vision_llm_inspector(image_path):
    api_key = os.getenv("GEMINI_API_KEY", None)

    if not api_key:
        return {
            "detector_name": "vision_llm_sanity_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": "GEMINI_API_KEY missing. Vision LLM scene inspection bypassed."
        }

    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """
        You are an expert digital forensics visual examiner. Inspect this image for AI generative artifacts:
        1. Human anatomy: count fingers, check hand symmetry, teeth structure, and ear geometry.
        2. Scene logic & semantic sanity: shadows, atmospheric consistency, object perspective, text/logo legibility.
        
        Respond STRICTLY with a raw JSON object (no markdown, no backticks):
        {"score": <float between 0.0 for perfectly natural real photo to 1.0 for synthetic AI>, "explanation": "<1-2 sentence plain text verdict>"}
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
                ]
            }]
        }

        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        res_data = res.json()
        
        raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
        parsed = json.loads(raw_text)

        return {
            "detector_name": "vision_llm_sanity_analysis",
            "score": float(parsed.get("score", 0.5)),
            "confidence": "high",
            "explanation": parsed.get("explanation", "Vision LLM scan complete.")
        }

    except Exception as e:
        return {
            "detector_name": "vision_llm_sanity_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Vision LLM inspection failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_vision_llm_inspector(target), indent=2))