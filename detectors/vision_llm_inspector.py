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
            "explanation": "GEMINI_API_KEY missing. Vision LLM scene inspection bypassed.",
            "status": "unavailable"
        }

    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """
        You are a border-control document examiner. Inspect this identity document image for tampering or fraud risk.
        Look for: replaced or pasted photograph, retyped or misaligned text, altered dates or numbers,
        suspicious stamps, layout inconsistencies, or signs the page was digitally edited.
        Do NOT focus on general AI-art artifacts (fingers, fantasy scenes) unless relevant to an ID photo.

        Respond STRICTLY with a raw JSON object (no markdown, no backticks):
        {"score": <float 0.0 = looks genuine to 1.0 = strong tampering risk>, "explanation": "<1-2 sentence plain text>"}
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
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

        if "candidates" not in res_data:
            # Show the real API error instead of crashing on a missing key
            return {
                "detector_name": "vision_llm_sanity_analysis",
                "score": 0.5,
                "confidence": "low",
                "explanation": f"Gemini API returned no candidates: {json.dumps(res_data)[:300]}",
                "status": "failed"
            }

        raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_text)

        return {
            "detector_name": "vision_llm_sanity_analysis",
            "score": float(parsed.get("score", 0.5)),
            "confidence": "high",
            "explanation": parsed.get("explanation", "Vision LLM scan complete."),
            "status": "passed" if float(parsed.get("score", 0.5)) < 0.4 else "flagged"
        }

    except Exception as e:
        return {
            "detector_name": "vision_llm_sanity_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Vision LLM inspection failed: {str(e)}",
            "status": "failed"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_vision_llm_inspector(target), indent=2))