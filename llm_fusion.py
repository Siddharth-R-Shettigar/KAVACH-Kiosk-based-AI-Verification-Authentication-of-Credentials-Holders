import os
import json
import requests


def generate_human_summary(report: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY", None)
    if not api_key:
        return "GEMINI_API_KEY missing; skipping AI-generated summary."

    prompt = f"""
You are a digital forensics analyst. Below is a JSON report containing scores (0.0 = authentic,
1.0 = synthetic/manipulated) and plain-language explanations from multiple independent,
imperfect image-forensics detectors.

Some detectors are more reliable than others, and any single detector can be wrong on a given
image. Weigh corroborating evidence more than a single outlier signal. Note any detectors that
failed, returned low confidence, or were skipped. Explain your reasoning in plain language a
non-technical person can understand.

Report:
{json.dumps(report, indent=2)}

Write a short (4-6 sentence) plain-English verdict: is this image likely authentic, likely
AI-generated/manipulated, or inconclusive — and why, citing the 2-3 most important pieces
of evidence.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        res_data = res.json()
        return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"LLM summary generation failed: {e}"