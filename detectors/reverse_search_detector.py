import json
import sys
import os
import requests

def run_reverse_search(image_path):
    api_key = os.getenv("TINEYE_API_KEY", None)
    
    if not api_key:
        return {
            "detector_name": "reverse_image_search",
            "score": 0.5,
            "confidence": "low",
            "explanation": "TINEYE_API_KEY environment variable missing. Online context lookup bypassed."
        }

    try:
        # Secure API call using environment variable
        with open(image_path, "rb") as f:
            response = requests.post(
                "https://api.tineye.com/rest/search/",
                headers={"X-API-Key": api_key},
                files={"image": f}
            )

        data = response.json()
        matches = data.get("results", {}).get("matches", [])

        if len(matches) > 0:
            return {
                "detector_name": "reverse_image_search",
                "score": 0.1,  # Matches online = real pre-existing published image
                "confidence": "high",
                "explanation": f"Found {len(matches)} exact or modified visual matches online across indexed web repositories."
            }
        else:
            return {
                "detector_name": "reverse_image_search",
                "score": 0.6,  # Unique image = potential new synthetic generation
                "confidence": "medium",
                "explanation": "Zero web matches found. Image is completely unique or recently generated."
            }

    except Exception as e:
        return {
            "detector_name": "reverse_image_search",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Reverse image search failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_reverse_search(target), indent=2))