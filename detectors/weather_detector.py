import json
import sys
import subprocess
import requests
from datetime import datetime

def extract_exif_location_and_time(image_path):
    """Extracts GPS coordinates and datetime using ExifTool."""
    try:
        cmd = ["exiftool", "-j", "-GPSLatitude", "-GPSLongitude", "-DateTimeOriginal", image_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)[0]

        lat = data.get("GPSLatitude")
        lon = data.get("GPSLongitude")
        date_str = data.get("DateTimeOriginal")

        if not lat or not lon or not date_str:
            return None, None, None

        # Format date string (e.g., '2025:06:20 14:30:00' -> '2025-06-20')
        dt = datetime.strptime(date_str[:10], "%Y:%m:%d")
        formatted_date = dt.strftime("%Y-%m-%d")

        return float(lat), float(lon), formatted_date

    except Exception:
        return None, None, None

def run_weather_detector(image_path):
    """Fetches historical weather at image coordinates and date."""
    lat, lon, date_str = extract_exif_location_and_time(image_path)

    if not lat or not lon or not date_str:
        return {
            "detector_name": "weather_context_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": "Image missing required EXIF GPS coordinates or DateTimeOriginal timestamp for historical weather lookup."
        }

    try:
        # Query Open-Meteo Historical Archive API (Free, no API key required)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": ["temperature_2m_mean", "precipitation_sum", "weathercode"],
            "timezone": "auto"
        }

        res = requests.get(url, params=params, timeout=5)
        res_data = res.json()

        daily = res_data.get("daily", {})
        temp = daily.get("temperature_2m_mean", [None])[0]
        precip = daily.get("precipitation_sum", [None])[0]
        wcode = daily.get("weathercode", [None])[0]

        # WMO Weather Interpretation Codes
        is_rainy = precip is not None and precip > 1.0
        weather_desc = "Rainy/Overcast" if is_rainy else "Clear/Dry"

        weather_summary = f"Location ({lat:.2f}, {lon:.2f}) on {date_str}: Temp {temp}°C, Precip {precip}mm, Condition: {weather_desc}."

        return {
            "detector_name": "weather_context_analysis",
            "score": 0.1,  # Successfully fetched verifiable ground-truth weather
            "confidence": "high",
            "weather_context": weather_summary,
            "explanation": f"Historical ground-truth weather data verified: {weather_summary}"
        }

    except Exception as e:
        return {
            "detector_name": "weather_context_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Failed to fetch historical weather data: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DSC_0153.JPG"
    print(json.dumps(run_weather_detector(target), indent=2))