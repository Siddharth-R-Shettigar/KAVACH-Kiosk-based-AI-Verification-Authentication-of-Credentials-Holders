import os
from kavach_engine import analyze_media

TEST_FOLDER = "test_images"

def run_folder(folder, label):
    print(f"\n--- {label} ---")
    if not os.path.isdir(folder):
        print(f"(folder {folder} does not exist yet)")
        return
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        report = analyze_media(path)
        prob = report.get("probabilities", {}).get("synthetic_ai_generated", "N/A")
        print(f"{fname}: predicted synthetic = {prob}")

if __name__ == "__main__":
    run_folder(os.path.join(TEST_FOLDER, "real"), "KNOWN REAL PHOTOS (should score low)")
    run_folder(os.path.join(TEST_FOLDER, "fake"), "KNOWN AI-GENERATED (should score high)")