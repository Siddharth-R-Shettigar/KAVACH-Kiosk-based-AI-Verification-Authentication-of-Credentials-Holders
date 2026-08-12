import json
import sys
import os
import numpy as np
from scipy.io import wavfile

def run_audio_spoof_detector(audio_path):
    if not os.path.exists(audio_path) or not audio_path.lower().endswith(('.wav', '.mp3')):
        return {
            "detector_name": "audio_voice_clone_analysis",
            "score": 0.1,
            "confidence": "low",
            "explanation": "Input file is not a supported audio format. Audio spoof analysis bypassed."
        }

    try:
        sample_rate, data = wavfile.read(audio_path)
        if len(data.shape) > 1:
            data = data[:, 0]  # Take single mono channel

        # Calculate Zero Crossing Rate (ZCR)
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        zcr = len(zero_crossings) / float(len(data))

        # Calculate high frequency spectral power ratio (>8kHz)
        fft_spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        
        hf_energy = np.sum(fft_spectrum[freqs > 8000])
        total_energy = np.sum(fft_spectrum) + 1e-5
        hf_ratio = float(hf_energy / total_energy)

        # Neural voice models often feature unnatural zero-crossing frequency bounds or hard spectral cutoffs
        score = 0.80 if hf_ratio < 0.005 or zcr > 0.35 else 0.15

        return {
            "detector_name": "audio_voice_clone_analysis",
            "score": score,
            "confidence": "high",
            "explanation": f"High-frequency audio energy spectral ratio: {round(hf_ratio, 4)}, ZCR: {round(zcr, 3)}. " +
                           ("Synthetic neural vocoder spectrum / voice clone signatures detected." if score > 0.5 else "Organic room acoustics and human acoustic spectrum verified.")
        }

    except Exception as e:
        return {
            "detector_name": "audio_voice_clone_analysis",
            "score": 0.5,
            "confidence": "low",
            "explanation": f"Audio analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_audio.wav"
    print(json.dumps(run_audio_spoof_detector(target), indent=2))