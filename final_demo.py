# final_demo.py
import time
import os
import sys

# Ensure the system path finds all modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.demo_utils import setup_demo_assets, run_asr_on_file, generate_voice_confirmation
from main import run_hybrid_extraction_pipeline # Import the core logic from main.py

# --- CONFIGURATION ---
TEST_AUDIO_FILENAME = "Voice input.m4a" # <-- UPDATE THIS FILENAME
TTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "demo_output.wav")
# ---------------------

def run_full_voice_demo():
    """Executes the end-to-end voice bot pipeline."""
    print("---  VOICE BOT DEMO START ---")
    
    # 1. Initialize all models (ASR, Transliteration, TTS)
    assets = setup_demo_assets()
    if not assets.get('asr_client'):
        print("\n CRITICAL ERROR: ASR client failed to load. Cannot start demo.")
        return

    print("\n---  STAGE 1: VOICE INPUT & TRANSLITERATION ---")
    
    # 2. Run ASR on the sample file
    transcript, asr_latency, audio_duration, _ = run_asr_on_file(
        TEST_AUDIO_FILENAME, assets
    )
    
    if transcript.startswith("ERROR"):
        print(f" ASR/Transliteration Failed: {transcript}. Check audio file path.")
        return

    print(f" ASR Latency: {asr_latency:.4f}s")
    print(f" Final Normalized Transcript: \"{transcript}\"")
    
    
    print("\n---  STAGE 2: HYBRID EXTRACTION (NLP/MERCURY) ---")

    # 3. Run Hybrid Extraction Pipeline on the cleaned transcript
    hybrid_metrics = run_hybrid_extraction_pipeline(transcript)

    if not hybrid_metrics['success']:
        print(" EXTRACTION FAILED: Hybrid pipeline could not extract data.")
        return

    data = hybrid_metrics['data']
    
    print(f" Extraction Method: {hybrid_metrics['method']} (Latency: {hybrid_metrics['latency_sec']:.4f}s)")
    print(f" Extracted Date: {data.get('date', 'N/A')}, Lead: {data.get('lead_name', 'N/A')}")
    
    
    print("\n---  STAGE 3: VOICE OUTPUT (TTS Confirmation) ---")

    # 4. Generate Voice Confirmation
    data['extraction_method'] = hybrid_metrics['method'] # Add method to data for TTS feedback
    
    audio_file = generate_voice_confirmation(
        data, assets, output_path=TTS_OUTPUT_PATH
    )
    
    if audio_file:
        print(f" Confirmation audio saved to {audio_file}")
        print("\nDEMO COMPLETE. Check the project root for 'demo_output.wav'.")
    else:
        print(" TTS failed to generate audio.")


if __name__ == "__main__":
    # Ensure this variable matches the path where you saved the audio file name!
    run_full_voice_demo()