# client_runner.py
import requests
import json
import os
import sys
import time

# Add root for module access
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ASR functions from the local models/demo_utils.py file
from models.demo_utils import setup_demo_assets, run_asr_on_file 

# --- CONFIGURATION ---
FASTAPI_ENDPOINT = "http://127.0.0.1:8000/api/utilities/extract-data"
# Ensure this filename matches the audio file you placed in tests/sample_audio/
TEST_AUDIO_FILENAME = "Voice_input.m4a" 
# --- END CONFIGURATION ---


def run_full_extraction_client():
    """
    Simulates the client/frontend by running ASR locally against the audio file
    and sending the resulting text transcript to the live FastAPI backend.
    """
    print("--- 🎙️ STAGE 1: LOCAL ASR (Processing Audio File) ---")
    
    # 1. Setup ASR Models
    assets = setup_demo_assets()
    if not assets.get('asr_available'):
        print("❌ ASR models failed to load. Check installation (Whisper/Dependencies).")
        return

    # 2. Run Local ASR on the file
    # This step uses the code we fixed to run Whisper and Indic Transliteration locally.
    transcript, asr_latency, duration, _ = run_asr_on_file(TEST_AUDIO_FILENAME, assets)
    
    if transcript.startswith("ERROR"):
        print(f"❌ ASR FAILED: {transcript}")
        return

    print(f"✅ Transcript Generated: \"{transcript[:70]}...\"")
    print(f"✅ ASR Latency (Time spent locally): {asr_latency:.4f}s")
    
    # --- STAGE 2: SENDING TRANSCRIPT TO LIVE BACKEND ---
    print("\n--- 🌐 STAGE 2: Sending POST request to FastAPI ---")
    
    payload = {
        "transcript": transcript
    }
    
    start_api_time = time.time()
    
    try:
        # Send the clean text transcript to the live FastAPI endpoint
        response = requests.post(
            FASTAPI_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() 
        
        total_api_latency = time.time() - start_api_time
        response_json = response.json()
        
        # --- STAGE 3: FINAL REPORT ---
        print("\n--- ✅ EXTRACTION SUCCESS (End-to-End) ---")
        print(f"Method Used: {response_json.get('method')}")
        print(f"Total Extraction Latency (API Call): {total_api_latency:.4f}s")
        print(f"Extracted Lead: {response_json['data'].get('lead_name')}")
        print(f"Extracted Date: {response_json['data'].get('date')}")
        print(f"Server Response Code: {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ FATAL ERROR: Could not connect to FastAPI server at {FASTAPI_ENDPOINT}.")
        print("   Ensure 'uvicorn api.main:app --reload' is running in another terminal.")


if __name__ == "__main__":
    # You must run the server in a separate terminal before running this client script.
    run_full_extraction_client()