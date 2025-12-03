# final_demo.py
import os
import sys
import re 
import time 
import json # Added for general use

# CRITICAL: Path appending allows modules to be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.demo_utils import setup_demo_assets, run_asr_on_file, generate_voice_confirmation
from main import run_hybrid_extraction_pipeline 

# --- CONFIGURATION ---
TEST_AUDIO_FILENAME = "Voice_input.m4a"
TTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "demo_output.mp3")
# --- END CONFIGURATION ---


def user_feedback_for_names(names_detected):
    """
    Prompts user to confirm or correct each detected name entity.
    Returns a dictionary mapping original (misspelled) to corrected names.
    """
    corrections = {}
    print("\n--- 📝 USER CORRECTION REQUIRED ---")
    for name in set(names_detected):
        print(f"Name detected: '{name}'. Enter correct spelling or press Enter to accept (Case Sensitive):")
        # Use input() to pause the execution and wait for the user
        correction = input().strip()
        # Ensure the corrected name is used if provided, otherwise use the detected name
        corrections[name] = correction if correction else name
    print("------------------------------------")
    return corrections

def apply_corrections_to_transcript(transcript, corrections):
    """
    Replace names in transcript with user corrections.
    """
    for orig_name, corrected_name in corrections.items():
        # Use regex to replace whole words to ensure accurate substitution
        transcript = re.sub(r'\b' + re.escape(orig_name) + r'\b', corrected_name, transcript, flags=re.IGNORECASE)
    return transcript

def run_full_voice_demo():
    print("---  VOICE BOT DEMO START (Interactive) ---")

    # 1. Setup Models (ASR, NER, Transliteration)
    assets = setup_demo_assets()
    if not assets.get('asr_available'):
        print("\n❌ CRITICAL ERROR: ASR model failed to load. Cannot start demo.")
        return

    print("\n---  STAGE 1: VOICE INPUT & TRANSLITERATION ---")

    # 2. Run ASR
    transcript, asr_latency, audio_duration, _ = run_asr_on_file(TEST_AUDIO_FILENAME, assets)

    if transcript.startswith("ERROR"):
        print(f"❌ ASR/Transliteration Failed: {transcript}. Check audio file path.")
        return

    print(f"✅ Raw ASR Output: \"{transcript}\"")

    # 3. INTERACTIVE CORRECTION LOOP (Human-in-the-Loop)
    ner = assets.get('ner_pipeline')
    names_detected = []
    
    if ner:
        # Step A: Detect Person Entities (Names)
        ner_results = ner(transcript)
        # Only extract the 'PER' (Person) entities
        names_detected = [ent['word'].strip() for ent in ner_results if ent['entity_group'] == 'PER']
        print(f"🟡 NER Detected Names: {names_detected}")
    else:
        print("🟡 WARNING: NER Pipeline not loaded. Skipping automatic name detection.")
    
    if names_detected:
        # Step B: Get User Input
        correction_start_time = time.time()
        corrections = user_feedback_for_names(names_detected)
        correction_time = time.time() - correction_start_time
        
        # Step C: Apply corrections to the transcript before extraction
        transcript = apply_corrections_to_transcript(transcript, corrections)
        print(f"✅ Transcript after User Correction: \"{transcript}\" (Correction Time: {correction_time:.2f}s)")

    else:
         # Fallback logic if NER fails but we want to proceed
         print("Proceeding to extraction with Transliterated transcript...")


    print("\n---  STAGE 2: HYBRID EXTRACTION (NLP/MERCURY) ---")

    # 4. Run Hybrid Extraction
    hybrid_metrics = run_hybrid_extraction_pipeline(transcript)

    if not hybrid_metrics['success']:
        print("❌ EXTRACTION FAILED: Hybrid pipeline could not extract data.")
        return

    data = hybrid_metrics['data']

    print(f"✅ Extraction Method: {hybrid_metrics['method']} (Latency: {hybrid_metrics['latency_sec']:.4f}s)")
    print(f"✅ Extracted Date: {data.get('date', 'N/A')}, Lead: {data.get('lead_name', 'N/A')}")

    print("\n---  STAGE 3: VOICE OUTPUT (TTS Confirmation) ---")

    # 5. Generate Voice Confirmation
    data['extraction_method'] = hybrid_metrics['method']
    audio_file = generate_voice_confirmation(data, assets, output_path=TTS_OUTPUT_PATH)

    if audio_file:
        print(f"✅ Confirmation audio saved to {audio_file}")
    else:
        print("TTS confirmation printed to console.")


if __name__ == "__main__":
    run_full_voice_demo()