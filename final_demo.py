import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.demo_utils import setup_demo_assets, run_asr_on_file, generate_voice_confirmation
from main import run_hybrid_extraction_pipeline

TEST_AUDIO_FILENAME = "Voice_input.m4a"
TTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "demo_output.wav")

def user_feedback_for_names(names_detected):
    """
    Prompt user to confirm or correct each detected name entity.
    Returns a dictionary mapping original to corrected names.
    """
    corrections = {}
    for name in set(names_detected):
        print(f"Detected name: '{name}'. Confirm or provide correction (or press Enter to accept):")
        correction = input().strip()
        corrections[name] = correction if correction else name
    return corrections

def apply_corrections_to_transcript(transcript, corrections):
    """
    Replace names in transcript with user corrections.
    """
    for orig_name, corrected_name in corrections.items():
        transcript = transcript.replace(orig_name, corrected_name)
    return transcript

def run_full_voice_demo():
    print("---  VOICE BOT DEMO START ---")

    assets = setup_demo_assets()
    if not assets.get('asr_available'):
        print("\n CRITICAL ERROR: ASR model failed to load. Cannot start demo.")
        return

    print("\n---  STAGE 1: VOICE INPUT & TRANSLITERATION ---")

    transcript, asr_latency, audio_duration, _ = run_asr_on_file(TEST_AUDIO_FILENAME, assets)

    if transcript.startswith("ERROR"):
        print(f" ASR/Transliteration Failed: {transcript}. Check audio file path.")
        return

    # Extract names using NER
    ner = assets.get('ner_pipeline')
    names_detected = []
    if ner:
        ner_results = ner(transcript)
        names_detected = [ent['word'] for ent in ner_results if ent['entity_group'] == 'PER']

    # User feedback to correct names
    if names_detected:
        corrections = user_feedback_for_names(names_detected)
        transcript = apply_corrections_to_transcript(transcript, corrections)

    print(f" ASR Latency: {asr_latency:.4f}s")
    print(f" Final Normalized Transcript (after corrections): \"{transcript}\"")

    print("\n---  STAGE 2: HYBRID EXTRACTION (NLP/MERCURY) ---")

    hybrid_metrics = run_hybrid_extraction_pipeline(transcript)

    if not hybrid_metrics['success']:
        print(" EXTRACTION FAILED: Hybrid pipeline could not extract data.")
        return

    data = hybrid_metrics['data']

    print(f" Extraction Method: {hybrid_metrics['method']} (Latency: {hybrid_metrics['latency_sec']:.4f}s)")
    print(f" Extracted Date: {data.get('date', 'N/A')}, Lead: {data.get('lead_name', 'N/A')}")

    print("\n---  STAGE 3: VOICE OUTPUT (TTS Confirmation) ---")

    data['extraction_method'] = hybrid_metrics['method']

    audio_file = generate_voice_confirmation(data, assets, output_path=TTS_OUTPUT_PATH)

    if audio_file:
        print(f" Confirmation audio saved to {audio_file}")
        print("\nDEMO COMPLETE. Check the project root for 'demo_output.wav'.")
    else:
        print(" TTS failed to generate audio or is disabled.")

if __name__ == "__main__":
    run_full_voice_demo()
