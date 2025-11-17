import os
import time
import re
import torch
import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- GLOBAL SETUP DICTIONARY ---
DEMO_ASSETS = {}

# --- CORE UTILITIES ---

def setup_demo_assets():
    """Load ASR model and processor for local inference, set transliteration flag."""
    global DEMO_ASSETS
    try:
        DEMO_ASSETS['asr_processor'] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        DEMO_ASSETS['asr_model'] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
        DEMO_ASSETS['asr_available'] = True
        print(" Hugging Face Wav2Vec2 ASR Model Loaded Locally.")
    except Exception as e:
        print(f" ERROR loading ASR model: {e}")
        DEMO_ASSETS['asr_available'] = False

    DEMO_ASSETS['xlit_engine_available'] = True
    print(" Indic Transliteration Logic Initialized.")

    DEMO_ASSETS['tts_available'] = False
    print(" TTS functionality disabled due to system download issues.")

    return DEMO_ASSETS

# --- TRANSLITERATION FUNCTION ---
def normalize_transcript_names(transcript: str):
    if not DEMO_ASSETS.get('xlit_engine_available'):
        return transcript

    words = transcript.split()
    normalized_words = []
    SRC_SCHEME = 'itrans'
    TGT_SCHEME = 'itrans'

    for word in words:
        if word[0].isupper() and len(word) > 2 and re.match(r'^[A-Za-z]+$', word):
            normalized_words.append(word.capitalize())
        else:
            normalized_words.append(word)

    return " ".join(normalized_words)

# --- ASR FUNCTION (Using Local Model) ---
def run_asr_on_file(filename: str, assets: dict):
    """Transcribe audio file locally using wav2vec2 model and processor."""
    if not assets.get('asr_available'):
        return "ERROR: ASR Model not initialized.", 0.0, 0.0, 0.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.normpath(os.path.join(script_dir, os.pardir, "tests", "sample_audio", filename))

    if not os.path.exists(audio_file_path):
        return f"ERROR: File not found at {audio_file_path}", 0.0, 0.0, 0.0

    try:
        speech, sampling_rate = sf.read(audio_file_path)
        audio_duration = len(speech) / sampling_rate
    except Exception:
        audio_duration = 0.0

    processor = assets['asr_processor']
    model = assets['asr_model']

    start_time = time.time()

    try:
        input_values = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        latency = time.time() - start_time

        final_transcript = normalize_transcript_names(transcription)

        return final_transcript, latency, audio_duration, audio_duration
    except Exception as e:
        latency = time.time() - start_time
        return f"ERROR: ASR Local Inference Failed. {e}", latency, audio_duration, audio_duration

# --- TTS FUNCTION (Disabled) ---
def generate_voice_confirmation(extracted_data_json: dict, assets: dict, output_path: str = "/tmp/confirmation_output.wav"):
    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")

    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"Processing used the {extracted_data_json.get('extraction_method', 'AI')} path."
    )

    print(f"Bot Confirmation: {confirmation_message}")
    return confirmation_message  # Return text instead of audio file path
