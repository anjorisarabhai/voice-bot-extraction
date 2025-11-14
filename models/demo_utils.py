# models/demo_utils.py
import time
import os
import soundfile as sf
import warnings
import re
import requests 

# Cloud/Service Clients
from huggingface_hub import InferenceClient
from indic_transliteration.sanscript import transliterate, ITRANS # Core Transliteration Logic
from config import HF_API_TOKEN # Imports the secure token from config

# --- CONFIGURATION ---
ASR_MODEL_ID = "facebook/wav2vec2-base-960h" 
HF_API_BASE_URL = "https://api-inference.huggingface.co/models"
# --- END CONFIGURATION ---

# --- GLOBAL SETUP DICTIONARY ---
DEMO_ASSETS = {}


# --- CORE UTILITIES ---

def setup_demo_assets():
    """Initializes clients and checks token availability."""
    
    global DEMO_ASSETS
    
    # 1. ASR Client Setup (Checks Token Security)
    if HF_API_TOKEN and HF_API_TOKEN != "PLACEHOLDER_FOR_SECURITY_CHECK":
        DEMO_ASSETS['asr_available'] = True
        print(" Hugging Face ASR Setup Initialized.")
    else:
        print(" CRITICAL ERROR: HF API Token is missing or invalid. Check your .env file.")
        DEMO_ASSETS['asr_available'] = False
        return DEMO_ASSETS

    # 2. Transliteration Setup 
    DEMO_ASSETS['xlit_engine_available'] = True
    print(" Indic Transliteration Logic Initialized.")
    
    # 3. TTS Model (Disabled)
    DEMO_ASSETS['tts_available'] = False
    print(" TTS functionality disabled due to system download issues.")

    return DEMO_ASSETS


# --- TRANSLITERATION FUNCTION (Logic used for Indian Name Accuracy) ---

def normalize_transcript_names(transcript: str):
    """
    Identifies capitalized words and standardizes their Romanized spelling 
    using the indic-transliteration library.
    """
    if not DEMO_ASSETS.get('xlit_engine_available'):
        return transcript

    words = transcript.split()
    normalized_words = []
    
    SRC_SCHEME = ITRANS 
    TGT_SCHEME = ITRANS 

    for word in words:
        # Heuristic: Only target capitalized words that look like names
        if word[0].isupper() and len(word) > 2 and re.match(r'^[A-Za-z]+$', word):
            try:
                # Use the simple, direct transliterate function
                normalized_word = transliterate(word, SRC_SCHEME, TGT_SCHEME)
                
                if normalized_word and re.match(r'^[A-Za-z\s]+$', normalized_word):
                     normalized_words.append(normalized_word.capitalize())
                     continue
            except Exception:
                pass 

        normalized_words.append(word)
        
    return " ".join(normalized_words)


# --- ASR FUNCTION (FIXED: Direct Requests Call) ---

def run_asr_on_file(filename: str, assets: dict):
    """
    Transcribes audio from a file path using a direct HTTPS call to the 
    Hugging Face Inference API and applies Transliteration.
    """
    if not assets.get('asr_available'):
        return f"ERROR: ASR Client not initialized.", 0.0, 0.0, 0.0

    # 1. Path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.normpath(os.path.join(
        script_dir, os.pardir, "tests", "sample_audio", filename
    ))

    if not os.path.exists(audio_file_path):
        return f"ERROR: File not found at {audio_file_path}", 0.0, 0.0, 0.0
    
    # Get audio duration
    try:
        audio_info = sf.info(audio_file_path)
        audio_duration = audio_info.duration
    except Exception:
        audio_duration = 0.0
        
    start_time = time.time()
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        API_URL = f"{HF_API_BASE_URL}/{ASR_MODEL_ID}"

        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        # CRITICAL FIX: Direct POST request using the robust requests library
        response = requests.post(
            API_URL,
            headers=headers,
            data=audio_bytes,
            timeout=30 # Set a timeout
        )
        response.raise_for_status() # Raise error for bad status codes

        response_json = response.json()
        raw_transcript = response_json.get('text', '').strip()
        
        latency_sec = time.time() - start_time
        
        # 2. Apply Transliteration (Name Normalization)
        final_transcript = normalize_transcript_names(raw_transcript)
        
        return final_transcript, latency_sec, audio_duration, audio_duration
    
    except Exception as e:
        latency_sec = time.time() - start_time
        return f"ERROR: API Call Failed. {e}", latency_sec, audio_duration, audio_duration


# --- TTS FUNCTION (Disabled) ---

def generate_voice_confirmation(extracted_data_json: dict, assets: dict, output_path: str = "/tmp/confirmation_output.wav"):
    """Generates a text confirmation as TTS functionality is disabled."""
    
    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")
    
    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"Processing used the {extracted_data_json.get('extraction_method', 'AI')} path."
    )
    
    print(f"Bot Confirmation: {confirmation_message}")
    return confirmation_message # Returns text instead of file path