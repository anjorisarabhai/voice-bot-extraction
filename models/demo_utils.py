# models/demo_utils.py
import time
import os
import warnings
import re
import numpy as np
from pydub import AudioSegment 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI 
import io # NEW: Needed to handle raw audio bytes

# Imports from config (Necessary to resolve the previous ImportError)
from config import ELEVENLABS_API_KEY 

# --- CONFIGURATION ---
ASR_MODEL_ID = "openai/whisper-large-v3"
# --- END CONFIGURATION ---

# --- GLOBAL SETUP DICTIONARY ---
DEMO_ASSETS = {}


# --- CORE UTILITIES ---

def setup_demo_assets():
    """Initializes ASR models and checks availability."""
    
    global DEMO_ASSETS
    
    # 1. ASR Setup (Local Whisper)
    try:
        DEMO_ASSETS['asr_processor'] = WhisperProcessor.from_pretrained(ASR_MODEL_ID)
        DEMO_ASSETS['asr_model'] = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_ID)
        DEMO_ASSETS['asr_available'] = True
        print("✅ Hugging Face Whisper ASR Model Loaded Locally.")
    except Exception as e:
        DEMO_ASSETS['asr_available'] = False
        print(f"❌ ERROR loading ASR model (Whisper): {e}")

    # 2. Transliteration Setup 
    DEMO_ASSETS['xlit_engine_available'] = True
    print("✅ Indic Transliteration Logic Initialized.")
    
    # 3. NER Pipeline Setup (Placeholder, removed for core ASR stability)
    DEMO_ASSETS['ner_pipeline'] = None
    
    # 4. TTS Setup (Disabled)
    DEMO_ASSETS['tts_available'] = False
    print("❌ TTS functionality disabled.")

    return DEMO_ASSETS


# --- TRANSLITERATION FUNCTION ---

def normalize_transcript_names(transcript: str):
    """
    Standardizes Romanized spelling of proper nouns using the Devanagari bridge.
    """
    if not DEMO_ASSETS.get('xlit_engine_available'):
        return transcript

    words = transcript.split()
    normalized_words = []
    
    SRC_SCHEME = ITRANS 
    TGT_SCHEME = ITRANS 
    DEVANAGARI_SCHEME = DEVANAGARI 

    for word in words:
        if word[0].isupper() and len(word) > 2 and re.match(r'^[A-Za-z]+$', word):
            try:
                devanagari_word = transliterate(word, SRC_SCHEME, DEVANAGARI_SCHEME)
                normalized_word = transliterate(devanagari_word, DEVANAGARI_SCHEME, TGT_SCHEME)
                
                if normalized_word and re.match(r'^[A-Za-z\s]+$', normalized_word):
                     normalized_words.append(normalized_word.capitalize())
                     continue
            except Exception:
                pass 

        normalized_words.append(word)
        
    return " ".join(normalized_words)


# --- ASR FUNCTION (CRITICAL: Handles Bytes Input) ---

def run_asr_on_bytes(audio_bytes: bytes, assets: dict):
    """
    Transcribes audio bytes received directly from the FastAPI upload endpoint.
    """
    if not assets.get('asr_available'):
        return "ERROR: ASR Model not initialized.", 0.0, 0.0, 0.0

    start_time = time.time()
    
    # 1. Process Raw Bytes
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io) 
        
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        speech = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sampling_rate = 16000
        audio_duration = len(speech) / sampling_rate
        
    except Exception as e:
        return f"ERROR: Failed to process audio bytes (pydub/ffmpeg error): {e}", 0.0, 0.0, 0.0
    
    processor = assets['asr_processor']
    model = assets['asr_model']

    try:
        # ASR Inference
        input_features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_features
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - start_time
        
        # Apply Transliteration
        final_transcript = normalize_transcript_names(transcription)
        
        return final_transcript, latency, audio_duration, audio_duration
    except Exception as e:
        latency = time.time() - start_time
        return f"ERROR: ASR Local Inference Failed. {e}", latency, audio_duration, audio_duration


# --- TTS FUNCTION (Disabled) ---

def generate_voice_confirmation(extracted_data_json: dict, assets: dict, output_path: str = "demo_output.mp3"):
    """Generates a text confirmation as TTS functionality is disabled."""
    
    if not assets.get('tts_available'):
        print("Warning: TTS functionality is disabled, confirmation printed as text.")
        
    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")
    
    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"This was processed by the {extracted_data_json.get('extraction_method', 'AI')} system."
    )
    
    print(f"Bot Confirmation: {confirmation_message}")
    return confirmation_message