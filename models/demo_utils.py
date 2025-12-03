# models/demo_utils.py
import time
import os
import soundfile as sf
import warnings
import re
import numpy as np
from pydub import AudioSegment 
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from indic_transliteration.sanscript import transliterate, ITRANS 

# --- CRITICAL ELEVENLABS IMPORTS ---
from elevenlabs.client import ElevenLabs 
from elevenlabs.types import Voice as ElevenLabsVoice 
# --- END CRITICAL IMPORTS ---

# Imports from config
from config import ELEVENLABS_API_KEY 

# --- CONFIGURATION ---
ASR_MODEL_ID = "openai/whisper-base"
# --- END CONFIGURATION ---

# --- GLOBAL SETUP DICTIONARY ---
DEMO_ASSETS = {}


# --- CORE UTILITIES ---

def setup_demo_assets():
    """Initializes clients and checks token availability."""
    
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
    
    # 3. ElevenLabs TTS Setup
    if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "PLACEHOLDER_FOR_SECURITY_CHECK":
        # Pass the key directly to the client constructor
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        DEMO_ASSETS['tts_client'] = elevenlabs_client
        DEMO_ASSETS['tts_available'] = True
        # CRITICAL FIX: Using a universally available Voice ID ('Adam')
        MONIKA_VOICE_ID = "1qEiC6qsybMkmnNdVMbK" 
        DEMO_ASSETS['tts_voice'] = ElevenLabsVoice(voice_id=MONIKA_VOICE_ID, name="Monika") 
        print("✅ ElevenLabs TTS Client Initialized.")
    else:
        DEMO_ASSETS['tts_available'] = False
        print("❌ ElevenLabs TTS disabled: API key missing or invalid.")

    return DEMO_ASSETS


# --- TRANSLITERATION FUNCTION ---

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
        if word[0].isupper() and len(word) > 2 and re.match(r'^[A-Za-z]+$', word):
            try:
                normalized_word = transliterate(word, SRC_SCHEME, TGT_SCHEME)
                
                # Check for successful output before appending
                if normalized_word and re.match(r'^[A-Za-z\s]+$', normalized_word):
                     normalized_words.append(normalized_word.capitalize())
                     continue
            except Exception:
                pass 

        normalized_words.append(word)
        
    return " ".join(normalized_words)


# --- ASR FUNCTION ---

def run_asr_on_file(filename: str, assets: dict):
    """
    Transcribes audio from a file path using the local Whisper model 
    and applies Indic Transliteration.
    """
    if not assets.get('asr_available'):
        return "ERROR: ASR Model not initialized.", 0.0, 0.0, 0.0

    # 1. Path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.normpath(os.path.join(
        script_dir, os.pardir, "tests", "sample_audio", filename
    ))

    if not os.path.exists(audio_file_path):
        return f"ERROR: File not found at {audio_file_path}", 0.0, 0.0, 0.0
    
    # 2. Audio loading (Requires pydub/ffmpeg setup)
    try:
        audio = AudioSegment.from_file(audio_file_path)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        sampling_rate = 16000
        speech = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    except Exception as e:
        return f"ERROR: Failed to read audio file: {e}", 0.0, 0.0, 0.0
    
    processor = assets['asr_processor']
    model = assets['asr_model']

    start_time = time.time()
    try:
        # ASR Inference
        input_features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_features
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        latency = time.time() - start_time
        
        # Apply Transliteration
        final_transcript = normalize_transcript_names(transcription)
        
        return final_transcript, latency, len(speech) / sampling_rate, len(speech) / sampling_rate
    except Exception as e:
        latency = time.time() - start_time
        return f"ERROR: ASR Local Inference Failed. {e}", latency, len(speech) / sampling_rate, len(speech) / sampling_rate


# --- TTS FUNCTION (ElevenLabs API) ---

def generate_voice_confirmation(extracted_data_json: dict, assets: dict, output_path: str = "demo_output.mp3"):
    """Generates high-quality voice confirmation using the ElevenLabs API."""
    
    if not assets.get('tts_available'):
        print("Warning: TTS functionality is disabled, confirmation printed as text.")
        return None

    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")
    
    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"This was processed by the {extracted_data_json.get('extraction_method', 'AI')} system."
    )
    
    print(f"Bot Confirmation: {confirmation_message}")
    
    try:
        # Generate audio via ElevenLabs API (returns a generator/stream)
        audio = assets['tts_client'].generate(
            text=confirmation_message,
            voice=assets['tts_voice'],
            model="eleven_multilingual_v2" 
        )
        
        # --- CRITICAL FIX: CONSOLIDATE AUDIO STREAM ---
        if isinstance(audio, (bytes, bytearray)):
            audio_bytes = audio
        else:
            # If it's a generator/stream, consolidate the chunks into a single bytes object
            audio_bytes = b"".join(audio) 
        # --- END CRITICAL FIX ---
        
        # Save the file
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
            
        print(f"✅ High-quality audio saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to generate TTS audio via ElevenLabs API: {e}")
        return None