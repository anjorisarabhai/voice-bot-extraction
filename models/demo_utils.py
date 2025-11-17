import os
import time
import re
import torch
import numpy as np
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from indic_transliteration.sanscript import transliterate, ITRANS
from gtts import gTTS

# --- GLOBAL SETUP DICTIONARY ---
DEMO_ASSETS = {}

# --- CORE UTILITIES ---

def setup_demo_assets():
    global DEMO_ASSETS
    try:
        # ASR
        DEMO_ASSETS['asr_processor'] = WhisperProcessor.from_pretrained("openai/whisper-base")
        DEMO_ASSETS['asr_model'] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        DEMO_ASSETS['asr_available'] = True
        print(" Hugging Face Whisper ASR Model Loaded Locally.")
    except Exception as e:
        print(f" ERROR loading ASR model: {e}")
        DEMO_ASSETS['asr_available'] = False

    DEMO_ASSETS['xlit_engine_available'] = True
    print(" Indic Transliteration Logic Initialized.")

    DEMO_ASSETS['tts_available'] = True
    print(" TTS functionality disabled due to system download issues.")

    try:
        # Load NER pipeline, fine-tune or choose appropriate model as needed
        DEMO_ASSETS['ner_pipeline'] = pipeline("ner", grouped_entities=True)
        print(" NER Pipeline Loaded.")
    except Exception as e:
        print(f" ERROR loading NER pipeline: {e}")
        DEMO_ASSETS['ner_pipeline'] = None

    return DEMO_ASSETS

def normalize_transcript_names(transcript: str):
    if not DEMO_ASSETS.get('xlit_engine_available'):
        return transcript

    words = transcript.split()
    normalized_words = []

    SRC_SCHEME = ITRANS
    TGT_SCHEME = ITRANS

    for word in words:
        # Transliterate capitalized words (names)
        if word[0].isupper() and len(word) > 2 and re.match(r'^[A-Za-z]+$', word):
            try:
                normalized_word = transliterate(word, SRC_SCHEME, TGT_SCHEME)
                if normalized_word and re.match(r'^[A-Za-z\s]+$', normalized_word):
                    normalized_words.append(normalized_word.capitalize())
                    continue
            except Exception:
                pass
        normalized_words.append(word)

    normalized_transcript = " ".join(normalized_words)

    # Use NER to extract names and other entities for correction if NER exists
    ner = DEMO_ASSETS.get('ner_pipeline')
    if ner:
        ner_results = ner(normalized_transcript)
        # Example: Replace person entity words with confirmed spelling if needed
        # This can be extended with correction logic or feedback loop

    return normalized_transcript

def run_asr_on_file(filename: str, assets: dict):
    if not assets.get('asr_available'):
        return "ERROR: ASR Model not initialized.", 0.0, 0.0, 0.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.normpath(os.path.join(script_dir, os.pardir, "tests", "sample_audio", filename))

    print(f"Resolved audio file path: {audio_file_path}")
    print(f"File exists: {os.path.exists(audio_file_path)}")

    if not os.path.exists(audio_file_path):
        return f"ERROR: File not found at {audio_file_path}", 0.0, 0.0, 0.0

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
        input_features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_features
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        latency = time.time() - start_time

        final_transcript = normalize_transcript_names(transcription)

        return final_transcript, latency, len(speech) / sampling_rate, len(speech) / sampling_rate
    except Exception as e:
        latency = time.time() - start_time
        return f"ERROR: ASR Local Inference Failed. {e}", latency, len(speech) / sampling_rate, len(speech) / sampling_rate

def generate_voice_confirmation(extracted_data_json: dict, assets: dict, output_path: str = "/tmp/confirmation_output.wav"):
    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")

    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"Processing used the {extracted_data_json.get('extraction_method', 'AI')} path."
    )

    print(f"Bot Confirmation: {confirmation_message}")

    if not assets.get('tts_available'):
        print("TTS functionality is disabled, no audio file generated.")
        return None

    # Use gTTS to generate audio
    try:
        tts = gTTS(confirmation_message, lang='en')
        tts.save(output_path)
        print(f"TTS audio saved at {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to generate TTS audio: {e}")
        return None
