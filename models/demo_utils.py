# models/demo_utils.py
import torch
import torchaudio
import whisper
import time
import os
import soundfile as sf
from TTS.api import TTS

# --- ASR MODEL (WHISPER) ---
def setup_whisper_model(model_size="medium"):
    """Initializes and returns the Whisper ASR model."""
    asr_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load Whisper model (medium for best balance if available)
        whisper_model = whisper.load_model(model_size, device=asr_device)
        return whisper_model, asr_device
    except Exception as e:
        print(f" Whisper Model Load Failed: {e}. Falling back to 'small'.")
        return whisper.load_model("small", device=asr_device), asr_device

def run_asr_on_file(filename: str, whisper_model, asr_device):
    """
    Transcribes audio from the specified file located in the tests/sample_audio directory.
    
    Args:
        filename (str): The name of the audio file (e.g., 'command.wav').
    """
    # CRITICAL FIX: Construct the absolute path from the project root.
    # We navigate two levels up from 'models' to the project root, then down to 'tests/sample_audio'.
    
    # Get the directory of the current script (models/demo_utils.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path construction: project_root/tests/sample_audio/filename
    audio_file_path = os.path.join(
        script_dir,
        os.pardir,  # Moves up to the 'voice-bot-extraction' root
        "tests",
        "sample_audio",
        filename
    )
    
    # Normalize the path to handle cross-platform issues
    audio_file_path = os.path.normpath(audio_file_path)

    if whisper_model is None or not os.path.exists(audio_file_path):
        print(f"ERROR: Audio file not found at {audio_file_path}")
        return "ERROR", 0.0, 0.0, 0.0 # Added audio_duration to return 4 items

    # Get audio duration
    try:
        audio_info = sf.info(audio_file_path)
        audio_duration = audio_info.duration
    except Exception:
        audio_duration = 0.0
        
    start_time = time.time()
    result = whisper_model.transcribe(audio_file_path, fp16=(asr_device == "cuda"))
    latency_sec = time.time() - start_time
    transcript = result["text"].strip()
    
    return transcript, latency_sec, audio_duration, audio_duration # Return the expected 4 items (transcript, latency, rtf, duration)


# --- TTS MODEL (SILERO) ---
def setup_tts_model():
    """Initializes and returns the Silero TTS model and related components."""
    tts_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model_id = 'silero_tts'
        language = 'en'
        speaker = 'lj_16khz'
        # Load the model from PyTorch Hub
        model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model=model_id,
            language=language,
            speaker=speaker
        )
        model = model.to(tts_device)
        return model, sample_rate, apply_tts
    except Exception as e:
        print(f" Silero TTS Model Load Failed: {e}. TTS functionality unavailable.")
        return None, None, None

def generate_voice_confirmation(extracted_data_json: dict, tts_components, output_path: str = "/tmp/confirmation_output.wav"):
    """Generates a voice confirmation based on the successful JSON extraction."""
    model, sample_rate, apply_tts = tts_components
    if model is None: return None
        
    lead_name = extracted_data_json.get("lead_name", "the client")
    visit_type = extracted_data_json.get("visit_type", "meeting")
    date = extracted_data_json.get("date", "N/A")
    
    confirmation_message = (
        f"Success! The {visit_type} visit with {lead_name} is scheduled for {date}. "
        f"Processing used the {extracted_data_json.get('extraction_method', 'AI')} path."
    )
    
    audio_tensor = apply_tts(texts=[confirmation_message], model=model, sample_rate=sample_rate, symbols=model.symbols, device=model.device)[0]
    torchaudio.save(output_path, audio_tensor.unsqueeze(0).cpu(), sample_rate=sample_rate)
    return output_path