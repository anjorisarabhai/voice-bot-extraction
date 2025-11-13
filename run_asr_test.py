# run_asr_test.py
import os
import sys
import time
import torch
import warnings

# CRITICAL: Add the project root to the path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress the numerous Whisper warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module='whisper')

# Import the necessary utility functions
from models.demo_utils import setup_whisper_model, run_asr_on_file

# --- CONFIGURATION ---
# !!! IMPORTANT: PUT YOUR AUDIO FILENAME HERE !!!
# 1. Place your test audio file (e.g., 'scheduling.wav') inside the: tests/sample_audio/ directory.
# 2. Update the filename below to match your file.
AUDIO_FILENAME = "test_command.wav" 
# ---------------------

def run_asr_benchmark():
    """Initializes Whisper and runs transcription on the sample audio file."""
    print("---  ASR BENCHMARK START ---")
    
    # Check if GPU is available (Whisper needs this to be fast)
    if not torch.cuda.is_available():
        print(" WARNING: CUDA not detected. Running ASR on CPU will be extremely slow.")

    # 1. Setup Model
    whisper_model, asr_device = setup_whisper_model(model_size="small")

    if whisper_model is None:
        print(" ERROR: Whisper model could not be loaded. Check dependencies.")
        return

    # 2. Run ASR and measure performance
    print(f"Loading and transcribing file: {AUDIO_FILENAME}")
    
    # The run_asr_on_file function handles the path construction internally.
    transcript, latency, duration, _ = run_asr_on_file(
        AUDIO_FILENAME, 
        whisper_model, 
        asr_device
    )

    if transcript == "ERROR":
        print(" ASR failed. Check if the audio file exists at the path shown in the console.")
        return

    # 3. Report Results
    print("\n--- ASR RESULTS ---")
    print(f" Input Processed. Duration: {duration:.2f}s")
    print(f" ASR Latency: {latency:.4f}s")
    print(f" Final Transcript: **\"{transcript}\"**")
    print("-------------------")


if __name__ == "__main__":
    run_asr_benchmark()