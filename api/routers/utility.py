# api/routers/utility.py
import time
from fastapi import APIRouter, HTTPException, UploadFile, File
import httpx 
from typing import List, Dict, Any, Union

# Import core business logic and schemas
from api.schemas import TranscriptInput, VisitExtractionResponse, AttendeeSearchResult
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback 
from models.demo_utils import setup_demo_assets, run_asr_on_bytes, normalize_transcript_names # ASR/Transliteration

router = APIRouter()

# --- MOCK DATABASE FOR SEARCH DEMO ---
MOCK_ATTENDEES: List[Dict[str, str]] = [
    {"id": "L001", "name": "Anjori Sarabhai"},
    {"id": "L002", "name": "Rajiv Sharma"},
    {"id": "L003", "name": "Akash Gupta"},
    {"id": "L004", "name": "Rahul Verma"},
    {"id": "L005", "name": "Amit Shah"},
]

# Initialize ASR assets once globally
ASR_ASSETS = setup_demo_assets()


# ==========================================================
# 1. AUDIO UPLOAD AND TRANSCRIPTION ENDPOINT
# ==========================================================

@router.post("/transcribe-audio")
async def transcribe_audio_file(audio_file: UploadFile = File(...)):
    """
    Receives raw audio data, runs local Whisper ASR and Indic Transliteration, 
    and returns the clean text transcript, latency, and duration.
    """
    if not ASR_ASSETS.get('asr_available'):
        raise HTTPException(status_code=503, detail="ASR Service is not loaded on the backend. Check model download.")

    try:
        # Read the audio file bytes asynchronously
        audio_bytes = await audio_file.read()
        
        # Run local ASR and Transliteration (latency and duration are calculated inside run_asr_on_bytes)
        # Note: The function name MUST match the import: run_asr_on_bytes
        transcript, asr_latency, duration, _ = run_asr_on_bytes(audio_bytes, ASR_ASSETS)

        if transcript.startswith("ERROR"):
            raise HTTPException(status_code=500, detail=transcript)
            
        # The latency is now correctly returned to the frontend
        return {
            "transcript": transcript,
            "asr_latency": round(asr_latency * 1000, 2), # Returning latency in milliseconds for readability
            "duration": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription Failed: {e}")


# ==========================================================
# 2. CORE VOICE EXTRACTION ENDPOINT (Main Logic)
# ==========================================================

@router.post("/extract-data", response_model=VisitExtractionResponse)
async def extract_voice_data(input_data: TranscriptInput):
    """
    Receives a transcript and executes the Hybrid Extraction Pipeline.
    """
    transcript = input_data.transcript
    
    # 1. Attempt FAST PATH (NLP)
    nlp_data, nlp_latency = run_nlp_fast_path(transcript)
    
    if nlp_data:
        total_latency = nlp_latency
        method_used = "NLP_RULES"
        extracted_data = nlp_data
        success = True
        
    else:
        # 2. FALLBACK to Mercury dLLM
        try:
            llm_data, llm_latency = await extract_via_mercury_fallback(transcript) 
            
            total_latency = nlp_latency + llm_latency
            method_used = "MERCURY_dLLM"
            extracted_data = llm_data
            success = llm_data is not None
        
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"LLM Fallback Service Error: {e.__class__.__name__}: {e}"
            )

    if not success:
        raise HTTPException(status_code=422, detail="Extraction failed for both NLP and LLM paths.")

    # --- HIL Validation (CRM Check) ---
    lead_name = extracted_data.get("lead_name", "").strip()
    validation_status = "N/A"
    h_i_l_prompt = ""
    
    if lead_name:
        if any(att["name"].lower() == lead_name.lower() for att in MOCK_ATTENDEES):
            validation_status = "EXISTS"
            h_i_l_prompt = f"Welcome back, {lead_name}! Details loaded from CRM."
        else:
            validation_status = "NEW_CONTACT_REQUIRED"
            h_i_l_prompt = "Name not found in CRM. Please provide Email and Phone Number."

    # Final result compilation
    extracted_data["validation_status"] = validation_status
    extracted_data["h_i_l_prompt"] = h_i_l_prompt

    # Return the structured response
    return VisitExtractionResponse(
        data=extracted_data,
        method=method_used,
        latency_sec=total_latency,
        success=success
    )


# ==========================================================
# 3. ATTENDEE SEARCH AND VALIDATION ENDPOINTS
# ==========================================================

@router.get("/search/attendees", response_model=List[AttendeeSearchResult], tags=["Search"])
async def search_attendees(query: str):
    """
    Performs real-time, predictive search for attendees.
    """
    if not query: return []
    query_lower = query.lower()
    
    results = [
        AttendeeSearchResult(id=att["id"], full_name=att["name"])
        for att in MOCK_ATTENDEES
        if att["name"].lower().startswith(query_lower)
    ]
    return results[:5]


@router.post("/search/validate_name", tags=["Search"])
async def validate_name(payload: Dict[str, Any]):
    """
    Checks if a full name already exists in the system.
    """
    name_to_check = payload.get("lead_name", "").strip().lower()
    
    for att in MOCK_ATTENDEES:
        if att["name"].lower() == name_to_check:
            return {"status": "EXISTS", "user_id": att["id"]}
            
    return {"status": "NEW_CONTACT", "user_id": None}