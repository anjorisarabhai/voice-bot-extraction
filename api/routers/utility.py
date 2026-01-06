# api/routers/utility.py
import time
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any

# 1. New Import: Pulling the database hook
from api.database import get_lead_by_name 

# Import core business logic and schemas
from api.schemas import TranscriptInput, VisitExtractionResponse, AttendeeSearchResult
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback 
from models.demo_utils import setup_demo_assets, run_asr_on_bytes # ASR/Transliteration

router = APIRouter()

# Initialize ASR assets once globally
ASR_ASSETS = setup_demo_assets()


# ==========================================================
# 1. AUDIO UPLOAD AND TRANSCRIPTION ENDPOINT
# ==========================================================

@router.post("/transcribe-audio")
async def transcribe_audio_file(audio_file: UploadFile = File(...)):
    """
    Receives raw audio data, runs local Whisper ASR and returns clean text.
    """
    if not ASR_ASSETS.get('asr_available'):
        raise HTTPException(status_code=503, detail="ASR Service is not loaded.")

    try:
        audio_bytes = await audio_file.read()
        
        # Run local ASR (latency/duration calculated inside run_asr_on_bytes)
        transcript, asr_latency, duration, _ = run_asr_on_bytes(audio_bytes, ASR_ASSETS)

        if transcript.startswith("ERROR"):
            raise HTTPException(status_code=500, detail=transcript)
            
        return {
            "transcript": transcript,
            "asr_latency": round(asr_latency * 1000, 2), # ms
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
            raise HTTPException(status_code=500, detail=f"LLM Fallback Error: {e}")

    if not success:
        raise HTTPException(status_code=422, detail="Extraction failed.")

    # --- UPDATED HIL Validation & Data Enrichment ---
    lead_name = extracted_data.get("lead_name", "").strip()
    validation_status = "N/A"
    h_i_l_prompt = ""
    
    # Check the lead name against the database (Mock or Real)
    db_record = await get_lead_by_name(lead_name)
    
    if db_record:
        # DATA ENRICHMENT: Map CRM details to the response
        validation_status = "EXISTS"
        extracted_data["user_id"] = db_record.get("id")
        
        # Override transcript "N/A" with actual CRM contact info
        extracted_data["email"] = db_record.get("email", "N/A")
        extracted_data["phone_number"] = db_record.get("phone", "N/A")
        
        h_i_l_prompt = f"Welcome back, {lead_name}! Details loaded from CRM."
    else:
        # NO MATCH: Keep existing extracted data (likely N/As) and prompt for input
        validation_status = "NEW_CONTACT_REQUIRED"
        h_i_l_prompt = "Name not found in CRM. Please provide Email and Phone Number."

    # Final result compilation
    extracted_data["validation_status"] = validation_status
    extracted_data["h_i_l_prompt"] = h_i_l_prompt

    # Return structured response
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
    Predictive search for attendees using the centralized Mock DB.
    """
    if not query: return []
    query_lower = query.lower()
    
    # Importing from database file to maintain single source of truth
    from api.database import MOCK_ATTENDEES
    results = [
        AttendeeSearchResult(id=att["id"], full_name=att["name"])
        for att in MOCK_ATTENDEES
        if att["name"].lower().startswith(query_lower)
    ]
    return results[:5]


@router.post("/search/validate_name", tags=["Search"])
async def validate_name(payload: Dict[str, Any]):
    """
    Checks if a full name already exists using the Database Hook.
    """
    name_to_check = payload.get("lead_name", "").strip()
    
    # Call the database layer
    db_record = await get_lead_by_name(name_to_check)
    
    if db_record:
        return {
            "status": "EXISTS", 
            "user_id": db_record.get("id"),
            "email": db_record.get("email"),
            "phone": db_record.get("phone")
        }
            
    return {"status": "NEW_CONTACT", "user_id": None}