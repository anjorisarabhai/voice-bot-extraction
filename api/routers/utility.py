# api/routers/utility.py
import time
import json
from fastapi import APIRouter, HTTPException, Depends
import httpx # Asynchronous HTTP client
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import core business logic and schemas
from api.schemas import TranscriptInput, VisitExtractionResponse
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback # Already async
from models.schema import VisitDetails # Used for validation


router = APIRouter()

# --- NEW SCHEMA FOR ATTENDEE SEARCH ---
class AttendeeSearchResult(BaseModel):
    id: str = Field(..., description="Unique ID of the attendee.")
    full_name: str = Field(..., description="The matching name.")

# --- MOCK DATABASE FOR SEARCH DEMO ---
MOCK_ATTENDEES = [
    {"id": "L001", "name": "Anjori Sarabhai"},
    {"id": "L002", "name": "Rajiv Sharma"},
    {"id": "L003", "name": "Akash Gupta"},
    {"id": "L004", "name": "Rahul Verma"},
    {"id": "L005", "name": "Amit Shah"},
    {"id": "L006", "name": "Dr. Patel"},
]



# 1. CORE VOICE EXTRACTION ENDPOINT


@router.post("/extract-data", response_model=VisitExtractionResponse)
async def extract_voice_data(input_data: TranscriptInput):
    """
    Receives a transcript and executes the Hybrid Extraction Pipeline.
    Prioritizes the fast NLP path and falls back to Mercury dLLM for complex inputs.
    """
    transcript = input_data.transcript
    
    # 1. Attempt FAST PATH (NLP)
    # run_nlp_fast_path is synchronous and returns data or None
    nlp_data, nlp_latency = run_nlp_fast_path(transcript)
    
    if nlp_data:
        # NLP SUCCESS: Return data from the fast path
        total_latency = nlp_latency
        method_used = "NLP_RULES"
        extracted_data = nlp_data
        success = True
        
    else:
        # 2. FALLBACK to Mercury dLLM for complex data processing (Async Call)
        try:
            llm_data, llm_latency = await extract_via_mercury_fallback(transcript) 
            
            total_latency = nlp_latency + llm_latency
            method_used = "MERCURY_dLLM"
            extracted_data = llm_data
            success = llm_data is not None
        
        except httpx.HTTPStatusError as e:
            # Catches API errors (401, 429, etc.)
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"MERCURY API Error: {e.response.status_code} - {e.response.text[:100]}"
            )
        except Exception as e:
            # Catches network, parsing, or generic exceptions
            raise HTTPException(
                status_code=500, 
                detail=f"LLM Fallback Service Error: {e.__class__.__name__}: {e}"
            )

    if not success:
        # Final failure if LLM returned None after all attempts
        raise HTTPException(status_code=422, detail="Extraction failed for both NLP and LLM paths.")

    # Return the structured response
    return VisitExtractionResponse(
        data=extracted_data,
        method=method_used,
        latency_sec=total_latency,
        success=success
    )



# 2. ATTENDEE SEARCH AND VALIDATION ENDPOINTS


# --- Endpoint for Real-Time Predictive Search (Chunk Validation) ---
@router.get("/search/attendees", response_model=List[AttendeeSearchResult], tags=["Search"])
async def search_attendees(query: str):
    """
    Performs real-time, predictive search for attendees based on a partial chunk (query).
    (Simulates Trigram index search in Python for speed)
    """
    if not query:
        return []

    query_lower = query.lower()
    
    # 1. SIMULATED INDEX SEARCH (Prefix Matching for speed)
    results = [
        AttendeeSearchResult(id=att["id"], full_name=att["name"])
        for att in MOCK_ATTENDEES
        if att["name"].lower().startswith(query_lower)
    ]

    # 2. Return limited results for performance (e.g., top 5)
    return results[:5]


# --- Endpoint for Full Name Validation (Exists / New Contact Decision) ---
@router.post("/search/validate_name", tags=["Search"])
async def validate_name(payload: Dict[str, Any]):
    """
    Checks if a full name (extracted via voice) already exists in the system.
    This dictates whether the frontend should prompt for new contact info (email/phone).
    """
    name_to_check = payload.get("lead_name", "").strip().lower()
    
    # 1. SIMULATED DATABASE LOOKUP (Exact Match)
    for att in MOCK_ATTENDEES:
        if att["name"].lower() == name_to_check:
            # Attendee exists
            return {"status": "EXISTS", "user_id": att["id"]}
    
    # 2. New Contact Decision
    return {"status": "NEW_CONTACT", "user_id": None}