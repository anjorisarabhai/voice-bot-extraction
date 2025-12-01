# api/routers/utility.py
from fastapi import APIRouter, HTTPException
from api.schemas import TranscriptInput, VisitExtractionResponse
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback # We will only use Mercury here

router = APIRouter()

# --- Endpoint to run the core Hybrid Extraction Pipeline ---
@router.post("/extract-data", response_model=VisitExtractionResponse)
async def extract_voice_data(input_data: TranscriptInput):
    """
    Receives a transcript and executes the Hybrid Extraction Pipeline.
    Prioritizes the fast NLP path and falls back to Mercury dLLM for complex inputs.
    """
    transcript = input_data.transcript
    
    # 1. Attempt FAST PATH (NLP)
    nlp_data, nlp_latency = run_nlp_fast_path(transcript)
    
    if nlp_data:
        # NLP SUCCESS: Return data from the fast path
        total_latency = nlp_latency
        method_used = "NLP_RULES"
        extracted_data = nlp_data
        success = True
        
    else:
        # 2. FALLBACK to Mercury dLLM for complex data processing
        try:
            # Note: This is an async I/O operation (API call), so we must use 'await'
            llm_data, llm_latency = await extract_via_mercury_fallback(transcript)
            
            total_latency = nlp_latency + llm_latency
            method_used = "MERCURY_dLLM"
            extracted_data = llm_data
            success = llm_data is not None
        
        except Exception as e:
            # Handle potential API timeouts or connection errors
            raise HTTPException(status_code=503, detail=f"LLM Fallback Service Error: {e}")

    if not success:
        raise HTTPException(status_code=422, detail="Extraction failed for both NLP and LLM paths.")

    # Return the structured response
    return VisitExtractionResponse(
        data=extracted_data,
        method=method_used,
        latency_sec=total_latency,
        success=success
    )