# api/routers/utility.py
from fastapi import APIRouter, HTTPException
from api.schemas import TranscriptInput, VisitExtractionResponse
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback # Now an async function
import httpx # Import httpx for potential error handling

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
    # Note: run_nlp_fast_path is synchronous and does not need await
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
            llm_data, llm_latency = await extract_via_mercury_fallback(transcript) # <-- CORRECT AWAIT CALL
            
            total_latency = nlp_latency + llm_latency
            method_used = "MERCURY_dLLM"
            extracted_data = llm_data
            success = llm_data is not None
        
        except httpx.HTTPStatusError as e:
            # Catches errors like 401 Unauthorized, 404 Not Found, 429 Rate Limit
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"MERCURY API Error: {e.response.status_code} - {e.response.text[:100]}"
            )
        except Exception as e:
            # Catches general connection or parsing errors (JSON/Pydantic validation)
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