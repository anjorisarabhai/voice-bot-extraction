# api/schemas.py
from pydantic import BaseModel, Field

# Schema for the incoming voice transcript from the frontend
class TranscriptInput(BaseModel):
    """Data structure for incoming voice transcript."""
    transcript: str = Field(..., description="The ASR-generated text transcription of the user's voice command.")

# Schema for the data returned by the backend (includes metrics)
class VisitExtractionResponse(BaseModel):
    """Data structure for the final extracted data and performance metrics."""
    data: dict = Field(..., description="The structured JSON data (VisitDetails) extracted by the LLM/NLP.")
    method: str = Field(..., description="The method used: 'NLP_RULES' or 'MERCURY_dLLM'.")
    latency_sec: float = Field(..., description="The total latency for the extraction process (in seconds).")
    success: bool = Field(..., description="True if extraction was successful.")