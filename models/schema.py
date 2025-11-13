# models/schema.py
from pydantic import BaseModel, Field
from typing import Literal

# --- Core Schema for Visit Scheduling ---
class VisitDetails(BaseModel):
    """The strict data schema required by the Log Visit form."""
    title: str = Field(description="A brief summary/title of the visit's purpose.")
    visit_type: Literal["OPERATION", "BUSINESS", "N/A"] = Field(description="Must be one of the allowed Visit Types: OPERATION or BUSINESS.")
    lead_name: str = Field(description="The full name of the lead/client.")
    date: str = Field(description="The date of the visit in YYYY-MM-DD format.")
    start_time: str = Field(description="The exact start time in 24-hour format (HH:MM).")
    end_time: str = Field(description="The exact end time in 24-hour format (HH:MM). If not specified, set to 'N/A'.")
    email: str = Field(description="Extracted email address, if mentioned. Default to 'N/A'.")
    phone_number: str = Field(description="Extracted phone number, if mentioned. Default to 'N/A'.")

# --- Schema for Voice Note Summarization (Phase 2 logic) ---
class NoteSummary(BaseModel):
    """Schema for summarizing a voice note."""
    lead_name: str = Field(description="The full name of the lead mentioned.")
    summary_of_note: str = Field(description="A concise summary of the key points in the note (20 words max).")
    action_required: Literal["Yes", "No"] = Field(description="Set to 'Yes' if the note implies a future action is needed.")