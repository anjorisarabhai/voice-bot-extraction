# models/schema.py
from pydantic import BaseModel, Field
from typing import Literal

# --- Core Schema for Visit Scheduling (SIMPLIFIED) ---
class VisitDetails(BaseModel):
    """The strict data schema required by the Log Visit form."""
    
    # KEEP: Event/Meeting Name (Maps to 'title')
    title: str = Field(description="A brief summary/title of the visit's purpose.")
    
    # KEEP: Visit Type (Needed for internal classification, even if not on main form)
    visit_type: Literal["OPERATION", "BUSINESS", "N/A"] = Field(description="Must be one of the allowed Visit Types: OPERATION or BUSINESS.")
    
    # KEEP: Attendees (Maps to 'lead_name')
    lead_name: str = Field(description="The full name of the lead/client.")
    
    # REMOVED: date, start_time, end_time (No longer visible on the form)
    
    # KEEP: New Contact Info (If Attendees don't exist)
    email: str = Field(description="Extracted email address, if mentioned. Default to 'N/A'.")
    phone_number: str = Field(description="Extracted phone number, if mentioned. Default to 'N/A'.")

# --- Schema for Voice Note Summarization remains the same ---
class NoteSummary(BaseModel):
    """Schema for summarizing a voice note."""
    lead_name: str = Field(description="The full name of the lead mentioned.")
    summary_of_note: str = Field(description="A concise summary of the key points in the note (20 words max).")
    action_required: Literal["Yes", "No"] = Field(description="Set to 'Yes' if the note implies a future action is needed.")